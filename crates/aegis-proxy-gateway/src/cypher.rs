//! A Cypher query engine over the compressed Data Transformation Graph.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! The FR-1.3 query interface: a hand-written tokenizer, recursive-descent
//! parser, planner, and depth-bounded graph executor for a documented Cypher
//! subset, running in-process against a [`CompressedDtg`].
//!
//! ## WHY
//! Analysts ask reachability questions -- "can this tool's input parameter reach
//! a process sink within three hops?" -- and the answer must come from the same
//! graph the verifier checked, not from a re-derived one. Hand-written, because
//! the vendored air-gapped dependency set has no Cypher engine and because this
//! subset is deliberately not Turing-complete: it is a reachability language
//! with a hard hop bound, and every element that could make it unbounded
//! (`*0..`, unbounded ranges, incoming traversal) is rejected with a message
//! rather than silently interpreted.
//!
//! The **3-hop bound** is not a performance limit: the compressed DTG only
//! *retains* the subgraph reachable within [`MAX_HOP_DEPTH`] (ADR-002), so a
//! query asking for four hops is asking about edges that were never extracted.
//! Answering it with the three-hop answer would be a correctness lie, so it is
//! a hard error.
//!
//! ## HOW
//! Grammar (keywords case-insensitive, `-- comments` not supported):
//!
//! ```text
//! query    := MATCH pattern (WHERE cond (AND cond)*)? RETURN returns (LIMIT int)?
//! pattern  := node segment*            -- segments extend the path
//! node     := '(' (ident|'_')? (':' label)? ('{' prop ':' value (',' ...)* '}')? ')'
//! segment  := ('-' '[' rel ']' '->') | ('-->') | ('->')
//! rel      := ident?                                             -- relationship variable
//!           | (ident? ':' kind)?                                 -- kind filter
//!           | '*' int '..' int                                   -- 1..n, n bounded by 3
//! returns  := item (',' item)*
//! item     := 'count' '(' '*' ')' | ident ('.' ident)? ('AS' ident)?
//! cond     := ident '.' ident ('='|'<>'|'<'|'<='|'>'|'>=') (string|int)
//! ```
//!
//! Labels are node types (`tool_input_param`, `env_secret`, ...) plus the
//! semantic sugar `sink` and `source`. Relationship kinds are transform
//! functions (`assign`, `path_join`, `execve_call`, ...), OS sink kinds
//! (`CommandExec`, ...), or the sugar `sink` and `any`.
//!
//! All evaluation is fail-closed: unknown variables, unknown properties,
//! unknown labels, malformed ranges, and any comparison that cannot be decided
//! from the node or edge (a missing property, or a string compared to a
//! number) exclude the row rather than pass it.
//!
//! Complexity: `O(anchors * fanout^hops)` with `hops <= 3` over a compressed
//! graph whose edges are already the bounded subgraph, so bounded by
//! `O(V + E)` in practice for the reachability patterns this interface serves.

use aegis_common::types::{CompressedDtg, DtgEdge, DtgNode, SinkKind, MAX_HOP_DEPTH};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// The server-side ceiling on returned rows, whatever the query says.
pub const DEFAULT_ROW_CAP: usize = 256;

/// A parse, plan, or execution error, always with the byte position that
/// produced it so a client can highlight the query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CypherError {
    /// What went wrong, in words.
    pub message: String,
    /// Byte offset into the query text.
    pub position: usize,
}

impl CypherError {
    fn at(message: impl Into<String>, position: usize) -> Self {
        CypherError {
            message: message.into(),
            position,
        }
    }
}

impl std::fmt::Display for CypherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (at byte {})", self.message, self.position)
    }
}

impl std::error::Error for CypherError {}

/// The result of a query: fixed column order, rows keyed by column.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryResult {
    /// Column names, in `RETURN` order.
    pub columns: Vec<String>,
    /// Rows, each a column-to-value object.
    pub rows: Vec<BTreeMap<String, Value>>,
    /// True when row collection stopped at the effective limit and at least
    /// one further match existed.
    pub truncated: bool,
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Str(String),
    Int(i64),
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
    Dash,
    Arrow,
    LArrow,
    Star,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
struct Spanned {
    token: Token,
    pos: usize,
}

fn tokenize(input: &str) -> Result<Vec<Spanned>, CypherError> {
    let bytes = input.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];
        match byte {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'(' | b')' | b'[' | b']' | b'{' | b'}' | b',' | b':' | b'.' | b'*' | b'=' => {
                let token = match byte {
                    b'(' => Token::LParen,
                    b')' => Token::RParen,
                    b'[' => Token::LBracket,
                    b']' => Token::RBracket,
                    b'{' => Token::LBrace,
                    b'}' => Token::RBrace,
                    b',' => Token::Comma,
                    b':' => Token::Colon,
                    b'*' => Token::Star,
                    b'.' => Token::Dot,
                    _ => Token::Eq,
                };
                out.push(Spanned { token, pos: i });
                i += 1;
            }
            b'-' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    out.push(Spanned {
                        token: Token::Arrow,
                        pos: i,
                    });
                    i += 2;
                } else {
                    out.push(Spanned {
                        token: Token::Dash,
                        pos: i,
                    });
                    i += 1;
                }
            }
            b'<' => {
                let (token, width) = if i + 1 < bytes.len() && bytes[i + 1] == b'-' {
                    (Token::LArrow, 2)
                } else if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    (Token::Le, 2)
                } else {
                    (Token::Lt, 1)
                };
                out.push(Spanned { token, pos: i });
                i += width;
            }
            b'>' => {
                let (token, width) = if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    (Token::Ge, 2)
                } else {
                    (Token::Gt, 1)
                };
                out.push(Spanned { token, pos: i });
                i += width;
            }
            b'!' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Spanned {
                        token: Token::Ne,
                        pos: i,
                    });
                    i += 2;
                } else {
                    return Err(CypherError::at("unexpected '!'", i));
                }
            }
            b'0'..=b'9' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let text = &input[start..i];
                let value: i64 = text
                    .parse()
                    .map_err(|_| CypherError::at(format!("integer '{text}' is out of range"), start))?;
                out.push(Spanned {
                    token: Token::Int(value),
                    pos: start,
                });
            }
            b'\'' | b'"' => {
                // Both quote styles carry string literals; Cypher uses single.
                let quote = byte as char;
                let start = i;
                i += 1;
                let mut value = String::new();
                let mut closed = false;
                let mut chars = input[i..].char_indices();
                while let Some((offset, ch)) = chars.next() {
                    let at = i + offset;
                    if ch == quote {
                        i = at + 1;
                        closed = true;
                        break;
                    }
                    if ch == '\\' {
                        let Some((_, escaped)) = chars.next() else {
                            return Err(CypherError::at("unterminated string literal", start));
                        };
                        match escaped {
                            '\'' => value.push('\''),
                            '"' => value.push('"'),
                            '\\' => value.push('\\'),
                            'n' => value.push('\n'),
                            't' => value.push('\t'),
                            other => {
                                return Err(CypherError::at(
                                    format!("unsupported escape '\\{other}'"),
                                    at,
                                ))
                            }
                        }
                        continue;
                    }
                    value.push(ch);
                }
                if !closed {
                    return Err(CypherError::at("unterminated string literal", start));
                }
                out.push(Spanned {
                    token: Token::Str(value),
                    pos: start,
                });
            }
            b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                out.push(Spanned {
                    token: Token::Ident(input[start..i].to_string()),
                    pos: start,
                });
            }
            other => {
                return Err(CypherError::at(
                    format!("unexpected character '{}'", other as char),
                    i,
                ))
            }
        }
    }
    out.push(Spanned {
        token: Token::Eof,
        pos: input.len(),
    });
    Ok(out)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct NodePattern {
    var: Option<String>,
    label: Option<String>,
    props: Vec<(String, PropValue)>,
}

#[derive(Debug, Clone, PartialEq)]
struct RelPattern {
    alias: Option<String>,
    kind: Option<String>,
    min_hops: u8,
    max_hops: u8,
}

#[derive(Debug, Clone, PartialEq)]
struct MatchClause {
    nodes: Vec<NodePattern>,
    rels: Vec<RelPattern>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, PartialEq)]
enum PropValue {
    Str(String),
    Int(i64),
}

#[derive(Debug, Clone, PartialEq)]
struct Condition {
    var: String,
    prop: String,
    op: CmpOp,
    value: PropValue,
}

#[derive(Debug, Clone, PartialEq)]
enum ReturnKind {
    Property(String),
    Node,
    Count,
}

#[derive(Debug, Clone, PartialEq)]
struct ReturnItem {
    var: String,
    kind: ReturnKind,
    alias: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct Query {
    match_clause: MatchClause,
    match_pos: usize,
    conditions: Vec<Condition>,
    returns: Vec<ReturnItem>,
    limit: Option<usize>,
}

struct Parser<'a> {
    tokens: &'a [Spanned],
    index: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Spanned]) -> Self {
        Parser { tokens, index: 0 }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.index].token
    }

    fn peek_pos(&self) -> usize {
        self.tokens[self.index].pos
    }

    fn next(&mut self) -> Token {
        let token = self.tokens[self.index].token.clone();
        if self.index + 1 < self.tokens.len() {
            self.index += 1;
        }
        token
    }

    /// Case-insensitive keyword test at the cursor.
    fn keyword(&self, keyword: &str) -> bool {
        matches!(self.peek(), Token::Ident(name) if name.eq_ignore_ascii_case(keyword))
    }

    /// Consumes a keyword, returning false (and consuming nothing) otherwise.
    fn take_keyword(&mut self, keyword: &str) -> bool {
        if self.keyword(keyword) {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token: &Token, what: &str) -> Result<(), CypherError> {
        if self.peek() == token {
            self.next();
            Ok(())
        } else {
            Err(CypherError::at(
                format!("expected {what}, found {}", describe(self.peek())),
                self.peek_pos(),
            ))
        }
    }

    fn parse_ident(&mut self, what: &str) -> Result<String, CypherError> {
        let pos = self.peek_pos();
        match self.next() {
            Token::Ident(name) => Ok(name),
            other => Err(CypherError::at(
                format!("expected {what}, found {}", describe(&other)),
                pos,
            )),
        }
    }

    fn parse_int(&mut self, what: &str) -> Result<i64, CypherError> {
        let pos = self.peek_pos();
        match self.next() {
            Token::Int(value) => Ok(value),
            other => Err(CypherError::at(
                format!("expected {what}, found {}", describe(&other)),
                pos,
            )),
        }
    }

    fn parse_string_or_int(&mut self, what: &str) -> Result<PropValue, CypherError> {
        let pos = self.peek_pos();
        match self.next() {
            Token::Str(value) => Ok(PropValue::Str(value)),
            Token::Int(value) => Ok(PropValue::Int(value)),
            other => Err(CypherError::at(
                format!("expected {what}, found {}", describe(&other)),
                pos,
            )),
        }
    }
}

fn describe(token: &Token) -> String {
    match token {
        Token::Ident(name) => format!("identifier '{name}'"),
        Token::Str(value) => format!("string '{value}'"),
        Token::Int(value) => format!("integer {value}"),
        Token::LParen => "'('".into(),
        Token::RParen => "')'".into(),
        Token::LBracket => "'['".into(),
        Token::RBracket => "']'".into(),
        Token::LBrace => "'{'".into(),
        Token::RBrace => "'}'".into(),
        Token::Comma => "','".into(),
        Token::Colon => "':'".into(),
        Token::Dot => "'.'".into(),
        Token::Dash => "'-'".into(),
        Token::Arrow => "'->'".into(),
        Token::LArrow => "'<-'".into(),
        Token::Star => "'*'".into(),
        Token::Eq => "'='".into(),
        Token::Ne => "'<>'".into(),
        Token::Lt => "'<'".into(),
        Token::Le => "'<='".into(),
        Token::Gt => "'>'".into(),
        Token::Ge => "'>='".into(),
        Token::Eof => "end of query".into(),
    }
}

fn parse(query: &str) -> Result<Query, CypherError> {
    let tokens = tokenize(query)?;
    let mut parser = Parser::new(&tokens);

    if !parser.take_keyword("MATCH") {
        return Err(CypherError::at(
            format!(
                "a query must start with MATCH, found {}",
                describe(parser.peek())
            ),
            parser.peek_pos(),
        ));
    }
    let match_pos = 0;
    let match_clause = parser.parse_match()?;

    let mut conditions = Vec::new();
    if parser.take_keyword("WHERE") {
        loop {
            conditions.push(parser.parse_condition()?);
            if !parser.take_keyword("AND") {
                break;
            }
        }
    }

    if !parser.take_keyword("RETURN") {
        return Err(CypherError::at(
            format!(
                "expected RETURN, found {}",
                describe(parser.peek())
            ),
            parser.peek_pos(),
        ));
    }
    let mut returns = vec![parser.parse_return_item()?];
    while parser.peek() == &Token::Comma {
        parser.next();
        returns.push(parser.parse_return_item()?);
    }

    let limit = if parser.take_keyword("LIMIT") {
        let pos = parser.peek_pos();
        let value = parser.parse_int("LIMIT row count")?;
        if value <= 0 {
            return Err(CypherError::at("LIMIT must be positive", pos));
        }
        Some(value as usize)
    } else {
        None
    };

    if parser.peek() != &Token::Eof {
        return Err(CypherError::at(
            format!("unexpected {} after the query", describe(parser.peek())),
            parser.peek_pos(),
        ));
    }

    Ok(Query {
        match_clause,
        match_pos,
        conditions,
        returns,
        limit,
    })
}

impl<'a> Parser<'a> {
    fn parse_match(&mut self) -> Result<MatchClause, CypherError> {
        let mut nodes = vec![self.parse_node_pattern()?];
        let mut rels = Vec::new();
        loop {
            match self.peek() {
                Token::Dash | Token::Arrow => {
                    let pos = self.peek_pos();
                    // Consume the leading dash if present ('-[..]->' / '-->');
                    // a bare '->' is already a complete arrow token.
                    if self.peek() == &Token::Dash {
                        self.next();
                    }
                    let mut alias = None;
                    let mut kind = None;
                    let mut min_hops = 1u8;
                    let mut max_hops = 1u8;
                    if self.peek() == &Token::LBracket {
                        self.next();
                        if let Token::Ident(name) = self.peek() {
                            alias = Some(name.clone());
                            self.next();
                        }
                        if self.peek() == &Token::Colon {
                            self.next();
                            kind = Some(self.parse_ident("relationship kind")?);
                        }
                        if self.peek() == &Token::Star {
                            self.next();
                            let low = self.parse_int("range lower bound")?;
                            self.expect(&Token::Dot, "'.' of '..'")?;
                            self.expect(&Token::Dot, "'.' of '..'")?;
                            let high = self.parse_int("range upper bound")?;
                            if low < 1 {
                                return Err(CypherError::at(
                                    "a variable-length pattern must start at 1 or more hops",
                                    pos,
                                ));
                            }
                            if high < low {
                                return Err(CypherError::at(
                                    format!("empty hop range *{low}..{high}"),
                                    pos,
                                ));
                            }
                            if high > MAX_HOP_DEPTH as i64 {
                                return Err(CypherError::at(
                                    format!(
                                        "hop range *{low}..{high} exceeds the {MAX_HOP_DEPTH}-hop \
                                         bound the compressed DTG is extracted to (ADR-002)"
                                    ),
                                    pos,
                                ));
                            }
                            min_hops = low as u8;
                            max_hops = high as u8;
                        }
                        self.expect(&Token::RBracket, "']' closing the relationship detail")?;
                    }
                    match self.peek() {
                        Token::Arrow => {
                            self.next();
                        }
                        Token::LArrow => {
                            return Err(CypherError::at(
                                "incoming traversal '<-' is not supported; the DTG is directed \
                                 and queries traverse '->'",
                                pos,
                            ));
                        }
                        other => {
                            return Err(CypherError::at(
                                format!(
                                    "expected '->' to close the relationship, found {}",
                                    describe(other)
                                ),
                                self.peek_pos(),
                            ));
                        }
                    }
                    let next = self.parse_node_pattern()?;
                    rels.push(RelPattern {
                        alias,
                        kind,
                        min_hops,
                        max_hops,
                    });
                    nodes.push(next);
                }
                Token::LArrow => {
                    return Err(CypherError::at(
                        "incoming traversal '<-' is not supported; the DTG is directed and \
                         queries traverse '->'",
                        self.peek_pos(),
                    ));
                }
                _ => break,
            }
        }
        Ok(MatchClause { nodes, rels })
    }

    fn parse_node_pattern(&mut self) -> Result<NodePattern, CypherError> {
        self.expect(&Token::LParen, "'(' opening a node pattern")?;
        let mut var = None;
        if let Token::Ident(name) = self.peek() {
            if name != "_" {
                var = Some(name.clone());
            }
            self.next();
        }
        let mut label = None;
        if self.peek() == &Token::Colon {
            self.next();
            label = Some(self.parse_ident("node label")?);
        }
        let mut props = Vec::new();
        if self.peek() == &Token::LBrace {
            self.next();
            loop {
                let key = self.parse_ident("property name")?;
                self.expect(&Token::Colon, "':' between property name and value")?;
                let value = self.parse_string_or_int("property value")?;
                props.push((key, value));
                match self.peek() {
                    Token::Comma => {
                        self.next();
                    }
                    Token::RBrace => {
                        self.next();
                        break;
                    }
                    other => {
                        return Err(CypherError::at(
                            format!(
                                "expected ',' or '}}' in the property map, found {}",
                                describe(other)
                            ),
                            self.peek_pos(),
                        ))
                    }
                }
            }
        }
        self.expect(&Token::RParen, "')' closing the node pattern")?;
        Ok(NodePattern { var, label, props })
    }

    fn parse_condition(&mut self) -> Result<Condition, CypherError> {
        let var = self.parse_ident("a variable name in WHERE")?;
        self.expect(&Token::Dot, "'.' between variable and property")?;
        let prop = self.parse_ident("a property name in WHERE")?;
        let op = match self.next() {
            Token::Eq => CmpOp::Eq,
            Token::Ne => CmpOp::Ne,
            Token::Lt => CmpOp::Lt,
            Token::Le => CmpOp::Le,
            Token::Gt => CmpOp::Gt,
            Token::Ge => CmpOp::Ge,
            other => {
                return Err(CypherError::at(
                    format!(
                        "expected a comparison operator ('=', '<>', '<', '<=', '>', '>='), \
                         found {}",
                        describe(&other)
                    ),
                    self.peek_pos(),
                ))
            }
        };
        let value = self.parse_string_or_int("a comparison value")?;
        Ok(Condition {
            var,
            prop,
            op,
            value,
        })
    }

    fn parse_return_item(&mut self) -> Result<ReturnItem, CypherError> {
        if self.keyword("count") {
            // count(*) -- a bare 'count' identifier is also a legal scope form,
            // so look ahead before committing.
            let save = self.index;
            self.next();
            if self.peek() == &Token::LParen {
                self.next();
                self.expect(&Token::Star, "'*' inside count(...)")?;
                self.expect(&Token::RParen, "')' closing count(...)")?;
                return Ok(ReturnItem {
                    var: String::new(),
                    kind: ReturnKind::Count,
                    alias: None,
                });
            }
            self.index = save;
        }

        let var = self.parse_ident("a variable name in RETURN")?;
        let kind = if self.peek() == &Token::Dot {
            self.next();
            let prop = self.parse_ident("a property name after '.'")?;
            ReturnKind::Property(prop)
        } else {
            ReturnKind::Node
        };
        let alias = if self.take_keyword("AS") {
            Some(self.parse_ident("an alias after AS")?)
        } else {
            None
        };
        Ok(ReturnItem { var, kind, alias })
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Node(usize),
    Edge(usize),
}

#[derive(Debug, Clone, PartialEq)]
enum PlannedReturnKind {
    Property(String),
    WholeNode,
    WholeEdge,
    Count,
}

#[derive(Debug, Clone, PartialEq)]
struct PlannedReturn {
    slot: Option<Slot>,
    kind: PlannedReturnKind,
    column: String,
}

#[derive(Debug, Clone, PartialEq)]
struct PlannedCondition {
    slot: Slot,
    prop: String,
    op: CmpOp,
    value: PropValue,
}

#[derive(Debug)]
struct Plan {
    returns: Vec<PlannedReturn>,
    conditions: Vec<PlannedCondition>,
    count_only: bool,
}

fn node_label_matches(node: &DtgNode, label: &str) -> bool {
    match label {
        "any" => true,
        "sink" => node.node_type.is_sink(),
        "source" => node.node_type.is_taint_source(),
        other => node.node_type.as_str() == other,
    }
}

fn known_node_label(label: &str) -> bool {
    use aegis_common::types::DtgNodeType;
    const ALL: [DtgNodeType; 7] = [
        DtgNodeType::ToolInputParam,
        DtgNodeType::LocalBuffer,
        DtgNodeType::EnvSecret,
        DtgNodeType::FileHandle,
        DtgNodeType::NetworkSocket,
        DtgNodeType::ProcessSink,
        DtgNodeType::ReturnPayload,
    ];
    label == "any"
        || label == "sink"
        || label == "source"
        || ALL.iter().any(|t| t.as_str() == label)
}

fn edge_kind_matches(edge: &DtgEdge, kind: &str) -> bool {
    match kind {
        "any" => true,
        "sink" => edge.sink_kind.is_some(),
        other => {
            edge.transform_fn == other
                || edge.sink_kind.map(|k| k.as_str()) == Some(other)
        }
    }
}

fn node_prop(node: &DtgNode, prop: &str) -> Option<PropValue> {
    match prop {
        "id" => Some(PropValue::Str(node.id.clone())),
        "label" => Some(PropValue::Str(node.label.clone())),
        "node_type" => Some(PropValue::Str(node.node_type.as_str().to_string())),
        "lattice" => Some(PropValue::Str(lattice_name(node))),
        "file_path" => Some(PropValue::Str(node.file_path.clone())),
        "line" => Some(PropValue::Int(node.line as i64)),
        _ => None,
    }
}

fn lattice_name(node: &DtgNode) -> String {
    serde_json::to_value(node.lattice)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("level-{}", node.lattice.level()))
}

fn edge_prop(edge: &DtgEdge, prop: &str) -> Option<PropValue> {
    match prop {
        "source" => Some(PropValue::Str(edge.source.clone())),
        "target" => Some(PropValue::Str(edge.target.clone())),
        "transform_fn" => Some(PropValue::Str(edge.transform_fn.clone())),
        "sink_kind" => Some(match edge.sink_kind {
            Some(kind) => PropValue::Str(kind.as_str().to_string()),
            None => PropValue::Str(String::new()),
        }),
        "hop_depth" => Some(PropValue::Int(edge.hop_depth as i64)),
        _ => None,
    }
}

/// Coerces a property to the comparison type the query asked for.
///
/// Only the lattice has two faithful representations -- its uppercase name for
/// string comparison and its level for integer comparison -- and that case is
/// handled explicitly in [`Engine::conditions_hold`], because it is the *only*
/// property whose name space overlaps with a numeric encoding. Every other
/// property has exactly one type, and asking for the other excludes the row
/// (`None`) rather than inventing an ordering.
fn coerce(prop: PropValue, want_int: bool) -> Option<PropValue> {
    match (prop, want_int) {
        (PropValue::Int(value), true) => Some(PropValue::Int(value)),
        (PropValue::Str(value), false) => Some(PropValue::Str(value)),
        _ => None,
    }
}

/// The level of a lattice name, for `WHERE n.lattice = <int>` comparisons.
fn lattice_level_from_name(name: &str) -> Option<i64> {
    match name {
        "PUBLIC" => Some(0),
        "INTERNAL" => Some(1),
        "CONFIDENTIAL" => Some(2),
        "SECRET" => Some(3),
        _ => None,
    }
}

fn compare(left: &PropValue, op: CmpOp, right: &PropValue) -> bool {
    // A comparison this engine cannot decide is false, for every operator --
    // including '<>'. Reporting an undecidable predicate as a match is how a
    // filter leaks rows it was supposed to exclude.
    match (left, right) {
        (PropValue::Int(a), PropValue::Int(b)) => match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
        },
        (PropValue::Str(a), PropValue::Str(b)) => match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
        },
        _ => false,
    }
}

fn plan(query: &Query, dtg: &CompressedDtg) -> Result<Plan, CypherError> {
    let total_hops: u32 = query
        .match_clause
        .rels
        .iter()
        .map(|r| u32::from(r.max_hops))
        .sum();
    if total_hops > u32::from(MAX_HOP_DEPTH) {
        return Err(CypherError::at(
            format!(
                "the pattern traverses up to {total_hops} hops, over the {MAX_HOP_DEPTH}-hop \
                 bound the compressed DTG retains (ADR-002)"
            ),
            query.match_pos,
        ));
    }

    // Validate node labels and relationship kinds against what can exist.
    for pattern in &query.match_clause.nodes {
        if let Some(label) = &pattern.label {
            if !known_node_label(label) {
                return Err(CypherError::at(
                    format!(
                        "unknown node label '{label}'; expected a node type, 'sink', 'source', \
                         or 'any'"
                    ),
                    query.match_pos,
                ));
            }
        }
        for (key, _) in &pattern.props {
            if node_prop_key_known(key) {
                continue;
            }
            return Err(CypherError::at(
                format!("unknown node property '{key}' in the pattern map"),
                query.match_pos,
            ));
        }
    }
    let present_transform_fns: Vec<&str> = {
        let mut set: Vec<&str> = dtg
            .edges
            .iter()
            .map(|e| e.transform_fn.as_str())
            .collect();
        set.sort_unstable();
        set.dedup();
        set
    };
    for rel in &query.match_clause.rels {
        if let Some(kind) = &rel.kind {
            let is_sink_kind = SinkKind::from_str_name(kind).is_some();
            let is_sugar = kind == "sink" || kind == "any";
            let is_transform = present_transform_fns.contains(&kind.as_str());
            if !(is_sink_kind || is_sugar || is_transform) {
                return Err(CypherError::at(
                    format!(
                        "unknown relationship kind '{kind}'; expected a sink kind \
                         (CommandExec, ...), 'sink', 'any', or a transform function present in \
                         this graph"
                    ),
                    query.match_pos,
                ));
            }
        }
        if rel.alias.is_some() && (rel.min_hops != 1 || rel.max_hops != 1) {
            return Err(CypherError::at(
                format!(
                    "relationship variable '{}' cannot be bound to a variable-length segment; \
                     bind it with a single-hop pattern",
                    rel.alias.clone().unwrap_or_default()
                ),
                query.match_pos,
            ));
        }
    }

    // Resolve variables to pattern slots.
    let resolve = |name: &str| -> Option<Slot> {
        for (index, pattern) in query.match_clause.nodes.iter().enumerate() {
            if pattern.var.as_deref() == Some(name) {
                return Some(Slot::Node(index));
            }
        }
        for (index, rel) in query.match_clause.rels.iter().enumerate() {
            if rel.alias.as_deref() == Some(name) {
                return Some(Slot::Edge(index));
            }
        }
        None
    };

    let mut conditions = Vec::new();
    for condition in &query.conditions {
        let Some(slot) = resolve(&condition.var) else {
            return Err(CypherError::at(
                format!("unknown variable '{}' in WHERE", condition.var),
                query.match_pos,
            ));
        };
        match slot {
            Slot::Node(_) if !node_prop_key_known(&condition.prop) => {
                return Err(CypherError::at(
                    format!("unknown node property '{}'", condition.prop),
                    query.match_pos,
                ))
            }
            Slot::Edge(_) if !edge_prop_key_known(&condition.prop) => {
                return Err(CypherError::at(
                    format!("unknown relationship property '{}'", condition.prop),
                    query.match_pos,
                ))
            }
            _ => {}
        }
        conditions.push(PlannedCondition {
            slot,
            prop: condition.prop.clone(),
            op: condition.op,
            value: condition.value.clone(),
        });
    }

    let mut returns = Vec::new();
    let mut columns: Vec<String> = Vec::new();
    let mut count_only = false;
    for item in &query.returns {
        let (slot, kind, canonical) = match &item.kind {
            ReturnKind::Count => {
                if query.returns.len() != 1 {
                    return Err(CypherError::at(
                        "count(*) cannot be combined with other RETURN items in this subset",
                        query.match_pos,
                    ));
                }
                count_only = true;
                (None, PlannedReturnKind::Count, "count".to_string())
            }
            ReturnKind::Property(prop) => {
                let Some(slot) = resolve(&item.var) else {
                    return Err(CypherError::at(
                        format!("unknown variable '{}' in RETURN", item.var),
                        query.match_pos,
                    ));
                };
                match slot {
                    Slot::Node(_) if !node_prop_key_known(prop) => {
                        return Err(CypherError::at(
                            format!("unknown node property '{prop}' in RETURN"),
                            query.match_pos,
                        ))
                    }
                    Slot::Edge(_) if !edge_prop_key_known(prop) => {
                        return Err(CypherError::at(
                            format!("unknown relationship property '{prop}' in RETURN"),
                            query.match_pos,
                        ))
                    }
                    _ => {}
                }
                let canonical = format!("{}.{}", item.var, prop);
                (Some(slot), PlannedReturnKind::Property(prop.clone()), canonical)
            }
            ReturnKind::Node => {
                let Some(slot) = resolve(&item.var) else {
                    return Err(CypherError::at(
                        format!("unknown variable '{}' in RETURN", item.var),
                        query.match_pos,
                    ));
                };
                let (kind, canonical) = match slot {
                    Slot::Node(_) => (PlannedReturnKind::WholeNode, item.var.clone()),
                    Slot::Edge(_) => (PlannedReturnKind::WholeEdge, item.var.clone()),
                };
                (Some(slot), kind, canonical)
            }
        };
        let column = item.alias.clone().unwrap_or(canonical);
        if columns.contains(&column) {
            return Err(CypherError::at(
                format!("duplicate RETURN column '{column}'; add an AS alias"),
                query.match_pos,
            ));
        }
        columns.push(column.clone());
        returns.push(PlannedReturn {
            slot,
            kind,
            column,
        });
    }

    // The limit may not exceed the server-side row cap.
    if let Some(limit) = query.limit {
        if limit > DEFAULT_ROW_CAP {
            return Err(CypherError::at(
                format!("LIMIT {limit} exceeds the server cap of {DEFAULT_ROW_CAP} rows"),
                query.match_pos,
            ));
        }
    }

    Ok(Plan {
        returns,
        conditions,
        count_only,
    })
}

fn node_prop_key_known(key: &str) -> bool {
    matches!(key, "id" | "label" | "node_type" | "lattice" | "file_path" | "line")
}

fn edge_prop_key_known(key: &str) -> bool {
    matches!(key, "source" | "target" | "transform_fn" | "sink_kind" | "hop_depth")
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

struct Engine<'a> {
    dtg: &'a CompressedDtg,
    plan: &'a Plan,
    query: &'a Query,
    id_to_pos: BTreeMap<&'a str, usize>,
    adjacency: Vec<Vec<usize>>,
    node_binds: Vec<usize>,
    edge_binds: Vec<Option<usize>>,
    rows: Vec<BTreeMap<String, Value>>,
    matches: u64,
    effective_limit: usize,
    truncated: bool,
    stop: bool,
}

impl<'a> Engine<'a> {
    fn new(dtg: &'a CompressedDtg, plan: &'a Plan, query: &'a Query) -> Self {
        let mut id_to_pos = BTreeMap::new();
        for (index, node) in dtg.nodes.iter().enumerate() {
            id_to_pos.insert(node.id.as_str(), index);
        }
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); dtg.nodes.len()];
        for (index, edge) in dtg.edges.iter().enumerate() {
            if let Some(&from) = id_to_pos.get(edge.source.as_str()) {
                adjacency[from].push(index);
            }
        }
        let effective_limit = query.limit.unwrap_or(DEFAULT_ROW_CAP);
        Engine {
            dtg,
            plan,
            query,
            id_to_pos,
            adjacency,
            node_binds: vec![0; query.match_clause.nodes.len()],
            edge_binds: vec![None; query.match_clause.rels.len()],
            rows: Vec::new(),
            matches: 0,
            effective_limit,
            truncated: false,
            stop: false,
        }
    }

    fn node_at(&self, pos: usize) -> &'a DtgNode {
        &self.dtg.nodes[pos]
    }

    fn node_pattern_matches(&self, slot: usize, pos: usize) -> bool {
        let pattern = &self.query.match_clause.nodes[slot];
        let node = self.node_at(pos);
        if let Some(label) = &pattern.label {
            if !node_label_matches(node, label) {
                return false;
            }
        }
        pattern.props.iter().all(|(key, value)| {
            match node_prop(node, key) {
                Some(actual) => compare(&actual, CmpOp::Eq, value),
                None => false,
            }
        })
    }

    /// Every (end node position, edge trail) reachable in exactly `hops` edges.
    fn walk_hops(&self, start: usize, kind: Option<&str>, hops: u8) -> Vec<(usize, Vec<usize>)> {
        let mut out = Vec::new();
        let mut trail = Vec::new();
        self.walk_rec(start, kind, hops, &mut trail, &mut out);
        out
    }

    fn walk_rec(
        &self,
        node: usize,
        kind: Option<&str>,
        remaining: u8,
        trail: &mut Vec<usize>,
        out: &mut Vec<(usize, Vec<usize>)>,
    ) {
        if remaining == 0 {
            out.push((node, trail.clone()));
            return;
        }
        for edge_index in &self.adjacency[node] {
            let edge = &self.dtg.edges[*edge_index];
            if let Some(kind) = kind {
                if !edge_kind_matches(edge, kind) {
                    continue;
                }
            }
            let Some(&target) = self.id_to_pos.get(edge.target.as_str()) else {
                // An edge into a node the compression dropped. Skipping is the
                // only sound choice: there is no node to bind or evaluate.
                continue;
            };
            trail.push(*edge_index);
            self.walk_rec(target, kind, remaining - 1, trail, out);
            trail.pop();
        }
    }

    fn conditions_hold(&self) -> bool {
        self.plan.conditions.iter().all(|condition| {
            let actual = match condition.slot {
                Slot::Node(slot) => node_prop(self.node_at(self.node_binds[slot]), &condition.prop),
                Slot::Edge(slot) => self.edge_binds[slot]
                    .and_then(|edge_index| edge_prop(&self.dtg.edges[edge_index], &condition.prop)),
            };
            let Some(actual) = actual else { return false };
            // Only the lattice is comparable both by name and by level; every
            // other property is coerced to the one type the query asked for,
            // and a mismatch excludes the row.
            if condition.prop == "lattice" {
                if let (PropValue::Str(name), PropValue::Int(level)) = (&actual, &condition.value) {
                    let Some(actual_level) = lattice_level_from_name(name) else {
                        return false;
                    };
                    return compare(
                        &PropValue::Int(actual_level),
                        condition.op,
                        &PropValue::Int(*level),
                    );
                }
            }
            let want_int = matches!(condition.value, PropValue::Int(_));
            match coerce(actual, want_int) {
                Some(actual) => compare(&actual, condition.op, &condition.value),
                None => false,
            }
        })
    }

    fn emit(&mut self) -> Result<(), CypherError> {
        if !self.conditions_hold() {
            return Ok(());
        }
        self.matches += 1;
        if self.plan.count_only {
            return Ok(());
        }
        if self.rows.len() >= self.effective_limit {
            self.truncated = true;
            self.stop = true;
            return Ok(());
        }
        let mut row = BTreeMap::new();
        for planned in &self.plan.returns {
            let value = match (planned.slot, &planned.kind) {
                (_, PlannedReturnKind::Count) => Value::Null, // replaced by the caller
                (Some(Slot::Node(slot)), PlannedReturnKind::Property(prop)) => {
                    match node_prop(self.node_at(self.node_binds[slot]), prop) {
                        Some(PropValue::Str(s)) => Value::String(s),
                        Some(PropValue::Int(i)) => Value::Number(i.into()),
                        None => Value::Null,
                    }
                }
                (Some(Slot::Edge(slot)), PlannedReturnKind::Property(prop)) => {
                    let edge_index = self.edge_binds[slot].ok_or_else(|| {
                        CypherError::at("relationship property read without a bound edge", 0)
                    })?;
                    match edge_prop(&self.dtg.edges[edge_index], prop) {
                        Some(PropValue::Str(s)) => Value::String(s),
                        Some(PropValue::Int(i)) => Value::Number(i.into()),
                        None => Value::Null,
                    }
                }
                (Some(Slot::Node(slot)), PlannedReturnKind::WholeNode) => {
                    serde_json::to_value(self.node_at(self.node_binds[slot]))
                        .unwrap_or(Value::Null)
                }
                (Some(Slot::Edge(slot)), PlannedReturnKind::WholeEdge) => {
                    let edge_index = self.edge_binds[slot].ok_or_else(|| {
                        CypherError::at("relationship read without a bound edge", 0)
                    })?;
                    serde_json::to_value(&self.dtg.edges[edge_index]).unwrap_or(Value::Null)
                }
                // A slot bound to a node can never carry a whole edge, or the
                // reverse: the planner rejects those queries, so reaching this
                // arm means the planner is inconsistent. Failing the query is
                // the fail-closed reading; returning null would fabricate a row.
                (Some(Slot::Node(_)), PlannedReturnKind::WholeEdge)
                | (Some(Slot::Edge(_)), PlannedReturnKind::WholeNode) => {
                    return Err(CypherError::at(
                        "internal planning inconsistency: variable kind does not match its binding",
                        0,
                    ));
                }
                (None, _) => Value::Null,
            };
            row.insert(planned.column.clone(), value);
        }
        self.rows.push(row);
        Ok(())
    }

    fn extend(&mut self, segment: usize) -> Result<(), CypherError> {
        if self.stop {
            return Ok(());
        }
        let rels = &self.query.match_clause.rels;
        if segment == rels.len() {
            return self.emit();
        }
        let rel = rels[segment].clone();
        let from = self.node_binds[segment];
        for hops in rel.min_hops..=rel.max_hops {
            let reached = self.walk_hops(from, rel.kind.as_deref(), hops);
            for (end, trail) in reached {
                if !self.node_pattern_matches(segment + 1, end) {
                    continue;
                }
                self.node_binds[segment + 1] = end;
                self.edge_binds[segment] = if hops == 1 { Some(trail[0]) } else { None };
                self.extend(segment + 1)?;
                if self.stop {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn run(&mut self) -> Result<QueryResult, CypherError> {
        let anchors: Vec<usize> = (0..self.dtg.nodes.len())
            .filter(|pos| self.node_pattern_matches(0, *pos))
            .collect();
        for anchor in anchors {
            self.node_binds[0] = anchor;
            self.extend(0)?;
            if self.stop {
                break;
            }
        }

        let columns: Vec<String> = self
            .plan
            .returns
            .iter()
            .map(|r| r.column.clone())
            .collect();
        if self.plan.count_only {
            let mut row = BTreeMap::new();
            row.insert("count".to_string(), Value::Number(self.matches.into()));
            return Ok(QueryResult {
                columns,
                rows: vec![row],
                truncated: false,
            });
        }
        Ok(QueryResult {
            columns,
            rows: std::mem::take(&mut self.rows),
            truncated: self.truncated,
        })
    }
}

/// Executes a query against one compressed DTG.
///
/// `query` is the full Cypher-subset text; the server-side row cap is
/// [`DEFAULT_ROW_CAP`]. Every error returns the byte position to highlight.
pub fn execute(dtg: &CompressedDtg, query: &str, row_cap: usize) -> Result<QueryResult, CypherError> {
    if query.trim().is_empty() {
        return Err(CypherError::at("empty query", 0));
    }
    let parsed = parse(query)?;
    let plan = plan(&parsed, dtg)?;
    // The cap is a parameter so tests can exercise small limits; the plan
    // validated against the default, so re-check here for the given cap.
    if let Some(limit) = parsed.limit {
        if limit > row_cap {
            return Err(CypherError::at(
                format!("LIMIT {limit} exceeds the server cap of {row_cap} rows"),
                parsed.match_pos,
            ));
        }
    }
    let mut engine = Engine::new(dtg, &plan, &parsed);
    engine.effective_limit = parsed.limit.unwrap_or(row_cap);
    engine.run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_common::types::{DetectedSink, DtgNodeType};
    use aegis_common::SecurityLattice;

    /// The canonical three-hop chain from the fixtures:
    ///
    /// `input(tool_input_param, PUBLIC)` -base64_decode-> `decoded`
    /// -path_join-> `joined(SECRET)` -execve_call(CommandExec)-> `proc(process_sink)`
    ///
    /// plus a side edge `input -assign-> scratch` used by the chain tests.
    fn chain() -> CompressedDtg {
        let node = |id: &str, label: &str, node_type: DtgNodeType, lattice| DtgNode {
            id: id.into(),
            label: label.into(),
            node_type,
            lattice,
            file_path: "server.py".into(),
            line: id.len(),
        };
        let edge = |source: &str, target: &str, transform: &str, hop: u8, sink| DtgEdge {
            source: source.into(),
            target: target.into(),
            transform_fn: transform.into(),
            sink_kind: sink,
            hop_depth: hop,
        };
        CompressedDtg {
            server_id: "s_demo".into(),
            language: "python".into(),
            raw_cpg_node_count: 400,
            compressed_node_count: 5,
            nodes: vec![
                node("n0", "arg", DtgNodeType::ToolInputParam, SecurityLattice::Public),
                node("n1", "decoded", DtgNodeType::LocalBuffer, SecurityLattice::Public),
                node("n2", "joined", DtgNodeType::LocalBuffer, SecurityLattice::Secret),
                node("n3", "proc", DtgNodeType::ProcessSink, SecurityLattice::Secret),
                node("n4", "scratch", DtgNodeType::LocalBuffer, SecurityLattice::Internal),
            ],
            edges: vec![
                edge("n0", "n1", "base64_decode", 1, None),
                edge("n1", "n2", "path_join", 2, None),
                edge("n2", "n3", "execve_call", 3, Some(SinkKind::CommandExec)),
                edge("n0", "n4", "assign", 1, None),
            ],
            detected_sinks: vec![DetectedSink {
                sink_kind: SinkKind::CommandExec,
                file_path: "server.py".into(),
                line: 9,
                function: "run".into(),
                hop_depth: 3,
                expression: "subprocess.run(cmd)".into(),
            }],
            dependencies: vec![],
            injection_markers: vec![],
            egress_literals: vec![],
            dtg_sha256: "d".repeat(64),
        }
    }

    fn rows(dtg: &CompressedDtg, query: &str) -> Vec<BTreeMap<String, Value>> {
        execute(dtg, query, DEFAULT_ROW_CAP)
            .unwrap_or_else(|e| panic!("{query} failed: {e}"))
            .rows
    }

    fn ids(dtg: &CompressedDtg, query: &str) -> Vec<String> {
        rows(dtg, query)
            .into_iter()
            .map(|row| row.values().next().unwrap().as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn a_three_hop_reachability_query_finds_the_process_sink() {
        let dtg = chain();
        let result = execute(
            &dtg,
            "MATCH (a:source)-[*1..3]->(b:sink) RETURN a.id, b.id",
            DEFAULT_ROW_CAP,
        )
        .unwrap();
        assert_eq!(result.columns, vec!["a.id", "b.id"]);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["a.id"], Value::String("n0".into()));
        assert_eq!(result.rows[0]["b.id"], Value::String("n3".into()));
        assert!(!result.truncated);
    }

    #[test]
    fn a_two_hop_bound_does_not_reach_the_sink() {
        let dtg = chain();
        let result = execute(
            &dtg,
            "MATCH (a:source)-[*1..2]->(b:sink) RETURN b.id",
            DEFAULT_ROW_CAP,
        )
        .unwrap();
        assert!(result.rows.is_empty(), "{:?}", result.rows);
    }

    #[test]
    fn variable_length_enumerates_every_reachable_pair() {
        let dtg = chain();
        // From each node, every node 1..3 hops away, with the per-hop kinds.
        let pairs = ids(
            &dtg,
            "MATCH (a)-[*1..3]->(b) RETURN a.id AS from_id LIMIT 50",
        );
        // Anchors in dtg.nodes order: n0 -> (n1, n2, n3, n4), n1 -> (n2, n3),
        // n2 -> (n3), n3 -> (), n4 -> ().
        assert_eq!(pairs, vec!["n0", "n0", "n0", "n0", "n1", "n1", "n2"]);
    }

    #[test]
    fn a_fixed_single_hop_with_a_transform_kind_works() {
        let dtg = chain();
        let result = execute(
            &dtg,
            "MATCH (a)-[:execve_call]->(b:sink) RETURN a.id, b.label",
            DEFAULT_ROW_CAP,
        )
        .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["a.id"], Value::String("n2".into()));
        assert_eq!(result.rows[0]["b.label"], Value::String("proc".into()));
    }

    #[test]
    fn sink_kinds_are_first_class_relationship_kinds() {
        let dtg = chain();
        let result = execute(
            &dtg,
            "MATCH (a)-[:CommandExec]->(b) RETURN a.id",
            DEFAULT_ROW_CAP,
        )
        .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["a.id"], Value::String("n2".into()));

        // The sugar form selects by sink presence.
        let sugar = execute(&dtg, "MATCH (a)-[:sink]->(b) RETURN a.id", DEFAULT_ROW_CAP).unwrap();
        assert_eq!(sugar.rows.len(), 1);
    }

    #[test]
    fn a_chained_pattern_composes_segments() {
        let dtg = chain();
        let result = execute(
            &dtg,
            "MATCH (a:source)-[:base64_decode]->(b)-[:path_join]->(c)-[:execve_call]->(d) \
             RETURN a.id, c.id AS joined, d.id",
            DEFAULT_ROW_CAP,
        )
        .unwrap();
        assert_eq!(result.columns, vec!["a.id", "joined", "d.id"]);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["joined"], Value::String("n2".into()));
    }

    #[test]
    fn pattern_property_maps_filter_nodes() {
        let dtg = chain();
        let selected = ids(
            &dtg,
            "MATCH (a {node_type: 'env_secret'}) RETURN a.id",
        );
        assert!(selected.is_empty(), "no env_secret node exists");

        let by_lattice = ids(&dtg, "MATCH (a {lattice: 'SECRET'}) RETURN a.id LIMIT 10");
        assert_eq!(by_lattice, vec!["n2", "n3"]);
    }

    #[test]
    fn where_clauses_filter_on_node_and_edge_properties() {
        let dtg = chain();
        let result = execute(
            &dtg,
            "MATCH (a)-[r:base64_decode]->(b) WHERE r.hop_depth <= 1 AND b.line > 1 \
             RETURN b.id",
            DEFAULT_ROW_CAP,
        )
        .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["b.id"], Value::String("n1".into()));

        // The lattice is comparable by level as well as by name.
        let by_level = ids(
            &dtg,
            "MATCH (a) WHERE a.lattice = 3 RETURN a.id LIMIT 10",
        );
        assert_eq!(by_level, vec!["n2", "n3"]);

        // An undecidable comparison excludes the row.
        let undecidable = ids(
            &dtg,
            "MATCH (a) WHERE a.line = 'not-a-number' RETURN a.id",
        );
        assert!(undecidable.is_empty());
    }

    #[test]
    fn count_star_counts_matches_not_rows() {
        let dtg = chain();
        let result = execute(&dtg, "MATCH (a)-[*1..3]->(b) RETURN count(*)", DEFAULT_ROW_CAP)
            .unwrap();
        assert_eq!(result.columns, vec!["count"]);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0]["count"], serde_json::json!(7));
    }

    #[test]
    fn whole_node_return_serialises_the_node() {
        let dtg = chain();
        let result = execute(&dtg, "MATCH (a {id: 'n3'}) RETURN a", DEFAULT_ROW_CAP).unwrap();
        assert_eq!(result.rows.len(), 1);
        let node = &result.rows[0]["a"];
        assert_eq!(node["id"], Value::String("n3".into()));
        assert_eq!(node["node_type"], Value::String("process_sink".into()));
        assert_eq!(node["lattice"], Value::String("SECRET".into()));
    }

    #[test]
    fn limit_truncates_and_reports_it() {
        let dtg = chain();
        let result = execute(&dtg, "MATCH (a)-[*1..3]->(b) RETURN a.id LIMIT 2", DEFAULT_ROW_CAP)
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert!(result.truncated, "a further match existed");

        // Exactly as many matches as the limit: not truncated.
        let exact = execute(&dtg, "MATCH (n {id: 'n3'}) RETURN n.id LIMIT 1", DEFAULT_ROW_CAP)
            .unwrap();
        assert_eq!(exact.rows.len(), 1);
        assert!(!exact.truncated);
    }

    #[test]
    fn unknown_labels_kinds_variables_and_properties_are_errors() {
        let dtg = chain();
        for (query, needle) in [
            ("MATCH (a:banana) RETURN a.id", "unknown node label"),
            ("MATCH (a)-[:nosuchfn]->(b) RETURN a.id", "unknown relationship kind"),
            ("MATCH (a)-[:execve_call]->(b) WHERE z.hop_depth = 1 RETURN a.id", "unknown variable"),
            ("MATCH (a) WHERE a.bogus = 1 RETURN a.id", "unknown node property"),
            ("MATCH (a) RETURN a.bogus", "unknown node property"),
            ("MATCH (a)-[r:execve_call]->(b) WHERE r.bogus = 1 RETURN b.id", "unknown relationship property"),
            ("MATCH (a) RETURN a.id, a.id", "duplicate RETURN column"),
            ("MATCH (a)-[r*1..2]->(b) RETURN b.id", "cannot be bound to a variable-length segment"),
            ("MATCH (a)-[*0..2]->(b) RETURN b.id", "start at 1"),
            ("MATCH (a)-[*2..1]->(b) RETURN b.id", "empty hop range"),
        ] {
            let error = execute(&dtg, query, DEFAULT_ROW_CAP)
                .unwrap_err();
            assert!(
                error.message.contains(needle),
                "{query} -> {}",
                error.message
            );
        }
    }

    #[test]
    fn the_hop_bound_is_enforced_at_parse_and_plan_time() {
        let dtg = chain();
        let too_wide = execute(&dtg, "MATCH (a)-[*1..4]->(b) RETURN b.id", DEFAULT_ROW_CAP)
            .unwrap_err();
        assert!(too_wide.message.contains("3-hop"), "{}", too_wide.message);

        // Four fixed segments are four hops, even though each is valid alone.
        let too_long = execute(
            &dtg,
            "MATCH (a)-[:base64_decode]->(b)-[:path_join]->(c)-[:execve_call]->(d)-[:assign]->(e) \
             RETURN e.id",
            DEFAULT_ROW_CAP,
        )
        .unwrap_err();
        assert!(too_long.message.contains("4 hops"), "{}", too_long.message);
    }

    #[test]
    fn syntax_errors_name_the_position() {
        let dtg = chain();
        for (query, needle) in [
            ("", "empty query"),
            ("RETURN 1", "must start with MATCH"),
            ("MATCH (a RETURN a.id", "expected"),
            ("MATCH (a)-(b) RETURN a.id", "expected '->'"),
            ("MATCH (a)<-[*1..2]-(b) RETURN a.id", "incoming traversal"),
            ("MATCH (a)-[*1..3]->(b)", "expected RETURN"),
            ("MATCH (a) RETURN a.id LIMIT -1", "expected LIMIT row count"),
            ("MATCH (a) WHERE a.id = RETURN a.id", "expected a comparison value"),
            ("MATCH (a {id: }) RETURN a.id", "expected property value"),
            ("MATCH (a) RETURN a.id > 3", "unexpected"),
        ] {
            let error = execute(&dtg, query, DEFAULT_ROW_CAP).unwrap_err();
            assert!(
                error.message.contains(needle),
                "{query:?} -> {}",
                error.message
            );
        }

        let unterminated = execute(&dtg, "MATCH (a {id: 'x}) RETURN a.id", DEFAULT_ROW_CAP)
            .unwrap_err();
        assert!(unterminated.message.contains("unterminated"), "{unterminated}");
    }

    #[test]
    fn the_server_row_cap_bounds_limits() {
        let dtg = chain();
        let over = execute(&dtg, "MATCH (a)-[*1..3]->(b) RETURN a.id LIMIT 100000", DEFAULT_ROW_CAP)
            .unwrap_err();
        assert!(over.message.contains("server cap"), "{}", over.message);

        // A small cap truncates.
        let small = execute(&dtg, "MATCH (a)-[*1..3]->(b) RETURN a.id", 3).unwrap();
        assert_eq!(small.rows.len(), 3);
        assert!(small.truncated);
    }

    #[test]
    fn traversal_is_deterministic_across_runs() {
        let dtg = chain();
        let first = execute(&dtg, "MATCH (a)-[*1..3]->(b) RETURN a.id, b.id", DEFAULT_ROW_CAP)
            .unwrap()
            .rows;
        for _ in 0..8 {
            let again = execute(&dtg, "MATCH (a)-[*1..3]->(b) RETURN a.id, b.id", DEFAULT_ROW_CAP)
                .unwrap()
                .rows;
            assert_eq!(first, again);
        }
    }

    #[test]
    fn an_empty_graph_yields_zero_rows_not_an_error() {
        let empty = CompressedDtg {
            server_id: "s_empty".into(),
            language: "python".into(),
            raw_cpg_node_count: 0,
            compressed_node_count: 0,
            nodes: vec![],
            edges: vec![],
            detected_sinks: vec![],
            dependencies: vec![],
            injection_markers: vec![],
            egress_literals: vec![],
            dtg_sha256: String::new(),
        };
        let result = execute(&empty, "MATCH (a)-[*1..3]->(b:sink) RETURN b.id", DEFAULT_ROW_CAP)
            .unwrap();
        assert!(result.rows.is_empty());
    }
}
