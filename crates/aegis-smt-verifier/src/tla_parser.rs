//! # TLA+ fragment parser
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! A lexer and Pratt parser for the TLA+ fragment used by
//! `specs/MCPShieldLTS.tla` and its `.cfg` model configurations: module
//! headers, operator definitions (with parameters), conjunctions and
//! disjunctions in list form, quantifiers, `CHOOSE`, `LET/IN`, tuples,
//! sets, set maps, function constructors, function spaces, `EXCEPT`
//! updates, primes, action subscripts `[A]_v`, fairness wrappers
//! `WF_v(A)`, and the temporal `[]` operator.
//!
//! ## WHY
//! The alternative to parsing the spec is to re-implement it in Rust, and a
//! re-implementation that drifts from the file it claims to check is worse
//! than no checker. This parser makes the `.tla` file the executable source
//! of truth: the model checker evaluates the AST the reader audits.
//!
//! ## HOW
//! [`lex`] produces a flat token stream with comments stripped; [`Parser`]
//! builds [`Expr`] trees with TLA+'s operator precedence (disjunction below
//! conjunction, conjunction below comparison, comparison below set
//! operators). Definition boundaries fall out of the token stream: an
//! expression parser stops at the first token it cannot consume, and the
//! next `Name (params)? ==` starts the next definition.

use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

/// Errors raised while lexing, parsing, or loading a specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlaError {
    /// Lexer failure with 1-based line number.
    Lex { line: usize, message: String },
    /// Parser failure with 1-based line number.
    Parse { line: usize, message: String },
    /// A `.cfg` invariant name is not defined in the `.tla` module.
    UndefinedInvariant(String),
    /// A `.cfg` references an operator or constant that does not exist.
    UnknownOperator(String),
    /// A `.cfg` is missing a required directive (SPECIFICATION / INIT / NEXT).
    MissingDirective(String),
    /// Evaluation failure (type mismatch, missing binding, ...).
    Eval(String),
    /// The `Next` relation has a shape the engine cannot enumerate.
    UnsupportedNext(String),
    /// An action does not fully determine every state variable.
    IncompleteAction { action: String, var: String },
}

impl fmt::Display for TlaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TlaError::Lex { line, message } => write!(f, "lex error at line {line}: {message}"),
            TlaError::Parse { line, message } => write!(f, "parse error at line {line}: {message}"),
            TlaError::UndefinedInvariant(n) => {
                write!(f, "invariant `{n}` is declared in the .cfg but not defined in the .tla")
            }
            TlaError::UnknownOperator(n) => write!(f, "unknown operator or constant `{n}`"),
            TlaError::MissingDirective(n) => write!(f, "configuration is missing `{n}`"),
            TlaError::Eval(m) => write!(f, "evaluation error: {m}"),
            TlaError::UnsupportedNext(m) => write!(f, "unsupported Next shape: {m}"),
            TlaError::IncompleteAction { action, var } => {
                write!(f, "action `{action}` leaves `{var}` undetermined")
            }
        }
    }
}

impl std::error::Error for TlaError {}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

/// Token kinds produced by [`lex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Str(String),
    DefEq,      // ==
    Prime,      // '
    And,        // /\
    Or,         // \/
    Implies,    // =>
    Iff,        // <=>
    Not,        // ~
    Eq,         // =
    Neq,        // #
    In,         // \in
    NotIn,      // \notin
    Le,         // <= / =<
    Ge,         // >=
    Lt,         // <
    Gt,         // >
    Cup,        // \cup
    Cap,        // \cap
    SubsetEq,   // \subseteq
    Times,      // \X
    Forall,     // \A
    Exists,     // \E
    Range,      // ..
    Map,        // |->
    Arrow,      // ->
    Box,        // []
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LAngle,
    RAngle,
    Comma,
    Colon,
    Bang,       // !
    At,         // @
    Underscore, // _
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Ident(s) => write!(f, "`{s}`"),
            Tok::Int(v) => write!(f, "{v}"),
            Tok::Str(s) => write!(f, "\"{s}\""),
            other => write!(f, "{other:?}"),
        }
    }
}

/// One token with its 1-based source line.
#[derive(Debug, Clone)]
pub struct SpannedTok {
    pub tok: Tok,
    pub line: usize,
}

fn strip_comments(src: &str) -> Result<String, TlaError> {
    // TLA+ comments: `\* to end of line` and `(* ... *)` with nesting.
    let mut out = String::with_capacity(src.len());
    let bytes: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    let mut line = 1usize;
    let mut depth = 0usize;
    let mut in_string = false;
    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            out.push(c);
            if c == '"' {
                in_string = false;
            }
            if c == '\n' {
                line += 1;
            }
            i += 1;
            continue;
        }
        match c {
            '"' if depth == 0 => {
                in_string = true;
                out.push(c);
                i += 1;
            }
            '\\' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == '*' => {
                while i < bytes.len() && bytes[i] != '\n' {
                    i += 1;
                }
            }
            '(' if i + 1 < bytes.len() && bytes[i + 1] == '*' => {
                depth += 1;
                i += 2;
            }
            '*' if depth > 0 && i + 1 < bytes.len() && bytes[i + 1] == ')' => {
                depth -= 1;
                i += 2;
            }
            _ if depth > 0 => {
                if c == '\n' {
                    line += 1;
                }
                i += 1;
            }
            _ => {
                if c == '\n' {
                    line += 1;
                }
                out.push(c);
                i += 1;
            }
        }
    }
    if depth > 0 {
        return Err(TlaError::Lex {
            line,
            message: "unterminated (* comment".to_string(),
        });
    }
    Ok(out)
}

/// Tokenizes a full `.tla` module or a `.cfg` value expression.
pub fn lex(src: &str) -> Result<Vec<SpannedTok>, TlaError> {
    let clean = strip_comments(src)?;
    let chars: Vec<char> = clean.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '-' && i + 3 < chars.len() && chars[i + 1] == '-' && chars[i + 2] == '-' {
            // `---- MODULE ... ----` separator; skip the whole line.
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '=' && i + 3 < chars.len() && chars[i + 1] == '=' && chars[i + 2] == '=' {
            // `====` module footer; skip the line.
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        let start_line = line;
        let two = |a: char, b: char| i + 1 < chars.len() && chars[i] == a && chars[i + 1] == b;
        if c == '"' {
            let mut s = String::new();
            i += 1;
            let mut closed = false;
            while i < chars.len() {
                let ch = chars[i];
                if ch == '"' {
                    closed = true;
                    i += 1;
                    break;
                }
                if ch == '\\' && i + 1 < chars.len() && matches!(chars[i + 1], '"' | '\\') {
                    s.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                if ch == '\n' {
                    line += 1;
                }
                s.push(ch);
                i += 1;
            }
            if !closed {
                return Err(TlaError::Lex {
                    line: start_line,
                    message: "unterminated string literal".to_string(),
                });
            }
            toks.push(SpannedTok {
                tok: Tok::Str(s),
                line: start_line,
            });
            continue;
        }
        if c.is_ascii_digit() {
            let mut n: i64 = 0;
            while i < chars.len() && chars[i].is_ascii_digit() {
                n = n
                    .checked_mul(10)
                    .and_then(|v| v.checked_add((chars[i] as u8 - b'0') as i64))
                    .ok_or_else(|| TlaError::Lex {
                        line: start_line,
                        message: "integer literal overflow".to_string(),
                    })?;
                i += 1;
            }
            toks.push(SpannedTok {
                tok: Tok::Int(n),
                line: start_line,
            });
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let mut s = String::new();
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                s.push(chars[i]);
                i += 1;
            }
            toks.push(SpannedTok {
                tok: Tok::Ident(s),
                line: start_line,
            });
            continue;
        }
        // Multi-character operators first.
        let (tok, consumed) = if two('=', '=') {
            (Tok::DefEq, 2)
        } else if two('<', '=') {
            if i + 2 < chars.len() && chars[i + 2] == '>' {
                (Tok::Iff, 3)
            } else {
                (Tok::Le, 2)
            }
        } else if two('=', '<') {
            (Tok::Le, 2)
        } else if two('>', '=') {
            (Tok::Ge, 2)
        } else if two('=', '>') {
            (Tok::Implies, 2)
        } else if two('|', '-') {
            if i + 2 < chars.len() && chars[i + 2] == '>' {
                (Tok::Map, 3)
            } else {
                return Err(TlaError::Lex {
                    line: start_line,
                    message: "`|-` is not a TLA+ token; did you mean `|->`?".to_string(),
                });
            }
        } else if two('-', '>') {
            (Tok::Arrow, 2)
        } else if two('.', '.') {
            (Tok::Range, 2)
        } else if two('<', '<') {
            (Tok::LAngle, 2)
        } else if two('>', '>') {
            (Tok::RAngle, 2)
        } else if two('[', ']') {
            (Tok::Box, 2)
        } else if c == '\\' {
            // Backslash-led keywords. The longest is `\subseteq` (9 chars),
            // so the lookahead must cover it entirely.
            let rest: String = chars[i..].iter().take(12).collect();
            let table: &[(&str, Tok)] = &[
                ("\\subseteq", Tok::SubsetEq),
                ("\\notin", Tok::NotIn),
                ("\\cup", Tok::Cup),
                ("\\cap", Tok::Cap),
                ("\\in", Tok::In),
                ("\\A", Tok::Forall),
                ("\\E", Tok::Exists),
                ("\\X", Tok::Times),
                ("\\lnot", Tok::Not),
                ("\\neg", Tok::Not),
                ("\\/", Tok::Or),
                ("/\\", Tok::And),
            ];
            match table.iter().find(|(pat, _)| rest.starts_with(pat)) {
                Some((pat, t)) => (t.clone(), pat.chars().count()),
                None => {
                    return Err(TlaError::Lex {
                        line: start_line,
                        message: format!("unknown escape sequence near `{}`", rest.chars().take(6).collect::<String>()),
                    });
                }
            }
        } else {
            let t = match c {
                '\'' => Tok::Prime,
                '#' => Tok::Neq,
                '~' => Tok::Not,
                '=' => Tok::Eq,
                '<' => Tok::Lt,
                '>' => Tok::Gt,
                '/' => {
                    if i + 1 < chars.len() && chars[i + 1] == '\\' {
                        // `/\\` written without the leading backslash.
                        toks.push(SpannedTok {
                            tok: Tok::And,
                            line: start_line,
                        });
                        i += 2;
                        continue;
                    }
                    return Err(TlaError::Lex {
                        line: start_line,
                        message: r"unexpected `/` (conjunction is `/\`)".to_string(),
                    });
                }
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                '{' => Tok::LBrace,
                '}' => Tok::RBrace,
                '[' => Tok::LBracket,
                ']' => Tok::RBracket,
                ',' => Tok::Comma,
                ':' => Tok::Colon,
                '!' => Tok::Bang,
                '@' => Tok::At,
                '_' => Tok::Underscore,
                other => {
                    return Err(TlaError::Lex {
                        line: start_line,
                        message: format!("unexpected character `{other}`"),
                    });
                }
            };
            (t, 1)
        };
        toks.push(SpannedTok {
            tok,
            line: start_line,
        });
        i += consumed;
    }
    Ok(toks)
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

/// Parsed TLA+ expression in the checker's fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Bool(bool),
    Int(i64),
    Str(String),
    /// Variable, constant, parameter, or defined-operator reference.
    Ident(String),
    /// A primed state variable (`x'`).
    Prime(Box<Expr>),
    /// `@` inside an EXCEPT update.
    At,
    Not(Box<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Implies(Box<Expr>, Box<Expr>),
    Iff(Box<Expr>, Box<Expr>),
    Cmp(CmpOp, Box<Expr>, Box<Expr>),
    SetOp(SetOp, Box<Expr>, Box<Expr>),
    /// `S1 \X S2 \X ...` with three or more factors: TLA+ defines `\X` as
    /// associative, so elements are *flat* n-tuples, not nested pairs.
    Cartesian(Vec<Expr>),
    Range(Box<Expr>, Box<Expr>),
    Tuple(Vec<Expr>),
    SetLit(Vec<Expr>),
    /// `{ body : x \in domain }`
    SetMap {
        body: Box<Expr>,
        var: String,
        domain: Box<Expr>,
    },
    /// `{ x \in domain : predicate }`
    SetFilter {
        var: String,
        domain: Box<Expr>,
        predicate: Box<Expr>,
    },
    /// `[ x \in domain |-> body ]`
    FunCtor {
        var: String,
        domain: Box<Expr>,
        body: Box<Expr>,
    },
    /// `[ domain -> range ]`
    FunSpace(Box<Expr>, Box<Expr>),
    /// `[ base EXCEPT ![p1][p2] = value, ... ]`
    Except {
        base: Box<Expr>,
        updates: Vec<(Vec<Expr>, Expr)>,
    },
    /// `[ k1 |-> v1, k2 |-> v2 ]` (cfg-style function literal).
    FunLit(Vec<(Expr, Expr)>),
    /// `UNCHANGED x` or `UNCHANGED <<x, y>>`.
    Unchanged(Box<Expr>),
    /// `f[a, b]`
    Apply(Box<Expr>, Vec<Expr>),
    Forall {
        binders: Vec<(String, Expr)>,
        body: Box<Expr>,
    },
    Exists {
        binders: Vec<(String, Expr)>,
        body: Box<Expr>,
    },
    Choose {
        var: String,
        domain: Box<Expr>,
        predicate: Box<Expr>,
    },
    Let {
        defs: Vec<(String, Expr)>,
        body: Box<Expr>,
    },
    /// `UNION S` (or `UNION { e : x \in S }` via SetMap).
    Union(Box<Expr>),
    /// `SUBSET S`
    Powerset(Box<Expr>),
    /// `Seq(S)`
    SeqOf(Box<Expr>),
    /// `WF_v(A)` / `SF_v(A)`; referenced, never evaluated by the checker.
    Fairness {
        kind: String,
        vars: String,
        action: Box<Expr>,
    },
    /// `A /\ [][Next]_vars` temporals; referenced, never evaluated.
    Always(Box<Expr>),
    ActionSubscript {
        action: Box<Expr>,
        vars: Box<Expr>,
    },
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
    SubsetEq,
}

/// Binary set operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOp {
    Cup,
    Cap,
}

/// One operator definition from the module.
#[derive(Debug, Clone)]
pub struct Operator {
    pub name: String,
    pub params: Vec<String>,
    pub body: Rc<Expr>,
}

/// All operator definitions of a module, keyed by name.
pub type Operators = BTreeMap<String, Operator>;

/// Parses a single TLA+ expression (used for `.cfg` constant values).
pub fn parse_expression(src: &str) -> Result<Expr, TlaError> {
    let toks = lex(src)?;
    let mut p = Parser::new(toks);
    let e = p.parse_expr()?;
    if !p.eof() {
        return Err(p.err(&format!(
            "trailing tokens after the expression: {:?}",
            p.peek()
        )));
    }
    Ok(e)
}

/// Parses every `Name (params)? == body` definition of a module.
///
/// The parser skips the module header (`---- MODULE ...`, `EXTENDS ...`) and
/// returns definitions in source order.
pub fn parse_module(src: &str) -> Result<Operators, TlaError> {
    let toks = lex(src)?;
    let mut p = Parser::new(toks);
    let mut defs = Operators::new();
    // Skip header tokens until the first definition-like pair
    // `Ident (Ident, ...)? ==`.
    loop {
        p.skip_extends_line();
        if p.eof() {
            return Ok(defs);
        }
        let save = p.pos;
        let name = match p.peek() {
            Some(Tok::Ident(n)) => n.clone(),
            _ => {
                // Header material between definitions: the `----` module
                // delimiters, commas, and keywords of the
                // `CONSTANTS A, B, C` / `VARIABLES x, y` bare-name lists.
                // Advance one token and keep looking; a definition always
                // starts with an identifier, so nothing is masked.
                p.advance();
                continue;
            }
        };
        p.advance();
        let mut params = Vec::new();
        if p.eat(&Tok::LParen) {
            loop {
                match p.peek() {
                    Some(Tok::Ident(n)) => {
                        params.push(n.clone());
                        p.advance();
                    }
                    _ => return Err(p.err("expected a parameter name")),
                }
                if p.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
            p.expect(&Tok::RParen)?;
        }
        if !p.eat(&Tok::DefEq) {
            // Not a definition (e.g. leftover header words); restart after name.
            p.pos = save;
            p.advance();
            continue;
        }
        let body = p.parse_expr()?;
        defs.insert(
            name.clone(),
            Operator {
                name,
                params,
                body: Rc::new(body),
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser {
    toks: Vec<SpannedTok>,
    pos: usize,
}

impl Parser {
    fn new(toks: Vec<SpannedTok>) -> Self {
        Parser { toks, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.toks.len()
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|t| &t.tok)
    }

    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.pos + n).map(|t| &t.tok)
    }

    fn line(&self) -> usize {
        self.toks
            .get(self.pos.min(self.toks.len().saturating_sub(1)))
            .map(|t| t.line)
            .unwrap_or(0)
    }

    fn advance(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t.map(|s| s.tok)
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if self.peek() == Some(want) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, want: &Tok) -> Result<(), TlaError> {
        if self.eat(want) {
            Ok(())
        } else {
            Err(self.err(&format!("expected {want}, found {:?}", self.peek())))
        }
    }

    fn err(&self, message: &str) -> TlaError {
        TlaError::Parse {
            line: self.line(),
            message: message.to_string(),
        }
    }

    /// Skips an `EXTENDS ...` header line if present.
    fn skip_extends_line(&mut self) {
        if matches!(self.peek(), Some(Tok::Ident(n)) if n == "EXTENDS" || n == "MODULE") {
            while let Some(t) = self.peek() {
                if matches!(t, Tok::Ident(n) if n != "EXTENDS" && n != "MODULE") {
                    break;
                }
                self.advance();
            }
            // Consume comma-separated module names.
            while matches!(self.peek(), Some(Tok::Ident(_))) {
                self.advance();
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, TlaError> {
        self.parse_iff()
    }

    fn parse_iff(&mut self) -> Result<Expr, TlaError> {
        let mut left = self.parse_implies()?;
        while self.eat(&Tok::Iff) {
            let right = self.parse_implies()?;
            left = Expr::Iff(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_implies(&mut self) -> Result<Expr, TlaError> {
        let left = self.parse_or()?;
        if self.eat(&Tok::Implies) {
            let right = self.parse_implies()?;
            Ok(Expr::Implies(Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    /// `\/` binds looser than `/\`, in both list form (`\/ A \/ B`) and
    /// inline form (`A \/ B`).
    fn parse_or(&mut self) -> Result<Expr, TlaError> {
        if self.eat(&Tok::Or) {
            let mut items = vec![self.parse_and()?];
            while self.eat(&Tok::Or) {
                items.push(self.parse_and()?);
            }
            return Ok(Expr::Or(items));
        }
        let mut left = self.parse_and()?;
        while self.eat(&Tok::Or) {
            let right = self.parse_and()?;
            match &mut left {
                Expr::Or(items) => items.push(right),
                _ => left = Expr::Or(vec![left, right]),
            }
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, TlaError> {
        if self.eat(&Tok::And) {
            let mut items = vec![self.parse_not()?];
            while self.eat(&Tok::And) {
                items.push(self.parse_not()?);
            }
            return Ok(Expr::And(items));
        }
        let mut left = self.parse_not()?;
        while self.eat(&Tok::And) {
            let right = self.parse_not()?;
            match &mut left {
                Expr::And(items) => items.push(right),
                _ => left = Expr::And(vec![left, right]),
            }
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, TlaError> {
        if self.eat(&Tok::Not) {
            Ok(Expr::Not(Box::new(self.parse_not()?)))
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<Expr, TlaError> {
        let mut left = self.parse_setop()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Eq) => CmpOp::Eq,
                Some(Tok::Neq) => CmpOp::Neq,
                Some(Tok::Lt) => CmpOp::Lt,
                Some(Tok::Le) => CmpOp::Le,
                Some(Tok::Gt) => CmpOp::Gt,
                Some(Tok::Ge) => CmpOp::Ge,
                Some(Tok::In) => CmpOp::In,
                Some(Tok::NotIn) => CmpOp::NotIn,
                Some(Tok::SubsetEq) => CmpOp::SubsetEq,
                _ => break,
            };
            self.advance();
            let right = self.parse_setop()?;
            left = Expr::Cmp(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// A `\X` product binds tighter than `\cap`/`\cup` and is associative.
    fn parse_times(&mut self) -> Result<Expr, TlaError> {
        let mut factors = vec![self.parse_postfix()?];
        while self.eat(&Tok::Times) {
            factors.push(self.parse_postfix()?);
        }
        if factors.len() == 1 {
            Ok(factors.pop().expect("one factor"))
        } else {
            Ok(Expr::Cartesian(factors))
        }
    }

    fn parse_setop(&mut self) -> Result<Expr, TlaError> {
        let mut left = self.parse_times()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Cup) => SetOp::Cup,
                Some(Tok::Cap) => SetOp::Cap,
                _ => break,
            };
            self.advance();
            let right = self.parse_times()?;
            left = Expr::SetOp(op, Box::new(left), Box::new(right));
        }
        if self.eat(&Tok::Range) {
            let right = self.parse_postfix()?;
            return Ok(Expr::Range(Box::new(left), Box::new(right)));
        }
        Ok(left)
    }

    fn parse_postfix(&mut self) -> Result<Expr, TlaError> {
        let mut e = self.parse_primary()?;
        loop {
            if self.eat(&Tok::Prime) {
                e = Expr::Prime(Box::new(e));
                continue;
            }
            if self.peek() == Some(&Tok::LBracket) && self.is_application_start() {
                self.advance();
                let mut args = vec![self.parse_expr()?];
                while self.eat(&Tok::Comma) {
                    args.push(self.parse_expr()?);
                }
                self.expect(&Tok::RBracket)?;
                e = Expr::Apply(Box::new(e), args);
                continue;
            }
            break;
        }
        Ok(e)
    }

    /// Distinguishes `f[e]` (application) from a following `[A]_v` or
    /// definition start. After a completed primary, `[` is always an
    /// application bracket in this fragment.
    fn is_application_start(&self) -> bool {
        true
    }

    fn parse_primary(&mut self) -> Result<Expr, TlaError> {
        let tok = self
            .peek()
            .cloned()
            .ok_or_else(|| self.err("unexpected end of input"))?;
        match tok {
            Tok::Int(v) => {
                self.advance();
                Ok(Expr::Int(v))
            }
            Tok::Str(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Tok::Forall | Tok::Exists => {
                self.advance();
                let is_forall = matches!(tok, Tok::Forall);
                let mut binders = Vec::new();
                loop {
                    let var = match self.advance() {
                        Some(Tok::Ident(n)) => n,
                        _ => return Err(self.err("expected a bound variable")),
                    };
                    self.expect(&Tok::In)?;
                    let domain = self.parse_expr()?;
                    binders.push((var, domain));
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::Colon)?;
                // The body extends maximally, matching TLA+ precedence.
                let body = self.parse_expr()?;
                if is_forall {
                    Ok(Expr::Forall {
                        binders,
                        body: Box::new(body),
                    })
                } else {
                    Ok(Expr::Exists {
                        binders,
                        body: Box::new(body),
                    })
                }
            }
            Tok::At => {
                self.advance();
                Ok(Expr::At)
            }
            Tok::Box => {
                self.advance();
                let inner = self.parse_postfix()?;
                Ok(Expr::Always(Box::new(inner)))
            }
            Tok::LParen => {
                self.advance();
                let inner = self.parse_expr()?;
                self.expect(&Tok::RParen)?;
                Ok(inner)
            }
            Tok::LAngle => {
                self.advance();
                let mut items = Vec::new();
                if !self.eat(&Tok::RAngle) {
                    items.push(self.parse_expr()?);
                    while self.eat(&Tok::Comma) {
                        items.push(self.parse_expr()?);
                    }
                    self.expect(&Tok::RAngle)?;
                }
                Ok(Expr::Tuple(items))
            }
            Tok::LBrace => {
                self.advance();
                if self.eat(&Tok::RBrace) {
                    return Ok(Expr::SetLit(vec![]));
                }
                // `{ x \in S : P }` — the filter form. Try it first; if no
                // colon follows the domain, reparse as a general expression
                // (which may itself be a membership test inside a singleton).
                if let (Some(Tok::Ident(_)), Some(Tok::In)) =
                    (self.peek(), self.peek_at(1))
                {
                    let save = self.pos;
                    let var = match self.advance() {
                        Some(Tok::Ident(n)) => n,
                        _ => unreachable!(),
                    };
                    self.expect(&Tok::In)?;
                    let domain = self.parse_expr()?;
                    if self.eat(&Tok::Colon) {
                        let predicate = self.parse_expr()?;
                        self.expect(&Tok::RBrace)?;
                        return Ok(Expr::SetFilter {
                            var,
                            domain: Box::new(domain),
                            predicate: Box::new(predicate),
                        });
                    }
                    self.pos = save;
                }
                let first = self.parse_expr()?;
                if self.eat(&Tok::Colon) {
                    // { body : x \in S }
                    let var = match self.advance() {
                        Some(Tok::Ident(n)) => n,
                        _ => return Err(self.err("expected a bound variable in the set map")),
                    };
                    self.expect(&Tok::In)?;
                    let domain = self.parse_expr()?;
                    self.expect(&Tok::RBrace)?;
                    return Ok(Expr::SetMap {
                        body: Box::new(first),
                        var,
                        domain: Box::new(domain),
                    });
                }
                let mut items = vec![first];
                while self.eat(&Tok::Comma) {
                    items.push(self.parse_expr()?);
                }
                self.expect(&Tok::RBrace)?;
                Ok(Expr::SetLit(items))
            }
            Tok::LBracket => self.parse_bracket_expr(),
            Tok::Ident(name) => self.parse_ident_expr(name),
            other => Err(self.err(&format!("unexpected token {other}"))),
        }
    }

    fn parse_bracket_expr(&mut self) -> Result<Expr, TlaError> {
        self.expect(&Tok::LBracket)?;
        // [ x \in S |-> e ]
        if let (Some(Tok::Ident(_)), Some(Tok::In)) = (self.peek(), self.peek_at(1)) {
            let var = match self.advance() {
                Some(Tok::Ident(n)) => n,
                _ => unreachable!(),
            };
            self.expect(&Tok::In)?;
            let domain = self.parse_expr()?;
            self.expect(&Tok::Map)?;
            let body = self.parse_expr()?;
            self.expect(&Tok::RBracket)?;
            return Ok(Expr::FunCtor {
                var,
                domain: Box::new(domain),
                body: Box::new(body),
            });
        }
        let first = self.parse_expr()?;
        if self.eat(&Tok::Arrow) {
            let range = self.parse_expr()?;
            self.expect(&Tok::RBracket)?;
            return Ok(Expr::FunSpace(Box::new(first), Box::new(range)));
        }
        if self.eat(&Tok::Map) {
            // `[ k1 |-> v1, k2 |-> v2, ... ]`
            let mut entries = vec![(first, self.parse_expr()?)];
            while self.eat(&Tok::Comma) {
                let k = self.parse_expr()?;
                self.expect(&Tok::Map)?;
                let v = self.parse_expr()?;
                entries.push((k, v));
            }
            self.expect(&Tok::RBracket)?;
            return Ok(Expr::FunLit(entries));
        }
        if matches!(self.peek(), Some(Tok::Ident(n)) if n == "EXCEPT") {
            self.advance();
            let mut updates = Vec::new();
            loop {
                self.expect(&Tok::Bang)?;
                let mut path = Vec::new();
                loop {
                    self.expect(&Tok::LBracket)?;
                    path.push(self.parse_expr()?);
                    self.expect(&Tok::RBracket)?;
                    if self.peek() != Some(&Tok::LBracket) {
                        break;
                    }
                }
                self.expect(&Tok::Eq)?;
                let value = self.parse_expr()?;
                updates.push((path, value));
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.expect(&Tok::RBracket)?;
            return Ok(Expr::Except {
                base: Box::new(first),
                updates,
            });
        }
        if self.eat(&Tok::RBracket) {
            // `[A]_v`: the subscript is either `_` `v` (when `_` lexed alone)
            // or a single identifier `_v` (since `_` can start an identifier).
            let vars = match self.advance() {
                Some(Tok::Underscore) => self.parse_postfix()?,
                Some(Tok::Ident(n)) if n.starts_with('_') && n.len() > 1 => {
                    let rest = n[1..].to_string();
                    if self.peek() == Some(&Tok::LParen) {
                        self.advance();
                        let mut args = vec![self.parse_expr()?];
                        while self.eat(&Tok::Comma) {
                            args.push(self.parse_expr()?);
                        }
                        self.expect(&Tok::RParen)?;
                        Expr::Apply(Box::new(Expr::Ident(rest)), args)
                    } else {
                        Expr::Ident(rest)
                    }
                }
                other => {
                    return Err(self.err(&format!(
                        "expected `_vars` after the bracketed action, found {other:?}"
                    )));
                }
            };
            return Ok(Expr::ActionSubscript {
                action: Box::new(first),
                vars: Box::new(vars),
            });
        }
        Err(self.err("malformed bracketed expression"))
    }

    fn parse_ident_expr(&mut self, name: String) -> Result<Expr, TlaError> {
        match name.as_str() {
            "TRUE" => {
                self.advance();
                return Ok(Expr::Bool(true));
            }
            "FALSE" => {
                self.advance();
                return Ok(Expr::Bool(false));
            }
            "BOOLEAN" => {
                self.advance();
                return Ok(Expr::SetLit(vec![Expr::Bool(false), Expr::Bool(true)]));
            }
            "CHOOSE" => {
                self.advance();
                let var = match self.advance() {
                    Some(Tok::Ident(n)) => n,
                    _ => return Err(self.err("expected a variable after CHOOSE")),
                };
                self.expect(&Tok::In)?;
                let domain = self.parse_expr()?;
                self.expect(&Tok::Colon)?;
                let predicate = self.parse_expr()?;
                return Ok(Expr::Choose {
                    var,
                    domain: Box::new(domain),
                    predicate: Box::new(predicate),
                });
            }
            "LET" => {
                self.advance();
                let mut defs = Vec::new();
                loop {
                    let dname = match self.advance() {
                        Some(Tok::Ident(n)) => n,
                        _ => return Err(self.err("expected an operator name after LET")),
                    };
                    self.expect(&Tok::DefEq)?;
                    let body = self.parse_expr()?;
                    defs.push((dname, body));
                    if matches!(self.peek(), Some(Tok::Ident(n)) if n == "IN") {
                        self.advance();
                        break;
                    }
                }
                let body = self.parse_expr()?;
                return Ok(Expr::Let {
                    defs,
                    body: Box::new(body),
                });
            }
            "UNION" => {
                self.advance();
                let inner = self.parse_postfix()?;
                return Ok(Expr::Union(Box::new(inner)));
            }
            "UNCHANGED" => {
                self.advance();
                let inner = self.parse_postfix()?;
                return Ok(Expr::Unchanged(Box::new(inner)));
            }
            "SUBSET" => {
                self.advance();
                let inner = self.parse_postfix()?;
                return Ok(Expr::Powerset(Box::new(inner)));
            }
            _ => {}
        }
        self.advance();
        if name == "Seq" && self.peek() == Some(&Tok::LParen) {
            self.advance();
            let inner = self.parse_expr()?;
            self.expect(&Tok::RParen)?;
            return Ok(Expr::SeqOf(Box::new(inner)));
        }
        if (name.starts_with("WF_") || name.starts_with("SF_")) && self.peek() == Some(&Tok::LParen)
        {
            self.advance();
            let action = self.parse_expr()?;
            self.expect(&Tok::RParen)?;
            return Ok(Expr::Fairness {
                kind: name[..2].to_string(),
                vars: name[3..].to_string(),
                action: Box::new(action),
            });
        }
        if self.peek() == Some(&Tok::LParen) {
            self.advance();
            let mut args = vec![self.parse_expr()?];
            while self.eat(&Tok::Comma) {
                args.push(self.parse_expr()?);
            }
            self.expect(&Tok::RParen)?;
            return Ok(Expr::Apply(Box::new(Expr::Ident(name)), args));
        }
        Ok(Expr::Ident(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(src: &str) -> Operators {
        parse_module(src).expect("module must parse")
    }

    #[test]
    fn lexes_a_range_and_a_string_with_a_backslash_comment() {
        let toks = lex("0..3 \\* comment to eol\n\"a\\\"b\"").unwrap();
        let kinds: Vec<Tok> = toks.into_iter().map(|t| t.tok).collect();
        assert_eq!(
            kinds,
            vec![
                Tok::Int(0),
                Tok::Range,
                Tok::Int(3),
                Tok::Str("a\"b".to_string()),
            ]
        );
    }

    #[test]
    fn strips_nested_block_comments() {
        let toks = lex("A (* outer (* inner *) still *) B").unwrap();
        let kinds: Vec<Tok> = toks.into_iter().map(|t| t.tok).collect();
        assert_eq!(
            kinds,
            vec![
                Tok::Ident("A".to_string()),
                Tok::Ident("B".to_string()),
            ]
        );
    }

    #[test]
    fn parses_operator_definitions_with_parameters() {
        let defs = module("F(a, b) == a /\\ b\nG == 1 .. 3");
        assert_eq!(defs["F"].params, vec!["a", "b"]);
        assert!(matches!(defs["F"].body.as_ref(), Expr::And(_)));
        assert!(matches!(defs["G"].body.as_ref(), Expr::Range(_, _)));
    }

    #[test]
    fn parses_function_constructors_and_except_updates() {
        let defs = module("H == [ x \\in S |-> [ y \\in T |-> {} ] ]\nU == [ H EXCEPT ![a][b] = @ \\cup {1} ]");
        assert!(matches!(defs["H"].body.as_ref(), Expr::FunCtor { .. }));
        match defs["U"].body.as_ref() {
            Expr::Except { updates, .. } => {
                assert_eq!(updates.len(), 1);
                assert_eq!(updates[0].0.len(), 2);
            }
            other => panic!("expected an EXCEPT, got {other:?}"),
        }
    }

    #[test]
    fn parses_quantifiers_choose_and_let() {
        let defs = module(
            "Q == \\A a \\in Agents, s \\in Servers : a # s\nC == CHOOSE h \\in H : TRUE\nL == LET e == 1 IN e = 1",
        );
        assert!(matches!(defs["Q"].body.as_ref(), Expr::Forall { .. }));
        assert!(matches!(defs["C"].body.as_ref(), Expr::Choose { .. }));
        assert!(matches!(defs["L"].body.as_ref(), Expr::Let { .. }));
    }

    #[test]
    fn parses_temporal_spec_line() {
        let defs = module("Spec == Init /\\ [][Next]_vars /\\ WF_vars(Next)");
        match defs["Spec"].body.as_ref() {
            Expr::And(items) => {
                assert_eq!(items.len(), 3);
                assert!(matches!(items[1], Expr::Always(_)));
                assert!(matches!(items[2], Expr::Fairness { .. }));
            }
            other => panic!("expected conjunction, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unbalanced_constructs_with_a_line_number() {
        let err = parse_module("F == (1 .. 3").unwrap_err();
        match err {
            TlaError::Parse { line, .. } => assert!(line >= 1),
            other => panic!("expected parse error, got {other:?}"),
        }
    }
}
