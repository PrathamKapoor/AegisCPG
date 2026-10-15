//! # TLA+ value semantics and configuration loader
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! The evaluator for the parsed TLA+ fragment, plus the loader for TLC-style
//! `.cfg` files (`SPECIFICATION`/`INIT`/`NEXT`, `CONSTANTS`, `INVARIANTS`,
//! `CHECK_DEADLOCK`).
//!
//! ## WHY
//! A parser without a semantics is decoration. The model checker must decide
//! `state |= invariant` and `(state, state') |= action` by evaluating the
//! *file's own formulas*; otherwise the file could drift from the code and
//! both would still "pass". Values are a first-class TLA+ data model
//! (unbounded integers, sets, functions, tuples, sequences) rather than
//! Rust structs, so the formulae evaluate exactly as TLA+ defines them.
//!
//! ## HOW
//! [`Value`] implements the TLA+ data model; [`Env`] is a lexical scope with
//! frames plus the module's constants, operators, model values, and the
//! next-state binding used to resolve primes. [`eval`] is a total structural
//! interpreter over [`crate::tla_parser::Expr`]. [`parse_config`] reads a
//! `.cfg`, evaluates each constant assignment, and collects model values
//! (bare identifiers such as `s_filesystem`) exactly as TLC does.

use crate::tla_parser::{parse_expression, Expr, Operator, Operators, TlaError};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// A TLA+ value: the data model the specification is evaluated over.
///
/// TLA+ sequences ARE tuples (functions on `1..n`), so there is a single
/// representation for both.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Str(String),
    Set(BTreeSet<Value>),
    Tuple(Vec<Value>),
    /// A function with an explicit, finite domain.
    Fun(BTreeMap<Value, Value>),
    /// The *space* `[D -> R]`, kept lazy for membership tests.
    FunSpace(Box<Value>, Box<Value>),
    /// The *space* `Seq(S)`, kept lazy for membership tests.
    SeqOf(Box<Value>),
    /// The *space* `S1 \X S2 \X ...`, kept lazy. TLA+ defines `\X` as
    /// associative (elements are flat tuples); materialising the six-fold
    /// product in `TypeOK` on every invariant evaluation dominated the model
    /// checker's runtime, so membership is checked factor by factor instead.
    CartesianSpace(Vec<Value>),
}

impl Value {
    /// Builds a set value from an iterator.
    pub fn set<I: IntoIterator<Item = Value>>(items: I) -> Value {
        Value::Set(items.into_iter().collect())
    }

    /// Returns the boolean payload, or a type error naming the expectation.
    pub fn as_bool(&self) -> Result<bool, TlaError> {
        match self {
            Value::Bool(b) => Ok(*b),
            other => Err(TlaError::Eval(format!(
                "expected a boolean, found {}",
                other.render()
            ))),
        }
    }

    /// Returns the integer payload.
    pub fn as_int(&self) -> Result<i64, TlaError> {
        match self {
            Value::Int(v) => Ok(*v),
            other => Err(TlaError::Eval(format!(
                "expected an integer, found {}",
                other.render()
            ))),
        }
    }

    /// Returns the set payload.
    pub fn as_set(&self) -> Result<&BTreeSet<Value>, TlaError> {
        match self {
            Value::Set(s) => Ok(s),
            other => Err(TlaError::Eval(format!(
                "expected a set, found {}",
                other.render()
            ))),
        }
    }

    /// Returns the function map.
    pub fn as_fun(&self) -> Result<&BTreeMap<Value, Value>, TlaError> {
        match self {
            Value::Fun(f) => Ok(f),
            other => Err(TlaError::Eval(format!(
                "expected a function, found {}",
                other.render()
            ))),
        }
    }

    /// Returns the set payload, materialising lazy `\X` spaces.
    pub fn into_set(self) -> Result<BTreeSet<Value>, TlaError> {
        match self {
            Value::Set(s) => Ok(s),
            Value::CartesianSpace(factors) => cartesian_product(&factors),
            other => Err(TlaError::Eval(format!(
                "expected a set, found {}",
                other.render()
            ))),
        }
    }

    /// Returns the sequence payload (a TLA+ sequence is a tuple).
    pub fn as_seq(&self) -> Result<&Vec<Value>, TlaError> {
        match self {
            Value::Tuple(s) => Ok(s),
            other => Err(TlaError::Eval(format!(
                "expected a sequence, found {}",
                other.render()
            ))),
        }
    }

    /// A compact TLA+ rendering used in counterexample traces and errors.
    pub fn render(&self) -> String {
        match self {
            Value::Bool(b) => b.to_string().to_uppercase(),
            Value::Int(v) => v.to_string(),
            Value::Str(s) => format!("\"{s}\""),
            Value::Set(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.render()).collect();
                format!("{{{}}}", parts.join(", "))
            }
            Value::Tuple(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.render()).collect();
                format!("<<{}>>", parts.join(", "))
            }
            Value::Fun(map) => {
                let parts: Vec<String> = map
                    .iter()
                    .map(|(k, v)| format!("{} |-> {}", k.render(), v.render()))
                    .collect();
                format!("[{}]", parts.join(", "))
            }
            Value::FunSpace(d, r) => format!("[{} -> {}]", d.render(), r.render()),
            Value::SeqOf(s) => format!("Seq({})", s.render()),
            Value::CartesianSpace(factors) => {
                let parts: Vec<String> = factors.iter().map(|v| v.render()).collect();
                format!("({})", parts.join(" \\X "))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Environments
// ---------------------------------------------------------------------------

/// Lexical evaluation environment.
pub struct Env<'a> {
    /// Nested scopes; the last frame is innermost.
    pub frames: Vec<BTreeMap<String, Value>>,
    pub constants: &'a BTreeMap<String, Value>,
    pub operators: &'a Operators,
    pub model_values: &'a BTreeSet<String>,
    /// Binding for primed variables; `None` in state-only contexts.
    pub next_state: Option<&'a BTreeMap<String, Value>>,
    /// Active operator expansions; a definition that recurses through its own
    /// name is a malformed module, not a computation.
    pub expansion_depth: usize,
}

impl<'a> Env<'a> {
    /// Creates a root environment over the module's constants.
    pub fn root(
        constants: &'a BTreeMap<String, Value>,
        operators: &'a Operators,
        model_values: &'a BTreeSet<String>,
    ) -> Self {
        Env {
            frames: vec![BTreeMap::new()],
            constants,
            operators,
            model_values,
            next_state: None,
            expansion_depth: 0,
        }
    }

    /// Pushes a fresh child scope and returns its index.
    pub fn push(&mut self) -> usize {
        self.frames.push(BTreeMap::new());
        self.frames.len() - 1
    }

    /// Pops the innermost scope.
    pub fn pop(&mut self) {
        self.frames.pop();
    }

    /// Binds a name in the innermost scope.
    pub fn bind(&mut self, name: &str, value: Value) {
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(name.to_string(), value);
        }
    }

    /// Resolves a bare identifier.
    pub fn lookup(&self, name: &str) -> Option<&Value> {
        for frame in self.frames.iter().rev() {
            if let Some(v) = frame.get(name) {
                return Some(v);
            }
        }
        if let Some(v) = self.constants.get(name) {
            return Some(v);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluates an expression in an environment.
pub fn eval(expr: &Expr, env: &mut Env<'_>) -> Result<Value, TlaError> {
    match expr {
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        Expr::Int(v) => Ok(Value::Int(*v)),
        Expr::Str(s) => Ok(Value::Str(s.clone())),
        Expr::At => env
            .lookup("@")
            .cloned()
            .ok_or_else(|| TlaError::Eval("`@` used outside an EXCEPT update".to_string())),
        Expr::Ident(name) => {
            if let Some(v) = env.lookup(name) {
                return Ok(v.clone());
            }
            if env.model_values.contains(name) {
                return Ok(Value::Str(name.clone()));
            }
            // Zero-argument module definitions (`SecurityLevels == 0..3`,
            // `vars == <<...>>`) are operators, resolved with their bodies.
            if let Some(operator) = env.operators.get(name) {
                if operator.params.is_empty() {
                    let operator = operator.clone();
                    return apply_operator_values(env, &operator, Vec::new());
                }
            }
            Err(TlaError::UnknownOperator(name.clone()))
        }
        Expr::Prime(inner) => match inner.as_ref() {
            Expr::Ident(name) => match env.next_state {
                Some(ns) => ns
                    .get(name)
                    .cloned()
                    .ok_or_else(|| TlaError::Eval(format!("no next-state binding for `{name}`"))),
                None => Err(TlaError::Eval(format!(
                    "`{name}'` evaluated outside an action context"
                ))),
            },
            other => Err(TlaError::Eval(format!(
                "priming a non-variable expression: {other:?}"
            ))),
        },
        Expr::Not(inner) => Ok(Value::Bool(!eval(inner, env)?.as_bool()?)),
        Expr::And(items) => {
            for item in items {
                if !eval(item, env)?.as_bool()? {
                    return Ok(Value::Bool(false));
                }
            }
            Ok(Value::Bool(true))
        }
        Expr::Or(items) => {
            for item in items {
                if eval(item, env)?.as_bool()? {
                    return Ok(Value::Bool(true));
                }
            }
            Ok(Value::Bool(false))
        }
        Expr::Implies(a, b) => {
            let left = eval(a, env)?.as_bool()?;
            if left {
                Ok(Value::Bool(eval(b, env)?.as_bool()?))
            } else {
                Ok(Value::Bool(true))
            }
        }
        Expr::Iff(a, b) => {
            let left = eval(a, env)?.as_bool()?;
            let right = eval(b, env)?.as_bool()?;
            Ok(Value::Bool(left == right))
        }
        Expr::Cmp(op, a, b) => {
            use crate::tla_parser::CmpOp;
            let left = eval(a, env)?;
            let right = eval(b, env)?;
            let result = match op {
                CmpOp::Eq => left == right,
                CmpOp::Neq => left != right,
                CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => {
                    let l = left.as_int()?;
                    let r = right.as_int()?;
                    match op {
                        CmpOp::Lt => l < r,
                        CmpOp::Le => l <= r,
                        CmpOp::Gt => l > r,
                        _ => l >= r,
                    }
                }
                CmpOp::In | CmpOp::NotIn => {
                    let r = value_in(&left, &right)?;
                    match op {
                        CmpOp::In => r,
                        _ => !r,
                    }
                }
                CmpOp::SubsetEq => {
                    let l = left.into_set()?;
                    match right {
                        Value::CartesianSpace(factors) => {
                            let mut subset = true;
                            for item in &l {
                                if !value_in_space(item, &factors)? {
                                    subset = false;
                                    break;
                                }
                            }
                            subset
                        }
                        other => l.is_subset(&other.into_set()?),
                    }
                }
            };
            Ok(Value::Bool(result))
        }
        Expr::SetOp(op, a, b) => {
            use crate::tla_parser::SetOp;
            let left = eval(a, env)?;
            let right = eval(b, env)?;
            match op {
                SetOp::Cup => {
                    let mut l = left.into_set()?;
                    l.extend(right.into_set()?);
                    Ok(Value::Set(l))
                }
                SetOp::Cap => {
                    let r = right.into_set()?;
                    Ok(Value::Set(
                        left.into_set()?
                            .into_iter()
                            .filter(|v| r.contains(v))
                            .collect(),
                    ))
                }
            }
        }
        Expr::Cartesian(factors) => {
            let mut evaluated = Vec::with_capacity(factors.len());
            for factor in factors {
                evaluated.push(eval(factor, env)?);
            }
            Ok(Value::CartesianSpace(evaluated))
        }
        Expr::Range(a, b) => {
            let lo = eval(a, env)?.as_int()?;
            let hi = eval(b, env)?.as_int()?;
            let mut out = BTreeSet::new();
            let mut i = lo;
            while i <= hi {
                out.insert(Value::Int(i));
                i += 1;
            }
            Ok(Value::Set(out))
        }
        Expr::Tuple(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(eval(item, env)?);
            }
            Ok(Value::Tuple(out))
        }
        Expr::SetLit(items) => {
            let mut out = BTreeSet::new();
            for item in items {
                out.insert(eval(item, env)?);
            }
            Ok(Value::Set(out))
        }
        Expr::SetMap { body, var, domain } => {
            let dom = eval(domain, env)?.into_set()?;
            let mut out = BTreeSet::new();
            for element in dom {
                env.push();
                env.bind(var, element);
                let v = eval(body, env);
                env.pop();
                out.insert(v?);
            }
            Ok(Value::Set(out))
        }
        Expr::SetFilter {
            var,
            domain,
            predicate,
        } => {
            let dom = eval(domain, env)?.into_set()?;
            let mut out = BTreeSet::new();
            for element in dom {
                env.push();
                env.bind(var, element.clone());
                let keep = eval(predicate, env);
                env.pop();
                if keep?.as_bool()? {
                    out.insert(element);
                }
            }
            Ok(Value::Set(out))
        }
        Expr::FunCtor { var, domain, body } => {
            let dom = eval(domain, env)?.into_set()?;
            let mut out = BTreeMap::new();
            for element in dom {
                env.push();
                env.bind(var, element.clone());
                let v = eval(body, env);
                env.pop();
                out.insert(element, v?);
            }
            Ok(Value::Fun(out))
        }
        Expr::FunLit(entries) => {
            let mut out = BTreeMap::new();
            for (k, v) in entries {
                out.insert(eval(k, env)?, eval(v, env)?);
            }
            Ok(Value::Fun(out))
        }
        Expr::FunSpace(d, r) => Ok(Value::FunSpace(
            Box::new(eval(d, env)?),
            Box::new(eval(r, env)?),
        )),
        Expr::SeqOf(s) => Ok(Value::SeqOf(Box::new(eval(s, env)?))),
        Expr::Powerset(s) => {
            let base = eval(s, env)?.into_set()?;
            let elements: Vec<Value> = base.into_iter().collect();
            let mut out = BTreeSet::new();
            for mask in 0u64..(1u64 << elements.len()) {
                let mut subset = BTreeSet::new();
                for (i, e) in elements.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        subset.insert(e.clone());
                    }
                }
                out.insert(Value::Set(subset));
            }
            Ok(Value::Set(out))
        }
        Expr::Union(inner) => {
            let sets = eval(inner, env)?.into_set()?;
            let mut out = BTreeSet::new();
            for s in sets {
                out.extend(s.into_set()?);
            }
            Ok(Value::Set(out))
        }
        Expr::Unchanged(inner) => Err(TlaError::Eval(format!(
            "UNCHANGED is a transition construct, not a value: {inner:?}"
        ))),
        Expr::Except { base, updates } => {
            let mut current = eval(base, env)?;
            for (path, value_expr) in updates {
                let mut keys = Vec::with_capacity(path.len());
                for k in path {
                    keys.push(eval(k, env)?);
                }
                let old = get_path(&current, &keys)?;
                env.push();
                env.bind("@", old);
                let new_value = eval(value_expr, env);
                env.pop();
                set_path(&mut current, &keys, new_value?)?;
            }
            Ok(current)
        }
        Expr::Apply(f, args) => {
            if let Expr::Ident(name) = f.as_ref() {
                match name.as_str() {
                    "Append" if args.len() == 2 => {
                        let mut seq = eval(&args[0], env)?.as_seq()?.clone();
                        seq.push(eval(&args[1], env)?);
                        return Ok(Value::Tuple(seq));
                    }
                    "Len" if args.len() == 1 => {
                        let seq = eval(&args[0], env)?;
                        return Ok(Value::Int(seq.as_seq()?.len() as i64));
                    }
                    "Head" if args.len() == 1 => {
                        let seq = eval(&args[0], env)?;
                        return seq
                            .as_seq()?
                            .first()
                            .cloned()
                            .ok_or_else(|| TlaError::Eval("Head of an empty sequence".to_string()));
                    }
                    "Seq" if args.len() == 1 => {
                        return Ok(Value::SeqOf(Box::new(eval(&args[0], env)?)));
                    }
                    _ => {}
                }
                if let Some(operator) = env.operators.get(name) {
                    let operator = operator.clone();
                    return apply_operator(env, &operator, args);
                }
                // Not an operator: the name must resolve to a *value*
                // (`grantedCapabilities[a]`, `ResourceLevel[r]`, `f[k]`).
                // Only single-argument application exists for those.
                if args.len() != 1 {
                    return Err(TlaError::UnknownOperator(name.clone()));
                }
            }
            // Function application f[x]: args.len() == 1.
            if args.len() == 1 {
                let fun = eval(f, env)?;
                let key = eval(&args[0], env)?;
                return match fun {
                    Value::Fun(map) => map.get(&key).cloned().ok_or_else(|| {
                        TlaError::Eval(format!("function is undefined at {}", key.render()))
                    }),
                    Value::FunSpace(dom, range) => {
                        let d = dom.into_set()?;
                        if d.contains(&key) {
                            Ok(*range.clone())
                        } else {
                            Err(TlaError::Eval(format!(
                                "function space applied outside its domain at {}",
                                key.render()
                            )))
                        }
                    }
                    Value::Tuple(seq) => {
                        let idx = key.as_int()?;
                        if idx >= 1 && (idx as usize) <= seq.len() {
                            Ok(seq[(idx - 1) as usize].clone())
                        } else {
                            Err(TlaError::Eval(format!(
                                "sequence index {idx} out of bounds (len {})",
                                seq.len()
                            )))
                        }
                    }
                    other => Err(TlaError::Eval(format!(
                        "applied a non-function {}",
                        other.render()
                    ))),
                };
            }
            Err(TlaError::Eval("malformed application".to_string()))
        }
        Expr::Forall { binders, body } => {
            let bindings = enumerate_binders(binders, env)?;
            for combo in bindings {
                env.push();
                for (name, value) in combo.iter() {
                    env.bind(name, value.clone());
                }
                let ok = eval(body, env);
                env.pop();
                if !ok?.as_bool()? {
                    return Ok(Value::Bool(false));
                }
            }
            Ok(Value::Bool(true))
        }
        Expr::Exists { binders, body } => {
            let bindings = enumerate_binders(binders, env)?;
            for combo in bindings {
                env.push();
                for (name, value) in combo.iter() {
                    env.bind(name, value.clone());
                }
                let ok = eval(body, env);
                env.pop();
                if ok?.as_bool()? {
                    return Ok(Value::Bool(true));
                }
            }
            Ok(Value::Bool(false))
        }
        Expr::Choose {
            var,
            domain,
            predicate,
        } => {
            let dom = eval(domain, env)?.into_set()?;
            for element in dom {
                env.push();
                env.bind(var, element.clone());
                let ok = eval(predicate, env);
                env.pop();
                if ok?.as_bool()? {
                    return Ok(element);
                }
            }
            Err(TlaError::Eval(
                "CHOOSE found no element satisfying its predicate".to_string(),
            ))
        }
        Expr::Let { defs, body } => {
            let scope = env.push();
            for (name, def_body) in defs {
                let v = eval(def_body, env)?;
                env.frames[scope].insert(name.clone(), v);
            }
            let out = eval(body, env)?;
            env.pop();
            Ok(out)
        }
        Expr::Fairness { .. } | Expr::Always(_) | Expr::ActionSubscript { .. } => Err(
            TlaError::Eval("temporal operators cannot be evaluated pointwise".to_string()),
        ),
    }
}

/// Applies a defined operator to already-evaluated arguments.
pub fn apply_operator(
    env: &mut Env<'_>,
    operator: &Operator,
    args: &[Expr],
) -> Result<Value, TlaError> {
    if operator.params.len() != args.len() {
        return Err(TlaError::Eval(format!(
            "operator `{}` expects {} arguments, got {}",
            operator.name,
            operator.params.len(),
            args.len()
        )));
    }
    let mut values = Vec::with_capacity(args.len());
    for a in args {
        values.push(eval(a, env)?);
    }
    apply_operator_values(env, operator, values)
}

/// Applies an operator to pre-evaluated argument values.
pub fn apply_operator_values(
    env: &mut Env<'_>,
    operator: &Operator,
    values: Vec<Value>,
) -> Result<Value, TlaError> {
    if env.expansion_depth >= 64 {
        return Err(TlaError::Eval(format!(
            "operator expansion depth exceeded at `{}` (recursive definition?)",
            operator.name
        )));
    }
    env.expansion_depth += 1;
    env.push();
    for (name, value) in operator.params.iter().zip(values) {
        env.bind(name, value);
    }
    let out = eval(&operator.body, env);
    env.pop();
    env.expansion_depth -= 1;
    out
}

/// Enumerates the Cartesian product of a quantifier's binders.
pub fn enumerate_binders(
    binders: &[(String, Expr)],
    env: &mut Env<'_>,
) -> Result<Vec<Vec<(String, Value)>>, TlaError> {
    let mut combos: Vec<Vec<(String, Value)>> = vec![Vec::new()];
    for (name, domain) in binders {
        let dom = eval(domain, env)?.as_set()?.clone();
        let mut next = Vec::new();
        for combo in &combos {
            for element in &dom {
                let mut c = combo.clone();
                c.push((name.clone(), element.clone()));
                next.push(c);
            }
        }
        combos = next;
    }
    Ok(combos)
}

fn value_in(element: &Value, container: &Value) -> Result<bool, TlaError> {
    match container {
        Value::Set(s) => Ok(s.contains(element)),
        Value::CartesianSpace(factors) => value_in_space(element, factors),
        Value::FunSpace(domain, range) => {
            // Membership in [D -> R]: x must be a function whose domain is D
            // and whose values all lie in R.
            let d = domain.clone().into_set()?;
            match element {
                Value::Fun(map) => {
                    if map.keys().ne(d.iter()) {
                        return Ok(false);
                    }
                    for v in map.values() {
                        if !value_in(v, range)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                _ => Ok(false),
            }
        }
        Value::SeqOf(s) => match element {
            Value::Tuple(items) => {
                for item in items {
                    if !value_in(item, s)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        },
        other => Err(TlaError::Eval(format!(
            "`in` applied to a non-set {}",
            other.render()
        ))),
    }
}

/// Membership in an `S1 \X ... \Xn` space: the element must be a flat
/// n-tuple whose components lie in the corresponding factors.
fn value_in_space(element: &Value, factors: &[Value]) -> Result<bool, TlaError> {
    match element {
        Value::Tuple(items) if items.len() == factors.len() => {
            for (item, factor) in items.iter().zip(factors.iter()) {
                if !value_in(item, factor)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Materialises an `S1 \X ... \Xn` space into the set of flat n-tuples.
fn cartesian_product(factors: &[Value]) -> Result<BTreeSet<Value>, TlaError> {
    let mut out: BTreeSet<Value> = BTreeSet::new();
    for (index, factor) in factors.iter().enumerate() {
        let set = factor.clone().into_set()?;
        if index == 0 {
            for v in set {
                out.insert(Value::Tuple(vec![v]));
            }
            continue;
        }
        let mut next: BTreeSet<Value> = BTreeSet::new();
        for tuple in &out {
            if let Value::Tuple(items) = tuple {
                for v in &set {
                    let mut items = items.clone();
                    items.push(v.clone());
                    next.insert(Value::Tuple(items));
                }
            }
        }
        out = next;
    }
    Ok(out)
}

fn get_path(value: &Value, keys: &[Value]) -> Result<Value, TlaError> {
    let mut current = value;
    for key in keys {
        match current {
            Value::Fun(map) => {
                current = map.get(key).ok_or_else(|| {
                    TlaError::Eval(format!("EXCEPT path is undefined at {}", key.render()))
                })?;
            }
            other => {
                return Err(TlaError::Eval(format!(
                    "EXCEPT path traverses a non-function {}",
                    other.render()
                )));
            }
        }
    }
    Ok(current.clone())
}

fn set_path(value: &mut Value, keys: &[Value], new_value: Value) -> Result<(), TlaError> {
    if keys.is_empty() {
        *value = new_value;
        return Ok(());
    }
    match value {
        Value::Fun(map) => {
            if keys.len() == 1 {
                map.insert(keys[0].clone(), new_value);
                Ok(())
            } else {
                let entry = map.get_mut(&keys[0]).ok_or_else(|| {
                    TlaError::Eval(format!(
                        "EXCEPT update targets an undefined key {}",
                        keys[0].render()
                    ))
                })?;
                set_path(entry, &keys[1..], new_value)
            }
        }
        other => Err(TlaError::Eval(format!(
            "EXCEPT update traverses a non-function {}",
            other.render()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Configuration files
// ---------------------------------------------------------------------------

/// A parsed TLC-style `.cfg` model configuration.
#[derive(Debug, Clone)]
pub struct TlaConfig {
    pub constants: BTreeMap<String, Value>,
    pub model_values: BTreeSet<String>,
    pub spec: Option<String>,
    pub init: Option<String>,
    pub next: Option<String>,
    pub invariants: Vec<String>,
    pub check_deadlock: Option<bool>,
}

/// Strips `\* comments` from a config line (outside string literals).
fn strip_line_comment(line: &str) -> String {
    let mut out = String::new();
    let mut in_string = false;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == '\\' && i + 1 < chars.len() && chars[i + 1] == '*' {
            break;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Parses a `.cfg` file body.
///
/// Constant assignments may span lines; continuation lines are indented.
pub fn parse_config(src: &str) -> Result<TlaConfig, TlaError> {
    #[derive(PartialEq)]
    enum Section {
        None,
        Constants,
        Invariants,
    }
    let mut cfg = TlaConfig {
        constants: BTreeMap::new(),
        model_values: BTreeSet::new(),
        spec: None,
        init: None,
        next: None,
        invariants: Vec::new(),
        check_deadlock: None,
    };
    let mut section = Section::None;
    let mut pending_name: Option<String> = None;
    let mut pending_value = String::new();
    let mut assignments: Vec<(String, String)> = Vec::new();

    let flush = |pending_name: &mut Option<String>,
                 pending_value: &mut String,
                 assignments: &mut Vec<(String, String)>| {
        if let Some(name) = pending_name.take() {
            assignments.push((name, std::mem::take(pending_value)));
        }
    };

    for raw_line in src.lines() {
        let without_comment = strip_line_comment(raw_line);
        let line = without_comment.trim_end();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // Blank lines are separators *inside* the CONSTANTS section, not
            // section terminators; the real `.cfg` files use one between
            // every assignment. Sections end at non-indented directives.
            continue;
        }
        let indented = line.starts_with(' ') || line.starts_with('\t');
        if !indented {
            if let Some(rest) = trimmed.strip_prefix("SPECIFICATION") {
                flush(&mut pending_name, &mut pending_value, &mut assignments);
                section = Section::None;
                cfg.spec = Some(rest.trim().to_string());
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("INIT") {
                if rest.chars().next().map_or(false, char::is_whitespace) {
                    flush(&mut pending_name, &mut pending_value, &mut assignments);
                    section = Section::None;
                    cfg.init = Some(rest.trim().to_string());
                    continue;
                }
            }
            if let Some(rest) = trimmed.strip_prefix("NEXT") {
                if rest.chars().next().map_or(false, char::is_whitespace) {
                    flush(&mut pending_name, &mut pending_value, &mut assignments);
                    section = Section::None;
                    cfg.next = Some(rest.trim().to_string());
                    continue;
                }
            }
            if let Some(rest) = trimmed.strip_prefix("INVARIANTS") {
                flush(&mut pending_name, &mut pending_value, &mut assignments);
                if rest.trim().is_empty() {
                    section = Section::Invariants;
                    continue;
                }
                return Err(TlaError::Parse {
                    line: 0,
                    message: "INVARIANTS takes no inline value".to_string(),
                });
            }
            if trimmed == "CONSTANTS" {
                flush(&mut pending_name, &mut pending_value, &mut assignments);
                section = Section::Constants;
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("CONSTANT") {
                flush(&mut pending_name, &mut pending_value, &mut assignments);
                if rest.trim().is_empty() {
                    section = Section::Constants;
                }
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("CHECK_DEADLOCK") {
                flush(&mut pending_name, &mut pending_value, &mut assignments);
                section = Section::None;
                match rest.trim() {
                    "TRUE" => cfg.check_deadlock = Some(true),
                    "FALSE" => cfg.check_deadlock = Some(false),
                    other => {
                        return Err(TlaError::Eval(format!(
                            "CHECK_DEADLOCK must be TRUE or FALSE, found `{other}`"
                        )));
                    }
                }
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("SYMMETRY") {
                flush(&mut pending_name, &mut pending_value, &mut assignments);
                section = Section::None;
                if rest.trim().is_empty() {
                    return Err(TlaError::Eval("SYMMETRY requires an operator".to_string()));
                }
                continue;
            }
            // Unknown top-level directive: fail closed rather than silently
            // ignoring a safety-relevant setting such as PROPERTY.
            return Err(TlaError::Parse {
                line: 0,
                message: format!("unknown configuration directive `{trimmed}`"),
            });
        }
        match section {
            Section::Constants => {
                // Continuation of the previous assignment, or a new one.
                if let Some(eq) = trimmed.find('=') {
                    let name: String = trimmed[..eq].trim().to_string();
                    let is_new_assignment = !name.is_empty()
                        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                        && pending_name.as_deref() != Some(name.as_str());
                    if is_new_assignment {
                        flush(&mut pending_name, &mut pending_value, &mut assignments);
                        pending_name = Some(name);
                        pending_value = trimmed[eq + 1..].trim().to_string();
                        continue;
                    }
                }
                if pending_name.is_none() {
                    return Err(TlaError::Parse {
                        line: 0,
                        message: format!("constant continuation without an assignment: `{trimmed}`"),
                    });
                }
                if !pending_value.is_empty() {
                    pending_value.push(' ');
                }
                pending_value.push_str(trimmed);
            }
            Section::Invariants => {
                cfg.invariants.push(trimmed.to_string());
            }
            Section::None => {
                return Err(TlaError::Parse {
                    line: 0,
                    message: format!("unexpected indented line outside a section: `{trimmed}`"),
                });
            }
        }
    }
    flush(&mut pending_name, &mut pending_value, &mut assignments);

    // Evaluate assignments in file order. Model values are bare identifiers;
    // they are collected by scanning the parsed value expressions.
    let mut parsed: Vec<(String, Expr)> = Vec::new();
    let mut model_values: BTreeSet<String> = BTreeSet::new();
    for (name, text) in &assignments {
        let expr = parse_expression(text).map_err(|e| TlaError::Eval(format!(
            "constant `{name}` does not parse: {e}"
        )))?;
        collect_value_idents(&expr, &mut model_values)?;
        parsed.push((name.clone(), expr));
    }
    let mut constants: BTreeMap<String, Value> = BTreeMap::new();
    for (name, expr) in &parsed {
        // Constants cannot use anything but earlier constants and model values.
        let operators = Operators::new();
        let mut env = Env::root(&constants, &operators, &model_values);
        let value = eval(expr, &mut env).map_err(|e| {
            TlaError::Eval(format!("constant `{name}` failed to evaluate: {e}"))
        })?;
        constants.insert(name.clone(), value);
    }
    // Identifiers that resolved to nothing are model values; drop the names
    // of actual constants from the model-value set.
    for name in constants.keys() {
        model_values.remove(name);
    }
    cfg.constants = constants;
    cfg.model_values = model_values;
    Ok(cfg)
}

/// Collects identifiers appearing in a configuration value expression.
///
/// Configuration values are data, not formulas: only literals, sets, tuples,
/// function literals, and bare identifiers are legal.
pub fn collect_value_idents(expr: &Expr, out: &mut BTreeSet<String>) -> Result<(), TlaError> {
    match expr {
        Expr::Ident(name) => {
            out.insert(name.clone());
            Ok(())
        }
        Expr::Bool(_) | Expr::Int(_) | Expr::Str(_) => Ok(()),
        Expr::SetLit(items) => {
            for item in items {
                collect_value_idents(item, out)?;
            }
            Ok(())
        }
        Expr::Tuple(items) => {
            for item in items {
                collect_value_idents(item, out)?;
            }
            Ok(())
        }
        Expr::FunLit(entries) => {
            for (k, v) in entries {
                collect_value_idents(k, out)?;
                collect_value_idents(v, out)?;
            }
            Ok(())
        }
        _ => Err(TlaError::Eval(
            "unsupported construct in a .cfg constant value".to_string(),
        )),
    }
}

/// Parses the `VARIABLES ...` block of a module.
///
/// Variables are the identifiers following the `VARIABLES` keyword up to the
/// next definition (`vars == ...` in our module, detected as an identifier
/// immediately followed by `==`).
pub fn parse_variables(src: &str) -> Result<Vec<String>, TlaError> {
    use crate::tla_parser::{lex, Tok};
    let toks = lex(src)?;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < toks.len() {
        if matches!(&toks[i].tok, Tok::Ident(n) if n == "VARIABLES") {
            i += 1;
            while i < toks.len() {
                match &toks[i].tok {
                    Tok::Ident(name) => {
                        // `Name == ` starts the next definition.
                        if matches!(toks.get(i + 1).map(|t| &t.tok), Some(Tok::DefEq)) {
                            break;
                        }
                        out.push(name.clone());
                        i += 1;
                        // Optional comma between names.
                        if matches!(toks.get(i).map(|t| &t.tok), Some(Tok::Comma)) {
                            i += 1;
                        }
                    }
                    _ => break,
                }
            }
            return Ok(out);
        }
        i += 1;
    }
    Err(TlaError::MissingDirective(
        "VARIABLES block".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tla_parser::parse_expression as pe;

    fn eval_src(src: &str, constants: &BTreeMap<String, Value>, model_values: &BTreeSet<String>) -> Value {
        let expr = pe(src).expect("expression must parse");
        let operators = Operators::new();
        let mut env = Env::root(constants, &operators, model_values);
        eval(&expr, &mut env).expect("expression must evaluate")
    }

    fn empty() -> (BTreeMap<String, Value>, BTreeSet<String>) {
        (BTreeMap::new(), BTreeSet::new())
    }

    #[test]
    fn evaluates_set_operations_lazily_and_eagerly() {
        let (c, m) = empty();
        let v = eval_src("{1, 2, 3} \\cup {3, 4}", &c, &m);
        assert_eq!(v.as_set().unwrap().len(), 4);
        let v = eval_src("{1, 2, 3} \\cap {3, 4}", &c, &m);
        assert_eq!(v.render(), "{3}");
        // `\X` stays lazy and membership is checked factor by factor...
        let v = eval_src("{1, 2} \\X {3}", &c, &m);
        assert!(matches!(v, Value::CartesianSpace(ref factors) if factors.len() == 2));
        assert_eq!(
            eval_src("<<1, 3>> \\in {1, 2} \\X {3}", &c, &m),
            Value::Bool(true)
        );
        // ...a flat tuple of the wrong arity is not an element...
        assert_eq!(
            eval_src("<<1, 3, 4>> \\in {1, 2} \\X {3}", &c, &m),
            Value::Bool(false)
        );
        // ...and materialisation produces flat tuples (TLA+ makes `\X`
        // associative: three or more factors are n-tuples, not nestings).
        let v = eval_src("{1, 2} \\X {3} \\X {4, 5}", &c, &m);
        let materialised = v.into_set().unwrap();
        assert_eq!(materialised.len(), 4);
        assert!(materialised.contains(&Value::Tuple(vec![
            Value::Int(1),
            Value::Int(3),
            Value::Int(4)
        ])));
        let v = eval_src("1..3", &c, &m);
        assert_eq!(v.as_set().unwrap().len(), 3);
    }

    #[test]
    fn evaluates_function_constructors_and_except_updates() {
        let (c, m) = empty();
        // Function constructor with a range domain.
        let v = eval_src("[ x \\in 1..2 |-> x ]", &c, &m);
        let f = v.as_fun().unwrap();
        assert_eq!(f.get(&Value::Int(1)), Some(&Value::Int(1)));
        // Nested EXCEPT update using `@`.
        let v = eval_src(
            "[ [ x \\in 1..2 |-> {x} ] EXCEPT ![1] = @ \\cup {9} ]",
            &c,
            &m,
        );
        let f = v.as_fun().unwrap();
        assert_eq!(f.get(&Value::Int(1)).unwrap().as_set().unwrap().len(), 2);
        // Function application and Len/Append builtins.
        let v = eval_src("Len(Append(<<1, 2>>, 3))", &c, &m);
        assert_eq!(v, Value::Int(3));
    }

    #[test]
    fn parses_and_evaluates_the_real_cfg_constant_shape() {
        let cfg_src = r#"
SPECIFICATION Spec

CONSTANTS
    Agents   = { a_editor, a_attacker }
    ServerOf = [ t_read_env     |-> s_filesystem,
                 t_read_doc      |-> s_filesystem ]
    ServerClearance = [ s_filesystem |-> 2,
                        s_markdown    |-> 1 ]
    GatewayEnforced = FALSE
    VerifiedDeclassifications = { <<r_dotenv, s_markdown>> }

INVARIANTS
    TypeOK
    ToolIntegrityInvariant

CHECK_DEADLOCK FALSE
"#;
        let cfg = parse_config(cfg_src).expect("cfg must parse");
        assert_eq!(cfg.invariants, vec!["TypeOK", "ToolIntegrityInvariant"]);
        assert_eq!(cfg.spec.as_deref(), Some("Spec"));
        assert_eq!(cfg.check_deadlock, Some(false));
        assert_eq!(
            cfg.constants.get("GatewayEnforced"),
            Some(&Value::Bool(false))
        );
        assert!(cfg.model_values.contains("a_editor"));
        assert!(cfg.model_values.contains("s_filesystem"));
        let agents = cfg.constants.get("Agents").unwrap().as_set().unwrap();
        assert_eq!(agents.len(), 2);
        let clr = cfg
            .constants
            .get("ServerClearance")
            .unwrap()
            .as_fun()
            .unwrap();
        assert_eq!(
            clr.get(&Value::Str("s_filesystem".to_string())),
            Some(&Value::Int(2))
        );
    }

    #[test]
    fn rejects_unknown_directives_fail_closed() {
        let err = parse_config("PROPERTY Liveness\n").unwrap_err();
        assert!(matches!(err, TlaError::Parse { .. }));
    }
}
