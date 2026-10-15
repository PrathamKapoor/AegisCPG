//! The RFC 9669 eBPF instruction set: an assembler, a verifier, and a VM.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! A self-contained implementation of the 64-bit eBPF ISA defined by
//! [RFC 9669](https://www.rfc-editor.org/rfc/rfc9669): instruction encoding
//! and decoding, a text assembler for the LLVM BPF dialect, a static verifier,
//! and an interpreter with a 512-byte stack, a context region, hash and
//! LPM-trie maps, helper calls, and fuel-bounded execution.
//!
//! ## WHY
//! The LSM enforcer's decision table exists in three places: the C hooks
//! (`ebpf-probes/lsm_enforcer.bpf.c`), the Rust reference
//! ([`crate::policy_compiler::KernelLsmSimulator`]), and the actual BPF program
//! the kernel would execute. The first two are compared by the loader's test
//! suite; the third cannot be loaded on a Windows host, so instead the *same
//! three decisions are written as BPF assembly* and executed here, against the
//! same compiled maps, and the verdicts are compared case by case in
//! `tests/shim_equivalence.rs`. If the assembly form of a decision and the C
//! form ever disagree, that test fails with the disagreement -- which is the
//! only way to find out before a kernel does.
//!
//! ## HOW
//! * [`assemble`] parses the LLVM-dialect subset this crate emits. Every form
//!   is byte-compared against `llvm-mc -triple=bpfel` in the test suite, so the
//!   encoder is pinned to LLVM's output, not to my reading of the spec.
//! * [`verify`] performs the checks that make execution safe rather than
//!   merely correct: instruction validity, jump targets inside the program and
//!   never off a wide-immediate slot pair, stack accesses inside the frame, the
//!   frame pointer read-only, dataflow-defined registers at every read, known
//!   helper ids, and no fallthrough off the end.
//! * [`Vm::run`] decodes, verifies, and executes. Memory access is checked
//!   against the region actually addressed (context, stack frame, map value),
//!   so a scalar used as a pointer is a runtime trap, not a read of host
//!   memory. Fuel bounds every run, so a compiler bug cannot hang a test.
//!
//! Out of scope, and rejected rather than approximated: BTF type info,
//! relocations (a wide-immediate load names a map id directly), tail calls,
//! atomic operations, and the BTF-extended ALU/MOV encodings.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// The eBPF stack frame size, in bytes.
pub const STACK_SIZE: usize = 512;
/// The frame pointer register, r10.
pub const FRAME_POINTER: u8 = 10;
/// The number of named registers, r0 through r10.
pub const REGISTER_COUNT: usize = 11;
/// The instruction budget one run gets unless overridden.
pub const DEFAULT_FUEL: u64 = 1_000_000;

/// Opcode bit fields from RFC 9669 section 3.
pub mod opcode {
    // Instruction classes (low three bits).
    /// 64-bit load (only `LDDW` exists).
    pub const LD: u8 = 0x00;
    /// Load into a register.
    pub const LDX: u8 = 0x01;
    /// Store an immediate.
    pub const ST: u8 = 0x02;
    /// Store a register.
    pub const STX: u8 = 0x03;
    /// 32-bit arithmetic.
    pub const ALU: u8 = 0x04;
    /// 64-bit jumps (and call/exit).
    pub const JMP: u8 = 0x05;
    /// 32-bit jumps.
    pub const JMP32: u8 = 0x06;
    /// 64-bit arithmetic.
    pub const ALU64: u8 = 0x07;

    // Operand sizes (bits 3-4).
    /// Word: 4 bytes.
    pub const W: u8 = 0x00;
    /// Half: 2 bytes.
    pub const H: u8 = 0x08;
    /// Byte: 1 byte.
    pub const B: u8 = 0x10;
    /// Double word: 8 bytes.
    pub const DW: u8 = 0x18;

    /// The memory mode selector.
    pub const MEM: u8 = 0x60;

    // ALU operations (bits 4-7).
    /// Addition.
    pub const ADD: u8 = 0x00;
    /// Subtraction.
    pub const SUB: u8 = 0x10;
    /// Multiplication.
    pub const MUL: u8 = 0x20;
    /// Unsigned division.
    pub const DIV: u8 = 0x30;
    /// Bitwise or.
    pub const OR: u8 = 0x40;
    /// Bitwise and.
    pub const AND: u8 = 0x50;
    /// Left shift.
    pub const LSH: u8 = 0x60;
    /// Logical right shift.
    pub const RSH: u8 = 0x70;
    /// Negation.
    pub const NEG: u8 = 0x80;
    /// Unsigned modulo.
    pub const MOD: u8 = 0x90;
    /// Bitwise exclusive or.
    pub const XOR: u8 = 0xa0;
    /// Move.
    pub const MOV: u8 = 0xb0;
    /// Arithmetic right shift.
    pub const ARSH: u8 = 0xc0;
    /// Endianness conversion.
    pub const END: u8 = 0xd0;

    // Jump operations (bits 4-7).
    /// Unconditional jump.
    pub const JA: u8 = 0x00;
    /// Jump if equal.
    pub const JEQ: u8 = 0x10;
    /// Jump if greater than, unsigned.
    pub const JGT: u8 = 0x20;
    /// Jump if greater or equal, unsigned.
    pub const JGE: u8 = 0x30;
    /// Jump if any bit of the mask is set.
    pub const JSET: u8 = 0x40;
    /// Jump if not equal.
    pub const JNE: u8 = 0x50;
    /// Jump if greater than, signed.
    pub const JSGT: u8 = 0x60;
    /// Jump if greater or equal, signed.
    pub const JSGE: u8 = 0x70;
    /// Helper call.
    pub const CALL: u8 = 0x80;
    /// Program exit.
    pub const EXIT: u8 = 0x90;
    /// Jump if less than, unsigned.
    pub const JLT: u8 = 0xa0;
    /// Jump if less or equal, unsigned.
    pub const JLE: u8 = 0xb0;
    /// Jump if less than, signed.
    pub const JSLT: u8 = 0xc0;
    /// Jump if less or equal, signed.
    pub const JSLE: u8 = 0xd0;

    /// Immediate source operand.
    pub const K: u8 = 0x00;
    /// Register source operand.
    pub const X: u8 = 0x08;

    /// The only 64-bit-immediate instruction; it occupies two slots.
    pub const LDDW: u8 = LD | DW;
}

/// Helper ids the VM implements, by their UAPI numbers.
pub mod helper {
    /// `bpf_map_lookup_elem`: r1 = map, r2 = key pointer, returns value pointer or 0.
    pub const MAP_LOOKUP_ELEM: i32 = 1;
    /// `bpf_map_update_elem`: r1 = map, r2 = key, r3 = value, r4 = flags.
    pub const MAP_UPDATE_ELEM: i32 = 2;
    /// `bpf_ktime_get_ns`: monotonic nanoseconds.
    pub const KTIME_GET_NS: i32 = 5;
    /// `bpf_get_current_pid_tgid`: tgid << 32 | pid.
    pub const GET_CURRENT_PID_TGID: i32 = 14;
    /// `bpf_get_current_uid_gid`: uid << 32 | gid.
    pub const GET_CURRENT_UID_GID: i32 = 15;
    /// `bpf_get_current_comm`: copies the process name.
    pub const GET_CURRENT_COMM: i32 = 16;
}

/// The UAPI name of a helper id, if the VM implements it.
pub fn helper_name(id: i32) -> Option<&'static str> {
    match id {
        helper::MAP_LOOKUP_ELEM => Some("bpf_map_lookup_elem"),
        helper::MAP_UPDATE_ELEM => Some("bpf_map_update_elem"),
        helper::KTIME_GET_NS => Some("bpf_ktime_get_ns"),
        helper::GET_CURRENT_PID_TGID => Some("bpf_get_current_pid_tgid"),
        helper::GET_CURRENT_UID_GID => Some("bpf_get_current_uid_gid"),
        helper::GET_CURRENT_COMM => Some("bpf_get_current_comm"),
        _ => None,
    }
}

fn helper_known(id: i32) -> bool {
    helper_name(id).is_some()
}

/// One 64-bit instruction slot, in the field form the ISA defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Insn {
    /// The full opcode byte.
    pub opcode: u8,
    /// Destination register (low nibble on the wire).
    pub dst: u8,
    /// Source register (high nibble on the wire).
    pub src: u8,
    /// Signed offset: branch displacement or memory offset.
    pub off: i16,
    /// Immediate or helper id.
    pub imm: i32,
}

impl Insn {
    /// A raw instruction.
    pub const fn new(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> Self {
        Insn { opcode, dst, src, off, imm }
    }

    /// The second slot of a wide-immediate load, carrying the high 32 bits.
    pub const fn wide_hi(bits: i32) -> Self {
        Insn::new(0, 0, 0, 0, bits)
    }

    /// The 8 wire bytes, little-endian throughout (the `bpfel` target).
    pub fn encode(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[0] = self.opcode;
        out[1] = (self.dst & 0x0f) | ((self.src & 0x0f) << 4);
        out[2..4].copy_from_slice(&self.off.to_le_bytes());
        out[4..8].copy_from_slice(&self.imm.to_le_bytes());
        out
    }

    /// Rebuilds a slot from its 8 wire bytes.
    pub fn from_bytes(bytes: &[u8; 8]) -> Self {
        Insn {
            opcode: bytes[0],
            dst: bytes[1] & 0x0f,
            src: bytes[1] >> 4,
            off: i16::from_le_bytes([bytes[2], bytes[3]]),
            imm: i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        }
    }

    /// The 64-bit immediate of a wide load, given its second slot.
    pub fn imm64(&self, hi: &Insn) -> u64 {
        u64::from(self.imm as u32) | (u64::from(hi.imm as u32) << 32)
    }
}

/// Assembles a whole program: one instruction per line (two slots per wide
/// load), `;` and `#` start comments, blank lines are ignored.
pub fn encode(insns: &[Insn]) -> Vec<u8> {
    let mut out = Vec::with_capacity(insns.len() * 8);
    for insn in insns {
        out.extend_from_slice(&insn.encode());
    }
    out
}

/// Decodes wire bytes into slots. A wide load stays two slots, so slot indices
/// -- and therefore every jump offset -- are exactly what the assembler saw.
pub fn decode(program: &[u8]) -> Result<Vec<Insn>, VerifyError> {
    if program.is_empty() {
        return Err(VerifyError::EmptyProgram);
    }
    if program.len() % 8 != 0 {
        return Err(VerifyError::MisalignedProgram { bytes: program.len() });
    }
    let mut insns = Vec::with_capacity(program.len() / 8);
    for chunk in program.chunks_exact(8) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        insns.push(Insn::from_bytes(&bytes));
    }
    Ok(insns)
}

/// A line that failed to assemble.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsmError {
    /// 1-based line number in the source text.
    pub line: usize,
    /// The offending line, comment stripped.
    pub source_line: String,
    /// What was wrong with it.
    pub message: String,
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {} (`{}`)", self.line, self.message, self.source_line)
    }
}

impl std::error::Error for AsmError {}

/// Strips a trailing comment, using the same rules the assembler applies.
pub(crate) fn strip_comment(line: &str) -> &str {
    let end = line.find([';', '#']).unwrap_or(line.len());
    &line[..end]
}

/// Assembles the LLVM BPF dialect subset this crate emits.
///
/// Grammar (one instruction per line):
/// * moves and arithmetic: `rN = <imm|rM>`, `rN = -rN`, `rN op= <imm|rM>`
///   with `op` in `+ - * / % & | ^ << >> s>>`, and `wN` for 32-bit forms.
/// * wide immediates: `rN = <imm64> ll` (the `ll` suffix, as LLVM prints it).
/// * memory: `rN = *(u8|u16|u32|u64 *)(rM [+|-] off)` and the store form.
/// * endianness: `rN = be16|be32|be64|le16|le32|le64 rN`.
/// * control: `goto [+|-]off`, `if <lhs> <op> <rhs> goto [+|-]off` with `op`
///   in `== != > < >= <= s> s< s>= s<= &`, `call <id>`, `exit`.
pub fn assemble(text: &str) -> Result<Vec<u8>, AsmError> {
    let mut insns = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim().to_string();
        if line.is_empty() {
            continue;
        }
        match parse_line(&line) {
            Ok(mut parsed) => insns.append(&mut parsed),
            Err(message) => {
                return Err(AsmError { line: index + 1, source_line: line, message });
            }
        }
    }
    Ok(encode(&insns))
}

fn parse_line(line: &str) -> Result<Vec<Insn>, String> {
    if line == "exit" {
        return Ok(vec![Insn::new(opcode::JMP | opcode::EXIT, 0, 0, 0, 0)]);
    }
    if let Some(rest) = line.strip_prefix("goto ") {
        return Ok(vec![Insn::new(opcode::JMP | opcode::JA, 0, 0, parse_offset(rest)?, 0)]);
    }
    if let Some(rest) = line.strip_prefix("if ") {
        return parse_conditional(rest);
    }
    if let Some(rest) = line.strip_prefix("call ") {
        return Ok(vec![Insn::new(opcode::JMP | opcode::CALL, 0, 0, 0, parse_imm32(rest)?)]);
    }
    if line.starts_with("*(") {
        return parse_store(line);
    }
    let (lhs, rhs) = line
        .split_once('=')
        .ok_or_else(|| format!("`{line}` is neither an assignment nor a jump"))?;
    let lhs = lhs.trim();
    let rhs = rhs.trim();
    let parts: Vec<&str> = lhs.split_whitespace().collect();
    let (dst, wide) = parse_register(parts.first().copied().unwrap_or(""))?;
    let alu_op = match parts.as_slice() {
        [_] => None,
        [_, op] => Some(match *op {
            "+" => opcode::ADD,
            "-" => opcode::SUB,
            "*" => opcode::MUL,
            "/" => opcode::DIV,
            "|" => opcode::OR,
            "&" => opcode::AND,
            "<<" => opcode::LSH,
            ">>" => opcode::RSH,
            "%" => opcode::MOD,
            "^" => opcode::XOR,
            "s>>" => opcode::ARSH,
            other => return Err(format!("`{other}` is not an operation")),
        }),
        _ => return Err(format!("`{lhs}` is not an assignment target")),
    };
    if let Some(alu_op) = alu_op {
        let class = if wide { opcode::ALU64 } else { opcode::ALU };
        if let Ok((src, src_wide)) = parse_register(rhs) {
            if src_wide != wide {
                return Err(format!("mixing widths: `{lhs} = {rhs}`"));
            }
            return Ok(vec![Insn::new(class | alu_op | opcode::X, dst, src, 0, 0)]);
        }
        return Ok(vec![Insn::new(class | alu_op | opcode::K, dst, 0, 0, parse_imm32(rhs)?)]);
    }
    parse_assignment(dst, wide, rhs)
}

fn parse_assignment(dst: u8, wide: bool, rhs: &str) -> Result<Vec<Insn>, String> {
    if let Some(number) = rhs.strip_suffix(" ll") {
        if !wide {
            return Err("the `ll` suffix needs a 64-bit destination register".into());
        }
        let imm = parse_imm64(number)?;
        return Ok(vec![
            Insn::new(opcode::LDDW, dst, 0, 0, imm as u32 as i32),
            Insn::wide_hi((imm >> 32) as u32 as i32),
        ]);
    }
    if rhs.starts_with("*(") {
        let (size_bits, _, base, off) = parse_mem_operand(rhs)?;
        return Ok(vec![Insn::new(opcode::LDX | opcode::MEM | size_bits, dst, base, off, 0)]);
    }
    if let Some(rest) = rhs.strip_prefix('-') {
        if let Ok((src, src_wide)) = parse_register(rest.trim()) {
            if src_wide != wide || src != dst {
                return Err(format!(
                    "negation is unary on its destination: write `r{dst} = -r{dst}`"
                ));
            }
            let class = if wide { opcode::ALU64 } else { opcode::ALU };
            return Ok(vec![Insn::new(class | opcode::NEG, dst, 0, 0, 0)]);
        }
        // A negative immediate, not a negated register: handled below.
    }
    let tokens: Vec<&str> = rhs.split_whitespace().collect();
    if tokens.len() == 2 && matches!(tokens[0], "be16" | "be32" | "be64" | "le16" | "le32" | "le64") {
        let width: i32 = tokens[0][2..]
            .parse()
            .map_err(|_| format!("`{}` is not a conversion width", tokens[0]))?;
        let (src, src_wide) = parse_register(tokens[1])?;
        if !src_wide || !wide || src != dst {
            return Err(format!(
                "endianness conversion is unary on a 64-bit register: write `r{dst} = {} r{dst}`",
                tokens[0]
            ));
        }
        let source = if tokens[0].starts_with("be") { opcode::X } else { opcode::K };
        return Ok(vec![Insn::new(opcode::ALU | opcode::END | source, dst, 0, 0, width)]);
    }
    if let Ok((src, src_wide)) = parse_register(rhs) {
        if src_wide != wide {
            return Err(format!(
                "mixing widths: `{rhs}` cannot be assigned to a {} register",
                if wide { "64-bit" } else { "32-bit" }
            ));
        }
        let class = if wide { opcode::ALU64 } else { opcode::ALU };
        return Ok(vec![Insn::new(class | opcode::MOV | opcode::X, dst, src, 0, 0)]);
    }
    let class = if wide { opcode::ALU64 } else { opcode::ALU };
    Ok(vec![Insn::new(class | opcode::MOV | opcode::K, dst, 0, 0, parse_imm32(rhs)?)])
}

fn parse_store(line: &str) -> Result<Vec<Insn>, String> {
    let (target, value) = line
        .split_once('=')
        .ok_or_else(|| format!("`{line}` is not a store"))?;
    let (size_bits, size, base, off) = parse_mem_operand(target.trim())?;
    if let Ok((src, src_wide)) = parse_register(value.trim()) {
        if !src_wide {
            return Err("stores take 64-bit source registers".into());
        }
        return Ok(vec![Insn::new(opcode::STX | opcode::MEM | size_bits, base, src, off, 0)]);
    }
    if size == 8 {
        return Err("there is no 64-bit immediate store (the immediate is 32 bits)".into());
    }
    let imm = parse_imm32(value.trim())?;
    Ok(vec![Insn::new(opcode::ST | opcode::MEM | size_bits, base, 0, off, imm)])
}

fn parse_conditional(rest: &str) -> Result<Vec<Insn>, String> {
    let (condition, offset) = rest
        .rsplit_once(" goto ")
        .ok_or_else(|| format!("`if {rest}` is missing `goto`"))?;
    let tokens: Vec<&str> = condition.split_whitespace().collect();
    let (lhs, op_text, rhs) = match tokens.as_slice() {
        [lhs, op, rhs] => (*lhs, *op, *rhs),
        _ => return Err(format!("`if {rest}` is not a comparison")),
    };
    let (dst, wide) = parse_register(lhs)?;
    let op_bits = match op_text {
        "==" => opcode::JEQ,
        "!=" => opcode::JNE,
        ">" => opcode::JGT,
        "<" => opcode::JLT,
        ">=" => opcode::JGE,
        "<=" => opcode::JLE,
        "s>" => opcode::JSGT,
        "s<" => opcode::JSLT,
        "s>=" => opcode::JSGE,
        "s<=" => opcode::JSLE,
        "&" => opcode::JSET,
        other => return Err(format!("`{other}` is not a comparison")),
    };
    let off = parse_offset(offset)?;
    let class = if wide { opcode::JMP } else { opcode::JMP32 };
    if let Ok((src, src_wide)) = parse_register(rhs) {
        if src_wide != wide {
            return Err(format!("mixing widths: `{lhs} {op_text} {rhs}`"));
        }
        return Ok(vec![Insn::new(class | op_bits | opcode::X, dst, src, off, 0)]);
    }
    Ok(vec![Insn::new(class | op_bits | opcode::K, dst, 0, off, parse_imm32(rhs)?)])
}

fn parse_register(token: &str) -> Result<(u8, bool), String> {
    let (wide, rest) = match token.as_bytes().first() {
        Some(b'r') => (true, token[1..].trim()),
        Some(b'w') => (false, token[1..].trim()),
        _ => return Err(format!("`{token}` is not a register (expected r0..r10 or w0..w10)")),
    };
    let number: u8 = rest
        .parse()
        .map_err(|_| format!("`{token}` is not a register (expected r0..r10 or w0..w10)"))?;
    if usize::from(number) >= REGISTER_COUNT {
        return Err(format!("`{token}` is outside r0..r10"));
    }
    Ok((number, wide))
}

fn parse_number(text: &str) -> Result<i128, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("expected a number".into());
    }
    let (negative, body) = if let Some(rest) = text.strip_prefix('-') {
        (true, rest.trim())
    } else if let Some(rest) = text.strip_prefix('+') {
        (false, rest.trim())
    } else {
        (false, text)
    };
    let magnitude = if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        i128::from_str_radix(hex, 16).map_err(|_| format!("`{text}` is not a number"))?
    } else {
        body.parse::<i128>().map_err(|_| format!("`{text}` is not a number"))?
    };
    Ok(if negative { -magnitude } else { magnitude })
}

fn parse_imm32(text: &str) -> Result<i32, String> {
    let value = parse_number(text)?;
    i32::try_from(value).map_err(|_| {
        format!("`{text}` does not fit in a 32-bit immediate (use the `ll` suffix for a 64-bit load)")
    })
}

fn parse_imm64(text: &str) -> Result<u64, String> {
    let value = parse_number(text)?;
    if value < i128::from(i64::MIN) || value > i128::from(u64::MAX) {
        return Err(format!("`{text}` does not fit in 64 bits"));
    }
    Ok(value as u64)
}

fn parse_offset(text: &str) -> Result<i16, String> {
    let text = text.trim();
    let value = parse_number(text)?;
    i16::try_from(value).map_err(|_| format!("jump offset `{text}` does not fit in 16 bits"))
}

/// Parses `*(u64 *)(r10 - 8)` into (size bits, size bytes, base register, offset).
fn parse_mem_operand(text: &str) -> Result<(u8, usize, u8, i16), String> {
    let inner = text
        .trim()
        .strip_prefix("*(")
        .ok_or_else(|| format!("`{text}` is not a memory operand"))?;
    let middle = inner
        .find("*)(")
        .ok_or_else(|| format!("`{text}` is not a memory operand"))?;
    let (size_bits, size) = match inner[..middle].trim() {
        "u8" => (opcode::B, 1),
        "u16" => (opcode::H, 2),
        "u32" => (opcode::W, 4),
        "u64" => (opcode::DW, 8),
        other => return Err(format!("`{other}` is not an eBPF access size")),
    };
    let after = &inner[middle + 3..];
    let close = after
        .rfind(')')
        .ok_or_else(|| format!("`{text}` is missing a closing parenthesis"))?;
    let address = after[..close].trim();
    let split = address.find(['+', '-']).unwrap_or(address.len());
    let base_token = address[..split].trim();
    let (base, wide) = parse_register(base_token)?;
    if !wide {
        return Err(format!("memory base `{base_token}` must be a 64-bit register"));
    }
    let rest = address[split..].trim();
    let off = if rest.is_empty() { 0 } else { parse_offset(rest)? };
    Ok((size_bits, size, base, off))
}

/// An ALU operation, after decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AluOp {
    Add,
    Sub,
    Mul,
    Div,
    Or,
    And,
    Lsh,
    Rsh,
    Neg,
    Mod,
    Xor,
    Mov,
    Arsh,
}

/// A jump (or call/exit) operation, after decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JmpOp {
    Ja,
    Jeq,
    Jgt,
    Jge,
    Jset,
    Jne,
    Jsgt,
    Jsge,
    Call,
    Exit,
    Jlt,
    Jle,
    Jslt,
    Jsle,
}

/// What an instruction does, after decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// A two-slot 64-bit immediate load.
    Lddw,
    /// A memory load.
    Ldx { size: usize },
    /// A memory store of an immediate.
    St { size: usize },
    /// A memory store of a register.
    Stx { size: usize },
    /// Arithmetic, 32- or 64-bit, immediate or register operand.
    Alu { op: AluOp, wide: bool, from_reg: bool },
    /// An endianness conversion.
    End { big_endian: bool, width: u8 },
    /// A jump, call, or exit.
    Jump { op: JmpOp, wide: bool, from_reg: bool },
}

/// Classifies one slot.
///
/// `None` means the opcode is not an instruction this implementation knows;
/// the verifier rejects such programs rather than guessing, so an encoding
/// outside the subset can never execute with invented semantics.
pub(crate) fn classify(insn: &Insn) -> Option<Class> {
    let opcode = insn.opcode;
    let class = opcode & 0x07;
    let source = opcode & opcode::X;
    match class {
        opcode::LD if opcode == opcode::LDDW => Some(Class::Lddw),
        opcode::LDX => {
            if opcode & opcode::MEM != opcode::MEM {
                return None;
            }
            match opcode & opcode::DW {
                opcode::W => Some(Class::Ldx { size: 4 }),
                opcode::H => Some(Class::Ldx { size: 2 }),
                opcode::B => Some(Class::Ldx { size: 1 }),
                opcode::DW => Some(Class::Ldx { size: 8 }),
                _ => None,
            }
        }
        opcode::ST => {
            if opcode & opcode::MEM != opcode::MEM {
                return None;
            }
            match opcode & opcode::DW {
                opcode::W => Some(Class::St { size: 4 }),
                opcode::H => Some(Class::St { size: 2 }),
                opcode::B => Some(Class::St { size: 1 }),
                // No 64-bit immediate store exists: the immediate is 32 bits.
                _ => None,
            }
        }
        opcode::STX => {
            if opcode & opcode::MEM != opcode::MEM {
                return None;
            }
            match opcode & opcode::DW {
                opcode::W => Some(Class::Stx { size: 4 }),
                opcode::H => Some(Class::Stx { size: 2 }),
                opcode::B => Some(Class::Stx { size: 1 }),
                opcode::DW => Some(Class::Stx { size: 8 }),
                _ => None,
            }
        }
        opcode::ALU | opcode::ALU64 => {
            let wide = class == opcode::ALU64;
            let op_bits = opcode & 0xf0;
            if op_bits == opcode::END {
                if wide {
                    // 64-bit END encodings belong to other extensions.
                    return None;
                }
                let width = match insn.imm {
                    16 => 16u8,
                    32 => 32,
                    64 => 64,
                    _ => return None,
                };
                return Some(Class::End { big_endian: source == opcode::X, width });
            }
            let op = match op_bits {
                opcode::ADD => AluOp::Add,
                opcode::SUB => AluOp::Sub,
                opcode::MUL => AluOp::Mul,
                opcode::DIV => AluOp::Div,
                opcode::OR => AluOp::Or,
                opcode::AND => AluOp::And,
                opcode::LSH => AluOp::Lsh,
                opcode::RSH => AluOp::Rsh,
                opcode::NEG => AluOp::Neg,
                opcode::MOD => AluOp::Mod,
                opcode::XOR => AluOp::Xor,
                opcode::MOV => AluOp::Mov,
                opcode::ARSH => AluOp::Arsh,
                _ => return None,
            };
            if op == AluOp::Neg && source != opcode::K {
                return None;
            }
            Some(Class::Alu { op, wide, from_reg: source == opcode::X })
        }
        opcode::JMP | opcode::JMP32 => {
            let wide = class == opcode::JMP;
            let op_bits = opcode & 0xf0;
            let op = match op_bits {
                opcode::JA => {
                    if !wide {
                        return None;
                    }
                    JmpOp::Ja
                }
                opcode::JEQ => JmpOp::Jeq,
                opcode::JGT => JmpOp::Jgt,
                opcode::JGE => JmpOp::Jge,
                opcode::JSET => JmpOp::Jset,
                opcode::JNE => JmpOp::Jne,
                opcode::JSGT => JmpOp::Jsgt,
                opcode::JSGE => JmpOp::Jsge,
                opcode::CALL => {
                    if !wide {
                        return None;
                    }
                    JmpOp::Call
                }
                opcode::EXIT => {
                    if !wide {
                        return None;
                    }
                    JmpOp::Exit
                }
                opcode::JLT => JmpOp::Jlt,
                opcode::JLE => JmpOp::Jle,
                opcode::JSLT => JmpOp::Jslt,
                opcode::JSLE => JmpOp::Jsle,
                _ => return None,
            };
            if matches!(op, JmpOp::Ja | JmpOp::Call | JmpOp::Exit) && source != opcode::K {
                return None;
            }
            Some(Class::Jump { op, wide, from_reg: source == opcode::X })
        }
        _ => None,
    }
}

/// A program the verifier refuses to let execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// A zero-length program.
    EmptyProgram,
    /// A program length that is not a whole number of instruction slots.
    MisalignedProgram {
        /// The offending length.
        bytes: usize,
    },
    /// An opcode outside the implemented subset.
    UndefinedOpcode {
        /// Slot index.
        pc: usize,
        /// The offending opcode.
        opcode: u8,
    },
    /// A register field naming something other than r0..r10.
    UnknownRegister {
        /// Slot index.
        pc: usize,
        /// The offending register number.
        register: u8,
    },
    /// A write to the read-only frame pointer.
    WriteToFramePointer {
        /// Slot index.
        pc: usize,
    },
    /// A fixed stack access outside the frame.
    StackAccessOutOfBounds {
        /// Slot index.
        pc: usize,
        /// The access offset.
        offset: i16,
        /// The access size.
        size: usize,
    },
    /// A branch target outside the program.
    JumpOutOfBounds {
        /// Slot index.
        pc: usize,
        /// Where the branch would land.
        target: i64,
    },
    /// A branch landing on the second slot of a wide load.
    JumpIntoWideImmediate {
        /// Slot index.
        pc: usize,
        /// The offending target.
        target: usize,
    },
    /// A wide load with no second slot after it.
    LddwWithoutSlot {
        /// Slot index.
        pc: usize,
    },
    /// A second wide-load slot that is not a bare high-word carrier.
    BadWideImmediateSlot {
        /// Slot index.
        pc: usize,
    },
    /// A division or modulo whose immediate divisor is zero.
    DivisionByZeroImmediate {
        /// Slot index.
        pc: usize,
    },
    /// A helper call to an id this VM cannot implement.
    UnknownHelper {
        /// Slot index.
        pc: usize,
        /// The helper id.
        id: i32,
    },
    /// A read of a register no reachable path has written.
    ReadOfUninitializedRegister {
        /// Slot index.
        pc: usize,
        /// The register read.
        register: u8,
    },
    /// Execution could fall off the end of the program.
    FallsThroughEnd {
        /// Slot index.
        pc: usize,
    },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerifyError::EmptyProgram => write!(f, "program is empty"),
            VerifyError::MisalignedProgram { bytes } => {
                write!(f, "program is {bytes} bytes, not a whole number of 8-byte slots")
            }
            VerifyError::UndefinedOpcode { pc, opcode } => {
                write!(f, "slot {pc}: opcode {opcode:#04x} is outside the implemented instruction set")
            }
            VerifyError::UnknownRegister { pc, register } => {
                write!(f, "slot {pc}: r{register} is not a register (r0..r10 exist)")
            }
            VerifyError::WriteToFramePointer { pc } => {
                write!(f, "slot {pc}: r10 is the read-only frame pointer")
            }
            VerifyError::StackAccessOutOfBounds { pc, offset, size } => write!(
                f,
                "slot {pc}: stack access at {offset} with size {size} leaves the {STACK_SIZE}-byte frame"
            ),
            VerifyError::JumpOutOfBounds { pc, target } => {
                write!(f, "slot {pc}: jump target {target} is outside the program")
            }
            VerifyError::JumpIntoWideImmediate { pc, target } => write!(
                f,
                "slot {pc}: jump target {target} is the second slot of a wide immediate"
            ),
            VerifyError::LddwWithoutSlot { pc } => {
                write!(f, "slot {pc}: wide immediate load runs past the end of the program")
            }
            VerifyError::BadWideImmediateSlot { pc } => {
                write!(f, "slot {pc}: the slot after a wide load must carry only its high 32 bits")
            }
            VerifyError::DivisionByZeroImmediate { pc } => {
                write!(f, "slot {pc}: division by an immediate zero is always a bug")
            }
            VerifyError::UnknownHelper { pc, id } => {
                write!(f, "slot {pc}: helper id {id} is not in the supported helper set")
            }
            VerifyError::ReadOfUninitializedRegister { pc, register } => {
                write!(f, "slot {pc}: r{register} may be read before it is written")
            }
            VerifyError::FallsThroughEnd { pc } => {
                write!(f, "slot {pc}: control reaches the end of the program without exiting")
            }
        }
    }
}

impl std::error::Error for VerifyError {}

fn check_registers(insn: &Insn, class: Class, pc: usize) -> Result<(), VerifyError> {
    let check = |register: u8| -> Result<(), VerifyError> {
        if register > FRAME_POINTER {
            Err(VerifyError::UnknownRegister { pc, register })
        } else {
            Ok(())
        }
    };
    match class {
        Class::Lddw => check(insn.dst)?,
        Class::Ldx { .. } | Class::Stx { .. } => {
            check(insn.dst)?;
            check(insn.src)?;
        }
        Class::St { .. } | Class::End { .. } => check(insn.dst)?,
        Class::Alu { .. } => {
            check(insn.dst)?;
            check(insn.src)?;
        }
        Class::Jump { from_reg, .. } => {
            check(insn.dst)?;
            if from_reg {
                check(insn.src)?;
            }
        }
    }
    Ok(())
}

fn check_stack_access(offset: i16, size: usize, pc: usize) -> Result<(), VerifyError> {
    let start = i64::from(offset);
    let end = start + size as i64;
    if start >= 0 || end > 0 || start < -(STACK_SIZE as i64) {
        return Err(VerifyError::StackAccessOutOfBounds { pc, offset, size });
    }
    Ok(())
}

fn jump_target(pc: usize, off: i16) -> i64 {
    pc as i64 + 1 + i64::from(off)
}

/// Verifies a decoded program.
///
/// The passes are: (1) every slot is an instruction this VM implements, every
/// register exists, every plain access is inside the frame, every branch lands
/// on a first slot inside the program, and control never falls off the end;
/// (2) no instruction writes r10 and no immediate division is by zero; (3) a
/// forward dataflow that proves every register read on some path is written on
/// every path to it. Only programs that pass all three execute.
pub fn verify(insns: &[Insn]) -> Result<(), VerifyError> {
    let count = insns.len();
    if count == 0 {
        return Err(VerifyError::EmptyProgram);
    }

    // Pass 1: structure.
    let mut classes: Vec<Option<Class>> = vec![None; count];
    let mut is_wide_hi = vec![false; count];
    let mut pc = 0usize;
    while pc < count {
        let insn = insns[pc];
        if insn.opcode == opcode::LDDW {
            if pc + 1 >= count {
                return Err(VerifyError::LddwWithoutSlot { pc });
            }
            let hi = insns[pc + 1];
            if hi.opcode != 0 || hi.dst != 0 || hi.src != 0 || hi.off != 0 {
                return Err(VerifyError::BadWideImmediateSlot { pc });
            }
            check_registers(&insn, Class::Lddw, pc)?;
            is_wide_hi[pc + 1] = true;
            classes[pc] = Some(Class::Lddw);
            pc += 2;
        } else {
            let class =
                classify(&insn).ok_or(VerifyError::UndefinedOpcode { pc, opcode: insn.opcode })?;
            check_registers(&insn, class, pc)?;
            classes[pc] = Some(class);
            pc += 1;
        }
    }

    // Pass 2: operand validity, branch targets, and fallthrough.
    for pc in 0..count {
        let Some(class) = classes[pc] else {
            continue;
        };
        let insn = insns[pc];
        match class {
            Class::Lddw => {
                if insn.dst == FRAME_POINTER {
                    return Err(VerifyError::WriteToFramePointer { pc });
                }
            }
            Class::Ldx { size } => {
                if insn.dst == FRAME_POINTER {
                    return Err(VerifyError::WriteToFramePointer { pc });
                }
                if insn.src == FRAME_POINTER {
                    check_stack_access(insn.off, size, pc)?;
                }
            }
            Class::St { size } | Class::Stx { size } => {
                if insn.dst == FRAME_POINTER {
                    check_stack_access(insn.off, size, pc)?;
                }
            }
            Class::Alu { op, from_reg, .. } => {
                if insn.dst == FRAME_POINTER {
                    return Err(VerifyError::WriteToFramePointer { pc });
                }
                if matches!(op, AluOp::Div | AluOp::Mod) && !from_reg && insn.imm == 0 {
                    return Err(VerifyError::DivisionByZeroImmediate { pc });
                }
            }
            Class::End { .. } => {
                if insn.dst == FRAME_POINTER {
                    return Err(VerifyError::WriteToFramePointer { pc });
                }
            }
            Class::Jump { op, .. } => match op {
                JmpOp::Exit => {}
                JmpOp::Call => {
                    if !helper_known(insn.imm) {
                        return Err(VerifyError::UnknownHelper { pc, id: insn.imm });
                    }
                }
                _ => {
                    let target = jump_target(pc, insn.off);
                    if target < 0 || target >= count as i64 {
                        return Err(VerifyError::JumpOutOfBounds { pc, target });
                    }
                    if is_wide_hi[target as usize] {
                        return Err(VerifyError::JumpIntoWideImmediate {
                            pc,
                            target: target as usize,
                        });
                    }
                }
            },
        }
        let step = if matches!(class, Class::Lddw) { 2 } else { 1 };
        let falls_through = match class {
            Class::Jump { op, .. } => !matches!(op, JmpOp::Ja | JmpOp::Exit),
            _ => true,
        };
        if falls_through && pc + step >= count {
            return Err(VerifyError::FallsThroughEnd { pc });
        }
    }

    // Pass 3: register dataflow.
    let everything: BTreeSet<u8> = (0u8..=FRAME_POINTER).collect();
    let mut entry: Vec<BTreeSet<u8>> = vec![everything; count];
    entry[0] = BTreeSet::from([1u8, FRAME_POINTER]);
    let mut changed = true;
    while changed {
        changed = false;
        for pc in 0..count {
            let Some(class) = classes[pc] else {
                continue;
            };
            let insn = insns[pc];
            for register in read_registers(&insn, class) {
                if !entry[pc].contains(&register) {
                    return Err(VerifyError::ReadOfUninitializedRegister { pc, register });
                }
            }
            let out = transfer(&entry[pc], &insn, class);
            for successor in successors(pc, &insn, class) {
                let merged: BTreeSet<u8> = entry[successor].intersection(&out).copied().collect();
                if merged != entry[successor] {
                    entry[successor] = merged;
                    changed = true;
                }
            }
        }
    }
    Ok(())
}

fn read_registers(insn: &Insn, class: Class) -> Vec<u8> {
    let mut out = Vec::new();
    match class {
        Class::Lddw => {}
        Class::Ldx { .. } => out.push(insn.src),
        Class::St { .. } => out.push(insn.dst),
        Class::Stx { .. } => {
            out.push(insn.dst);
            out.push(insn.src);
        }
        Class::Alu { op, from_reg, .. } => {
            if op != AluOp::Mov {
                out.push(insn.dst);
            }
            if from_reg {
                out.push(insn.src);
            }
        }
        Class::End { .. } => out.push(insn.dst),
        Class::Jump { op, from_reg, .. } => match op {
            JmpOp::Ja => {}
            JmpOp::Exit => out.push(0),
            JmpOp::Call => match insn.imm {
                helper::MAP_LOOKUP_ELEM => out.extend(1..=2),
                helper::MAP_UPDATE_ELEM => out.extend(1..=3),
                helper::GET_CURRENT_COMM => out.extend(1..=2),
                // bpf_ktime_get_ns, bpf_get_current_pid_tgid, and
                // bpf_get_current_uid_gid take no arguments.
                _ => {}
            },
            _ => {
                out.push(insn.dst);
                if from_reg {
                    out.push(insn.src);
                }
            }
        },
    }
    out
}

fn transfer(entry: &BTreeSet<u8>, insn: &Insn, class: Class) -> BTreeSet<u8> {
    let mut out = entry.clone();
    match class {
        Class::Lddw | Class::Ldx { .. } | Class::Alu { .. } | Class::End { .. } => {
            out.insert(insn.dst);
        }
        Class::St { .. } | Class::Stx { .. } => {}
        Class::Jump { op, .. } => match op {
            JmpOp::Call => {
                // The kernel ABI: arguments and scratch are clobbered.
                for register in 1..=5 {
                    out.remove(&register);
                }
                out.insert(0);
            }
            _ => {}
        },
    }
    out
}

fn successors(pc: usize, insn: &Insn, class: Class) -> Vec<usize> {
    let step = if matches!(class, Class::Lddw) { 2 } else { 1 };
    let target = |off: i16| jump_target(pc, off) as usize;
    match class {
        Class::Jump { op, .. } => match op {
            JmpOp::Exit => vec![],
            JmpOp::Ja => vec![target(insn.off)],
            JmpOp::Call => vec![pc + step],
            _ => vec![pc + step, target(insn.off)],
        },
        _ => vec![pc + step],
    }
}

/// A runtime trap. Executed programs are verified first, so these are for the
/// properties static analysis cannot see: what a register actually holds at
/// the moment it is dereferenced, and where a computed pointer lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VmError {
    /// The program failed verification and was not executed.
    Verify(VerifyError),
    /// The assembler rejected a source text.
    Assemble(String),
    /// The instruction budget ran out.
    FuelExhausted,
    /// The program counter left the program (a defensive check; the verifier
    /// already proves this cannot happen).
    PcOutOfBounds {
        /// Slot index.
        pc: usize,
    },
    /// A memory access outside the region it addressed.
    MemoryOutOfBounds {
        /// Slot index.
        pc: usize,
        /// Which region.
        region: &'static str,
        /// The effective offset.
        offset: i64,
        /// The access size.
        size: usize,
    },
    /// A store into the read-only context.
    ContextIsReadOnly {
        /// Slot index.
        pc: usize,
    },
    /// A scalar used where a pointer is required.
    DereferenceOfScalar {
        /// Slot index.
        pc: usize,
    },
    /// A non-pointer used where a pointer argument is required.
    ExpectedPointer {
        /// Slot index.
        pc: usize,
    },
    /// A non-scalar used where a scalar argument is required.
    ExpectedScalar {
        /// Slot index.
        pc: usize,
    },
    /// A non-map value used where a map reference is required.
    ExpectedMapReference {
        /// Slot index.
        pc: usize,
    },
    /// Arithmetic on a pointer that is not addition or subtraction.
    NonScalarArithmetic {
        /// Slot index.
        pc: usize,
    },
    /// A comparison involving something that is not a scalar.
    NonScalarComparison {
        /// Slot index.
        pc: usize,
    },
    /// A division or modulo by a runtime zero.
    DivisionByZero {
        /// Slot index.
        pc: usize,
    },
    /// A helper call naming a map that is not installed.
    UnknownMap {
        /// Slot index.
        pc: usize,
        /// The map id the program named.
        id: u64,
    },
    /// A map definition or lookup key with inconsistent sizes.
    MapSizeMismatch {
        /// Which of key/value/slot.
        detail: &'static str,
    },
    /// An LPM key whose declared prefix length exceeds its data.
    InvalidLpmPrefix {
        /// The declared length, in bits.
        prefix_len: u32,
        /// The maximum for this map.
        max_bits: usize,
    },
    /// An LPM lookup key whose declared prefix length exceeds its data.
    MapKeyMalformed {
        /// Slot index.
        pc: usize,
        /// The declared length, in bits.
        prefix_len: u32,
        /// The maximum for this map.
        key_bits: u32,
    },
    /// A helper call to an id with no implementation.
    UnknownHelper {
        /// Slot index.
        pc: usize,
        /// The helper id.
        id: i32,
    },
    /// `exit` with a pointer or map reference in r0.
    ReturnValueNotScalar {
        /// Slot index.
        pc: usize,
    },
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::Verify(error) => write!(f, "verification failed: {error}"),
            VmError::Assemble(message) => write!(f, "assembly failed: {message}"),
            VmError::FuelExhausted => write!(f, "instruction budget exhausted"),
            VmError::PcOutOfBounds { pc } => write!(f, "slot {pc} is outside the program"),
            VmError::MemoryOutOfBounds { pc, region, offset, size } => write!(
                f,
                "slot {pc}: {size}-byte access at {offset} leaves the {region} region"
            ),
            VmError::ContextIsReadOnly { pc } => {
                write!(f, "slot {pc}: the context is read-only for this program type")
            }
            VmError::DereferenceOfScalar { pc } => {
                write!(f, "slot {pc}: a scalar was used as a pointer")
            }
            VmError::ExpectedPointer { pc } => {
                write!(f, "slot {pc}: a helper argument must be a pointer")
            }
            VmError::ExpectedScalar { pc } => {
                write!(f, "slot {pc}: a helper argument must be a scalar")
            }
            VmError::ExpectedMapReference { pc } => {
                write!(f, "slot {pc}: a helper argument must be a map reference")
            }
            VmError::NonScalarArithmetic { pc } => {
                write!(f, "slot {pc}: arithmetic on a value that is not a scalar")
            }
            VmError::NonScalarComparison { pc } => {
                write!(f, "slot {pc}: comparison of a value that is not a scalar")
            }
            VmError::DivisionByZero { pc } => write!(f, "slot {pc}: division by zero"),
            VmError::UnknownMap { pc, id } => {
                write!(f, "slot {pc}: map {id} is not installed")
            }
            VmError::MapSizeMismatch { detail } => write!(f, "map {detail} has the wrong size"),
            VmError::InvalidLpmPrefix { prefix_len, max_bits } => write!(
                f,
                "LPM key declares {prefix_len} prefix bits but the map allows {max_bits}"
            ),
            VmError::MapKeyMalformed { pc, prefix_len, key_bits } => write!(
                f,
                "slot {pc}: LPM key declares {prefix_len} prefix bits but carries {key_bits}"
            ),
            VmError::UnknownHelper { pc, id } => {
                write!(f, "slot {pc}: helper id {id} has no implementation")
            }
            VmError::ReturnValueNotScalar { pc } => {
                write!(f, "slot {pc}: exit with a pointer in r0")
            }
        }
    }
}

impl std::error::Error for VmError {}

/// Where a pointer points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// The program context (read-only).
    Ctx,
    /// The 512-byte stack frame, addressed from `r10`.
    Stack,
    /// The value slot a map lookup returned.
    MapSlot {
        /// The map id.
        map: u64,
        /// The slot index inside the map.
        slot: usize,
    },
}

/// A pointer: a region plus a signed offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtrRef {
    /// The region addressed.
    pub region: Region,
    /// The byte offset from the region start.
    pub off: i64,
}

/// A register value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    /// An integer.
    Scalar(u64),
    /// A pointer into a known region.
    Ptr(PtrRef),
    /// A map reference produced by a wide-immediate load whose immediate names
    /// an installed map (the state after libbpf relocation).
    MapRef(u64),
}

fn region_name(region: Region) -> &'static str {
    match region {
        Region::Ctx => "context",
        Region::Stack => "stack",
        Region::MapSlot { .. } => "map value",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapKind {
    Hash,
    Lpm,
}

/// One map: a fixed key and value size, value slots, and a lookup index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VmMap {
    kind: MapKind,
    key_size: usize,
    value_size: usize,
    slots: Vec<Vec<u8>>,
    hash: BTreeMap<Vec<u8>, usize>,
    prefixes: Vec<(u32, Vec<u8>, usize)>,
}

impl VmMap {
    /// A hash map.
    pub fn hash(key_size: usize, value_size: usize) -> Self {
        VmMap {
            kind: MapKind::Hash,
            key_size,
            value_size,
            slots: Vec::new(),
            hash: BTreeMap::new(),
            prefixes: Vec::new(),
        }
    }

    /// An LPM trie. Keys are `[u32 prefix length in bits][data]`, exactly the
    /// `struct bpf_lpm_trie_key` layout the kernel and the C shim use.
    pub fn lpm(key_size: usize, value_size: usize) -> Self {
        VmMap {
            kind: MapKind::Lpm,
            key_size,
            value_size,
            slots: Vec::new(),
            hash: BTreeMap::new(),
            prefixes: Vec::new(),
        }
    }

    /// The key size, in bytes.
    pub fn key_size(&self) -> usize {
        self.key_size
    }

    /// The value size, in bytes.
    pub fn value_size(&self) -> usize {
        self.value_size
    }

    /// The number of value slots (entries ever inserted).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// The value bytes of a slot.
    pub fn slot_value(&self, slot: usize) -> Option<&[u8]> {
        self.slots.get(slot).map(|value| value.as_slice())
    }

    /// Inserts an entry. For an LPM map, the key's prefix length is what gets
    /// stored, and an identical prefix replaces its earlier value.
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), VmError> {
        if key.len() != self.key_size {
            return Err(VmError::MapSizeMismatch { detail: "key" });
        }
        if value.len() != self.value_size {
            return Err(VmError::MapSizeMismatch { detail: "value" });
        }
        let slot = self.slots.len();
        self.slots.push(value.to_vec());
        match self.kind {
            MapKind::Hash => {
                self.hash.insert(key.to_vec(), slot);
            }
            MapKind::Lpm => {
                let prefix_len = u32::from_le_bytes([key[0], key[1], key[2], key[3]]);
                let max_bits = (self.key_size - 4) * 8;
                if prefix_len as usize > max_bits {
                    return Err(VmError::InvalidLpmPrefix {
                        prefix_len,
                        max_bits,
                    });
                }
                let data = key[4..].to_vec();
                self.prefixes
                    .retain(|(bits, stored, _)| !(*bits == prefix_len && *stored == data));
                self.prefixes.push((prefix_len, data, slot));
            }
        }
        Ok(())
    }

    fn lookup(&self, key: &[u8], pc: usize) -> Result<Option<usize>, VmError> {
        match self.kind {
            MapKind::Hash => Ok(self.hash.get(key).copied()),
            MapKind::Lpm => {
                let prefix_len = u32::from_le_bytes([key[0], key[1], key[2], key[3]]);
                let key_bits = ((self.key_size - 4) * 8) as u32;
                if prefix_len > key_bits {
                    return Err(VmError::MapKeyMalformed {
                        pc,
                        prefix_len,
                        key_bits,
                    });
                }
                let data = &key[4..];
                let mut best: Option<(u32, usize)> = None;
                for (bits, prefix, slot) in &self.prefixes {
                    if *bits <= prefix_len && prefix_bits_match(prefix, *bits, data) {
                        if best.map_or(true, |(best_bits, _)| *bits > best_bits) {
                            best = Some((*bits, *slot));
                        }
                    }
                }
                Ok(best.map(|(_, slot)| slot))
            }
        }
    }
}

/// Whether the first `bits` bits of `prefix` and `key` agree.
///
/// This is the trie's comparison, bit for bit, so a partial trailing byte
/// matters. `crate::policy_compiler::LpmPrefix::matches_key` is the same rule
/// on the simulator side; `tests/shim_equivalence.rs` ties the two together
/// by construction.
fn prefix_bits_match(prefix: &[u8], bits: u32, key: &[u8]) -> bool {
    let full_bytes = (bits / 8) as usize;
    if key.len() < full_bytes || prefix.len() < full_bytes {
        return false;
    }
    if key[..full_bytes] != prefix[..full_bytes] {
        return false;
    }
    let remainder = bits % 8;
    if remainder == 0 {
        return true;
    }
    if key.len() <= full_bytes || prefix.len() <= full_bytes {
        return false;
    }
    let mask = 0xffu8 << (8 - remainder);
    key[full_bytes] & mask == prefix[full_bytes] & mask
}

/// The maps a program can name.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MapTable {
    maps: BTreeMap<u64, VmMap>,
}

impl MapTable {
    /// An empty table.
    pub fn new() -> Self {
        MapTable::default()
    }

    /// Installs a map under the id a wide-immediate load will name.
    pub fn insert(&mut self, id: u64, map: VmMap) {
        self.maps.insert(id, map);
    }

    /// Whether an id is installed.
    pub fn contains(&self, id: u64) -> bool {
        self.maps.contains_key(&id)
    }

    /// A map by id.
    pub fn get(&self, id: u64) -> Option<&VmMap> {
        self.maps.get(&id)
    }

    /// A map by id, mutably.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut VmMap> {
        self.maps.get_mut(&id)
    }
}

fn region_slice<'a>(maps: &'a MapTable, stack: &'a [u8], ctx: &'a [u8], region: Region) -> &'a [u8] {
    match region {
        Region::Ctx => ctx,
        Region::Stack => stack,
        Region::MapSlot { map, slot } => maps
            .get(map)
            .and_then(|map| map.slots.get(slot))
            .map(|value| value.as_slice())
            .unwrap_or(&[]),
    }
}

fn bounds_check(
    bytes: &[u8],
    off: i64,
    size: usize,
    region: Region,
    pc: usize,
) -> Result<(), VmError> {
    let end = off + size as i64;
    if off < 0 || end > bytes.len() as i64 {
        return Err(VmError::MemoryOutOfBounds {
            pc,
            region: region_name(region),
            offset: off,
            size,
        });
    }
    Ok(())
}

fn read_memory(
    maps: &MapTable,
    stack: &[u8],
    ctx: &[u8],
    ptr: PtrRef,
    insn_off: i16,
    size: usize,
    pc: usize,
) -> Result<u64, VmError> {
    let off = ptr.off + i64::from(insn_off);
    let bytes = region_slice(maps, stack, ctx, ptr.region);
    bounds_check(bytes, off, size, ptr.region, pc)?;
    let mut value = 0u64;
    for (index, byte) in bytes[off as usize..off as usize + size].iter().enumerate() {
        value |= u64::from(*byte) << (index * 8);
    }
    Ok(value)
}

fn read_bytes(
    maps: &MapTable,
    stack: &[u8],
    ctx: &[u8],
    ptr: PtrRef,
    size: usize,
    pc: usize,
) -> Result<Vec<u8>, VmError> {
    let bytes = region_slice(maps, stack, ctx, ptr.region);
    bounds_check(bytes, ptr.off, size, ptr.region, pc)?;
    Ok(bytes[ptr.off as usize..ptr.off as usize + size].to_vec())
}

fn store_memory(
    maps: &mut MapTable,
    stack: &mut [u8],
    ptr: PtrRef,
    insn_off: i16,
    size: usize,
    value: u64,
    pc: usize,
) -> Result<(), VmError> {
    if matches!(ptr.region, Region::Ctx) {
        return Err(VmError::ContextIsReadOnly { pc });
    }
    let off = ptr.off + i64::from(insn_off);
    let region_len = match ptr.region {
        Region::Ctx => unreachable!("checked above"),
        Region::Stack => stack.len(),
        Region::MapSlot { map, slot } => maps
            .get(map)
            .and_then(|map| map.slots.get(slot))
            .map(|value| value.len())
            .unwrap_or(0),
    };
    let end = off + size as i64;
    if off < 0 || end > region_len as i64 {
        return Err(VmError::MemoryOutOfBounds {
            pc,
            region: region_name(ptr.region),
            offset: off,
            size,
        });
    }
    let bytes = value.to_le_bytes();
    match ptr.region {
        Region::Ctx => unreachable!("checked above"),
        Region::Stack => stack[off as usize..end as usize].copy_from_slice(&bytes[..size]),
        Region::MapSlot { map, slot } => {
            let target = maps
                .get_mut(map)
                .and_then(|map| map.slots.get_mut(slot))
                .expect("region existence is checked above");
            target[off as usize..end as usize].copy_from_slice(&bytes[..size]);
        }
    }
    Ok(())
}

fn write_bytes(
    maps: &mut MapTable,
    stack: &mut [u8],
    ptr: PtrRef,
    bytes: &[u8],
    pc: usize,
) -> Result<(), VmError> {
    if matches!(ptr.region, Region::Ctx) {
        return Err(VmError::ContextIsReadOnly { pc });
    }
    let region_len = match ptr.region {
        Region::Ctx => unreachable!("checked above"),
        Region::Stack => stack.len(),
        Region::MapSlot { map, slot } => maps
            .get(map)
            .and_then(|map| map.slots.get(slot))
            .map(|value| value.len())
            .unwrap_or(0),
    };
    let end = ptr.off + bytes.len() as i64;
    if ptr.off < 0 || end > region_len as i64 {
        return Err(VmError::MemoryOutOfBounds {
            pc,
            region: region_name(ptr.region),
            offset: ptr.off,
            size: bytes.len(),
        });
    }
    match ptr.region {
        Region::Ctx => unreachable!("checked above"),
        Region::Stack => stack[ptr.off as usize..end as usize].copy_from_slice(bytes),
        Region::MapSlot { map, slot } => {
            let target = maps
                .get_mut(map)
                .and_then(|map| map.slots.get_mut(slot))
                .expect("region existence is checked above");
            target[ptr.off as usize..end as usize].copy_from_slice(bytes);
        }
    }
    Ok(())
}

fn sign_extend(imm: i32) -> u64 {
    imm as i64 as u64
}

fn expect_ptr(regs: &[Value; REGISTER_COUNT], register: u8, pc: usize) -> Result<PtrRef, VmError> {
    match regs[register as usize] {
        Value::Ptr(ptr) => Ok(ptr),
        _ => Err(VmError::DereferenceOfScalar { pc }),
    }
}

fn expect_scalar(regs: &[Value; REGISTER_COUNT], register: u8, pc: usize) -> Result<u64, VmError> {
    match regs[register as usize] {
        Value::Scalar(value) => Ok(value),
        _ => Err(VmError::NonScalarArithmetic { pc }),
    }
}

fn execute_alu(
    regs: &mut [Value; REGISTER_COUNT],
    insn: &Insn,
    op: AluOp,
    wide: bool,
    from_reg: bool,
    pc: usize,
) -> Result<(), VmError> {
    let dst = insn.dst as usize;
    let operand = if from_reg {
        regs[insn.src as usize]
    } else {
        Value::Scalar(sign_extend(insn.imm))
    };
    if op == AluOp::Mov {
        regs[dst] = match operand {
            // A 32-bit move zero-extends its result, so the high half of a
            // scalar is discarded; a 64-bit move copies all of it.
            Value::Scalar(value) if !wide => Value::Scalar(value & 0xffff_ffff),
            Value::Scalar(value) => Value::Scalar(value),
            // A pointer cannot move into a 32-bit register without being
            // truncated, which the kernel forbids; fail closed.
            Value::Ptr(_) | Value::MapRef(_) if wide => operand,
            Value::Ptr(_) | Value::MapRef(_) => {
                return Err(VmError::NonScalarArithmetic { pc });
            }
        };
        return Ok(());
    }
    let result = match (regs[dst], operand) {
        (Value::Ptr(ptr), Value::Scalar(delta)) if matches!(op, AluOp::Add | AluOp::Sub) => {
            let signed = delta as i64;
            let delta = if op == AluOp::Add { signed } else { signed.wrapping_neg() };
            Value::Ptr(PtrRef { region: ptr.region, off: ptr.off + delta })
        }
        (Value::Scalar(left), Value::Scalar(right)) => {
            Value::Scalar(alu_scalar(op, wide, left, right, pc)?)
        }
        _ => return Err(VmError::NonScalarArithmetic { pc }),
    };
    regs[dst] = result;
    Ok(())
}

fn alu_scalar(op: AluOp, wide: bool, left: u64, right: u64, pc: usize) -> Result<u64, VmError> {
    if wide {
        let value = match op {
            AluOp::Add => left.wrapping_add(right),
            AluOp::Sub => left.wrapping_sub(right),
            AluOp::Mul => left.wrapping_mul(right),
            AluOp::Div => {
                if right == 0 {
                    return Err(VmError::DivisionByZero { pc });
                }
                left / right
            }
            AluOp::Or => left | right,
            AluOp::And => left & right,
            AluOp::Lsh => left << (right & 63),
            AluOp::Rsh => left >> (right & 63),
            AluOp::Neg => left.wrapping_neg(),
            AluOp::Mod => {
                if right == 0 {
                    return Err(VmError::DivisionByZero { pc });
                }
                left % right
            }
            AluOp::Xor => left ^ right,
            AluOp::Arsh => ((left as i64) >> (right & 63)) as u64,
            AluOp::Mov => unreachable!("mov is handled before arithmetic"),
        };
        Ok(value)
    } else {
        let left = left as u32;
        let right = right as u32;
        let value: u32 = match op {
            AluOp::Add => left.wrapping_add(right),
            AluOp::Sub => left.wrapping_sub(right),
            AluOp::Mul => left.wrapping_mul(right),
            AluOp::Div => {
                if right == 0 {
                    return Err(VmError::DivisionByZero { pc });
                }
                left / right
            }
            AluOp::Or => left | right,
            AluOp::And => left & right,
            AluOp::Lsh => left << (right & 31),
            AluOp::Rsh => left >> (right & 31),
            AluOp::Neg => left.wrapping_neg(),
            AluOp::Mod => {
                if right == 0 {
                    return Err(VmError::DivisionByZero { pc });
                }
                left % right
            }
            AluOp::Xor => left ^ right,
            AluOp::Arsh => ((left as i32) >> (right & 31)) as u32,
            AluOp::Mov => unreachable!("mov is handled before arithmetic"),
        };
        Ok(u64::from(value))
    }
}

fn eval_condition(
    regs: &[Value; REGISTER_COUNT],
    insn: &Insn,
    op: JmpOp,
    wide: bool,
    from_reg: bool,
    pc: usize,
) -> Result<bool, VmError> {
    let left = regs[insn.dst as usize];
    let right = if from_reg {
        regs[insn.src as usize]
    } else {
        Value::Scalar(sign_extend(insn.imm))
    };
    let (left, right) = match (left, right) {
        (Value::Scalar(left), Value::Scalar(right)) => (left, right),
        // A pointer or map reference can only be compared against zero: that
        // is the null test every program performs after a lookup. Any other
        // comparison over a non-scalar is meaningless in this model, and
        // traps rather than inventing an ordering.
        (Value::Ptr(_) | Value::MapRef(_), Value::Scalar(0))
        | (Value::Scalar(0), Value::Ptr(_) | Value::MapRef(_))
            if matches!(op, JmpOp::Jeq | JmpOp::Jne) =>
        {
            return Ok(op == JmpOp::Jne);
        }
        _ => return Err(VmError::NonScalarComparison { pc }),
    };
    if wide {
        Ok(match op {
            JmpOp::Jeq => left == right,
            JmpOp::Jne => left != right,
            JmpOp::Jgt => left > right,
            JmpOp::Jlt => left < right,
            JmpOp::Jge => left >= right,
            JmpOp::Jle => left <= right,
            JmpOp::Jsgt => (left as i64) > (right as i64),
            JmpOp::Jslt => (left as i64) < (right as i64),
            JmpOp::Jsge => (left as i64) >= (right as i64),
            JmpOp::Jsle => (left as i64) <= (right as i64),
            JmpOp::Jset => (left & right) != 0,
            _ => unreachable!("call, exit, and ja are not conditions"),
        })
    } else {
        let left = left as u32;
        let right = right as u32;
        Ok(match op {
            JmpOp::Jeq => left == right,
            JmpOp::Jne => left != right,
            JmpOp::Jgt => left > right,
            JmpOp::Jlt => left < right,
            JmpOp::Jge => left >= right,
            JmpOp::Jle => left <= right,
            JmpOp::Jsgt => (left as i32) > (right as i32),
            JmpOp::Jslt => (left as i32) < (right as i32),
            JmpOp::Jsge => (left as i32) >= (right as i32),
            JmpOp::Jsle => (left as i32) <= (right as i32),
            JmpOp::Jset => (left & right) != 0,
            _ => unreachable!("call, exit, and ja are not conditions"),
        })
    }
}

/// An interpreter for verified programs.
///
/// `run` decodes, verifies, and executes; it never executes unverified code.
/// Map references come from wide-immediate loads: an immediate that names an
/// installed map becomes [`Value::MapRef`], the post-relocation state, and any
/// other immediate is a scalar -- the same ambiguity a kernel has when a
/// constant happens to equal a pointer, made explicit here.
#[derive(Debug)]
pub struct Vm<'a> {
    maps: &'a mut MapTable,
    fuel: u64,
    ticks: u64,
}

impl<'a> Vm<'a> {
    /// A VM over a map table, with the default instruction budget.
    pub fn new(maps: &'a mut MapTable) -> Self {
        Vm { maps, fuel: DEFAULT_FUEL, ticks: 0 }
    }

    /// Overrides the instruction budget.
    pub fn with_fuel(mut self, fuel: u64) -> Self {
        self.fuel = fuel;
        self
    }

    /// Decodes, verifies, and runs `program` against `ctx`, returning r0.
    pub fn run(&mut self, program: &[u8], ctx: &[u8]) -> Result<u64, VmError> {
        let insns = decode(program).map_err(VmError::Verify)?;
        verify(&insns).map_err(VmError::Verify)?;
        self.execute(&insns, ctx)
    }

    /// Assembles, verifies, and runs a source text.
    pub fn run_assembly(&mut self, text: &str, ctx: &[u8]) -> Result<u64, VmError> {
        let program = assemble(text).map_err(|error| VmError::Assemble(error.to_string()))?;
        self.run(&program, ctx)
    }

    fn execute(&mut self, insns: &[Insn], ctx: &[u8]) -> Result<u64, VmError> {
        let mut regs = [Value::Scalar(0); REGISTER_COUNT];
        regs[1] = Value::Ptr(PtrRef { region: Region::Ctx, off: 0 });
        // r10 addresses the top of the frame: `r10 - 8` is stack offset 504.
        regs[FRAME_POINTER as usize] = Value::Ptr(PtrRef {
            region: Region::Stack,
            off: STACK_SIZE as i64,
        });
        let mut stack = [0u8; STACK_SIZE];
        let mut pc = 0usize;
        loop {
            if self.fuel == 0 {
                return Err(VmError::FuelExhausted);
            }
            self.fuel -= 1;
            let Some(insn) = insns.get(pc).copied() else {
                return Err(VmError::PcOutOfBounds { pc });
            };
            let class = classify(&insn).ok_or(VmError::PcOutOfBounds { pc })?;
            match class {
                Class::Lddw => {
                    let hi = insns
                        .get(pc + 1)
                        .copied()
                        .ok_or(VmError::PcOutOfBounds { pc })?;
                    let immediate = insn.imm64(&hi);
                    regs[insn.dst as usize] = if self.maps.contains(immediate) {
                        Value::MapRef(immediate)
                    } else {
                        Value::Scalar(immediate)
                    };
                    pc += 2;
                }
                Class::Ldx { size } => {
                    let ptr = expect_ptr(&regs, insn.src, pc)?;
                    let value =
                        read_memory(self.maps, &stack, ctx, ptr, insn.off, size, pc)?;
                    regs[insn.dst as usize] = Value::Scalar(value);
                    pc += 1;
                }
                Class::St { size } => {
                    let ptr = expect_ptr(&regs, insn.dst, pc)?;
                    store_memory(
                        &mut self.maps,
                        &mut stack,
                        ptr,
                        insn.off,
                        size,
                        sign_extend(insn.imm),
                        pc,
                    )?;
                    pc += 1;
                }
                Class::Stx { size } => {
                    let ptr = expect_ptr(&regs, insn.dst, pc)?;
                    let value = expect_scalar(&regs, insn.src, pc)?;
                    store_memory(&mut self.maps, &mut stack, ptr, insn.off, size, value, pc)?;
                    pc += 1;
                }
                Class::Alu { op, wide, from_reg } => {
                    execute_alu(&mut regs, &insn, op, wide, from_reg, pc)?;
                    pc += 1;
                }
                Class::End { big_endian, width } => {
                    let value = expect_scalar(&regs, insn.dst, pc)?;
                    let converted = match (big_endian, width) {
                        (false, 16) => value & 0xffff,
                        (false, 32) => value & 0xffff_ffff,
                        (false, 64) => value,
                        (true, 16) => u64::from((value as u16).swap_bytes()),
                        (true, 32) => u64::from((value as u32).swap_bytes()),
                        (true, 64) => value.swap_bytes(),
                        _ => unreachable!("endianness widths are 16, 32, or 64"),
                    };
                    regs[insn.dst as usize] = Value::Scalar(converted);
                    pc += 1;
                }
                Class::Jump { op, wide, from_reg } => match op {
                    JmpOp::Exit => {
                        return match regs[0] {
                            Value::Scalar(value) => Ok(value),
                            _ => Err(VmError::ReturnValueNotScalar { pc }),
                        };
                    }
                    JmpOp::Call => {
                        self.execute_helper(insn.imm, &mut regs, &mut stack, ctx, pc)?;
                        pc += 1;
                    }
                    JmpOp::Ja => {
                        pc = jump_target(pc, insn.off) as usize;
                    }
                    _ => {
                        let taken = eval_condition(&regs, &insn, op, wide, from_reg, pc)?;
                        if taken {
                            pc = jump_target(pc, insn.off) as usize;
                        } else {
                            pc += 1;
                        }
                    }
                },
            }
        }
    }

    fn execute_helper(
        &mut self,
        id: i32,
        regs: &mut [Value; REGISTER_COUNT],
        stack: &mut [u8],
        ctx: &[u8],
        pc: usize,
    ) -> Result<(), VmError> {
        match id {
            helper::MAP_LOOKUP_ELEM => {
                let map_id = match regs[1] {
                    Value::MapRef(id) => id,
                    _ => return Err(VmError::ExpectedMapReference { pc }),
                };
                let key_ptr = match regs[2] {
                    Value::Ptr(ptr) => ptr,
                    _ => return Err(VmError::ExpectedPointer { pc }),
                };
                let key_size = {
                    let map = self
                        .maps
                        .get(map_id)
                        .ok_or(VmError::UnknownMap { pc, id: map_id })?;
                    map.key_size()
                };
                let key = read_bytes(self.maps, stack, ctx, key_ptr, key_size, pc)?;
                let map = self
                    .maps
                    .get(map_id)
                    .ok_or(VmError::UnknownMap { pc, id: map_id })?;
                let slot = map.lookup(&key, pc)?;
                regs[0] = match slot {
                    Some(slot) => Value::Ptr(PtrRef {
                        region: Region::MapSlot { map: map_id, slot },
                        off: 0,
                    }),
                    None => Value::Scalar(0),
                };
            }
            helper::MAP_UPDATE_ELEM => {
                let map_id = match regs[1] {
                    Value::MapRef(id) => id,
                    _ => return Err(VmError::ExpectedMapReference { pc }),
                };
                let key_ptr = match regs[2] {
                    Value::Ptr(ptr) => ptr,
                    _ => return Err(VmError::ExpectedPointer { pc }),
                };
                let value_ptr = match regs[3] {
                    Value::Ptr(ptr) => ptr,
                    _ => return Err(VmError::ExpectedPointer { pc }),
                };
                let (key_size, value_size) = {
                    let map = self
                        .maps
                        .get(map_id)
                        .ok_or(VmError::UnknownMap { pc, id: map_id })?;
                    (map.key_size(), map.value_size())
                };
                let key = read_bytes(self.maps, stack, ctx, key_ptr, key_size, pc)?;
                let value = read_bytes(self.maps, stack, ctx, value_ptr, value_size, pc)?;
                let map = self
                    .maps
                    .get_mut(map_id)
                    .ok_or(VmError::UnknownMap { pc, id: map_id })?;
                map.insert(&key, &value)?;
                regs[0] = Value::Scalar(0);
            }
            helper::KTIME_GET_NS => {
                self.ticks += 1_000;
                regs[0] = Value::Scalar(self.ticks);
            }
            helper::GET_CURRENT_PID_TGID => {
                regs[0] = Value::Scalar((4242u64 << 32) | 17);
            }
            helper::GET_CURRENT_UID_GID => {
                regs[0] = Value::Scalar((1000u64 << 32) | 1000);
            }
            helper::GET_CURRENT_COMM => {
                let dst = match regs[1] {
                    Value::Ptr(ptr) => ptr,
                    _ => return Err(VmError::ExpectedPointer { pc }),
                };
                let len = match regs[2] {
                    Value::Scalar(value) => value as usize,
                    _ => return Err(VmError::ExpectedScalar { pc }),
                };
                let name = b"aegis-vm\0";
                let take = len.min(name.len());
                write_bytes(self.maps, stack, dst, &name[..take], pc)?;
                regs[0] = Value::Scalar(0);
            }
            _ => return Err(VmError::UnknownHelper { pc, id }),
        }
        // The kernel helper ABI clobbers r1..r5; a program that depends on
        // them across a call is a program the verifier already rejected.
        for register in 1..=5 {
            regs[register] = Value::Scalar(0);
        }
        Ok(())
    }
}

/// `lsm/bprm_check_security`, as BPF assembly: allow an exec only when the
/// cgroup has an enforced policy that grants EXEC without denying it, and the
/// executable's inode is in the userspace-verified registry.
///
/// The context is `{ u64 cgroup; u64 inode; }`. Map 1 is the cgroup policy,
/// map 2 the verified-inode registry.
pub const BPRM_CHECK_SECURITY_ASM: &str = r#"r6 = r1                  ; 0
r7 = *(u64 *)(r6 + 0)    ; 1  cgroup id
r8 = *(u64 *)(r6 + 8)    ; 2  inode
*(u64 *)(r10 - 8) = r7   ; 3  policy key
r1 = 1 ll                ; 4  policy map
r2 = r10                 ; 6
r2 += -8                 ; 7
call 1                   ; 8
if r0 == 0 goto +18      ; 9  no policy -> deny
r1 = *(u32 *)(r0 + 0)    ; 10 allow bitmask
r1 &= 1                  ; 11 EXEC bit
if r1 == 0 goto +15      ; 12
r2 = *(u32 *)(r0 + 12)   ; 13 enforce mode
if r2 != 0 goto +13      ; 14 as-configured or mutated -> deny
r3 = *(u32 *)(r0 + 4)    ; 15 deny bitmask
r3 &= 1                  ; 16
if r3 != 0 goto +10      ; 17 deny wins over allow
*(u64 *)(r10 - 24) = r7  ; 18 registry key: cgroup
*(u64 *)(r10 - 16) = r8  ; 19                 inode
r1 = 2 ll                ; 20 registry map
r2 = r10                 ; 22
r2 += -24                ; 23
call 1                   ; 24
if r0 == 0 goto +2       ; 25 unregistered executable -> deny
r0 = 0                   ; 26
exit                     ; 27
r0 = -1                  ; 28
exit                     ; 29
"#;

/// `lsm/file_open`, as BPF assembly: the open mode selects the read or write
/// capability, then the path must take the longest LPM match, and that match
/// must belong to this cgroup and allow the access.
///
/// The context is `{ u64 cgroup; u64 flags; u8 path[128]; }`. Map 1 is the
/// cgroup policy, map 3 the path trie (values are `{ u32 cgroup; u32 allow;
/// u32 reserved[2]; }`).
pub const FILE_OPEN_ASM: &str = r#"r6 = r1                  ; 0
r7 = *(u64 *)(r6 + 0)    ; 1  cgroup id
r8 = *(u64 *)(r6 + 8)    ; 2  open flags
*(u64 *)(r10 - 8) = r7   ; 3  policy key
r1 = 1 ll                ; 4  policy map
r2 = r10                 ; 6
r2 += -8                 ; 7
call 1                   ; 8
if r0 == 0 goto +43      ; 9  no policy -> deny
r9 = r0                  ; 10 policy entry
r1 = *(u32 *)(r9 + 12)   ; 11 enforce mode
if r1 != 0 goto +40      ; 12
r5 = 8                   ; 13 FILE_READ bit
r1 = r8                  ; 14
r1 &= 3                  ; 15 O_WRONLY | O_RDWR
if r1 == 0 goto +1       ; 16
r5 = 16                  ; 17 FILE_WRITE bit
r1 = *(u32 *)(r9 + 4)    ; 18 deny bitmask
r1 &= r5                 ; 19
if r1 != 0 goto +32      ; 20
r1 = *(u32 *)(r9 + 0)    ; 21 allow bitmask
r1 &= r5                 ; 22
if r1 == 0 goto +29      ; 23
r1 = 0                   ; 24 path length
r3 = r6                  ; 25
r3 += 16                 ; 26 path cursor
r2 = *(u8 *)(r3 + 0)     ; 27 loop: next byte
if r2 == 0 goto +8       ; 28 end of path
if r1 == 127 goto +7     ; 29 the context holds 127 bytes at most
r4 = r10                 ; 30
r4 += -132               ; 31 LPM key data
r4 += r1                 ; 32
*(u8 *)(r4 + 0) = r2     ; 33
r1 += 1                  ; 34
r3 += 1                  ; 35
goto -10                 ; 36 -> 27
r2 = r1                  ; 37 prefix length in bits
r2 <<= 3                 ; 38
*(u32 *)(r10 - 136) = r2 ; 39
r1 = 3 ll                ; 40 path trie
r2 = r10                 ; 42
r2 += -136               ; 43
call 1                   ; 44
if r0 == 0 goto +7       ; 45 no prefix matches -> deny
r1 = *(u32 *)(r0 + 0)    ; 46 owning cgroup
r2 = r7                  ; 47
if w1 != w2 goto +4      ; 48 another cgroup's prefix
r1 = *(u32 *)(r0 + 4)    ; 49 allow flag
if r1 == 0 goto +2       ; 50 deny entry
r0 = 0                   ; 51
exit                     ; 52
r0 = -1                  ; 53
exit                     ; 54
"#;

/// `lsm/socket_connect`, as BPF assembly: AF_INET only, enforced policy that
/// grants NET_OUT without denying it, and the destination in this cgroup's
/// network map.
///
/// The context is `{ u64 cgroup; u64 address; u64 family; }`, the address in
/// network byte order as a `u32` value. Map 1 is the cgroup policy, map 4 the
/// destination set (keys are `{ u64 cgroup; u32 address; u32 reserved; }`).
pub const SOCKET_CONNECT_ASM: &str = r#"r6 = r1                  ; 0
r7 = *(u64 *)(r6 + 0)    ; 1  cgroup id
r8 = *(u64 *)(r6 + 8)    ; 2  address
r9 = *(u64 *)(r6 + 16)   ; 3  family
if r9 != 2 goto +26      ; 4  only AF_INET is policed
*(u64 *)(r10 - 8) = r7   ; 5  policy key
r1 = 1 ll                ; 6  policy map
r2 = r10                 ; 8
r2 += -8                 ; 9
call 1                   ; 10
if r0 == 0 goto +19      ; 11 no policy -> deny
r1 = *(u32 *)(r0 + 12)   ; 12 enforce mode
if r1 != 0 goto +17      ; 13
r1 = *(u32 *)(r0 + 0)    ; 14 allow bitmask
r1 &= 2                  ; 15 NET_OUT bit
if r1 == 0 goto +14      ; 16
r1 = *(u32 *)(r0 + 4)    ; 17 deny bitmask
r1 &= 2                  ; 18
if r1 != 0 goto +11      ; 19
*(u64 *)(r10 - 24) = r7  ; 20 destination key: cgroup
*(u32 *)(r10 - 16) = r8  ; 21                  address
*(u32 *)(r10 - 12) = 0   ; 22                  padding
r1 = 4 ll                ; 23 destination map
r2 = r10                 ; 25
r2 += -24                ; 26
call 1                   ; 27
if r0 == 0 goto +2       ; 28 destination not granted -> deny
r0 = 0                   ; 29
exit                     ; 30
r0 = -1                  ; 31
exit                     ; 32
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toolchain;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Assembles one line and returns its wire bytes.
    fn assemble_one(line: &str) -> Vec<u8> {
        assemble(line).unwrap_or_else(|error| panic!("`{line}`: {error}"))
    }

    /// Runs `llvm-mc -triple=bpfel -show-encoding` over `lines` and returns the
    /// encoding of each, in order. Panics if the tool is missing (the vendored
    /// toolchain must exist) or if any line fails to assemble.
    ///
    /// The scratch file name is derived from the corpus content, so two tests
    /// running in parallel never read each other's file.
    fn llvm_mc_encodings(lines: &[&str]) -> Vec<Vec<u8>> {
        let mc = toolchain::require_llvm_tool("llvm-mc");
        let directory = toolchain::repo_root().join("target").join("llvm-mc-check");
        std::fs::create_dir_all(&directory).expect("create llvm-mc scratch dir");
        let text = format!("{}\n", lines.join("\n"));
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in text.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let source = directory.join(format!("corpus-{hash:016x}.s"));
        std::fs::write(&source, &text).expect("write corpus");
        let output = std::process::Command::new(&mc)
            .arg("-triple=bpfel")
            .arg("-show-encoding")
            .arg(&source)
            .output()
            .expect("run llvm-mc");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && !stderr.contains("error"),
            "llvm-mc rejected the corpus:\n{stderr}\n{stdout}"
        );
        let mut encodings = Vec::new();
        for line in stdout.lines() {
            if let Some(start) = line.find("# encoding: [") {
                let rest = &line[start + "# encoding: [".len()..];
                let end = rest.find(']').expect("closing bracket in encoding row");
                let bytes: Vec<u8> = rest[..end]
                    .split(',')
                    .map(|byte| {
                        u8::from_str_radix(byte.trim().trim_start_matches("0x"), 16)
                            .expect("hex byte in encoding row")
                    })
                    .collect();
                encodings.push(bytes);
            }
        }
        encodings
    }

    /// Every instruction form the assembler accepts must be byte-identical to
    /// LLVM's. `llvm-mc` is the external oracle here: if my encoder and LLVM's
    /// disagree on any of these, the VM would execute a different program than
    /// one a real kernel loader would load.
    const CORPUS: [&str; 59] = [
        "r1 = 4660",
        "r1 = -1",
        "r1 = -5",
        "r1 = r2",
        "w1 = 5",
        "w2 = w3",
        "r1 += r2",
        "r1 += 7",
        "r1 += 0x10",
        "r1 -= 7",
        "r1 *= r2",
        "r1 /= r2",
        "r1 %= 8",
        "r1 &= 255",
        "r1 |= r2",
        "r1 ^= 3",
        "r1 <<= 4",
        "r1 >>= 4",
        "r1 s>>= 4",
        "r1 = -r1",
        "w1 += 2",
        "w1 /= 2",
        "w1 s>>= 1",
        "r1 = *(u8 *)(r10 - 1)",
        "r1 = *(u16 *)(r10 - 2)",
        "r1 = *(u32 *)(r10 - 4)",
        "r1 = *(u64 *)(r10 - 8)",
        "r1 = *(u64 *)(r10 + 0)",
        "r1 = *(u64 *)(r2 + 8)",
        "*(u8 *)(r10 - 1) = r1",
        "*(u16 *)(r10 - 2) = r1",
        "*(u32 *)(r10 - 4) = r1",
        "*(u64 *)(r10 - 8) = r1",
        "*(u32 *)(r10 - 8) = 7",
        "*(u16 *)(r10 - 2) = 3",
        "*(u8 *)(r10 - 1) = 4",
        "*(u64 *)(r10 - 8) = r3",
        "*(u64 *)(r2 + 0) = r1",
        "goto +3",
        "goto -3",
        "if r1 == r2 goto +1",
        "if r1 != 0 goto +1",
        "if r1 > 5 goto +1",
        "if r1 < r2 goto +1",
        "if r1 >= 0 goto +1",
        "if r1 <= 5 goto +1",
        "if r1 s> 0 goto +1",
        "if r1 s< 0 goto +1",
        "if r1 s>= 0 goto +1",
        "if r1 s<= 5 goto +1",
        "if r1 & 3 goto +1",
        "if w1 == w2 goto +1",
        "if w1 != 0 goto +1",
        "if w1 > 5 goto +1",
        "if w1 s< 0 goto +1",
        "r1 = 0x1122334455667788 ll",
        "r1 = 4660 ll",
        "call 1",
        "exit",
    ];

    #[test]
    fn every_instruction_form_encodes_exactly_as_llvm_mc_does() {
        let expected = llvm_mc_encodings(&CORPUS);
        assert_eq!(
            expected.len(),
            CORPUS.len(),
            "llvm-mc encoded a different number of lines than the corpus has"
        );
        for (line, want) in CORPUS.iter().zip(expected.iter()) {
            let mine = assemble_one(line);
            assert_eq!(mine, *want, "encoding mismatch for `{line}`");
        }
    }

    #[test]
    fn the_three_decision_programs_encode_exactly_as_llvm_mc_does() {
        for program in [BPRM_CHECK_SECURITY_ASM, FILE_OPEN_ASM, SOCKET_CONNECT_ASM] {
            let lines: Vec<&str> = program
                .lines()
                .map(strip_comment)
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect();
            let expected: Vec<u8> = llvm_mc_encodings(&lines).concat();
            let mine = assemble(program).expect("the looked-at program must assemble");
            assert_eq!(
                mine, expected,
                "a decision program encodes differently than LLVM encodes it"
            );
        }
    }

    #[test]
    fn the_decision_programs_have_the_slot_count_their_offsets_assume() {
        // Every jump offset in the sources was computed by hand against these
        // slot counts; if a line is added or removed, this fails before a
        // mis-targeted branch can execute.
        assert_eq!(assemble(BPRM_CHECK_SECURITY_ASM).unwrap().len(), 30 * 8);
        assert_eq!(assemble(FILE_OPEN_ASM).unwrap().len(), 55 * 8);
        assert_eq!(assemble(SOCKET_CONNECT_ASM).unwrap().len(), 33 * 8);
    }

    #[test]
    fn decoding_reproduces_the_assembler_byte_for_byte() {
        for program in [BPRM_CHECK_SECURITY_ASM, FILE_OPEN_ASM, SOCKET_CONNECT_ASM] {
            let bytes = assemble(program).unwrap();
            let insns = decode(&bytes).unwrap();
            verify(&insns).expect("the decision programs must verify");
            assert_eq!(encode(&insns), bytes, "round trip must be exact");
        }
        // A wide load stays a pair of slots, so jump offsets keep their
        // instruction indices.
        let insns = decode(&assemble("r1 = 0x1122334455667788 ll\nexit\n").unwrap()).unwrap();
        assert_eq!(insns.len(), 3);
        assert_eq!(insns[0].opcode, opcode::LDDW);
        assert_eq!(insns[0].imm64(&insns[1]), 0x1122334455667788);
        assert_eq!(insns[1].opcode, 0);
    }

    // ------------------------------------------------------------------
    // Verifier negatives: each names the property it protects.
    // ------------------------------------------------------------------

    fn verify_text(text: &str) -> Result<(), VerifyError> {
        verify(&decode(&assemble(text).unwrap()).unwrap())
    }

    #[test]
    fn rejects_empty_and_misaligned_programs() {
        assert_eq!(verify(&[]), Err(VerifyError::EmptyProgram));
        assert_eq!(
            decode(&[0x95, 0, 0, 0]),
            Err(VerifyError::MisalignedProgram { bytes: 4 })
        );
    }

    #[test]
    fn rejects_undefined_opcodes() {
        let mut insns = decode(&assemble("exit\n").unwrap()).unwrap();
        insns[0].opcode = 0xff;
        assert_eq!(
            verify(&insns),
            Err(VerifyError::UndefinedOpcode { pc: 0, opcode: 0xff })
        );
    }

    #[test]
    fn rejects_registers_outside_r0_to_r10() {
        let write = encode(&[Insn::new(opcode::ALU64 | opcode::MOV, 11, 0, 0, 1), Insn::new(opcode::JMP | opcode::EXIT, 0, 0, 0, 0)]);
        assert_eq!(
            verify(&decode(&write).unwrap()),
            Err(VerifyError::UnknownRegister { pc: 0, register: 11 })
        );
        let read = encode(&[Insn::new(opcode::ALU64 | opcode::MOV | opcode::X, 0, 12, 0, 0), Insn::new(opcode::JMP | opcode::EXIT, 0, 0, 0, 0)]);
        assert_eq!(
            verify(&decode(&read).unwrap()),
            Err(VerifyError::UnknownRegister { pc: 0, register: 12 })
        );
    }

    #[test]
    fn rejects_writes_to_the_frame_pointer() {
        assert_eq!(
            verify_text("r10 = 1\nexit\n"),
            Err(VerifyError::WriteToFramePointer { pc: 0 })
        );
        assert_eq!(
            verify_text("r10 += 8\nexit\n"),
            Err(VerifyError::WriteToFramePointer { pc: 0 })
        );
    }

    #[test]
    fn rejects_stack_accesses_outside_the_frame() {
        assert_eq!(
            verify_text("r1 = *(u64 *)(r10 + 0)\nexit\n"),
            Err(VerifyError::StackAccessOutOfBounds { pc: 0, offset: 0, size: 8 })
        );
        assert_eq!(
            verify_text("*(u64 *)(r10 - 513) = r1\nexit\n"),
            Err(VerifyError::StackAccessOutOfBounds { pc: 0, offset: -513, size: 8 })
        );
        // The deepest legal slot and a sub-word access must both pass.
        verify_text("r1 = *(u8 *)(r10 - 512)\nr0 = 0\nexit\n").expect("the last byte is in the frame");
        verify_text("*(u64 *)(r10 - 8) = r1\nr0 = 0\nexit\n").expect("the first slot is in the frame");
    }

    #[test]
    fn rejects_jumps_out_of_bounds_and_into_wide_slots() {
        assert_eq!(
            verify_text("goto +50\nexit\n"),
            Err(VerifyError::JumpOutOfBounds { pc: 0, target: 51 })
        );
        // Slot 0 jumps to slot 2, the second slot of the wide load at slot 1.
        assert_eq!(
            verify_text("goto +1\nr1 = 0x1122334455667788 ll\nexit\n"),
            Err(VerifyError::JumpIntoWideImmediate { pc: 0, target: 2 })
        );
        // A backward self-jump is in bounds: it is a loop, and fuel bounds it.
        verify_text("goto -1\nexit\n").expect("a self-jump is a legal loop");
    }

    #[test]
    fn rejects_malformed_wide_immediates() {
        let truncated = encode(&[Insn::new(opcode::LDDW, 1, 0, 0, 5)]);
        assert_eq!(
            verify(&decode(&truncated).unwrap()),
            Err(VerifyError::LddwWithoutSlot { pc: 0 })
        );
        let tampered = encode(&[
            Insn::new(opcode::LDDW, 1, 0, 0, 5),
            Insn::new(0, 3, 0, 0, 0),
            Insn::new(opcode::JMP | opcode::EXIT, 0, 0, 0, 0),
        ]);
        assert_eq!(
            verify(&decode(&tampered).unwrap()),
            Err(VerifyError::BadWideImmediateSlot { pc: 0 })
        );
    }

    #[test]
    fn rejects_immediate_division_by_zero() {
        assert_eq!(
            verify_text("r1 = 5\nr1 /= 0\nexit\n"),
            Err(VerifyError::DivisionByZeroImmediate { pc: 1 })
        );
        assert_eq!(
            verify_text("r1 = 5\nr1 %= 0\nexit\n"),
            Err(VerifyError::DivisionByZeroImmediate { pc: 1 })
        );
    }

    #[test]
    fn rejects_unknown_helpers() {
        assert_eq!(
            verify_text("call 999\nexit\n"),
            Err(VerifyError::UnknownHelper { pc: 0, id: 999 })
        );
    }

    #[test]
    fn rejects_fallthrough_off_the_end() {
        assert_eq!(
            verify_text("r0 = 0\n"),
            Err(VerifyError::FallsThroughEnd { pc: 0 })
        );
        // `goto +0` at the last slot targets one past the end, which is out of
        // bounds before the fallthrough check can even see it.
        assert_eq!(
            verify_text("r1 = 1\nif r1 == 1 goto +0\n"),
            Err(VerifyError::JumpOutOfBounds { pc: 1, target: 2 })
        );
    }

    #[test]
    fn rejects_reads_of_uninitialized_registers() {
        assert_eq!(
            verify_text("r0 = r7\nexit\n"),
            Err(VerifyError::ReadOfUninitializedRegister { pc: 0, register: 7 })
        );
        assert_eq!(
            verify_text("r2 = r2\nexit\n"),
            Err(VerifyError::ReadOfUninitializedRegister { pc: 0, register: 2 })
        );
        // Pass 3 applies on every path: the assignment below exists, but not
        // on the path that reaches the exit.
        assert_eq!(
            verify_text("goto +1\nr0 = 0\nexit\n"),
            Err(VerifyError::ReadOfUninitializedRegister { pc: 2, register: 0 })
        );
        verify_text("r7 = 1\nr0 = r7\nexit\n").expect("written before read must pass");
    }

    #[test]
    fn rejects_use_of_helper_clobbered_registers() {
        // r2 holds the key pointer across the call; the kernel ABI clobbers
        // r1..r5, so reading r2 after the call must be rejected. Slots: the
        // wide load takes 0-1, the move is 2, the call is 3, the read is 4.
        assert_eq!(
            verify_text("r1 = 1 ll\nr2 = r10\ncall 1\nr0 = r2\nexit\n"),
            Err(VerifyError::ReadOfUninitializedRegister { pc: 4, register: 2 })
        );
        // r6 is callee-saved by convention: the same shape must pass.
        verify_text("r6 = r10\nr1 = 1 ll\nr2 = r10\ncall 1\nr0 = r6\nexit\n")
            .expect("r6 survives a call");
    }

    #[test]
    fn accepts_loops_and_multi_slot_iteration() {
        verify_text("r1 = 0\nr1 += 1\nif r1 < 5 goto -3\nr0 = r1\nexit\n")
            .expect("a bounded loop is a normal program");
    }

    // ------------------------------------------------------------------
    // VM semantics
    // ------------------------------------------------------------------

    fn run_with(text: &str, ctx: &[u8], maps: &mut MapTable) -> Result<u64, VmError> {
        Vm::new(maps).run_assembly(text, ctx)
    }

    #[test]
    fn mov_sign_extends_64_bit_immediates() {
        let mut maps = MapTable::new();
        assert_eq!(run_with("r0 = -1\nexit\n", &[], &mut maps), Ok(u64::MAX));
        assert_eq!(run_with("r0 = -2\nexit\n", &[], &mut maps), Ok(u64::MAX - 1));
        assert_eq!(run_with("w0 = -1\nexit\n", &[], &mut maps), Ok(0xffff_ffff));
    }

    #[test]
    fn alu32_operations_wrap_at_thirty_two_bits() {
        let mut maps = MapTable::new();
        assert_eq!(
            run_with("w0 = 0xffff\nw0 += 1\nexit\n", &[], &mut maps),
            Ok(0x1_0000)
        );
        assert_eq!(
            run_with("r0 = 0xffffffffffffffff ll\nw0 += 1\nexit\n", &[], &mut maps),
            Ok(0)
        );
    }

    #[test]
    fn shift_counts_are_masked_like_the_isa_says() {
        let mut maps = MapTable::new();
        assert_eq!(run_with("r0 = 1\nr0 <<= 64\nexit\n", &[], &mut maps), Ok(1));
        assert_eq!(run_with("r0 = 2\nr0 <<= 65\nexit\n", &[], &mut maps), Ok(4));
        assert_eq!(
            run_with("r0 = 0x80000000 ll\nw0 >>= 32\nexit\n", &[], &mut maps),
            Ok(0x8000_0000)
        );
    }

    #[test]
    fn division_by_a_runtime_zero_traps() {
        let mut maps = MapTable::new();
        assert_eq!(
            run_with("r0 = 10\nr1 = 0\nr0 /= r1\nexit\n", &[], &mut maps),
            Err(VmError::DivisionByZero { pc: 2 })
        );
    }

    #[test]
    fn endianness_conversions_are_byte_exact() {
        let mut maps = MapTable::new();
        assert_eq!(
            run_with("r0 = 0x1122334455667788 ll\nr0 = be16 r0\nexit\n", &[], &mut maps),
            Ok(0x8877)
        );
        assert_eq!(
            run_with("r0 = 0x1122334455667788 ll\nr0 = be32 r0\nexit\n", &[], &mut maps),
            Ok(0x8877_6655)
        );
        assert_eq!(
            run_with("r0 = 0x1122334455667788 ll\nr0 = be64 r0\nexit\n", &[], &mut maps),
            Ok(0x8877_6655_4433_2211)
        );
        assert_eq!(
            run_with("r0 = 0x1122334455667788 ll\nr0 = le16 r0\nexit\n", &[], &mut maps),
            Ok(0x7788)
        );
        assert_eq!(
            run_with("r0 = 0x1122334455667788 ll\nr0 = le64 r0\nexit\n", &[], &mut maps),
            Ok(0x1122_3344_5566_7788)
        );
    }

    #[test]
    fn stack_stores_are_visible_to_every_load_width() {
        let mut maps = MapTable::new();
        let store = "r5 = 0x123456789abcdef0 ll\n*(u64 *)(r10 - 8) = r5\n";
        assert_eq!(
            run_with(&format!("{store}r0 = *(u64 *)(r10 - 8)\nexit\n"), &[], &mut maps),
            Ok(0x1234_5678_9abc_def0)
        );
        assert_eq!(
            run_with(&format!("{store}r0 = *(u8 *)(r10 - 8)\nexit\n"), &[], &mut maps),
            Ok(0xf0)
        );
        assert_eq!(
            run_with(&format!("{store}r0 = *(u16 *)(r10 - 8)\nexit\n"), &[], &mut maps),
            Ok(0xdef0)
        );
    }

    #[test]
    fn accesses_outside_the_context_trap() {
        let mut maps = MapTable::new();
        assert_eq!(
            run_with("r0 = *(u64 *)(r1 + 0)\nexit\n", &[0u8; 4], &mut maps),
            Err(VmError::MemoryOutOfBounds {
                pc: 0,
                region: "context",
                offset: 0,
                size: 8
            })
        );
    }

    #[test]
    fn a_scalar_used_as_a_pointer_traps() {
        let mut maps = MapTable::new();
        assert_eq!(
            run_with("r2 = 5\nr0 = *(u64 *)(r2 + 0)\nexit\n", &[], &mut maps),
            Err(VmError::DereferenceOfScalar { pc: 1 })
        );
    }

    #[test]
    fn the_context_is_read_only() {
        let mut maps = MapTable::new();
        assert_eq!(
            run_with("*(u8 *)(r1 + 0) = 7\nr0 = 0\nexit\n", &[0u8; 8], &mut maps),
            Err(VmError::ContextIsReadOnly { pc: 0 })
        );
    }

    #[test]
    fn fuel_bounds_a_loop() {
        let mut maps = MapTable::new();
        assert_eq!(
            Vm::new(&mut maps).with_fuel(64).run_assembly("goto -1\n", &[]),
            Err(VmError::FuelExhausted)
        );
    }

    #[test]
    fn map_lookup_distinguishes_absent_keys_from_present_ones() {
        let mut maps = MapTable::new();
        let mut map = VmMap::hash(8, 4);
        map.insert(&42u64.to_le_bytes(), &[1, 2, 3, 4]).unwrap();
        maps.insert(1, map);

        // An absent key answers scalar zero, and the program exits with it.
        let absent =
            "r3 = 7\n*(u64 *)(r10 - 8) = r3\nr1 = 1 ll\nr2 = r10\nr2 += -8\ncall 1\nexit\n";
        assert_eq!(run_with(absent, &[], &mut maps), Ok(0));

        // A present key answers a pointer into the value slot, and a load
        // through that pointer sees the stored bytes.
        let present = "r3 = 42\n*(u64 *)(r10 - 8) = r3\nr1 = 1 ll\nr2 = r10\nr2 += -8\ncall 1\n\
                       if r0 == 0 goto +3\nr1 = *(u8 *)(r0 + 2)\nr0 = r1\nexit\nr0 = 99\nexit\n";
        assert_eq!(run_with(present, &[], &mut maps), Ok(3));
    }

    #[test]
    fn lpm_lookup_returns_the_longest_stored_prefix() {
        let mut maps = MapTable::new();
        let mut trie = VmMap::lpm(4 + 128, 16);
        trie.insert(&lpm_key_for("/a"), &lpm_value(1, 1)).unwrap();
        trie.insert(&lpm_key_for("/a/b"), &lpm_value(2, 1)).unwrap();
        maps.insert(3, trie);

        // A query key of "/a/b": both stored prefixes match and the longer
        // one must win. Both paths are four bytes, so both are 32-bit keys.
        let query_ab = "r4 = 32\n*(u32 *)(r10 - 136) = r4\nr4 = 0x622f612f\n\
                        *(u32 *)(r10 - 132) = r4\nr1 = 3 ll\nr2 = r10\nr2 += -136\ncall 1\n\
                        r1 = *(u8 *)(r0 + 0)\nr0 = r1\nexit\n";
        assert_eq!(run_with(query_ab, &[], &mut maps), Ok(2));

        // A query key of "/a": only the short prefix matches.
        let query_a = "r4 = 16\n*(u32 *)(r10 - 136) = r4\nr4 = 0x0000612f\n\
                       *(u32 *)(r10 - 132) = r4\nr1 = 3 ll\nr2 = r10\nr2 += -136\ncall 1\n\
                       r1 = *(u8 *)(r0 + 0)\nr0 = r1\nexit\n";
        assert_eq!(run_with(query_a, &[], &mut maps), Ok(1));

        // A key outside every stored prefix: no match, and the lookup answers
        // null, which the program exits with.
        let query_x = "r4 = 16\n*(u32 *)(r10 - 136) = r4\nr4 = 0x0000782f\n\
                       *(u32 *)(r10 - 132) = r4\nr1 = 3 ll\nr2 = r10\nr2 += -136\ncall 1\n\
                       exit\n";
        assert_eq!(run_with(query_x, &[], &mut maps), Ok(0));
    }

    #[test]
    fn the_clock_helper_advances_between_calls() {
        let mut maps = MapTable::new();
        assert_eq!(run_with("call 5\ncall 5\nexit\n", &[], &mut maps), Ok(2000));
    }

    #[test]
    fn the_comm_helper_writes_into_the_frame() {
        let mut maps = MapTable::new();
        let program = "r1 = r10\nr1 += -16\nr2 = 8\ncall 16\nr0 = *(u64 *)(r10 - 16)\nexit\n";
        let name = run_with(program, &[], &mut maps).expect("the comm write must succeed");
        assert_eq!(name.to_le_bytes(), *b"aegis-vm");
    }

    // ------------------------------------------------------------------
    // The decision programs against the fixture aegis_userspace_shim.c
    // builds. Capability bits, guard modes, cgroups, paths, addresses,
    // and registry entries are copied from that file; the integration
    // test in tests/shim_equivalence.rs replays the same fixture by
    // parsing the harness's own MAP lines.
    // ------------------------------------------------------------------

    const CAP_EXEC: u32 = 1 << 0;
    const CAP_NET_OUT: u32 = 1 << 1;
    const CAP_FILE_READ: u32 = 1 << 3;
    const CAP_FILE_WRITE: u32 = 1 << 4;
    const ENFORCE_ON: u32 = 0;
    const ENFORCE_OBSERVE_ONLY: u32 = 1;

    fn policy_bytes(allow: u32, deny: u32, level: u32, mode: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        for word in [allow, deny, level, mode] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    fn lpm_key(bits: u32, path: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(4 + 128);
        key.extend_from_slice(&bits.to_le_bytes());
        let mut data = [0u8; 128];
        data[..path.len()].copy_from_slice(path.as_bytes());
        key.extend_from_slice(&data);
        key
    }

    /// A stored path entry's prefix length is always its byte length in bits,
    /// exactly as the hook's d_path loop computes it.
    fn lpm_key_for(path: &str) -> Vec<u8> {
        lpm_key((path.len() * 8) as u32, path)
    }

    /// `struct path_policy`: owning cgroup, allow flag, reserved.
    fn lpm_value(owner: u32, allow: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        for word in [owner, allow, 0, 0] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    fn fixture_maps() -> MapTable {
        let mut maps = MapTable::new();

        let mut policy = VmMap::hash(8, 16);
        {
            let mut put = |cgroup: u64, allow: u32, deny: u32, level: u32, mode: u32| {
                policy
                    .insert(&cgroup.to_le_bytes(), &policy_bytes(allow, deny, level, mode))
                    .expect("fixture policy insert");
            };
            put(42, CAP_FILE_READ, 0, 1, ENFORCE_ON);
            put(43, CAP_FILE_READ | CAP_FILE_WRITE | CAP_EXEC | CAP_NET_OUT, 0, 2, ENFORCE_ON);
            put(44, CAP_FILE_READ | CAP_EXEC, 0, 2, ENFORCE_OBSERVE_ONLY);
            put(45, CAP_EXEC | CAP_FILE_READ, CAP_EXEC, 3, ENFORCE_ON);
        }
        maps.insert(1, policy);

        let mut registry = VmMap::hash(16, 1);
        for (cgroup, inode) in [(42u64, 558u64), (43, 555), (44, 557), (45, 556)] {
            let mut key = Vec::with_capacity(16);
            key.extend_from_slice(&cgroup.to_le_bytes());
            key.extend_from_slice(&inode.to_le_bytes());
            registry.insert(&key, &[1]).expect("fixture registry insert");
        }
        maps.insert(2, registry);

        let mut trie = VmMap::lpm(4 + 128, 16);
        for (path, allow, owner) in [
            ("/workspace", 1u32, 42u32),
            ("/workspace/.env", 0, 42),
            ("/srv", 1, 43),
            ("/srv/.ssh", 0, 43),
            ("/other", 1, 43),
        ] {
            trie.insert(&lpm_key_for(path), &lpm_value(owner, allow))
                .expect("fixture trie insert");
        }
        maps.insert(3, trie);

        let mut net = VmMap::hash(16, 1);
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&43u64.to_le_bytes());
        key.extend_from_slice(&0x090071CBu32.to_le_bytes());
        key.extend_from_slice(&0u32.to_le_bytes());
        net.insert(&key, &[1]).expect("fixture net insert");
        maps.insert(4, net);

        maps
    }

    fn bprm_ctx(cgroup: u64, inode: u64) -> Vec<u8> {
        let mut ctx = cgroup.to_le_bytes().to_vec();
        ctx.extend_from_slice(&inode.to_le_bytes());
        ctx
    }

    fn file_open_ctx(cgroup: u64, flags: u64, path: &str) -> Vec<u8> {
        let mut ctx = cgroup.to_le_bytes().to_vec();
        ctx.extend_from_slice(&flags.to_le_bytes());
        let mut data = [0u8; 128];
        data[..path.len()].copy_from_slice(path.as_bytes());
        ctx.extend_from_slice(&data);
        ctx
    }

    fn socket_ctx(cgroup: u64, family: u64, address: u32) -> Vec<u8> {
        let mut ctx = cgroup.to_le_bytes().to_vec();
        ctx.extend_from_slice(&u64::from(address).to_le_bytes());
        ctx.extend_from_slice(&family.to_le_bytes());
        ctx
    }

    fn bprm_allows(maps: &mut MapTable, cgroup: u64, inode: u64) -> bool {
        Vm::new(maps)
            .run_assembly(BPRM_CHECK_SECURITY_ASM, &bprm_ctx(cgroup, inode))
            .expect("the decision program must run")
            == 0
    }

    fn file_open_allows(maps: &mut MapTable, cgroup: u64, flags: u64, path: &str) -> bool {
        Vm::new(maps)
            .run_assembly(FILE_OPEN_ASM, &file_open_ctx(cgroup, flags, path))
            .expect("the decision program must run")
            == 0
    }

    fn socket_allows(maps: &mut MapTable, cgroup: u64, family: u64, address: u32) -> bool {
        Vm::new(maps)
            .run_assembly(SOCKET_CONNECT_ASM, &socket_ctx(cgroup, family, address))
            .expect("the decision program must run")
            == 0
    }

    #[test]
    fn the_bprm_program_reproduces_the_harness_cases() {
        let mut maps = fixture_maps();
        // aegis_userspace_shim.c run_cases: verified binary allowed; never
        // registered, exec-not-granted, observe-only, deny-wins, and no
        // policy all denied.
        assert!(bprm_allows(&mut maps, 43, 555));
        assert!(!bprm_allows(&mut maps, 43, 556));
        assert!(!bprm_allows(&mut maps, 42, 558));
        assert!(!bprm_allows(&mut maps, 44, 557));
        assert!(!bprm_allows(&mut maps, 45, 556));
        assert!(!bprm_allows(&mut maps, 9999, 555));
    }

    #[test]
    fn the_file_open_program_reproduces_the_harness_cases() {
        let mut maps = fixture_maps();
        // The NULL path in the shim (d_path failure) is the empty string
        // here: an unresolvable path and an empty path both fall out of the
        // trie, and both deny.
        assert!(file_open_allows(&mut maps, 42, 0, "/workspace/a.txt"));
        assert!(!file_open_allows(&mut maps, 42, 0, "/etc/passwd"));
        assert!(!file_open_allows(&mut maps, 42, 0, "/workspace/.env"));
        assert!(!file_open_allows(&mut maps, 42, 0, "/workspace/.env.local"));
        assert!(!file_open_allows(&mut maps, 42, 2, "/workspace/a.txt"));
        assert!(file_open_allows(&mut maps, 43, 2, "/srv/b.txt"));
        assert!(!file_open_allows(&mut maps, 43, 0, "/srv/.ssh/key"));
        assert!(!file_open_allows(&mut maps, 43, 0, ""));
        assert!(!file_open_allows(&mut maps, 42, 0, "/other/x"));
        assert!(file_open_allows(&mut maps, 43, 0, "/other/x"));
        assert!(!file_open_allows(&mut maps, 44, 0, "/workspace/a.txt"));
        assert!(!file_open_allows(&mut maps, 9999, 0, "/workspace/a.txt"));
    }

    #[test]
    fn the_socket_program_reproduces_the_harness_cases() {
        let mut maps = fixture_maps();
        // The one listed destination allows; an unlisted address, a non-IPv4
        // family, and every cgroup without the grant all deny.
        assert!(socket_allows(&mut maps, 43, 2, 0x090071CB));
        assert!(!socket_allows(&mut maps, 43, 2, 0x0A0071CB));
        assert!(!socket_allows(&mut maps, 43, 10, 0));
        assert!(!socket_allows(&mut maps, 42, 2, 0x090071CB));
        assert!(!socket_allows(&mut maps, 44, 2, 0x090071CB));
        assert!(!socket_allows(&mut maps, 9999, 2, 0x090071CB));
    }
}

