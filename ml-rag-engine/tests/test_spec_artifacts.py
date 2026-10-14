"""Static validation of the formal specification and protocol artefacts.

WHAT:  Parses and structurally verifies every file under ``specs/``:
        - ``tacr.schema.json``       JSON Schema Draft 2020-12 well-formedness
        - ``lattice_rules.smt2``     SMT-LIBv2 parenthesisation and shape
        - ``MCPShieldLTS.tla``       TLA+ block balance and operator definition
        - ``proto/aegis.proto``      proto3 brace balance and RPC/service shape

WHY:   The four MCPShield safety properties are only as trustworthy as the
        specification that encodes them. A TLA+ module with an unbalanced
        ``\\*`` block comment, or an SMT-LIBv2 file with one missing paren, fails
        to load in TLC / Z3 and silently degrades into "we ran the verifier and
        it produced no counterexamples" -- which is indistinguishable from a
        clean bill of health. These tests are the difference between a
        specification that is checked and one that is merely present.

HOW:   Pure-stdlib parsers. TLC and Z3 are external binaries that are not
        installed on every developer host (and are not required to run the
        Rust suite, which carries its own product-automaton solver), so these
        tests enforce the structural invariants that guarantee the files are
        *loadable* wherever those binaries do exist.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS = REPO_ROOT / "specs"


# ---------------------------------------------------------------------------
# tacr.schema.json
# ---------------------------------------------------------------------------
def _load_schema() -> dict:
    with open(SPECS / "tacr.schema.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_tacr_schema_is_valid_json_and_declares_draft_2020_12() -> None:
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"].startswith("AegisCPG")


def test_tacr_schema_requires_every_mandated_field() -> None:
    schema = _load_schema()
    required = set(schema["required"])
    mandated = {
        "tool_id",
        "server_version",
        "ast_dtg_sha256",
        "semantic_divergence_score",
        "security_lattice_label",
        "granted_capabilities",
        "composition_policy",
        "ttl_seconds",
        "ecdsa_p256_signature",
    }
    assert mandated <= required, f"missing required fields: {mandated - required}"


def test_tacr_schema_granted_capabilities_has_all_four_prefix_lists() -> None:
    schema = _load_schema()
    caps = schema["properties"]["granted_capabilities"]
    assert set(caps["required"]) == {
        "file_read_prefixes",
        "file_write_prefixes",
        "allowed_exec_sha256",
        "allowed_net_cidrs",
    }


def test_tacr_schema_lattice_label_enum_matches_rust_lattice() -> None:
    schema = _load_schema()
    assert schema["properties"]["security_lattice_label"]["enum"] == [
        "PUBLIC",
        "INTERNAL",
        "CONFIDENTIAL",
        "SECRET",
    ]


def test_tacr_schema_encodes_the_adr005_fail_closed_contract() -> None:
    """Dynamic reflection must force both OS allowlists empty.

    This is the machine-checkable form of ADR-005. If someone relaxes the
    constraint to let a self-declared "trusted" server keep an exec allowlist,
    this test fails.
    """
    schema = _load_schema()
    conditions = schema.get("allOf", [])
    assert conditions, "schema lost its conditional constraints"

    dynamic = [c for c in conditions if "UnboundedDynamic" in json.dumps(c)]
    assert dynamic, "no conditional constrains UnboundedDynamic tools"
    enforced = dynamic[0]["then"]["properties"]["granted_capabilities"]["properties"]
    assert enforced["allowed_exec_sha256"]["maxItems"] == 0
    assert enforced["allowed_net_cidrs"]["maxItems"] == 0


def test_tacr_schema_rejects_unknown_top_level_properties() -> None:
    """An attacker must not be able to smuggle fields past verification."""
    assert _load_schema()["additionalProperties"] is False


def test_tacr_schema_rejects_sink_kind_outside_the_rust_enum() -> None:
    """SinkKind strings must stay in lockstep with aegis_common::types::SinkKind."""
    expected = {
        "FileRead",
        "FileWrite",
        "CommandExec",
        "NetworkOutbound",
        "NetworkInbound",
        "EnvAccess",
        "UnboundedDynamic",
    }
    schema = _load_schema()
    sink_items = schema["properties"]["detected_sinks"]["items"]
    assert set(sink_items["properties"]["sink_kind"]["enum"]) == expected


# ---------------------------------------------------------------------------
# lattice_rules.smt2
# ---------------------------------------------------------------------------
def _strip_smt_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def _test_smt_parentheses_balanced() -> bool:
    text = _strip_smt_comments((SPECS / "lattice_rules.smt2").read_text(encoding="utf-8"))
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def test_lattice_rules_smt2_parentheses_are_balanced() -> None:
    assert _test_smt_parentheses_balanced(), "unbalanced parentheses -> Z3 cannot load"


def test_lattice_rules_smt2_declares_the_required_definitions() -> None:
    """dtg_flows_to is deliberately an uninterpreted predicate (declare-fun)
    constrained by assertions, not a define-fun macro: whether taint actually
    flows between two tools is a property of the discovered DTG, not something
    the specification should hard-code. The rest are define-fun macros."""
    text = _strip_smt_comments((SPECS / "lattice_rules.smt2").read_text(encoding="utf-8"))
    for name in ("flows_to", "join", "meet", "can_chain"):
        assert f"(define-fun {name} " in text, f"missing define-fun {name}"
    for name in (
        "dtg_reads",
        "dtg_flows_to",
        "dtg_emits",
        "dtg_sinks_net",
        "granted_read",
        "granted_chain",
        "granted_net_sink",
        "declassified",
        "lattice_of",
        "clearance_of",
        "hosting_server",
    ):
        assert f"(declare-fun {name} " in text, f"missing declare-fun {name}"
    for name in (
        "tool_integrity_violated",
        "data_confinement_violated",
        "privilege_boundedness_violated",
        "context_isolation_violated",
        "any_property_violated",
    ):
        assert f"(define-fun {name} " in text, f"missing define-fun {name}"


def test_lattice_rules_smt2_has_the_three_analysis_sections() -> None:
    """PART A is the base context (no push needed); PART B and PART C each open
    their own pushed context. Every section must emit exactly one verdict."""
    text = _strip_smt_comments((SPECS / "lattice_rules.smt2").read_text(encoding="utf-8"))
    assert text.count("(push)") == 2
    assert text.count("(pop)") == 2
    for section in ("PART A", "PART B", "PART C"):
        assert f"{section}:" in text, section
    # Comments are stripped first, so the header's "(check-sat) => sat" prose
    # does not inflate the count. One verdict per section: A, B, and C.
    assert text.count("(check-sat)") == 3


def test_lattice_rules_smt2_uses_no_llm_or_network_symbols() -> None:
    """ADR-001: the formal core must stay hermetic. Comments are stripped first
    so the authorship header's repository URL is not mistaken for a call."""
    text = _strip_smt_comments((SPECS / "lattice_rules.smt2").read_text(encoding="utf-8")).lower()
    for banned in ("openai", "anthropic", "gemini", "ollama", "http://", "https://"):
        assert banned not in text, f"ADR-001 violation: '{banned}' in the SMT encoding"


# ---------------------------------------------------------------------------
# MCPShieldLTS.tla
# ---------------------------------------------------------------------------
_TLA_KEYWORDS = {
    "MODULE", "EXTENDS", "CONSTANTS", "VARIABLES", "ASSUME", "THEOREM",
    "LET", "IN", "IF", "THEN", "ELSE", "CHOOSE", "ENABLED", "UNCHANGED",
    "BOOLEAN", "TRUE", "FALSE", "STRING", "EXCEPT", "WITH",
}

# Built-ins that come from the standard TLA+ modules this file EXTENDS.
_TLA_BUILTINS = {
    "Nat", "Int", "Bool", "Seq", "SUBSET", "UNION", "DOMAIN", "Cardinality",
    "Len", "Append", "Head", "Tail", "SubSeq", "WF_vars", "SF_vars",
    "WF_vars", "ENABLED", "UNCHANGED", "Cross", "Times",
}


def _strip_tla_comments(text: str) -> str:
    """Remove (* block *) and \\* line comments.

    Necessary before scanning for operator references: prose in the module
    header mentions `H(def)` and "MCPShield (Acharya" which would otherwise be
    mistaken for unapplied operators.
    """
    text = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.DOTALL)
    return "\n".join(line.split("\\*", 1)[0] for line in text.splitlines())


def _tla_operators(module_text: str) -> set[str]:
    """Operator definitions, including parameterised ones.

    TLA+ allows both `Foo == expr` and `Foo(a, b) == expr`, so the parameter
    list is optional in the pattern.
    """
    return set(
        re.findall(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^=]*?\))?\s*==",
            module_text,
            re.MULTILINE,
        )
    )


def _tla_block_comments_closed(text: str) -> bool:
    return text.count("(*") == text.count("*)")


def test_tla_module_declares_the_correct_module_name() -> None:
    text = (SPECS / "MCPShieldLTS.tla").read_text(encoding="utf-8")
    assert re.search(r"^-+ MODULE MCPShieldLTS -+$", text, re.MULTILINE)
    assert text.rstrip().endswith("=")


def test_tla_block_comments_are_balanced() -> None:
    """An unclosed (* *) comment makes TLC fail to parse and emit no result."""
    text = (SPECS / "MCPShieldLTS.tla").read_text(encoding="utf-8")
    assert _tla_block_comments_closed(text)


def test_tla_declares_all_seven_required_state_variables() -> None:
    text = _strip_tla_comments((SPECS / "MCPShieldLTS.tla").read_text(encoding="utf-8"))
    for var in (
        "activeConnections",
        "agentKnowledge",
        "grantedCapabilities",
        "approvedToolHashes",
        "currentToolHashes",
        "declassifiedFlows",
        "traceLog",
    ):
        # traceLog is the final entry in the VARIABLES block, so it carries no
        # trailing comma; the separator is therefore optional.
        assert re.search(rf"^\s+{var}\s*,?\s*$", text, re.MULTILINE), f"missing {var}"


def test_tla_defines_all_five_transition_actions() -> None:
    ops = _tla_operators((SPECS / "MCPShieldLTS.tla").read_text(encoding="utf-8"))
    for action in (
        "InitializeSession",
        "InvokeToolRead",
        "InvokeToolWrite",
        "MutateToolDefinition",
        "Fido2Declassify",
    ):
        assert action in ops, f"missing transition action {action}"


def test_tla_defines_all_four_safety_invariants() -> None:
    ops = _tla_operators((SPECS / "MCPShieldLTS.tla").read_text(encoding="utf-8"))
    for invariant in (
        "ToolIntegrityInvariant",
        "DataConfinementInvariant",
        "PrivilegeBoundednessInvariant",
        "ContextIsolationInvariant",
    ):
        assert invariant in ops, f"missing invariant {invariant}"


def test_tla_every_referenced_operator_is_defined() -> None:
    """Catches the copy/paste class of bug where an action references a name
    that was never defined, which TLC reports as an opaque 'Unknown operator'
    rather than as a syntax error."""
    raw = (SPECS / "MCPShieldLTS.tla").read_text(encoding="utf-8")
    text = _strip_tla_comments(raw)
    defined = _tla_operators(raw) | _TLA_KEYWORDS | _TLA_BUILTINS
    referenced = set(re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*\(", text))
    undefined = referenced - defined
    assert not undefined, f"unbound operator references in TLA+: {sorted(undefined)}"


def test_tla_invoke_tool_write_binds_every_parameter_it_uses() -> None:
    """InvokeToolWrite(a, s, t, r, d) must not reference `s_target`, a name bound
    only in Fido2Declassify. This exact bug shipped in the first draft."""
    raw = (SPECS / "MCPShieldLTS.tla").read_text(encoding="utf-8")
    body = re.search(
        r"^InvokeToolWrite\(a, s, t, r, d\)\s*==(.*?)^MutateToolDefinition",
        raw,
        re.MULTILINE | re.DOTALL,
    )
    assert body, "InvokeToolWrite definition not found"
    assert "s_target" not in body.group(1), "InvokeToolWrite references unbound s_target"


@pytest.mark.parametrize("cfg_name", ["MCPShieldLTS.cfg", "MCPShieldLTS.safe.cfg"])
def test_tla_configs_check_all_four_invariants(cfg_name: str) -> None:
    text = (SPECS / cfg_name).read_text(encoding="utf-8")
    for invariant in (
        "ToolIntegrityInvariant",
        "DataConfinementInvariant",
        "PrivilegeBoundednessInvariant",
        "ContextIsolationInvariant",
    ):
        assert invariant in text


def test_tla_unprotected_and_protected_configs_differ_on_the_gateway_flag() -> None:
    """The vulnerability/fix pair is only meaningful if the two configs really
    toggle the control plane."""
    vulnerable = (SPECS / "MCPShieldLTS.cfg").read_text(encoding="utf-8")
    protected = (SPECS / "MCPShieldLTS.safe.cfg").read_text(encoding="utf-8")
    assert "GatewayEnforced = FALSE" in vulnerable
    assert "GatewayEnforced = TRUE" in protected
    assert "VerifiedDeclassifications = {}" in vulnerable
    assert "VerifiedDeclassifications" in protected


# ---------------------------------------------------------------------------
# proto/aegis.proto
# ---------------------------------------------------------------------------
def test_proto_braces_are_balanced() -> None:
    text = (SPECS / "proto" / "aegis.proto").read_text(encoding="utf-8")
    assert text.count("{") == text.count("}")
    assert text.count("(") == text.count(")")


def test_proto_is_proto3_and_defines_the_package() -> None:
    text = (SPECS / "proto" / "aegis.proto").read_text(encoding="utf-8")
    assert re.search(r'^syntax\s*=\s*"proto3";', text, re.MULTILINE)
    assert re.search(r"^package\s+aegis\.v1;", text, re.MULTILINE)


def test_proto_declares_the_three_mandated_rpcs() -> None:
    text = (SPECS / "proto" / "aegis.proto").read_text(encoding="utf-8")
    for rpc in (
        "EvaluateSemanticDivergence",
        "SynthesizeMinCutPolicy",
        "StreamKernelTelemetry",
    ):
        assert re.search(rf"rpc\s+{rpc}\s*\(", text), f"missing rpc {rpc}"


def test_proto_stream_kernel_telemetry_is_server_streaming() -> None:
    text = (SPECS / "proto" / "aegis.proto").read_text(encoding="utf-8")
    match = re.search(r"rpc\s+StreamKernelTelemetry\s*\([^)]*\)\s*returns\s*\(\s*stream\s+\w+\s*\)", text)
    assert match, "telemetry must be server-streaming or the portal loses ringbuf events"


def test_proto_field_numbers_are_unique_within_each_message() -> None:
    """protoc rejects duplicates; catching it here gives a better message."""
    text = (SPECS / "proto" / "aegis.proto").read_text(encoding="utf-8")
    for message in re.findall(r"message\s+(\w+)\s*\{(.*?)\n\}", text, re.DOTALL):
        name, body = message
        numbers = re.findall(r"=\s*(\d+)\s*;", body)
        assert len(numbers) == len(set(numbers)), f"duplicate field number in {name}"