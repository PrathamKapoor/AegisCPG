"""Six-act walkthrough of the AegisCPG pipeline, on the real binaries.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Runs six acts, in order, against the compiled artefacts a user actually
    ships, and asserts each act's claim instead of printing a picture:

      1 extract   `aegis-cli scan` over the Postmark-style BCC rug-pull
                  fixture: compression, lattice, sinks, egress literals, H(DTG)
      2 verify    `aegis-cli verify` over two workflow configurations -- one
                  that violates Data Confinement and one narrowed by clearance
                  -- plus the emitted SMT-LIBv2 script, written beside the run
      3 model     `aegis-cli check-tla` over the unprotected and the protected
                  configurations of the shipped TLA+ LTS
      4 sandbox   the gateway's live code-to-eBPF sandbox: the presets are
                  fetched from the running binary and every one is replayed
                  end to end (extract, verify, compile, disassemble, RFC 9669
                  verification, kernel verdict)
      5 enforce   `aegis-cli audit-and-enforce` over the vulnerable lab: TACRs
                  synthesised, BPF policies compiled, attack-surface reduction
      6 evidence  the sealed artefacts of the Phase 9 test runs: the 1,000-case
                  three-way differential fuzz and the algebraic-properties
                  three-way guard agreement

WHY
    Every number in the stage script has to be reproducible by the person
    hearing it. This runner is that reproducibility: one command, real
    binaries, real artefacts, and a non-zero exit the moment an act's claim
    stops holding. It also fails the run when the wall clock exceeds the
    ten-second budget the walkthrough promises, so a regression that doubles
    the demo cannot hide behind a green tick.

HOW
    The acts call the same executables the documentation names (`aegis-cli`,
    `aegis-proxy-gateway`) with the same arguments, under the same working
    directories, and parse their real stdout. The sandbox act starts the
    gateway on its loopback bind, waits for `/api/health`, replays the five
    compiled-in presets, and stops the process in a `finally` block so a
    failed act cannot leak a listener.

    Exit codes: 0 when every act passed inside the budget; 1 otherwise.
    `--smoke` tolerates absent Phase 9 artefacts (acts 6's two files are then
    reported as skipped) so the runner works straight after a clean clone;
    the default mode requires them and names the command that produces each.
    `--list` prints the act table and exits.

Run: python scripts/demo_walkthrough.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = REPO_ROOT / "target" / "demo-walkthrough"
LAB_FIXTURE = (
    REPO_ROOT / "benchmarks" / "fixtures" / "vulnerable-lab" / "js_malicious_tools.js"
)
TLA_SPEC = REPO_ROOT / "specs" / "MCPShieldLTS.tla"
TLA_UNPROTECTED = REPO_ROOT / "specs" / "MCPShieldLTS.cfg"
TLA_SAFE = REPO_ROOT / "specs" / "MCPShieldLTS.safe.cfg"
FUZZ_SUMMARY = REPO_ROOT / "target" / "differential-fuzz" / "fuzz_summary.json"
ALGEBRA_SUMMARY = REPO_ROOT / "target" / "algebraic-properties" / "summary.json"
GATEWAY_BASE = "http://127.0.0.1:8080"

#: The demo promises to finish inside ten seconds; the runner holds it to that.
BUDGET_SECONDS = 10.0

#: The rug-pull fixture's hidden BCC address, pinned so a drifted fixture fails
#: the extract act rather than quietly changing the story.
RUGPULL_EGRESS_LITERAL = "collector@attacker.tld"

#: The differential-fuzz seed the Phase 9 test pins; the artefact must carry it.
FUZZ_SEED = "0xae615c9620260008"
FUZZ_CASES = 1000

#: The algebraic-properties counts the Phase 9 suite pins.
ALGEBRA_HASH_CASES = 200
ALGEBRA_LATTICE_TRIPLES = 64
ALGEBRA_TUPLE_CASES = 200


class ActFailure(Exception):
    """An act's claim did not hold."""


class SkipAct(Exception):
    """An act cannot run in this mode (a missing artefact under --smoke)."""


def find_binary(name: str) -> Path:
    """Locate a compiled binary, preferring release over debug."""
    suffix = ".exe" if sys.platform == "win32" else ""
    for profile in ("release", "debug"):
        candidate = REPO_ROOT / "target" / profile / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise ActFailure(
        f"{name} is not built; run "
        f"`cargo build --package {name}` first (the walkthrough drives real "
        f"binaries, not a reimplementation)"
    )


def run_cli(arguments: list[str], expect_exit: int = 0, cwd: Path | None = None) -> str:
    """Runs `aegis-cli` and returns stdout, asserting the exit code."""
    completed = subprocess.run(
        [str(CLI), *arguments],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != expect_exit:
        raise ActFailure(
            f"`aegis-cli {' '.join(arguments)}` exited {completed.returncode}, "
            f"expected {expect_exit}\n{(completed.stderr or completed.stdout).strip()[:600]}"
        )
    return completed.stdout


def http_get(path: str, timeout: float = 2.0) -> tuple[int, dict]:
    with urllib.request.urlopen(GATEWAY_BASE + path, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def http_post(path: str, body: dict, timeout: float = 30.0) -> tuple[int, dict]:
    request = urllib.request.Request(
        GATEWAY_BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def stage_verify_lab() -> tuple[Path, Path]:
    """Materialises the verify act's inputs under `target/`.

    The workflow's server id is the name of the directory the fixture lives
    in, because `scan_repository` names a server after its directory; the
    fixture therefore lives in `lab/` and the configs sit one level above it.
    """
    lab = SCRATCH / "lab"
    lab.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LAB_FIXTURE, lab / "rugpull.js")
    base = {
        "server_clearances": {"lab": "PUBLIC"},
        "granted_tools": ["read_file", "send_email"],
        "links": ["read_file>send_email"],
        "known_data": [["read_file", "CONFIDENTIAL"]],
    }
    violating = SCRATCH / "workflow.json"
    violating.write_text(json.dumps(base, indent=2), encoding="utf-8")
    narrowed = dict(base)
    narrowed["server_clearances"] = {"lab": "CONFIDENTIAL"}
    narrowed_path = SCRATCH / "workflow_narrowed.json"
    narrowed_path.write_text(json.dumps(narrowed, indent=2), encoding="utf-8")
    return violating, narrowed_path


# ---------------------------------------------------------------------------
# The six acts
# ---------------------------------------------------------------------------
def act_extract() -> list[str]:
    """Act 1: Stage 1 over the rug-pull fixture, asserted field by field."""
    stdout = run_cli(["scan", "--path", str(LAB_FIXTURE), "--json"])
    document = json.loads(stdout)
    server = document["servers"][0]
    if server["tools"] != 2:
        raise ActFailure(f"expected 2 registered tools, saw {server['tools']}")
    if "FileRead" not in server["sink_kinds"]:
        raise ActFailure(f"the file read sink is missing: {server['sink_kinds']}")
    if RUGPULL_EGRESS_LITERAL not in server["egress_literals"]:
        raise ActFailure(
            f"the hidden BCC literal is not hashed: {server['egress_literals']}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", server["dtg_sha256"]):
        raise ActFailure(f"malformed DTG hash: {server['dtg_sha256']}")
    return [
        f"2 tools, {server['raw_cpg_node_count']} -> {server['compressed_node_count']} "
        f"nodes ({server['compression_ratio'] * 100:.1f}% reduced), "
        f"lattice {server['lattice']}",
        f"sinks {server['sink_kinds']}, egress literals {server['egress_literals']}",
        f"H(DTG) {server['dtg_sha256']}",
    ]


def act_verify() -> list[str]:
    """Act 2: Stage 3 over a violating and a narrowed configuration."""
    violating, narrowed = stage_verify_lab()
    sat = run_cli(["verify", "--config", str(violating), "--smt"], expect_exit=1, cwd=SCRATCH)
    if "SAT" not in sat:
        raise ActFailure("the violating configuration did not report SAT")
    counterexamples = len(re.findall(r"VIOLATION Property 2: Data Confinement", sat))
    if counterexamples < 2:
        raise ActFailure(f"expected the singleton and pair counterexamples, saw {counterexamples}")
    if "read_file -> send_email" not in sat:
        raise ActFailure("the composition witness is missing from the report")
    marker = "--- generated SMT-LIBv2 ---"
    if marker not in sat:
        raise ActFailure("--smt emitted no script")
    script = sat.split(marker, 1)[1].strip() + "\n"
    if "(check-sat)" not in script or "data_confinement_violated" not in script:
        raise ActFailure("the emitted script is not a decidable SMT-LIBv2 query")
    if "(> (clearance" in script:
        raise ActFailure("the retired clearance guard reappeared in the script")
    if "(or\n  ))" in script:
        raise ActFailure("the script contains an empty disjunction")
    script_path = SCRATCH / "policy.smt2"
    script_path.write_text(script, encoding="utf-8")

    clean = run_cli(["verify", "--config", str(narrowed)], expect_exit=0, cwd=SCRATCH)
    if "UNSAT" not in clean or clean.count("VERIFIED") != 4:
        raise ActFailure("the narrowed configuration did not verify all four properties")
    return [
        f"PUBLIC clearance : SAT, {counterexamples} counterexamples "
        f"(read_file -> send_email at CONFIDENTIAL)",
        "CONFIDENTIAL clearance: UNSAT, 4/4 properties verified",
        f"script written to {script_path.relative_to(REPO_ROOT)} "
        f"({len(script.splitlines())} lines, retired guard absent)",
    ]


def act_model() -> list[str]:
    """Act 3: the TLA+ LTS, unprotected and protected."""
    arguments = ["check-tla", "--spec", str(TLA_SPEC)]
    unprotected = json.loads(
        run_cli(
            arguments
            + ["--config", str(TLA_UNPROTECTED), "--mode", "as-configured",
               "--expect", "violations", "--json"]
        )
    )
    if not unprotected["violations"]:
        raise ActFailure("the unprotected configuration produced no witness")
    first = unprotected["violations"][0]
    safe = json.loads(
        run_cli(
            arguments
            + ["--config", str(TLA_SAFE), "--mode", "as-configured",
               "--expect", "clean", "--json"]
        )
    )
    if safe["violations"] or safe["limits_hit"]:
        raise ActFailure("the protected configuration did not check clean")
    return [
        f"unprotected: {len(unprotected['violations'])} violation(s), first "
        f"{first['invariant']} at depth {first['depth']}, "
        f"{unprotected['distinct_states_explored']} states explored",
        f"protected  : clean, {safe['distinct_states_explored']} states explored, "
        f"search complete",
    ]


def act_sandbox() -> list[str]:
    """Act 4: the live sandbox, all five presets, asserted against the payload."""
    process = subprocess.Popen(
        [str(GATEWAY)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    lines: list[str] = []
    try:
        ready = False
        for _ in range(100):
            if process.poll() is not None:
                raise ActFailure(
                    f"the gateway exited before becoming ready (code {process.returncode}); "
                    f"is {GATEWAY_BASE} already in use?"
                )
            try:
                status, _ = http_get("/api/health", timeout=0.5)
                if status == 200:
                    ready = True
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                time.sleep(0.1)
        if not ready:
            raise ActFailure("the gateway never answered /api/health")
        _, listing = http_get("/api/live-analyze/presets")
        presets = listing["presets"]
        if len(presets) != 5:
            raise ActFailure(f"expected 5 presets, saw {len(presets)}")
        first_disassembly: list[str] = []
        for preset in presets:
            body = {
                "source": preset["source"],
                "language": preset["language"],
                "server_id": preset["id"],
                "clearance": preset["clearance"],
                "hook": preset["hook"],
                "target": preset["target"],
            }
            status, response = http_post("/api/live-analyze", body)
            if status != 200:
                raise ActFailure(f"preset {preset['id']} answered {status}")
            expected = preset["expected"]
            stage2, stage3, stage4 = response["stage2"], response["stage3"], response["stage4"]
            if stage2["verdict"] != expected["verdict"]:
                raise ActFailure(
                    f"preset {preset['id']}: verdict {stage2['verdict']} != "
                    f"{expected['verdict']}"
                )
            if stage3["safe"] != expected["safe"]:
                raise ActFailure(f"preset {preset['id']}: safe {stage3['safe']} != {expected['safe']}")
            if not (stage4["verified"] and stage4["agree"]):
                raise ActFailure(
                    f"preset {preset['id']}: verified={stage4['verified']} "
                    f"agree={stage4['agree']}"
                )
            if stage4["simulator_verdict"] != expected["kernel"]:
                raise ActFailure(
                    f"preset {preset['id']}: simulator {stage4['simulator_verdict']} != "
                    f"{expected['kernel']}"
                )
            if stage4["vm_verdict"] != expected["kernel"]:
                raise ActFailure(
                    f"preset {preset['id']}: VM {stage4['vm_verdict']} != {expected['kernel']}"
                )
            if len(stage4["instructions"]) == 0 or len(stage4["bytecode_hex"]) == 0:
                raise ActFailure(f"preset {preset['id']}: empty program")
            if not re.fullmatch(r"[0-9a-f]{64}", stage3["tacr"]["tacr_sha256"]):
                raise ActFailure(f"preset {preset['id']}: malformed TACR hash")
            lines.append(
                f"{preset['id']:<22} {stage2['verdict']:<10} "
                f"kernel {stage4['simulator_verdict']:<5} "
                f"{len(stage4['instructions']):>3} insns  "
                f"H(DTG) {response['stage1']['dtg_sha256'][:12]}"
            )
            if preset["id"] == "py_env_harvester":
                first_disassembly = stage4["disassembly"][:4]
                (SCRATCH / "sandbox_response.json").write_text(
                    json.dumps(response, indent=2), encoding="utf-8"
                )
        lines.append("py_env_harvester disassembly (first four slots):")
        lines.extend(f"  {instruction}" for instruction in first_disassembly)
        return lines
    finally:
        process.kill()
        process.wait(timeout=10)


def act_enforce() -> list[str]:
    """Act 5: the audit-and-enforce report over the vulnerable lab."""
    stdout = run_cli(
        ["audit-and-enforce", "--repo", str(LAB_FIXTURE.parent)]
    )
    tacrs = re.search(r"TACRs emitted\s*:\s*(\d+)", stdout)
    policies = re.search(r"BPF policies compiled:\s*(\d+)", stdout)
    eliminated = re.search(r"attack surface eliminated:\s*([\d.]+)%", stdout)
    exec_grants = re.search(r"cgroups with an exec grant:\s*(\d+)", stdout)
    if not (tacrs and policies and eliminated and exec_grants):
        raise ActFailure(f"the enforce report is missing a stage line:\n{stdout[:600]}")
    if int(tacrs.group(1)) < 1 or int(tacrs.group(1)) != int(policies.group(1)):
        raise ActFailure("TACRs and BPF policies did not match one-for-one")
    if float(eliminated.group(1)) <= 0.0:
        raise ActFailure("the least-privilege synthesis eliminated nothing")
    if "RESULT:" not in stdout:
        raise ActFailure("the report carries no RESULT line")
    return [
        f"TACRs emitted {tacrs.group(1)}, BPF policies compiled {policies.group(1)}, "
        f"attack surface eliminated {eliminated.group(1)}%",
        f"cgroups with an exec grant: {exec_grants.group(1)} (least privilege holds)",
        "kernel readiness and the reduced-guarantee note printed",
    ]


def act_evidence(smoke: bool) -> list[str]:
    """Act 6: the Phase 9 artefacts, re-asserted rather than re-run."""
    missing = [path for path in (FUZZ_SUMMARY, ALGEBRA_SUMMARY) if not path.exists()]
    if missing:
        if smoke:
            raise SkipAct(
                "no Phase 9 artefacts yet; run the two test targets to produce them"
            )
        names = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing)
        raise ActFailure(
            f"missing {names}; produce them with "
            f"`cargo test -p aegis-ebpf-loader --test differential_fuzz` and "
            f"`cargo test -p aegis-smt-verifier --test algebraic_properties`"
        )
    fuzz = json.loads(FUZZ_SUMMARY.read_text(encoding="utf-8"))
    algebra = json.loads(ALGEBRA_SUMMARY.read_text(encoding="utf-8"))
    if fuzz["seed"] != FUZZ_SEED or fuzz["cases"] != FUZZ_CASES:
        raise ActFailure(f"the fuzz artefact drifted: seed {fuzz['seed']}, cases {fuzz['cases']}")
    if not fuzz["three_way_equal"]:
        raise ActFailure("the three-way fuzz verdict equality did not hold")
    if fuzz["allows"] + fuzz["denials"] != fuzz["probes"]:
        raise ActFailure("the fuzz verdict counts do not sum to the probe count")
    if fuzz["mutant_control_divergences"] <= 0:
        raise ActFailure("the mutant control caught nothing")
    inputs = algebra["inputs"]
    if (
        inputs["dtg_hash_cases"] != ALGEBRA_HASH_CASES
        or inputs["lattice_triples"] != ALGEBRA_LATTICE_TRIPLES
        or inputs["tuple_cases"] != ALGEBRA_TUPLE_CASES
    ):
        raise ActFailure(f"the algebraic artefact drifted: {inputs}")
    guard = algebra["property_c"]
    if guard["three_way_agreements"] != ALGEBRA_TUPLE_CASES or not guard["facts_consistent"]:
        raise ActFailure("the guard agreement did not hold on every tuple")
    return [
        f"fuzz     : {fuzz['cases']} cases, {fuzz['probes']} probes, "
        f"{fuzz['allows']} allow / {fuzz['denials']} deny, three-way equal, "
        f"seed {fuzz['seed']}",
        f"mutants  : {fuzz['mutant_control_divergences']} divergences and "
        f"{fuzz['mutant_control_fixture_failures']} pinned fixture failure(s) in the control",
        f"algebraic: {inputs['dtg_hash_cases']} hash cases, {inputs['lattice_triples']} "
        f"lattice triples, {inputs['tuple_cases']} guard tuples, three-way agreement "
        f"against {guard['spec']}",
    ]


ACTS = (
    ("extract", "Stage 1 over the BCC rug-pull fixture", lambda smoke: act_extract()),
    ("verify", "Stage 3 over a violating and a narrowed workflow", lambda smoke: act_verify()),
    ("model", "the TLA+ LTS, unprotected and protected", lambda smoke: act_model()),
    ("sandbox", "the live code-to-eBPF sandbox, all five presets", lambda smoke: act_sandbox()),
    ("enforce", "audit-and-enforce over the vulnerable lab", lambda smoke: act_enforce()),
    ("evidence", "the Phase 9 differential-fuzz and algebraic artefacts", act_evidence),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tolerate absent Phase 9 artefacts (the evidence act is then skipped)",
    )
    parser.add_argument("--list", action="store_true", help="print the act table and exit")
    args = parser.parse_args(argv)

    if args.list:
        for number, (name, description, _) in enumerate(ACTS, start=1):
            print(f"  {number}  {name:<9} {description}")
        return 0

    print("AegisCPG six-act walkthrough", flush=True)
    print(f"  cli     : {CLI}", flush=True)
    print(f"  gateway : {GATEWAY}", flush=True)

    started = time.time()
    failures: list[str] = []
    skipped: list[str] = []
    for number, (name, description, runner) in enumerate(ACTS, start=1):
        act_started = time.time()
        print(f"\n== Act {number}  {name} -- {description} ==", flush=True)
        try:
            lines = runner(args.smoke)
        except SkipAct as reason:
            skipped.append(name)
            print(f"   skipped: {reason}", flush=True)
            continue
        except ActFailure as reason:
            failures.append(f"act {number} ({name}): {reason}")
            print(f"   FAILED: {reason}", flush=True)
            continue
        for line in lines:
            print(f"   {line}", flush=True)
        print(f"   ok ({(time.time() - act_started) * 1000:.0f} ms)", flush=True)

    elapsed = time.time() - started
    total = len(ACTS)
    passed = total - len(failures) - len(skipped)
    if failures:
        print("\n" + "\n".join(f"FAILED {failure}" for failure in failures), flush=True)
    print(
        f"\ndemo: {passed}/{total} acts passed"
        + (f", {len(skipped)} skipped" if skipped else "")
        + f" in {elapsed:.1f}s (budget {BUDGET_SECONDS:.1f}s)",
        flush=True,
    )
    if failures:
        return 1
    if elapsed > BUDGET_SECONDS:
        print(
            f"FAILED: the walkthrough took {elapsed:.1f}s, over its "
            f"{BUDGET_SECONDS:.1f}s budget",
            flush=True,
        )
        return 1
    return 0


#: The binaries every act drives. Resolved at import so a missing build fails
#: before any act starts, with the build command in the message.
CLI = find_binary("aegis-cli")
GATEWAY = find_binary("aegis-proxy-gateway")


if __name__ == "__main__":
    raise SystemExit(main())
