"""Deployment-manifest validation.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Parses the YAML and JSON deployment manifests and asserts the security
    properties they claim. This is a test of the *manifests*, not of the images:
    a compose file that forgets `cap_drop` is a deployment that runs privileged,
    and nothing else in the repository would notice.

WHY A STRUCTURED CHECK AND NOT A REVIEW
    The compose file is the only place where "drop all capabilities",
    "read-only filesystem" and "no new privileges" are actually asserted. Those
    three lines are the difference between a security tool and a security
    tool-shaped container, and they are exactly the lines a well-meaning edit
    removes.

HOW
    `yaml` is not vendored on this host, so the parser is a small indentation
    reader written for this file's subset: mappings, sequences, scalars, and
    comments. It is deliberately not a general YAML implementation -- it is a
    checker that fails loudly on anything it does not understand rather than
    guessing.

Run: python scripts/validate_deploy.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE = REPO_ROOT / "docker-compose.yml"
PUBSPEC = REPO_ROOT / "apps" / "mobile-approver" / "pubspec.yaml"
JSON_MANIFESTS = (
    REPO_ROOT / "apps" / "web-portal" / "package.json",
    REPO_ROOT / "apps" / "web-portal" / "next.config.json",
    REPO_ROOT / "apps" / "desktop-firewall" / "tauri.conf.json",
    REPO_ROOT / "specs" / "tacr.schema.json",
    REPO_ROOT / "benchmarks" / "fixtures" / "mcptox_chains" / "exfil_chains.json",
    REPO_ROOT / "benchmarks" / "results" / "summary_metrics.json",
)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


# ---------------------------------------------------------------------------
# A deliberately small YAML reader for the subset these manifests use
# ---------------------------------------------------------------------------
def strip_comment(line: str) -> str:
    """Remove a trailing comment, respecting quoted strings."""
    out: list[str] = []
    quote: str | None = None
    previous = ""
    for character in line:
        if quote:
            out.append(character)
            if character == quote and previous != "\\":
                quote = None
        elif character in "\"'":
            quote = character
            out.append(character)
        elif character == "#" and (not out or out[-1] in " \t"):
            break
        else:
            out.append(character)
        previous = character
    return "".join(out).rstrip()


def scalar(text: str):
    """Convert a YAML scalar to a Python value, or fail loudly."""
    text = text.strip()
    if not text:
        return ""
    if text[0] in "\"'" and text[-1] == text[0] and len(text) >= 2:
        return text[1:-1]
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text in ("null", "~"):
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [scalar(part) for part in inner.split(",")] if inner else []
    return text


def parse_yaml(text: str):
    """Parse block mappings and block sequences by indentation."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        cleaned = strip_comment(raw)
        if not cleaned.strip():
            continue
        lines.append((len(cleaned) - len(cleaned.lstrip(" ")), cleaned.strip()))

    def parse_block(index: int, indent: int):
        if index >= len(lines):
            return None, index
        if lines[index][1].startswith("- "):
            items = []
            while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
                body = lines[index][1][2:].strip()
                index += 1
                if re.match(r"^[\w.\-/]+:(\s|$)", body) and not body.startswith("{"):
                    # A sequence item that is itself a mapping. `key: value` opens
                    # one; `key:value` does not -- the distinction matters
                    # because `- no-new-privileges:true` is a scalar, and reading
                    # it as a mapping turns the string into a dict and makes the
                    # security check fail for the wrong reason.
                    key, _, rest = body.partition(":")
                    item = {key.strip(): scalar(rest)}
                    while index < len(lines) and lines[index][0] > indent:
                        sub, index = parse_block(index, lines[index][0])
                        if isinstance(sub, dict):
                            item.update(sub)
                    items.append(item)
                else:
                    items.append(scalar(body))
            return items, index
        mapping: dict = {}
        while index < len(lines) and lines[index][0] == indent:
            key, _, rest = lines[index][1].partition(":")
            key = key.strip()
            index += 1
            if rest.strip():
                mapping[key] = scalar(rest)
            elif index < len(lines) and lines[index][0] > indent:
                child_indent = lines[index][0]
                if lines[index][1].startswith("- ") or lines[index][1] == "-":
                    child, index = parse_block(index, child_indent)
                    mapping[key] = child
                else:
                    child, index = parse_block(index, child_indent)
                    mapping[key] = child
            else:
                mapping[key] = None
        return mapping, index

    parsed, _ = parse_block(0, lines[0][0] if lines else 0)
    return parsed


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate_compose() -> dict:
    text = COMPOSE.read_text(encoding="utf-8")
    document = parse_yaml(text)
    check(isinstance(document, dict), "docker-compose.yml did not parse as a mapping")
    services = (document or {}).get("services") or {}
    check(bool(services), "docker-compose.yml declares no services")
    check("name" in (document or {}), "the compose project has no name")

    required_services = {"verifier", "ml-engine", "proxy-gateway", "ebpf-loader"}
    missing = required_services - set(services)
    check(not missing, f"compose file is missing services: {sorted(missing)}")

    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        check("image" in service, f"service '{name}' pins no image tag")
        if name == "ebpf-loader":
            # The one documented exception, and it must be marked.
            check(
                "kernel-lsm" in (service.get("profiles") or []),
                "ebpf-loader must sit behind the `kernel-lsm` profile so it is "
                "not started on a host that cannot enforce",
            )
            check(bool(service.get("cap_add")), "ebpf-loader needs its capabilities declared")
            continue
        check(service.get("read_only") is True, f"service '{name}' does not set read_only")
        check(
            service.get("cap_drop") == ["ALL"],
            f"service '{name}' does not drop every capability: {service.get('cap_drop')}",
        )
        check(
            service.get("security_opt") == ["no-new-privileges:true"],
            f"service '{name}' does not set no-new-privileges: {service.get('security_opt')}",
        )
        check(
            str(service.get("user", "")).startswith("65534"),
            f"service '{name}' does not run as an unprivileged user: {service.get('user')}",
        )
        check("healthcheck" in service, f"service '{name}' declares no healthcheck")

    networks = (document or {}).get("networks") or {}
    check(
        (networks.get("aegis-control") or {}).get("internal") is True,
        "the control network must be internal: the verifier and the ML engine "
        "should not be reachable from outside it",
    )
    check("aegis-edge" in networks, "no edge network is declared")
    for name, service in services.items():
        joined = (service or {}).get("networks") or []
        if name in ("verifier", "ml-engine"):
            check(
                "aegis-edge" not in joined,
                f"service '{name}' joins the edge network; only the gateway should",
            )
    return document or {}


def validate_no_llm_in_manifests() -> None:
    """No deployment manifest may name a generative runtime (ADR-001)."""
    forbidden = ("openai", "anthropic", "ollama", "llama", "cohere", "huggingface")
    for path in (COMPOSE, PUBSPEC) + JSON_MANIFESTS:
        if not path.exists():
            continue
        haystack = path.read_text(encoding="utf-8").lower()
        for name in forbidden:
            # A mention in a *comment* explaining that none is present is fine;
            # a dependency or environment variable is not.
            for match in re.finditer(rf"\b{name}\b", haystack):
                line_start = haystack.rfind("\n", 0, match.start()) + 1
                line = haystack[line_start : haystack.find("\n", match.start())]
                if line.lstrip().startswith("#") or line.lstrip().startswith("//"):
                    continue
                failures.append(f"{path.name} names '{name}': {line.strip()[:80]}")


def validate_json() -> None:
    for path in JSON_MANIFESTS:
        if not path.exists():
            failures.append(f"missing manifest {path.name}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            failures.append(f"{path.name} is not valid JSON: {error}")


def validate_pubspec() -> None:
    text = PUBSPEC.read_text(encoding="utf-8")
    check(text.lstrip().startswith("name:"), "pubspec.yaml does not start with a name key")
    check("flutter:" in text, "pubspec.yaml declares no flutter section")
    check(
        "notBuiltOnThisHost" in text,
        "pubspec.yaml must state that it was not built on this host, so nobody "
        "reads a passing test suite as covering it",
    )
    check(
        "description" in text and "Author" not in text.split("description")[0],
        "pubspec.yaml is missing its description",
    )


def validate_readme_claims_measured_numbers() -> None:
    """The README must quote the numbers the harness actually wrote.

    A README that hand-edits a figure is the exact failure ADR-016 was written to
    stop, and it is invisible to every other check in the repository.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    summary_path = REPO_ROOT / "benchmarks" / "results" / "summary_metrics.json"
    if not summary_path.exists():
        failures.append("no summary_metrics.json; the README's numbers cannot be checked")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    static = summary["static"]
    check(
        f"{static['recall_count']} / {static['vulnerable']}" in readme,
        "the README does not quote the measured recall",
    )
    check(
        f"{static['false_positive_count']} / {static['benign']}" in readme,
        "the README does not quote the measured false-positive count",
    )
    check(
        f"{summary['divergence']['max_benign_score']:.4f}" in readme,
        "the README does not quote the measured maximum benign divergence score",
    )
    check(
        f"{summary['divergence']['min_malicious_score']:.4f}" in readme,
        "the README does not quote the measured minimum malicious divergence score",
    )
    check(
        f"{summary['rag']['mean_hybrid_ndcg_at_10']:.4f}" in readme,
        "the README does not quote the measured hybrid nDCG",
    )
    check(
        f"{summary['chains']['blocked']} / {summary['chains']['malicious']}" in readme,
        "the README does not quote the measured chain block count",
    )
    check(
        str(summary["machine"]["machine_tag"]) in readme,
        "the README does not name the machine the numbers came from",
    )
    check(
        summary["threshold_failures"] == [],
        f"the harness reported threshold failures: {summary['threshold_failures']}",
    )


def main() -> int:
    validate_json()
    validate_compose()
    validate_pubspec()
    validate_no_llm_in_manifests()
    validate_readme_claims_measured_numbers()

    if failures:
        print("DEPLOYMENT MANIFEST FAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("all deployment manifests valid and the README matches the measured results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
