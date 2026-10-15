"""Ad-hoc probe: print the CLI's per-fixture DTG summary as JSON.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Runs `aegis-cli scan --json` over every fixture and prints one compact
    record per file, so the sink set, the dependency list, and the compression
    ratio can be read without scrolling past a full DTG dump.

WHY
    `benchmarks/run_all_benchmarks.py` needs the same data, but it also needs
    assertions and multiple output formats. This is the diagnostic view: no
    thresholds, no failure codes, just what the extractor actually saw.

HOW
    Subprocess per fixture, JSON parsed with `json.loads`, failures reported
    with the CLI's own stderr rather than swallowed.

Run: python scripts/scan_probe.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "benchmarks" / "fixtures" / "vulnerable-lab"
CLI = REPO_ROOT / "target" / "debug" / "aegis-cli.exe"


def main() -> int:
    if not CLI.exists():
        print(f"missing {CLI}; build with cargo build -p aegis-cli", file=sys.stderr)
        return 2

    records = []
    for path in sorted(FIXTURES.iterdir()):
        if path.suffix not in {".py", ".js", ".ts", ".rs"}:
            continue
        completed = subprocess.run(
            [str(CLI), "scan", "--path", str(path), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            records.append(
                {
                    "fixture": path.name,
                    "error": (completed.stderr or completed.stdout).strip().splitlines()[0:1],
                }
            )
            continue
        document = json.loads(completed.stdout)
        server = document["servers"][0]
        records.append(
            {
                "fixture": path.name,
                "language": server["language"],
                "tools": server["tools"],
                "raw": server["raw_cpg_node_count"],
                "dtg": server["compressed_node_count"],
                "compression_ratio_pct": round(server["compression_ratio"] * 100.0, 2),
                "lattice": server["lattice"],
                "sinks": server["sink_kinds"],
                "sink_count": server["sink_count"],
                "max_hop_depth": server["max_hop_depth"],
                "dependencies": server["dependencies"],
                "dtg_sha256": server["dtg_sha256"][:16],
            }
        )

    print(json.dumps(records, indent=2))
    failures = [r for r in records if "error" in r]
    print(f"\n{len(records)} fixtures, {len(failures)} extraction failure(s)", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())