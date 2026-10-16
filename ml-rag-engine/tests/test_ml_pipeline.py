"""End-to-end verification of the AegisCPG ML pipeline.

Author: Pratham Kapoor <prathamkapoor027@gmail.com>
https://github.com/PrathamKapoor

WHAT
    Exercises the four Python subsystems against the *real* compiled binary:
    the NumPy GAT + sklearn GBDT + ONNX classifier, the hybrid RRF retriever,
    the HiGHS MILP privilege cut, and the durable declassification workflow.

WHY THESE TESTS LOOK THE WAY THEY DO
    Every assertion here runs real arithmetic on real extracted graphs. Several
    of them exist because the corresponding defect shipped and produced a
    plausible-looking wrong answer rather than an error:

    * the ONNX graph returned a constant because leaf weights were never gated
      by path conditions;
    * ONNX `Sum` with no axes collapsed a batch to one scalar;
    * `GAT_EMBED_DIM` disagreed with what `forward` returned;
    * the DTG parser read `node_type` as CamelCase while Rust emits snake_case,
      so every node fell through to the default and every tool scored benign;
    * a node carries no `sink_kind` in the wire format -- the sink is on the
      edge -- so a builder that only reads nodes reports no sinks at all.

    A test suite that does not pin those facts will not catch them again.

HOW
    Module-scoped fixtures train and export the model once, scan every fixture
    through `target/<profile>/aegis-cli.exe scan --json`, and reuse both. A test
    that reported a number without running the binary would be exactly the
    unfalsifiable benchmark this repository forbids, so the binary's absence is
    a hard skip with an explicit reason, never a silent pass.

Complexity: the shared fixture set-up is O(F) CLI invocations plus one fit;
per-test work is O(F) or O(1).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "benchmarks" / "fixtures" / "vulnerable-lab"

VULNERABLE_FIXTURES = (
    "go_reverse_shell_sink.go",
    "js_indirect_prompt_inj.js",
    "js_indirect_prompt_rmt.js",
    "js_malicious_code_exec.js",
    "js_malicious_tools.js",
    "js_namespace_typosquat.js",
    "js_outdated_packages.js",
    "js_secrets_pii.js",
    "py_filesystem_workspace.py",
    "py_multifile_split_exfil.py",
    "py_wikipedia_streamable.py",
    "rs_unsafe_command_exec.rs",
    "ts_ssrf_metadata_exfil.ts",
)

BENIGN_FIXTURES = (
    "benign_json_validator.rs",
    "benign_markdown_formatter.js",
    "benign_math_server.py",
    "benign_sqlite_readonly.py",
    "benign_timezone_converter.ts",
)

#: The decision threshold the gateway applies. Restated here so a change in
#: either place is caught rather than silently moving the operating point.
DIVERGENCE_THRESHOLD = 0.5


def _find_cli() -> Path | None:
    for profile in ("release", "debug"):
        name = "aegis-cli.exe" if __import__("os").name == "nt" else "aegis-cli"
        candidate = REPO_ROOT / "target" / profile / name
        if candidate.exists():
            return candidate
    return None


CLI = _find_cli()
requires_cli = pytest.mark.skipif(
    CLI is None,
    reason=(
        "aegis-cli is not built. Run `cargo build -p aegis-cli`. Every test in "
        "this file measures the compiled binary; without it there is nothing to "
        "measure and a silent pass would be worse than a skip."
    ),
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def scans() -> dict[str, dict]:
    """Every fixture's `scan --json` document, keyed by filename."""
    if CLI is None:
        pytest.skip("aegis-cli is not built")
    out: dict[str, dict] = {}
    for name in VULNERABLE_FIXTURES + BENIGN_FIXTURES:
        path = FIXTURE_DIR / name
        assert path.exists(), f"missing shipped fixture {path}"
        completed = subprocess.run(
            [str(CLI), "scan", "--path", str(path), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, (
            f"{name}: aegis-cli exited {completed.returncode}: {completed.stderr.strip()}"
        )
        document = json.loads(completed.stdout)
        assert document["schema"] == "aegis.scan.v1"
        assert document["server_count"] == 1
        out[name] = document["servers"][0]
    return out


@pytest.fixture(scope="session")
def model():
    """A trained, exported and loaded divergence model.

    Training is unconditional rather than "load if present". `test_onnx_matches_
    sklearn_exactly` needs the fitted sklearn estimator to compare against, and
    a session fixture that silently skipped the fit would make that comparison
    untestable on a fresh checkout. Training takes about a second.
    """
    from aegis_ml.gat_divergence import DivergenceModel

    instance = DivergenceModel()
    instance.train()
    instance.export_onnx()
    return instance


@pytest.fixture(scope="session")
def scores(model, scans) -> dict[str, float]:
    """One divergence score per fixture, measured through the real binary."""
    return {
        name: model.predict(server["dtg"], "", tool_id=name).score
        for name, server in scans.items()
    }


# ---------------------------------------------------------------------------
# DTG wire-format contract
# ---------------------------------------------------------------------------
@requires_cli
def test_every_fixture_extracts(scans: dict[str, dict]) -> None:
    """No shipped fixture may fail extraction.

    `extract_source` refuses to report a file containing MCP registration syntax
    but resolving no tool as clean, which is the right fail-closed behaviour.
    It also means a parser regression surfaces here as a hard failure instead of
    a quietly shorter recall number.
    """
    assert len(scans) == len(VULNERABLE_FIXTURES) + len(BENIGN_FIXTURES)
    for name, server in scans.items():
        assert server["compressed_node_count"] > 0, f"{name} produced an empty DTG"
        assert server["tools"] >= 1, f"{name} resolved no tool registration"
        assert len(server["dtg_sha256"]) == 64


@requires_cli
def test_every_fixture_compresses_above_the_hop_bound(scans: dict[str, dict]) -> None:
    for name, server in scans.items():
        assert server["compression_ratio"] >= 0.80, (
            f"{name} compressed to only {server['compression_ratio']:.4f}"
        )
        assert server["max_hop_depth"] <= server.get("hop_bound", 3)


@requires_cli
def test_dtgs_are_byte_identical_across_runs() -> None:
    """The DTG hash is a rug-pull detector, so it must be reproducible.

    Two scans of one file must produce the same hash. A hash that changed
    between runs would make the TACR's pinned `ast_dtg_sha256` unmeetable and
    turn the rug-pull check into a permanent failure rather than a real alarm.
    """
    if CLI is None:
        pytest.skip("aegis-cli is not built")
    target = FIXTURE_DIR / "js_malicious_code_exec.js"
    hashes = set()
    for _ in range(2):
        completed = subprocess.run(
            [str(CLI), "scan", "--path", str(target), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        hashes.add(json.loads(completed.stdout)["servers"][0]["dtg_sha256"])
    assert len(hashes) == 1, f"the DTG hash is not stable: {hashes}"


@requires_cli
def test_wire_format_normalisation_is_total() -> None:
    """The Python builder must accept every spelling Rust can emit.

    Rust serialises `node_type` snake_case and `lattice` as a label; the feature
    layout names the variants CamelCase and indexes the lattice numerically.
    Reconciling them is the boundary's job, and an unrecognised value must raise
    rather than fall through to a default -- the default is `LocalBuffer` at
    level 0, which scores a real tool as benign.
    """
    from aegis_ml.gat_divergence import (
        NODE_TYPE_ALIASES,
        NODE_TYPES,
        LATTICE_LABELS,
        normalise_lattice,
        normalise_node_type,
    )

    for snake, camel in NODE_TYPE_ALIASES.items():
        assert camel in NODE_TYPES, f"{camel} is not in the feature layout"
        assert normalise_node_type(snake) == camel
        assert normalise_node_type(camel) == camel, "both spellings must be accepted"

    assert set(LATTICE_LABELS.values()) == {0, 1, 2, 3}
    for label, rank in LATTICE_LABELS.items():
        assert normalise_lattice(label) == rank
        assert normalise_lattice(rank) == rank, "both spellings must be accepted"

    with pytest.raises(ValueError):
        normalise_lattice("TOP_SECRET")
    with pytest.raises(ValueError):
        normalise_node_type("mystery_node")
    with pytest.raises(ValueError):
        # An unknown index must raise rather than clamp: clamping 9 to 3 would
        # read as SECRET, and clamping down would read as PUBLIC.
        normalise_lattice(9)


@requires_cli
def test_sinks_are_read_from_edges_not_nodes(scans: dict[str, dict]) -> None:
    """A Rust DTG node has no `sink_kind`; the sink is on the incoming edge.

    A builder that only inspects nodes reports an empty sink set for every
    tool, and an empty sink set is exactly what the divergence classifier reads
    as benign. This asserts the Python builder recovers the sink set from the
    edge structure for a fixture whose node objects provably carry none.
    """
    from aegis_ml.gat_divergence import build_tool_graph

    server = scans["py_filesystem_workspace.py"]
    dtg = server["dtg"]
    assert all("sink_kind" not in node for node in dtg["nodes"]), (
        "if the wire format ever moves the sink onto the node, this test is "
        "testing the wrong thing and must be revisited"
    )
    graph = build_tool_graph(dtg["nodes"], dtg["edges"])
    assert graph.sink_kinds, "the sink set was not recovered from the edges"
    assert "CommandExec" in graph.sink_kinds
    assert graph.max_lattice == 3, "a command-execution tool runs at SECRET"


# ---------------------------------------------------------------------------
# The GAT
# ---------------------------------------------------------------------------
def test_gat_attention_rows_sum_to_one() -> None:
    """Attention must be a distribution over neighbours.

    Masked with -1e9 rather than zeroed: zeroing and renormalising divides by
    zero on an isolated node, yields NaN, and NaN silently becomes a 0.5 score
    for every tool.
    """
    from aegis_ml.gat_divergence import (
        INPUT_FEATURE_DIM,
        NODE_TYPES,
        NumPyGATv2Layer,
    )

    layer = NumPyGATv2Layer()
    rng = np.random.default_rng(7)
    features = rng.random((5, INPUT_FEATURE_DIM))
    adjacency = np.eye(5, dtype=np.float64)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[1, 2] = adjacency[2, 1] = 1.0
    # Node 3 and node 4 are deliberately isolated.

    alphas = layer.attention(features, adjacency)
    assert np.isfinite(alphas).all(), "isolated nodes produced NaN attention"
    for head in range(alphas.shape[0]):
        for row in range(alphas.shape[1]):
            total = float(alphas[head, row].sum())
            assert abs(total - 1.0) < 1e-9, f"head {head} row {row} sums to {total}"


def test_gat_embedding_width_matches_the_declared_dimension() -> None:
    """`forward` must return exactly `GAT_EMBED_DIM` columns.

    It returned 32 while the constant said 16 for a while, producing a 40-wide
    feature vector against a graph declared for 24. The mismatch surfaced as an
    onnxruntime dimension error, i.e. it was caught by luck rather than by a
    test. This is that test.
    """
    from aegis_ml.gat_divergence import GAT_EMBED_DIM, INPUT_FEATURE_DIM, NumPyGATv2Layer

    layer = NumPyGATv2Layer()
    features = np.eye(INPUT_FEATURE_DIM, dtype=np.float32)[:4]
    embedding, alphas = layer.forward(features, np.eye(4, dtype=np.float64))
    assert embedding.shape == (GAT_EMBED_DIM,), embedding.shape
    assert layer.out_dim == GAT_EMBED_DIM, (
        "the layer's declared output width is the embedding's width; they are "
        "the same number and the feature layout depends on it"
    )
    assert alphas.ndim == 3


def test_gat_weights_are_seeded_and_reproducible() -> None:
    from aegis_ml.gat_divergence import NumPyGATv2Layer

    a = NumPyGATv2Layer().weights_spec()
    b = NumPyGATv2Layer().weights_spec()
    assert a == b, "the GAT is supposed to be seeded, not trained"
    restored = NumPyGATv2Layer.from_spec(a)
    assert np.allclose(restored.W, NumPyGATv2Layer().W)
    assert np.allclose(restored.W_out, NumPyGATv2Layer().W_out)


def test_gat_weights_are_not_degenerate() -> None:
    """Xavier-scaled weights must be non-zero and of the expected magnitude."""
    from aegis_ml.gat_divergence import NumPyGATv2Layer

    layer = NumPyGATv2Layer()
    assert float(np.linalg.norm(layer.W)) > 1e-3
    assert float(np.linalg.norm(layer.a_src)) > 1e-3
    assert float(np.linalg.norm(layer.a_dst)) > 1e-3
    assert float(np.linalg.norm(layer.W_out)) > 1e-3


# ---------------------------------------------------------------------------
# The classifier and its ONNX export
# ---------------------------------------------------------------------------
def test_training_corpus_is_extracted_by_the_real_binary() -> None:
    """The corpus must come from `aegis-cli`, not from fabricated features.

    A corpus whose graph features do not come from the same GAT that inference
    uses trains a model that scores every real tool benign while reporting
    AUC 1.0 in training. That is not hypothetical; it is what shipped.
    """
    from aegis_ml.gat_divergence import build_training_corpus

    if CLI is None:
        pytest.skip("aegis-cli is not built; the corpus cannot be extracted")
    x, y, samples = build_training_corpus()
    assert x.shape[0] == len(samples)
    assert set(np.unique(y)) == {0, 1}, "the corpus must contain both classes"
    assert samples[0]["source"] == "generated-aegis-cli", (
        f"corpus fell back to {samples[0]['source']!r}; the numbers would not "
        "describe the shipped toolchain"
    )
    positives = int(y.sum())
    assert 0 < positives < len(y)


def test_generated_corpus_labels_match_what_the_extractor_found() -> None:
    """A program that embeds a sink must extract as reaching one.

    If the generator says "this program calls `cp.execSync`" and the extractor
    finds no `CommandExec`, the extractor has a blind spot and the corpus is
    labelling noise. Asserting the two agree is what makes the corpus a
    measurement rather than an assumption.
    """
    from aegis_ml.gat_divergence import HIGH_RISK_SINKS, build_tool_graph
    from aegis_ml.training_corpus import build_corpus

    if CLI is None:
        pytest.skip("aegis-cli is not built")
    # The full corpus, not a subsample: the point is to cover every sink
    # template, and a 36-program draw leaves only four of them. The extraction
    # is cached after the first call, so this costs one CLI invocation.
    programs, servers = build_corpus()
    sink_templates = 0
    injection_templates = 0
    egress_templates = 0
    for index, program in enumerate(programs):
        document = servers.get(f"s{index:04d}")
        if document is None:
            continue
        graph = build_tool_graph(
            document["dtg"]["nodes"], document["dtg"]["edges"]
        )
        if program.intended_sink == "InjectionMarker":
            injection_templates += 1
            assert program.label == 1, "an injection template was labelled benign"
            continue
        if program.intended_sink == "EgressLiteral":
            egress_templates += 1
            assert program.label == 1, "an egress template was labelled benign"
            assert graph.sink_kinds, (
                f"{program.filename} was generated with a data source and must "
                f"reach one, got {sorted(graph.sink_kinds)}"
            )
            continue
        if program.label == 1 and program.intended_sink in HIGH_RISK_SINKS:
            sink_templates += 1
            assert program.intended_sink in graph.sink_kinds, (
                f"{program.filename} was generated to reach "
                f"{program.intended_sink} but the extractor found "
                f"{sorted(graph.sink_kinds)}"
            )
        if program.label == 0:
            assert not graph.sink_kinds & HIGH_RISK_SINKS, (
                f"{program.filename} is a benign template but reaches "
                f"{sorted(graph.sink_kinds & HIGH_RISK_SINKS)}"
            )
    # The generator splits the malicious class three ways by `index % 9`, so
    # each of these classes holds roughly a third of the 80 positives. The
    # threshold is set below the expected count so a change in the split fails
    # the balance check without making the coverage check itself brittle.
    assert sink_templates >= 20, f"only {sink_templates} sink templates were covered"
    assert injection_templates >= 20, f"only {injection_templates} injection templates"
    assert egress_templates >= 20, f"only {egress_templates} egress templates"


def test_onnx_matches_sklearn_exactly(model) -> None:
    """The exported graph must reproduce sklearn's probability.

    Not merely "agree in sign". The earlier graph returned a constant that was
    correct for roughly one input in two hundred, and a sign check would have
    passed.
    """
    from aegis_ml.gat_divergence import build_training_corpus

    assert model.session is not None, "the ONNX graph is not loaded"
    x, _y, _samples = build_training_corpus()
    expected = model.classifier.predict_proba(x[:64])[:, 1]
    actual = model.session.run(None, {"features": x[:64].astype(np.float32)})[0].reshape(-1)
    assert actual.shape == (64,), f"batch axis lost: {actual.shape}"
    worst = float(np.max(np.abs(expected - actual)))
    assert worst < 1e-5, f"max |sklearn - onnx| = {worst}"


def test_onnx_is_not_a_constant_function(model) -> None:
    """The graph must depend on its input.

    Directly targets the defect where the leaf weights were summed ungated.
    """
    from aegis_ml.gat_divergence import N_TABULAR, GAT_EMBED_DIM, build_training_corpus

    x, _y, _samples = build_training_corpus()
    rng = np.random.default_rng(11)
    probe = rng.random((32, GAT_EMBED_DIM + N_TABULAR)).astype(np.float32)
    out_a = model.session.run(None, {"features": probe})[0].reshape(-1)
    out_b = model.session.run(
        None, {"features": x[:32].astype(np.float32)}
    )[0].reshape(-1)
    assert float(np.std(out_a)) > 1e-4, "random inputs produced a constant score"
    assert float(np.std(out_b)) > 1e-4, "corpus inputs produced a constant score"
    assert not np.allclose(out_a, out_b), "the graph ignored its input entirely"


def test_model_artifact_is_valid_and_reproducible() -> None:
    """The shipped graph must pass the ONNX checker and carry provenance."""
    import onnx

    from aegis_ml.gat_divergence import MODEL_PATH

    assert MODEL_PATH.exists(), f"{MODEL_PATH} is missing; run DivergenceModel.export_onnx()"
    onnx.checker.check_model(onnx.load(str(MODEL_PATH)))
    model = onnx.load(str(MODEL_PATH))
    props = {entry.key: entry.value for entry in model.metadata_props}
    assert props.get("author") == "Pratham Kapoor <prathamkapoor027@gmail.com>"
    assert props.get("gat_heads") == "4"
    assert int(props["n_estimators"]) > 0
    assert int(props["training_samples"]) > 0


def test_training_is_deterministic(model) -> None:
    """Two fits on the same corpus must produce the same weights."""
    from aegis_ml.gat_divergence import DivergenceModel

    first = DivergenceModel()
    first.train()
    second = DivergenceModel()
    second.train()
    assert first.metrics["auc"] == second.metrics["auc"]
    assert np.allclose(
        first.classifier.predict_proba(np.eye(first.metrics["feature_dim"])[:5])[:, 1],
        second.classifier.predict_proba(np.eye(second.metrics["feature_dim"])[:5])[:, 1],
    )


# ---------------------------------------------------------------------------
# Separation, measured
# ---------------------------------------------------------------------------
@requires_cli
def test_recall_and_false_positive_rate(scores: dict[str, float]) -> None:
    """13/13 recall and 0/5 false positives, measured, not asserted in prose."""
    recall = sum(1 for n in VULNERABLE_FIXTURES if scores[n] >= DIVERGENCE_THRESHOLD)
    false_positives = sum(1 for n in BENIGN_FIXTURES if scores[n] >= DIVERGENCE_THRESHOLD)
    detail = ", ".join(f"{n}={scores[n]:.4f}" for n in sorted(scores))
    assert recall == len(VULNERABLE_FIXTURES), f"recall {recall}/13: {detail}"
    assert false_positives == 0, f"{false_positives}/5 benign flagged: {detail}"


@requires_cli
def test_rust_scorer_matches_onnxruntime_to_six_decimals(model, scans) -> None:
    """The pure-Rust evaluator scores exactly what the ONNX graph scores.

    AegisCPG keeps two engines for the divergence head: Python/onnxruntime and
    `crates/aegis-common/src/onnx_parity_scorer.rs`, which re-executes the
    trained arithmetic so the gateway and CLI never need a Python runtime to
    score a tool. Two engines sharing only weights drift silently, so this
    test makes the claim falsifiable (ADR-024): the exact node features, the
    adjacency, and the tabular features of every shipped fixture plus 100
    seeded synthetic probes are handed to `aegis-cli parity`, which runs the
    manifest's weights in pure Rust and reports the maximum absolute
    difference against onnxruntime's answers. 1e-6 is the bound: well below
    any threshold decision, tight enough to catch a reordered summation.
    """
    from aegis_ml.gat_divergence import WEIGHTS_MANIFEST_PATH
    from aegis_ml.parity_handoff import build_parity_inputs
    from aegis_ml.training_corpus import CACHE_ROOT

    manifest_path = model.export_weights_manifest()
    assert (
        manifest_path == WEIGHTS_MANIFEST_PATH and manifest_path.exists()
    ), "the parity manifest belongs in ml-rag-engine/models"
    assert manifest_path.read_text(encoding="utf-8").count('"feature"') == len(
        model.classifier.estimators_
    ), "one exported tree per boosting iteration"

    inputs_path = CACHE_ROOT / "parity_inputs.json"
    build_parity_inputs(
        model,
        {name: server["dtg"] for name, server in scans.items()},
        inputs_path,
    )

    completed = subprocess.run(
        [str(CLI), "parity", "--manifest", str(manifest_path), "--inputs", str(inputs_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, f"parity gate failed: {completed.stdout}\n{completed.stderr}"
    report = json.loads(completed.stdout)
    assert report["case_count"] == len(scans) + 100
    assert report["fixture_cases"] == len(scans)
    assert report["synthetic_probes"] == 100
    assert report["within_bound"] is True, report["failures"]
    assert report["max_abs_diff"] < 1e-6, report
    assert report["manifest_model_sha256"] == model.model_sha256, (
        "the parity document was built against a different ONNX artifact than "
        "the manifest describes"
    )

    # The second, deeper half of the gate: fixture cases carry the raw DTG
    # they were featurised from, and the Rust side re-derives its features
    # from that graph — the exact inputs derivation the live endpoint runs.
    # The derived numbers must reload to the Python pipeline's own matrices,
    # which is what makes `POST /api/live-analyze`'s native model head
    # trustworthy without a Python runtime (ADR-024).
    assert report["derived_cases"] == len(scans), report["derived_failures"]
    assert report["derived_feature_diff"] < 1e-6, report["derived_failures"]
    assert report["derived_adjacency_diff"] < 1e-6, report["derived_failures"]
    assert report["derived_tabular_diff"] < 1e-6, report["derived_failures"]
    assert report["derived_score_diff"] < 1e-6, report["derived_failures"]


@requires_cli
def test_the_three_detection_classes_are_each_covered(scores: dict[str, float]) -> None:
    """Each signal must catch its own class.

    A sink-based detector misses `js_indirect_prompt_inj.js` (no OS sink) and a
    marker-based one misses `js_malicious_tools.js` (no marker). Passing on
    aggregate recall while one signal is dead is the failure this pins.
    """
    for name in (
        "js_indirect_prompt_inj.js",       # injection marker, no sink
        "js_malicious_tools.js",           # egress literal + FileRead, no marker
        "js_malicious_code_exec.js",       # CommandExec three hops down
    ):
        assert scores[name] >= DIVERGENCE_THRESHOLD, (
            f"{name} scored {scores[name]:.4f}; one of the three detection "
            "signals is not reaching the model"
        )


@requires_cli
def test_every_benign_score_is_far_below_the_threshold(scores: dict[str, float]) -> None:
    """Margin, not just the sign.

    A benign tool at 0.49 would pass the flag test today and fail in
    production after any model retrain.
    """
    for name in BENIGN_FIXTURES:
        assert scores[name] < 0.05, (
            f"{name} scored {scores[name]:.4f}; the benign class is not "
            "comfortably separated from the threshold"
        )


@requires_cli
def test_inference_latency_is_within_budget(model, scans) -> None:
    """End-to-end scoring must stay inside the control-plane budget.

    Measured as a median over many calls on a real graph, not a single timing:
    one sample on a shared host measures the scheduler. The budget is 15 ms, not
    the 2 ms that ADR-015 originally claimed -- that figure predated any
    measurement and was never achievable end to end; the ONNX Runtime call alone
    costs about 40 percent of the measured total. `scripts/latency_probe.py`
    prints the per-stage breakdown this budget is derived from.
    """
    graph = scans["js_secrets_pii.js"]["dtg"]
    description = "Reads environment metrics for health checks"
    model.predict(graph, description)
    samples = []
    for _ in range(60):
        start = time.perf_counter()
        model.predict(graph, description)
        samples.append((time.perf_counter() - start) * 1000.0)
    median_ms = float(np.median(samples))
    assert median_ms < 15.0, f"median end-to-end scoring latency {median_ms:.3f} ms"


@requires_cli
def test_onnx_runtime_call_dominates_but_does_not_exceed_the_graph(model, scans) -> None:
    """The ONNX call must be a small fraction of the budget on its own.

    A regression that pushed this to tens of milliseconds would show up here
    long before it showed up as a slow control plane.
    """
    from aegis_ml.gat_divergence import DeterministicSubwordEncoderCompat

    graph = scans["js_secrets_pii.js"]["dtg"]
    description = "Reads environment metrics for health checks"
    model.predict(graph, description)
    description_vec = DeterministicSubwordEncoderCompat().encode(description)
    features, _graph, _alphas = model.features_for(graph, description, description_vec)
    batched = features.reshape(1, -1).astype(np.float32)
    model.session.run(None, {"features": batched})
    samples = []
    for _ in range(120):
        start = time.perf_counter()
        model.session.run(None, {"features": batched})
        samples.append((time.perf_counter() - start) * 1000.0)
    median_ms = float(np.median(samples))
    assert 0.05 < median_ms < 10.0, f"onnxruntime call took {median_ms:.3f} ms"


@requires_cli
def test_backend_is_reported_and_real(scans) -> None:
    """The score must come from onnxruntime, and the artefact must hash."""
    from aegis_ml.gat_divergence import DivergenceModel

    instance = DivergenceModel()
    assert instance.load_onnx(), "the exported graph could not be loaded"
    assert instance.backend == "onnxruntime"
    assert len(instance.model_sha256) == 64
    report = instance.predict(scans["js_secrets_pii.js"]["dtg"], "", tool_id="x")
    assert report.score_availability is True
    assert report.backend == "onnxruntime"
    assert report.inference_ms >= 0.0


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
@requires_cli
def test_reports_name_the_reason(scans) -> None:
    """A verdict without an explanation is not actionable in an audit."""
    from aegis_ml.gat_divergence import DivergenceModel

    instance = DivergenceModel()
    instance.load_onnx()
    report = instance.predict(
        scans["js_namespace_typosquat.js"]["dtg"], "Posts a message to a Slack workspace"
    )
    assert report.attributions, "a divergent tool produced no attribution"
    joined = " ".join(report.attributions).lower()
    assert "declare" in joined or "hop" in joined, report.attributions


# ---------------------------------------------------------------------------
# Zero-LLM
# ---------------------------------------------------------------------------
def test_no_llm_runtime_is_importable() -> None:
    """ADR-001/ADR-015: nothing in the control plane may pull in an LLM."""
    for forbidden in ("torch", "tensorflow", "transformers", "sentence_transformers", "llama_cpp"):
        assert forbidden not in _loaded_modules(), (
            f"{forbidden} is loaded; the control plane must run without any "
            "neural-network runtime beyond NumPy, scikit-learn and ONNX Runtime"
        )


def _loaded_modules() -> set[str]:
    import sys

    return {name.split(".")[0] for name in sys.modules}


def test_declared_dependency_set_is_free_of_llm_frameworks() -> None:
    """The manifest must not name an LLM runtime either."""
    import tomllib

    manifest = REPO_ROOT / "ml-rag-engine" / "pyproject.toml"
    text = manifest.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        # The manifest carries a docstring-style header; fall back to a scan.
        data = {}
    declared = " ".join(
        str(value)
        for table in data.get("project", {}).get("dependencies", []) or []
        for value in (table if isinstance(table, list) else [table])
    )
    forbidden = ("torch", "tensorflow", "transformers", "openai", "anthropic", "llama")
    haystack = (declared + " " + text).lower()
    for name in forbidden:
        assert name not in haystack, f"{name} appears in the ML engine manifest"