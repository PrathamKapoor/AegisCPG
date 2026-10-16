//! Pure-Rust mirror of the trained AegisCPG divergence head.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Loads `ml-rag-engine/models/aegis_divergence_weights.json` — the exact
//! trained weights — and re-executes the model's arithmetic: the multi-head
//! GATv2 attention over the tool graph, the entry/sink pooled embedding, the
//! fused graph+tabular feature vector, and the gradient-boosted decision
//! function under the sigmoid.
//!
//! ## WHY
//! Two inference engines that only share *weights* drift: a different softmax
//! mask here, a different accumulation order there, and the drift surfaces as
//! a different verdict for the same tool exactly near the 0.5 operating point.
//! The Rust control plane must be able to score a tool with no Python runtime
//! in the loop, and it must be *demonstrably* the same score that the shipped
//! onnxruntime graph produces: the parity gate (`aegis-cli parity`, ADR-024)
//! runs both engines over identical case inputs and requires
//! `max_abs_diff < 1e-6`.
//!
//! ## HOW
//! The numeric contracts are copied, not approximated:
//!
//! * the attention layer runs in `f64`, exactly as the NumPy reference does,
//!   and casts the final embedding to `f32` — the same cast the Python path
//!   performs before concatenating the tabular features;
//! * the tree branch runs in `f32` against the float32-rounded thresholds the
//!   manifest exports, matching the ONNX graph's initializer precision, so
//!   every leaf decision is identical rather than merely in expectation;
//! * the export's leaf multiplier is re-applied (`value * learning_rate`,
//!   then cast to `f32`), and the log-odds are summed left to right in the
//!   export's order: base, tree 0, tree 1, … — the binary `Add` chain the
//!   exported graph emits.
//!
//! Zero Python dependencies, zero network calls: the evaluator is a pure
//! function of the manifest and the case inputs (ADR-001).

use crate::divergence_features;
use crate::types::CompressedDtg;
use serde::Deserialize;

/// One exported decision tree from the gradient-boosting ensemble.
///
/// Arrays are node-indexed; `children_left[node] == children_right[node] ==
/// -1` marks a leaf, whose stored `value` is the raw leaf estimate the export
/// multiplies by the learning rate.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct DecisionTree {
    #[serde(rename = "feature")]
    pub features: Vec<i64>,
    pub threshold: Vec<f64>,
    #[serde(rename = "children_left")]
    pub children_left: Vec<i64>,
    #[serde(rename = "children_right")]
    pub children_right: Vec<i64>,
    pub value: Vec<f64>,
}

/// The exported trained model, exactly as `DivergenceModel.weights_manifest()`
/// writes it.
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct WeightsManifest {
    #[serde(rename = "schema")]
    pub schema: String,
    #[serde(rename = "model_sha256", default)]
    pub model_sha256: String,
    #[serde(rename = "n_heads")]
    pub heads: usize,
    #[serde(rename = "hidden_dim")]
    pub hidden_dim: usize,
    #[serde(rename = "out_dim")]
    pub out_dim: usize,
    #[serde(rename = "in_dim")]
    pub in_dim: usize,
    #[serde(rename = "alpha_slope")]
    pub alpha_slope: f64,
    #[serde(rename = "W")]
    pub w: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "a_src")]
    pub a_src: Vec<Vec<f64>>,
    #[serde(rename = "a_dst")]
    pub a_dst: Vec<Vec<f64>>,
    #[serde(rename = "W_out")]
    pub w_out: Vec<Vec<f64>>,
    #[serde(rename = "learning_rate")]
    pub learning_rate: f64,
    #[serde(rename = "base_log_odds")]
    pub base_log_odds: f64,
    #[serde(rename = "n_estimators")]
    pub n_estimators: usize,
    #[serde(rename = "trees")]
    pub trees: Vec<DecisionTree>,
}

/// One parity case: the graph the GAT sees, the tabular features, and the
/// score onnxruntime reports for the fusion.
///
/// `dtg` is optional: fixture cases carry the *raw compressed graph* they were
/// featurised from, so the gate can also re-derive its features natively and
/// compare against the matrix the Python pipeline computed. Synthetic probes
/// are raw matrices by construction and carry none.
#[derive(Debug, Clone, Deserialize)]
pub struct ParityCase {
    #[serde(default)]
    pub kind: String,
    pub case_id: String,
    #[serde(rename = "node_features")]
    pub node_features: Vec<Vec<f64>>,
    pub adjacency: Vec<Vec<f64>>,
    pub tabular: Vec<f64>,
    #[serde(rename = "expected_score")]
    pub expected_score: f64,
    /// The description the case was featurised with (empty for the fixture
    /// protocol, which scores graph-only inputs).
    #[serde(default)]
    pub description: String,
    /// The raw graph, when the case has one to re-derive from.
    #[serde(default)]
    pub dtg: Option<CompressedDtg>,
}

/// The parity case document `aegis_ml/parity_handoff.py` writes.
#[derive(Debug, Clone, Deserialize)]
pub struct ParityInputs {
    #[serde(rename = "schema")]
    pub schema: String,
    #[serde(default)]
    pub model_sha256: String,
    #[serde(default)]
    pub cases: Vec<ParityCase>,
}

/// One derived-features drift: everything the native featurization got wrong
/// for one case, so a fix starts from the exact numbers.
#[derive(Debug, Clone)]
pub struct DerivedFailure {
    pub case_id: String,
    pub feature_diff: f64,
    /// `(row, col)` of the largest feature drift.
    pub feature_slot: (usize, usize),
    pub adjacency_diff: f64,
    pub tabular_diff: f64,
    /// Slot of the largest tabular drift.
    pub tabular_slot: usize,
    pub score_diff: f64,
}

/// The measurement the parity gate records.
#[derive(Debug, Clone)]
pub struct ParityReport {
    pub case_count: usize,
    pub fixture_cases: usize,
    pub synthetic_probes: usize,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub worst_case: String,
    /// `(case_id, rust_score, expected_score, abs_diff)` per failing case,
    /// capped so a report stays a report and not a data dump.
    pub failures: Vec<(String, f64, f64, f64)>,
    /// How many cases carried a raw DTG and were re-featurised natively.
    pub derived_cases: usize,
    /// Largest |rust-derived − python-stored| over the `(N, 21)` features.
    pub derived_feature_diff: f64,
    /// Where that largest feature drift lives.
    pub derived_feature_slot: (usize, usize),
    /// Largest |rust-derived − python-stored| over the adjacency.
    pub derived_adjacency_diff: f64,
    /// Largest |rust-derived − python-stored| over the nine tabular features.
    pub derived_tabular_diff: f64,
    /// Where that largest tabular drift lives.
    pub derived_tabular_slot: usize,
    /// Largest |native score − onnxruntime score| over the derived cases.
    pub derived_score_diff: f64,
    /// The case owning the largest combined derived drift.
    pub derived_worst_case: String,
    /// Cases with any component at or past the bound, capped at eight.
    pub derived_failures: Vec<DerivedFailure>,
}

/// The largest `|a − b|` over two matrices of the same shape, with the
/// `(row, col)` of the largest drift; a shape mismatch is an error, not a
/// drift, because it names the derivation problem rather than a magnitude.
fn matrix_diff(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    case_id: &str,
    what: &str,
) -> Result<(f64, usize, usize), String> {
    if a.len() != b.len() {
        return Err(format!(
            "case {}: derived {what} has {} rows, stored has {}",
            case_id,
            a.len(),
            b.len()
        ));
    }
    let mut worst = (0.0f64, 0usize, 0usize);
    for (row, (arow, brow)) in a.iter().zip(b.iter()).enumerate() {
        if arow.len() != brow.len() {
            return Err(format!(
                "case {}: derived {what} row {row} is {} wide, stored is {}",
                case_id,
                arow.len(),
                brow.len()
            ));
        }
        for (col, (av, bv)) in arow.iter().zip(brow.iter()).enumerate() {
            let d = (av - bv).abs();
            if d > worst.0 {
                worst = (d, row, col);
            }
        }
    }
    Ok(worst)
}

impl WeightsManifest {
    /// Parses and shape-checks a manifest from its JSON bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|e| format!("invalid weights manifest: {e}"))?;
        if manifest.w.len() != manifest.heads
            || manifest.a_src.len() != manifest.heads
            || manifest.a_dst.len() != manifest.heads
        {
            return Err(format!(
                "manifest declares {} heads but W/a_src/a_dst hold {} / {} / {}",
                manifest.heads,
                manifest.w.len(),
                manifest.a_src.len(),
                manifest.a_dst.len()
            ));
        }
        for (index, w) in manifest.w.iter().enumerate() {
            if w.len() != manifest.in_dim
                || w.iter().any(|row| row.len() != manifest.hidden_dim)
            {
                return Err(format!(
                    "head {index} W is not ({}, {})",
                    manifest.in_dim, manifest.hidden_dim
                ));
            }
        }
        if manifest.w_out.len() != manifest.heads * manifest.hidden_dim
            || manifest
                .w_out
                .iter()
                .any(|row| row.len() != manifest.out_dim)
        {
            return Err("W_out must be (heads*hidden_dim, out_dim)".to_string());
        }
        if manifest.trees.len() != manifest.n_estimators {
            return Err(format!(
                "manifest declares {} estimators but exports {} trees",
                manifest.n_estimators,
                manifest.trees.len()
            ));
        }
        Ok(manifest)
    }

    /// Validates one case's shape against the manifest's declared dimensions.
    pub fn validate_case(&self, case: &ParityCase) -> Result<(), String> {
        let rows = case.node_features.len();
        if rows == 0 {
            return Err(format!("case {} has no nodes", case.case_id));
        }
        if case.node_features.iter().any(|r| r.len() != self.in_dim) {
            return Err(format!(
                "case {} rows are not {} wide (the manifest's in_dim)",
                case.case_id, self.in_dim
            ));
        }
        if case.adjacency.len() != rows || case.adjacency.iter().any(|r| r.len() != rows) {
            return Err(format!("case {} adjacency is not {rows}x{rows}", case.case_id));
        }
        if case.tabular.len() != 9 {
            return Err(format!(
                "case {} carries {} tabular features, expected 9",
                case.case_id,
                case.tabular.len()
            ));
        }
        Ok(())
    }
}

/// The `LeakyReLU` the attention scores pass through, with the layer's slope.
fn leaky_relu(value: f64, slope: f64) -> f64 {
    if value >= 0.0 {
        value
    } else {
        slope * value
    }
}

/// `ELU`, defined through `expm1` for the negative arm exactly like `np.expm1`.
fn elu(value: f64) -> f64 {
    if value > 0.0 {
        value
    } else {
        value.exp_m1()
    }
}

/// Runs the multi-head GATv2 layer and returns the pooled embedding.
///
/// Per head `k`: `z = h @ W[k]`; attention logits
/// `e = LeakyReLU(z_i a_src + z_j a_dst)` masked with `-1e9` where
/// `A_ij <= 0`, then softmaxed row-wise with the max-subtraction the NumPy
/// reference uses (which keeps an isolated node's row finite); messages
/// `(alphas @ h) @ W[k]` passed through `ELU`; concatenated across heads; and
/// projected by `W_out`. The embedding is the entry node's first half joined
/// with the node-wise column maximum's second half, cast to `f32`.
pub fn gat_embedding(
    manifest: &WeightsManifest,
    node_features: &[Vec<f64>],
    adjacency: &[Vec<f64>],
) -> Vec<f32> {
    let heads = manifest.heads;
    let hidden = manifest.hidden_dim;
    let in_dim = manifest.in_dim;
    let out_dim = manifest.out_dim;
    let n = node_features.len();
    let half = out_dim / 2;

    // concatenated[i][k*hidden + c] holds head k's ELU output for node i, the
    // exact layout `np.concatenate(outputs, axis=1)` produces.
    let mut concatenated = vec![vec![0f64; heads * hidden]; n];
    for k in 0..heads {
        let w = &manifest.w[k];
        // z = h @ W[k]: (n, hidden), accumulated over the input axis in order.
        let mut z = vec![vec![0f64; hidden]; n];
        for i in 0..n {
            for c in 0..hidden {
                let mut acc = 0.0;
                for d in 0..in_dim {
                    acc += node_features[i][d] * w[d][c];
                }
                z[i][c] = acc;
            }
        }
        // e_src/z a_src[k] and e_dst = z @ a_dst[k]: (n,) each.
        let e_src = z
            .iter()
            .map(|row| row.iter().zip(&manifest.a_src[k]).map(|(z, a)| z * a).sum())
            .collect::<Vec<f64>>();
        let e_dst = z
            .iter()
            .map(|row| row.iter().zip(&manifest.a_dst[k]).map(|(z, a)| z * a).sum())
            .collect::<Vec<f64>>();

        for i in 0..n {
            // Logits, in row order, masked after the LeakyReLU exactly as the
            // reference does (`e = where(a > 0, e, -1e9)`).
            let mut logits = vec![0f64; n];
            for j in 0..n {
                let e = leaky_relu(e_src[i] + e_dst[j], manifest.alpha_slope);
                logits[j] = if adjacency[i][j] > 0.0 { e } else { -1e9 };
            }
            // Softmax with max subtraction; the guard keeps an all-masked row
            // finite (NumPy guards the same zero denominator to 1.0).
            let row_max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0;
            let exps = logits
                .iter()
                .map(|l| {
                    let x = (l - row_max).exp();
                    denom += x;
                    x
                })
                .collect::<Vec<f64>>();
            let denom = if denom == 0.0 { 1.0 } else { denom };

            // messages_i = (alphas_row @ h) — pooled over the input axis —
            // then @ W[k], then ELU. Two separate steps, in that order.
            let mut pooled = vec![0f64; in_dim];
            for j in 0..n {
                let alpha = exps[j] / denom * if adjacency[i][j] > 0.0 { 1.0 } else { 0.0 };
                if alpha != 0.0 {
                    for d in 0..in_dim {
                        pooled[d] += alpha * node_features[j][d];
                    }
                }
            }
            for c in 0..hidden {
                let mut projected = 0.0;
                for d in 0..in_dim {
                    projected += pooled[d] * w[d][c];
                }
                concatenated[i][k * hidden + c] = elu(projected);
            }
        }
    }

    // projected = concatenated @ W_out: (n, out_dim).
    let mut projected = vec![0f64; n * out_dim];
    for i in 0..n {
        for o in 0..out_dim {
            let mut acc = 0.0;
            for idx in 0..heads * hidden {
                acc += concatenated[i][idx] * manifest.w_out[idx][o];
            }
            projected[i * out_dim + o] = acc;
        }
    }

    // embedding = [projected[0][:half], max_over_nodes(half:)], cast to f32.
    let mut col_max = vec![f64::NEG_INFINITY; out_dim];
    for i in 0..n {
        for o in 0..out_dim {
            col_max[o] = col_max[o].max(projected[i * out_dim + o]);
        }
    }
    let mut embedding = Vec::with_capacity(out_dim);
    for o in 0..half {
        embedding.push(projected[o] as f32);
    }
    for o in half..out_dim {
        embedding.push(col_max[o] as f32);
    }
    embedding
}

/// Walks one tree to its reached leaf with the ONNX graph's own comparison
/// rule: `x <= threshold` goes left (`LessOrEqual`), `x > threshold` goes
/// right (`Greater`). Thresholds and features are both `f32`, matching the
/// graph's initializer and operand precision.
fn reached_leaf(tree: &DecisionTree, features: &[f32]) -> f64 {
    let mut node = 0usize;
    while tree.children_left[node] != -1 {
        let feature = tree.features[node];
        let threshold = tree.threshold[node] as f32;
        let left = features[feature as usize] <= threshold;
        node = if left {
            tree.children_left[node] as usize
        } else {
            tree.children_right[node] as usize
        };
    }
    tree.value[node]
}

/// The tree ensemble's log-odds for one fused feature vector, in `f32` and in
/// the export's summation order: base first, then one contribution per tree.
pub fn gbdt_logit(manifest: &WeightsManifest, features: &[f32]) -> f32 {
    let mut combined = manifest.base_log_odds as f32;
    for tree in &manifest.trees {
        let value = reached_leaf(tree, features);
        // The ONNX export stores `leaf_value * learning_rate` as a float32
        // initializer; reproduce the multiplication in f64 and cast once.
        combined += (value * manifest.learning_rate) as f32;
    }
    combined
}

/// The final probability: `sigmoid(log-odds)`, in `f32` like the graph's
/// terminating `Sigmoid` node.
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Scores one case the way the Rust control plane does: GAT embedding from
/// the graph, fused with the tabular features, then the tree head.
pub fn score_case(manifest: &WeightsManifest, case: &ParityCase) -> Result<f64, String> {
    manifest.validate_case(case)?;
    let mut fused = gat_embedding(manifest, &case.node_features, &case.adjacency);
    fused.extend(case.tabular.iter().map(|x| *x as f32));
    Ok(sigmoid(gbdt_logit(manifest, &fused)) as f64)
}

/// Runs the whole parity measurement and rounds it into a [`ParityReport`].
pub fn evaluate_parity(
    manifest: &WeightsManifest,
    inputs: &ParityInputs,
) -> Result<ParityReport, String> {
    if inputs.cases.is_empty() {
        return Err("the parity document holds no cases".to_string());
    }
    let mut fixtures = 0usize;
    let mut synthetic = 0usize;
    let mut worst = (0.0f64, String::new());
    let mut total = 0.0f64;
    let mut failures = Vec::new();

    // Derived-feature bookkeeping: the case-local worst sets the *interesting*
    // case; the report-level paths take the max over cases.
    let mut derived_cases = 0usize;
    let mut derived_feature_diff = 0.0f64;
    let mut derived_feature_slot = (0usize, 0usize);
    let mut derived_adjacency_diff = 0.0f64;
    let mut derived_tabular_diff = 0.0f64;
    let mut derived_tabular_slot = 0usize;
    let mut derived_score_diff = 0.0f64;
    let mut derived_worst_case = String::new();
    let mut derived_worst_key = 0.0f64;
    let mut derived_failures: Vec<DerivedFailure> = Vec::new();

    for case in &inputs.cases {
        let rust = score_case(manifest, case)?;
        let diff = (rust - case.expected_score).abs();
        total += diff;
        if diff > worst.0 {
            worst = (diff, case.case_id.clone());
        }
        if diff >= 1e-6 {
            failures.push((case.case_id.clone(), rust, case.expected_score, diff));
        }
        if case.kind == "fixture" {
            fixtures += 1;
        } else {
            synthetic += 1;
        }

        // The second, deeper measurement: where the case carries the raw DTG
        // it was featurised from, re-derive every input natively and compare
        // against the matrix the Python pipeline stored. A drift here is a
        // featurization bug, catching it is what makes the live path's model
        // score trustworthy (ADR-024).
        if let Some(dtg) = &case.dtg {
            derived_cases += 1;
            let entropy = divergence_features::max_sink_entropy(&dtg.detected_sinks);
            let graph = divergence_features::build_tool_graph(dtg, entropy);
            let description_vec = divergence_features::encode(&case.description);
            let tabular = divergence_features::tabular_features(
                dtg,
                &case.description,
                &graph,
                &description_vec,
                entropy,
            );
            let (feature_diff, feature_row, feature_col) =
                matrix_diff(&graph.features, &case.node_features, &case.case_id, "features")?;
            let (adjacency_diff, _, _) =
                matrix_diff(&graph.adjacency, &case.adjacency, &case.case_id, "adjacency")?;
            if tabular.len() != case.tabular.len() {
                return Err(format!(
                    "case {}: derived tabular is {} wide, stored is {}",
                    case.case_id,
                    tabular.len(),
                    case.tabular.len()
                ));
            }
            let (tabular_diff, tabular_slot) = tabular
                .iter()
                .zip(case.tabular.iter())
                .enumerate()
                .fold((0.0f64, 0usize), |acc, (slot, (a, b))| {
                    let d = (f64::from(*a) - b).abs();
                    if d > acc.0 {
                        (d, slot)
                    } else {
                        acc
                    }
                });
            let mut fused = gat_embedding(manifest, &graph.features, &graph.adjacency);
            fused.extend_from_slice(&tabular);
            let derived_score = sigmoid(gbdt_logit(manifest, &fused)) as f64;
            let score_diff = (derived_score - case.expected_score).abs();

            derived_feature_diff = derived_feature_diff.max(feature_diff);
            if feature_diff == derived_feature_diff {
                derived_feature_slot = (feature_row, feature_col);
            }
            derived_adjacency_diff = derived_adjacency_diff.max(adjacency_diff);
            derived_tabular_diff = derived_tabular_diff.max(tabular_diff);
            if tabular_diff == derived_tabular_diff {
                derived_tabular_slot = tabular_slot;
            }
            derived_score_diff = derived_score_diff.max(score_diff);
            let key = feature_diff.max(adjacency_diff).max(tabular_diff).max(score_diff);
            if key > derived_worst_key {
                derived_worst_key = key;
                derived_worst_case = case.case_id.clone();
            }
            if feature_diff >= 1e-6
                || adjacency_diff >= 1e-6
                || tabular_diff >= 1e-6
                || score_diff >= 1e-6
            {
                derived_failures.push(DerivedFailure {
                    case_id: case.case_id.clone(),
                    feature_diff,
                    feature_slot: (feature_row, feature_col),
                    adjacency_diff,
                    tabular_diff,
                    tabular_slot,
                    score_diff,
                });
            }
        }
    }
    // A report stays a report: the first eight drifts name their case and
    // carry both scores, and anything beyond that is what `max_abs_diff`
    // already summarises.
    failures.truncate(8);
    derived_failures.truncate(8);
    Ok(ParityReport {
        case_count: inputs.cases.len(),
        fixture_cases: fixtures,
        synthetic_probes: synthetic,
        max_abs_diff: worst.0,
        mean_abs_diff: total / inputs.cases.len() as f64,
        worst_case: worst.1,
        failures,
        derived_cases,
        derived_feature_diff,
        derived_feature_slot,
        derived_adjacency_diff,
        derived_tabular_diff,
        derived_tabular_slot,
        derived_score_diff,
        derived_worst_case,
        derived_failures,
    })
}

/// Parses both documents and runs the measurement — the entry point the CLI
/// parity subcommand calls.
pub fn run_parity_check(
    manifest_bytes: &[u8],
    inputs_bytes: &[u8],
) -> Result<(WeightsManifest, ParityReport), String> {
    let manifest = WeightsManifest::parse(manifest_bytes)?;
    let inputs: ParityInputs = serde_json::from_slice(inputs_bytes)
        .map_err(|e| format!("invalid parity case document: {e}"))?;
    let report = evaluate_parity(&manifest, &inputs)?;
    Ok((manifest, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal valid manifest: 1 head, hidden 2, in 3, out 4, and a
    /// one-tree GBDT with two leaves split on feature 0.
    fn tiny_manifest() -> WeightsManifest {
        WeightsManifest {
            schema: "test".into(),
            model_sha256: String::new(),
            heads: 1,
            hidden_dim: 2,
            out_dim: 4,
            in_dim: 3,
            alpha_slope: 0.2,
            w: vec![vec![vec![0.5, -0.25], vec![1.0, 2.0], vec![-1.5, 0.75]]],
            a_src: vec![vec![0.25, -0.5]],
            a_dst: vec![vec![-0.125, 0.875]],
            w_out: vec![vec![1.0, 2.0, 3.0, 4.0], vec![-1.0, 0.5, -0.25, 0.125]],
            learning_rate: 0.12,
            base_log_odds: 0.0,
            n_estimators: 1,
            trees: vec![DecisionTree {
                features: vec![0, -1, -1],
                threshold: vec![0.5, 0.0, 0.0],
                children_left: vec![1, -1, -1],
                children_right: vec![2, -1, -1],
                value: vec![0.0, 1.0, 2.0],
            }],
        }
    }

    #[test]
    fn manifest_shape_checks_catch_a_head_count_mismatch() {
        let mut manifest = tiny_manifest();
        manifest.heads = 3;
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let error = WeightsManifest::parse(&bytes).unwrap_err();
        assert!(error.contains("heads"), "{error}");
    }

    /// A manifest wide enough to score a real featurized tool graph
    /// (`in_dim == 21`), for the derived-features tests.
    fn derived_manifest() -> WeightsManifest {
        WeightsManifest {
            schema: "test".into(),
            model_sha256: String::new(),
            heads: 1,
            hidden_dim: 2,
            out_dim: 4,
            in_dim: 21,
            alpha_slope: 0.2,
            w: vec![(0..21)
                .map(|i| vec![0.5 - 0.02 * i as f64, -0.25 + 0.03 * i as f64])
                .collect()],
            a_src: vec![vec![0.25, -0.5]],
            a_dst: vec![vec![-0.125, 0.875]],
            w_out: vec![vec![1.0, 2.0, 3.0, 4.0], vec![-1.0, 0.5, -0.25, 0.125]],
            learning_rate: 0.12,
            base_log_odds: 0.0,
            n_estimators: 1,
            trees: vec![DecisionTree {
                features: vec![4, -1, -1],
                threshold: vec![0.5, 0.0, 0.0],
                children_left: vec![1, -1, -1],
                children_right: vec![2, -1, -1],
                value: vec![0.0, 1.0, 2.0],
            }],
        }
    }

    /// One benign input node, the smallest derivable graph.
    fn tiny_dtg() -> CompressedDtg {
        use crate::lattice::SecurityLattice;
        use crate::types::{DtgEdge, DtgNode, DtgNodeType};
        let node = DtgNode {
            id: "p".into(),
            label: "in".into(),
            node_type: DtgNodeType::ToolInputParam,
            lattice: SecurityLattice::Public,
            file_path: "x.py".into(),
            line: 1,
        };
        CompressedDtg {
            server_id: "t".into(),
            language: "python".into(),
            raw_cpg_node_count: 1,
            compressed_node_count: 1,
            nodes: vec![node],
            edges: Vec::<DtgEdge>::new(),
            detected_sinks: Vec::new(),
            dependencies: Vec::new(),
            injection_markers: Vec::new(),
            egress_literals: Vec::new(),
            dtg_sha256: "0".repeat(64),
        }
    }

    /// The derived path over an exactly self-consistent case: re-deriving the
    /// features natively must reproduce both stored matrices and the stored
    /// score bit-for-`1e-6`. This is the plumbing regression guard: a nonzero
    /// diff here means the derivation loop broke, long before the fixture
    /// gate would notice.
    #[test]
    fn derived_features_reproduce_a_self_consistent_case() {
        use crate::divergence_features::{build_tool_graph, encode, max_sink_entropy, tabular_features};

        let manifest = derived_manifest();
        let dtg = tiny_dtg();
        let entropy = max_sink_entropy(&dtg.detected_sinks);
        let graph = build_tool_graph(&dtg, entropy);
        let tabular = tabular_features(&dtg, "", &graph, &encode(""), entropy);
        let mut fused = gat_embedding(&manifest, &graph.features, &graph.adjacency);
        fused.extend_from_slice(&tabular);
        let expected = sigmoid(gbdt_logit(&manifest, &fused)) as f64;

        let case = ParityCase {
            kind: "fixture".into(),
            case_id: "derived-ok".into(),
            node_features: graph.features.clone(),
            adjacency: graph.adjacency.clone(),
            tabular: tabular.iter().map(|v| f64::from(*v)).collect(),
            expected_score: expected,
            description: String::new(),
            dtg: Some(dtg),
        };
        let inputs = ParityInputs {
            schema: "t".into(),
            model_sha256: String::new(),
            cases: vec![case],
        };
        let report = evaluate_parity(&manifest, &inputs).unwrap();
        assert_eq!(report.derived_cases, 1);
        assert_eq!(report.derived_feature_diff, 0.0);
        assert_eq!(report.derived_adjacency_diff, 0.0);
        assert_eq!(report.derived_tabular_diff, 0.0);
        assert_eq!(report.derived_score_diff, 0.0);
        assert!(report.derived_failures.is_empty(), "{:?}", report.derived_failures);
    }

    /// A case whose stored features drift from what the DTG re-derives is
    /// named with every component of its drift, and the failure carries the
    /// case id — a featurization bug surfaces as data, not a lost assertion.
    #[test]
    fn a_derived_drift_is_reported_per_component() {
        let manifest = derived_manifest();
        // Stored: everything 0.9 — the derivation of the one-node graph puts
        // 1.0 into the type / lattice / sink one-hots instead, so the
        // feature diff is 0.1 and the tabular cosine is far off too.
        let tabular = vec![0.0; 9];
        let mut fused = gat_embedding(&manifest, &vec![vec![0.9; 21]], &vec![vec![1.0]]);
        fused.extend(tabular.iter().map(|v| *v as f32));
        let expected = sigmoid(gbdt_logit(&manifest, &fused)) as f64;

        let case = ParityCase {
            kind: "fixture".into(),
            case_id: "derived-drift".into(),
            node_features: vec![vec![0.9; 21]],
            adjacency: vec![vec![1.0]],
            tabular,
            expected_score: expected,
            description: String::new(),
            dtg: Some(tiny_dtg()),
        };
        let inputs = ParityInputs {
            schema: "t".into(),
            model_sha256: String::new(),
            cases: vec![case],
        };
        let report = evaluate_parity(&manifest, &inputs).unwrap();
        assert_eq!(report.derived_cases, 1);
        // The widest gap is the entropy slot: derived 0.0 against stored 0.9
        // (the one-hot and tabular slots sit at |1.0 - 0.9| = 0.099…8).
        assert!((report.derived_feature_diff - 0.9).abs() < 1e-12);
        assert!(report.derived_tabular_diff > 0.5);
        assert!(report.derived_failures.len() == 1);
        let failure = &report.derived_failures[0];
        assert_eq!(failure.case_id, "derived-drift");
        assert!(failure.score_diff >= 1e-6 || failure.feature_diff >= 1e-6);
        assert_eq!(report.derived_worst_case, "derived-drift");
    }

    #[test]
    fn a_single_node_graph_makes_the_embedding_exactly_the_projection() {
        let manifest = tiny_manifest();
        let h = vec![vec![1.0, 0.0, 2.0]];
        let adjacency = vec![vec![1.0]];

        // Hand-derived, deliberately independent of `gat_embedding`'s loops:
        // for a one-node graph the attention row has exactly one unmasked
        // entry (the self-loop), so its softmax weight is exactly 1.0 no
        // matter what the logit is, the message is `h` itself, and the
        // column maximum is the only row. The embedding is therefore the
        // entry's own projection, both halves of it.
        let w = &manifest.w[0];
        let head_out: Vec<f64> = (0..2)
            .map(|c| {
                elu(h[0][0] * w[0][c] + h[0][1] * w[1][c] + h[0][2] * w[2][c])
            })
            .collect();
        let projected: Vec<f64> = (0..4)
            .map(|o| head_out[0] * manifest.w_out[0][o] + head_out[1] * manifest.w_out[1][o])
            .collect();

        let embedding = gat_embedding(&manifest, &h, &adjacency);
        for (index, value) in projected.iter().enumerate() {
            // The embedding is the projection cast to f32 — bit for bit.
            assert_eq!(embedding[index], *value as f32, "dim {index}");
        }
    }

    #[test]
    fn isolated_rows_stay_finite_because_the_mask_beats_the_softmax() {
        let manifest = tiny_manifest();
        let n = 4;
        let h: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..3).map(|d| ((i + 1) * (d + 2)) as f64 / 5.0).collect())
            .collect();
        let mut adjacency = vec![vec![1.0f64; n]; n];
        // Drop every row beyond the diagonal: node 0 keeps only its self-loop,
        // node 3 keeps no edges at all. Zeroing a fully-masked row would
        // divide by zero and NaN out; masking with -1e9 must not.
        for row in adjacency.iter_mut() {
            for col in row.iter_mut() {
                *col = 0.0;
            }
        }
        for i in 0..n {
            adjacency[i][i] = 1.0;
        }

        let embedding = gat_embedding(&manifest, &h, &adjacency);
        assert!(
            embedding.iter().all(|x| x.is_finite()),
            "a masked softmax must never produce NaN"
        );
    }

    #[test]
    fn gbdt_traversal_matches_the_exported_comparison_rule() {
        let manifest = tiny_manifest();
        let features = vec![0.3f32, 0.0, 0.0];
        // 0.3 <= 0.5 goes left: base 0.0 + 0.12 * 1.0.
        assert_eq!(gbdt_logit(&manifest, &features), 0.12);
        let features = vec![0.9f32, 0.0, 0.0];
        // 0.9 > 0.5 goes right: base 0.0 + 0.12 * 2.0.
        assert_eq!(gbdt_logit(&manifest, &features), 0.24);
        assert_eq!(
            sigmoid(0.12),
            1.0 / (1.0 + (-0.12f32).exp()),
            "sigmoid is 1 / (1 + e^-x) in f32"
        );
    }

    #[test]
    fn the_fused_vector_is_embedding_first_tabular_second() {
        let mut manifest = tiny_manifest();
        // Retarget the root split to fused index 4, which is exactly the
        // first tabular slot (out_dim = 4): if the fusion ever re-ordered
        // embedding and tabular, this tree would read an embedding value and
        // the branch below would flip.
        manifest.trees[0].features[0] = 4;
        let graph = vec![vec![1.0, 0.0, 2.0]];
        let adjacency = vec![vec![1.0]];
        let low = ParityCase {
            kind: "synthetic".into(),
            case_id: "low".into(),
            node_features: graph.clone(),
            adjacency: adjacency.clone(),
            tabular: vec![0.1; 9],
            expected_score: 0.0,
            description: String::new(),
            dtg: None,
        };
        let high = ParityCase {
            kind: "synthetic".into(),
            case_id: "high".into(),
            node_features: graph,
            adjacency,
            tabular: vec![0.9; 9],
            expected_score: 0.0,
            description: String::new(),
            dtg: None,
        };
        // 0.1 <= 0.5 -> leaf 1 -> logit 0.12; 0.9 > 0.5 -> leaf 2 -> 0.24.
        assert_eq!(score_case(&manifest, &low).unwrap(), sigmoid(0.12) as f64);
        assert_eq!(score_case(&manifest, &high).unwrap(), sigmoid(0.24) as f64);
    }

    #[test]
    fn evaluate_parity_requires_cases_and_measures_the_exact_score() {
        let manifest = tiny_manifest();
        let empty = ParityInputs {
            schema: "t".into(),
            model_sha256: String::new(),
            cases: Vec::new(),
        };
        assert!(evaluate_parity(&manifest, &empty).is_err());

        // A case whose expected value is the Rust engine's own score must
        // measure as a perfect match.
        let mut case = ParityCase {
            kind: "fixture".into(),
            case_id: "self".into(),
            node_features: vec![vec![1.0, 0.0, 2.0]],
            adjacency: vec![vec![1.0]],
            tabular: vec![0.25; 9],
            expected_score: 0.0,
            description: String::new(),
            dtg: None,
        };
        case.expected_score = score_case(&manifest, &case).unwrap();
        let inputs = ParityInputs {
            schema: "t".into(),
            model_sha256: String::new(),
            cases: vec![case],
        };
        let report = evaluate_parity(&manifest, &inputs).unwrap();
        assert_eq!(report.case_count, 1);
        assert_eq!(report.fixture_cases, 1);
        assert_eq!(report.synthetic_probes, 0);
        assert_eq!(report.max_abs_diff, 0.0);
        assert!(report.failures.is_empty(), "{:?}", report.failures);

        // And a drift past the bound is reported with both scores, not
        // silently swallowed.
        let mut drifting = ParityCase {
            kind: "synthetic".into(),
            case_id: "off".into(),
            node_features: vec![vec![1.0, 0.0, 2.0]],
            adjacency: vec![vec![1.0]],
            tabular: vec![0.25; 9],
            expected_score: 0.0,
            description: String::new(),
            dtg: None,
        };
        let rust = score_case(&manifest, &drifting).unwrap();
        drifting.expected_score = rust - 1e-4;
        let drifted = ParityInputs {
            schema: "t".into(),
            model_sha256: String::new(),
            cases: vec![drifting],
        };
        let report = evaluate_parity(&manifest, &drifted).unwrap();
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.worst_case, "off");
        assert!((report.max_abs_diff - 1e-4).abs() < 1e-12);
    }
}
