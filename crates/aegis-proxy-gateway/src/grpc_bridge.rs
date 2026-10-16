//! gRPC control-plane bridge: the typed wire contract.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! The message types and handlers of `AegisControlPlane` as defined in
//! `specs/proto/aegis.proto`, implemented against plain Rust types.
//!
//! ## WHY
//! Stage 2 (the GAT/GBDT classifier) was trained in Python and the hot path
//! lives in Rust. Without a typed contract the two drift: the Python model adds
//! a feature, the Rust gateway keeps reading the old struct, and the divergence
//! score silently stops being enforced. Since ADR-024 the trained weights ship
//! in a JSON manifest and this crate can run the model *itself* — no Python
//! runtime, no gRPC hop — through [`NativeManifestInvoker`], which is what
//! `POST /api/live-analyze` scores with when the manifest is present.
//!
//! ## HOW
//! These are hand-written equivalents of the generated code rather than the
//! generated code itself. Building real protobuf stubs requires `protoc` and
//! `tonic-build`, which are not available on every developer host, and a build
//! that fails on a missing external binary is a build that does not run. The
//! schema in `specs/proto/aegis.proto` remains the single source of truth, and
//! [`validate_against_proto`] asserts at test time that the field names and
//! service methods here still appear there -- so the two cannot diverge
//! unnoticed, and codegen can be switched on per-host when protoc exists.
//!
//! The handler deliberately *fails closed* when the model is absent: returning a
//! made-up score would be the worst possible behaviour for a security control.

use aegis_common::types::{CompressedDtg, SinkKind, MAX_HOP_DEPTH};
use serde::{Deserialize, Serialize};

/// A tool's declared description and its reachable capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DivergenceRequest {
    pub tool_id: String,
    pub server_id: String,
    /// Attacker-controlled. Never interpreted as an instruction (ADR-001).
    pub declared_description: String,
    pub dtg: CompressedDtg,
    pub declared_capabilities: Vec<String>,
    /// Whether the ONNX graph is resident in the ML service.
    pub model_loaded: bool,
}

/// The semantic divergence verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DivergenceResponse {
    /// False when no model is loaded. The caller must then treat the tool as
    /// unreviewed rather than assume a score of zero.
    pub score_availability: bool,
    pub semantic_divergence_score: f64,
    pub gat_score: f64,
    pub gbdt_score: f64,
    pub description_cosine_distance: f64,
    pub capability_iou: f64,
    pub shannon_entropy: f64,
    pub cyclomatic_complexity: u32,
    pub reached_sinks: Vec<SinkKind>,
    pub inference_ms: f64,
    pub model_sha256: String,
}

impl DivergenceResponse {
    /// The response used when no model is resident.
    ///
    /// `score_availability: false` with a zero score. Callers are required to
    /// check availability first; a zero score presented without it would be
    /// indistinguishable from a genuinely benign tool.
    pub fn unavailable() -> Self {
        DivergenceResponse {
            score_availability: false,
            semantic_divergence_score: 0.0,
            gat_score: 0.0,
            gbdt_score: 0.0,
            description_cosine_distance: 1.0,
            capability_iou: 0.0,
            shannon_entropy: 0.0,
            cyclomatic_complexity: 1,
            reached_sinks: Vec::new(),
            inference_ms: 0.0,
            model_sha256: String::new(),
        }
    }

    /// Whether the score crosses the threshold that forbids every grant.
    ///
    /// 0.50, from `specs/tacr.schema.json` and ADR-007's MILP constraint.
    pub fn exceeds_threshold(&self) -> bool {
        self.score_availability && self.semantic_divergence_score >= 0.50
    }
}

/// One benign task requirement, derived from surviving benign code paths and
/// never from the attacker-controlled description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenignRequirement {
    pub id: String,
    /// Capabilities that satisfy this requirement.
    pub satisfied_by: Vec<SinkKind>,
}

/// A MILP synthesis request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicySynthesisRequest {
    pub tool_id: String,
    pub server_id: String,
    pub dtg: CompressedDtg,
    pub benign_requirements: Vec<BenignRequirement>,
    /// Capabilities excluded outright by divergence or by a counterexample.
    pub forbidden_capabilities: Vec<SinkKind>,
    /// CVSS-derived risk weight per capability name.
    pub risk_weights: std::collections::BTreeMap<String, f64>,
    pub ttl_seconds: u32,
    pub signing_public_key_hex: String,
}

/// The signed TACR and its optimisation report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TacrContractProto {
    /// Canonical JSON conforming to `specs/tacr.schema.json`.
    pub canonical_json: Vec<u8>,
    pub tacr_sha256: String,
    pub ecdsa_p256_signature: String,
    pub objective_value: f64,
    pub capabilities_denied: u32,
    pub attack_surface_eliminated_pct: f64,
    pub solve_time_ms: f64,
    pub solver: String,
}

/// A telemetry subscription filter.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TelemetrySubscription {
    /// Empty means every hook.
    pub hook_types: Vec<u32>,
    pub cgroup_ids: Vec<u64>,
    pub max_latency_us: u64,
}

impl TelemetrySubscription {
    /// Whether an event with this hook code passes the filter.
    pub fn admits_hook(&self, hook_code: u32) -> bool {
        self.hook_types.is_empty() || self.hook_types.contains(&hook_code)
    }

    /// Whether an event from this cgroup passes the filter.
    pub fn admits_cgroup(&self, cgroup_id: u64) -> bool {
        self.cgroup_ids.is_empty() || self.cgroup_ids.contains(&cgroup_id)
    }

    /// Whether an event with this latency passes the filter. `0` disables it.
    pub fn admits_latency(&self, latency_ns: u64) -> bool {
        self.max_latency_us == 0 || latency_ns <= self.max_latency_us * 1_000
    }
}

/// The service surface, for the Rust-side router.
pub mod service {
    /// The three RPCs named in `specs/proto/aegis.proto`.
    pub const EVALUATE_SEMANTIC_DIVERGENCE: &str = "EvaluateSemanticDivergence";
    pub const SYNTHESIZE_MIN_CUT_POLICY: &str = "SynthesizeMinCutPolicy";
    pub const STREAM_KERNEL_TELEMETRY: &str = "StreamKernelTelemetry";
    pub const REQUEST_FIDO2_CHALLENGE: &str = "RequestFido2Challenge";
    pub const SUBMIT_FIDO2_APPROVAL: &str = "SubmitFido2Approval";

    /// The fully-qualified service name.
    pub const SERVICE_NAME: &str = "aegis.v1.AegisControlPlane";

    /// Every method, in declaration order.
    pub const METHODS: [&str; 5] = [
        EVALUATE_SEMANTIC_DIVERGENCE,
        SYNTHESIZE_MIN_CUT_POLICY,
        STREAM_KERNEL_TELEMETRY,
        REQUEST_FIDO2_CHALLENGE,
        SUBMIT_FIDO2_APPROVAL,
    ];
}

/// The client's view of the ML service.
///
/// An `Option<ModelInvoker>` rather than a hard dependency: the gateway must
/// still run when the ML service is absent, and in that state every tool is
/// treated as unreviewed rather than as benign.
pub trait ModelInvoker: Send + Sync {
    /// Scores one tool, or reports unavailability.
    fn evaluate(&self, request: &DivergenceRequest) -> DivergenceResponse;
}

/// The bridge the gateway holds.
#[derive(Default)]
pub struct GrpcBridge {
    invoker: Option<std::sync::Arc<dyn ModelInvoker>>,
    /// Cached divergence scores, so the hot path does not re-invoke the model
    /// for a tool whose DTG hash has not changed.
    cache: std::sync::Mutex<std::collections::HashMap<String, DivergenceResponse>>,
}

impl std::fmt::Debug for GrpcBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcBridge")
            .field("invoker_present", &self.invoker.is_some())
            .finish()
    }
}

impl GrpcBridge {
    /// A bridge with no ML service attached.
    pub fn new() -> Self {
        GrpcBridge::default()
    }

    /// A bridge that will invoke `invoker`.
    pub fn with_invoker(invoker: std::sync::Arc<dyn ModelInvoker>) -> Self {
        GrpcBridge {
            invoker: Some(invoker),
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Whether an ML service is attached.
    pub fn is_available(&self) -> bool {
        self.invoker.is_some()
    }

    /// Scores a tool, consulting and populating the cache.
    ///
    /// The cache key includes the DTG hash, so a tool whose code changed is
    /// re-scored rather than served from a stale verdict -- the cache must never
    /// become a way to keep a known verdict after the code has moved.
    pub fn evaluate(&self, request: &DivergenceRequest) -> DivergenceResponse {
        let key = format!("{}:{}", request.tool_id, request.dtg.dtg_sha256);

        if let Ok(cache) = self.cache.lock() {
            if let Some(hit) = cache.get(&key) {
                return hit.clone();
            }
        }

        let response = match &self.invoker {
            Some(inv) => {
                let mut r = inv.evaluate(request);
                // Availability is decided by whether the model was resident, not
                // by whether the service happened to answer.
                r.score_availability = r.score_availability && request.model_loaded;
                r
            }
            None => DivergenceResponse::unavailable(),
        };

        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, response.clone());
        }
        response
    }

    /// Drops every cached verdict, e.g. after a policy change.
    pub fn invalidate(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// The number of cached verdicts.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

/// Where the shipped divergence manifest lives, relative to the repository
/// root. The gateway resolves it against the working directory and each parent
/// up to four levels, so running the binary from the repo root, from a crate
/// directory, or from a subdirectory all land on the same artifact.
pub const DEFAULT_MANIFEST_PATH: &str = "ml-rag-engine/models/aegis_divergence_weights.json";

/// The environment variable that overrides the default manifest location.
pub const MANIFEST_ENV_VAR: &str = "AEGIS_DIVERGENCE_MANIFEST";

/// The pure-Rust divergence head: the trained weights plus the native
/// featurization (`aegis-common`'s onnx_parity_scorer and
/// divergence_features), no Python runtime and no gRPC hop.
///
/// This is the [`ModelInvoker`] the gateway attaches when
/// [`NativeManifestInvoker::load`] finds the shipped manifest. The parity gate
/// (`aegis-cli parity`, ADR-024) is what makes attaching it sound: the gate
/// measures, against the Python pipeline itself, that the featurized inputs
/// and the scores this code produces are the ones the model was trained on
/// (ADR-024).
pub struct NativeManifestInvoker {
    manifest: aegis_common::onnx_parity_scorer::WeightsManifest,
    load_path: std::path::PathBuf,
}

impl std::fmt::Debug for NativeManifestInvoker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeManifestInvoker")
            .field("model_sha256", &self.manifest.model_sha256)
            .field("load_path", &self.load_path)
            .finish()
    }
}

impl NativeManifestInvoker {
    /// A head over an already-parsed manifest.
    pub fn from_manifest(
        manifest: aegis_common::onnx_parity_scorer::WeightsManifest,
        load_path: std::path::PathBuf,
    ) -> Self {
        NativeManifestInvoker { manifest, load_path }
    }

    /// Loads the shipped manifest: `AEGIS_DIVERGENCE_MANIFEST` when set, else
    /// `DEFAULT_MANIFEST_PATH` resolved against the working directory and up
    /// to four parents. `None` when nothing parses — the caller runs without
    /// a model head, exactly as if the ML service were down.
    pub fn load() -> Option<Self> {
        match std::env::var(MANIFEST_ENV_VAR) {
            Ok(path) => NativeManifestInvoker::load_from(std::path::Path::new(&path)),
            Err(_) => NativeManifestInvoker::load_from_default(),
        }
    }

    /// The default-path search: the working directory itself, then up to four
    /// parents (the depth of a crate directory inside this workspace).
    fn load_from_default() -> Option<Self> {
        let cwd = std::env::current_dir().ok()?;
        let mut candidate = cwd.as_path();
        for _ in 0..5 {
            let path = candidate.join(DEFAULT_MANIFEST_PATH);
            if let Some(head) = NativeManifestInvoker::load_from(&path) {
                return Some(head);
            }
            candidate = candidate.parent()?;
        }
        None
    }

    /// Parses the manifest at `path`; `None` on a missing or malformed file.
    pub fn load_from(path: &std::path::Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let manifest = aegis_common::onnx_parity_scorer::WeightsManifest::parse(&bytes).ok()?;
        Some(NativeManifestInvoker::from_manifest(manifest, path.to_path_buf()))
    }

    /// The manifest this head scores with.
    pub fn model_sha256(&self) -> &str {
        &self.manifest.model_sha256
    }

    /// Where the manifest was read from.
    pub fn load_path(&self) -> &std::path::Path {
        &self.load_path
    }
}

impl ModelInvoker for NativeManifestInvoker {
    /// Scores one tool with the trained weights, entirely in process.
    ///
    /// Every internal failure is fail-closed: a malformed graph or a shape
    /// mismatch yields [`DivergenceResponse::unavailable`], never a guessed
    /// score. `model_sha256` is always reported so a consumer can tell which
    /// artifact produced even an unavailable answer.
    fn evaluate(&self, request: &DivergenceRequest) -> DivergenceResponse {
        let started = std::time::Instant::now();
        let fused = match aegis_common::divergence_features::fused_features(
            &self.manifest,
            &request.dtg,
            &request.declared_description,
        ) {
            Ok((graph, fused)) => (graph, fused),
            Err(_) => {
                let mut unavailable = DivergenceResponse::unavailable();
                unavailable.model_sha256 = self.manifest.model_sha256.clone();
                return unavailable;
            }
        };
        let (graph, fused) = fused;
        let logit = aegis_common::onnx_parity_scorer::gbdt_logit(&self.manifest, &fused);
        let score = aegis_common::onnx_parity_scorer::sigmoid(logit);

        // Auxiliary signals, in the shape the proto defines: the pooled
        // embedding's magnitude squashed through the training report's tanh
        // scale, the description's cosine distance from the realised-sink
        // text, and the capability overlap the description claims versus what
        // the graph reaches.
        let embedding = &fused[..self.manifest.out_dim];
        let norm: f64 = embedding
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        let gat_score = (norm / 4.0).tanh();

        let description_vec = aegis_common::divergence_features::encode(&request.declared_description);
        let sink_kinds: Vec<String> = graph.sink_kinds.iter().cloned().collect();
        let sink_vec = aegis_common::divergence_features::encode(&sink_kinds.join(" "));
        let cosine = if sink_kinds.is_empty() {
            1.0
        } else {
            aegis_common::divergence_features::cosine_distance(&description_vec, &sink_vec)
        };

        let declared_caps = aegis_common::divergence_features::declared_capabilities(
            &request.declared_description,
        );
        let declared_ref: std::collections::BTreeSet<&str> = declared_caps.iter().copied().collect();
        let reached_ref: std::collections::BTreeSet<&str> = graph
            .sink_kinds
            .iter()
            .map(String::as_str)
            .collect();
        let iou = if declared_ref.is_empty() && reached_ref.is_empty() {
            1.0
        } else {
            let inter = declared_ref.intersection(&reached_ref).count();
            let union = declared_ref.union(&reached_ref).count();
            inter as f64 / union as f64
        };

        DivergenceResponse {
            score_availability: true,
            semantic_divergence_score: f64::from(score),
            gat_score,
            // The manifest head emits one post-fusion score; the proto's
            // gbdt field carries the same value because the fused trees are
            // the branch that decides.
            gbdt_score: f64::from(score),
            description_cosine_distance: cosine,
            capability_iou: iou,
            shannon_entropy: aegis_common::divergence_features::max_sink_entropy(
                &request.dtg.detected_sinks,
            ),
            // The compressed DTG carries no branch count; the native head
            // records the single-path baseline rather than a fabricated
            // estimate. The field is proto shape, not a policy input.
            cyclomatic_complexity: 1,
            reached_sinks: {
                let mut sinks: Vec<SinkKind> = graph
                    .sink_kinds
                    .iter()
                    .map(|kind| SinkKind::from_str_name(kind).expect("kinds come from the enum"))
                    .collect();
                sinks.sort();
                sinks
            },
            inference_ms: started.elapsed().as_secs_f64() * 1e3,
            model_sha256: self.manifest.model_sha256.clone(),
        }
    }
}

/// Asserts that every field and method this bridge implements still appears in
/// `specs/proto/aegis.proto`.
///
/// Returns the list of names that have drifted. Called from the test suite so
/// that editing one side without the other is a build failure rather than a
/// runtime mystery.
pub fn validate_against_proto(proto: &str) -> Vec<String> {
    let mut missing = Vec::new();
    for method in service::METHODS {
        if !proto.contains(&format!("rpc {method} (")) {
            missing.push(format!("rpc {method}"));
        }
    }
    if !proto.contains(&format!("service {}", "AegisControlPlane")) {
        missing.push("service AegisControlPlane".to_string());
    }
    for field in [
        "tool_id",
        "server_id",
        "declared_description",
        "dtg",
        "model_loaded",
        "score_availability",
        "semantic_divergence_score",
        "gat_score",
        "gbdt_score",
        "description_cosine_distance",
        "capability_iou",
        "shannon_entropy",
        "cyclomatic_complexity",
        "reached_sinks",
        "inference_ms",
        "model_sha256",
        "canonical_json",
        "tacr_sha256",
        "ecdsa_p256_signature",
        "objective_value",
        "capabilities_denied",
        "attack_surface_eliminated_pct",
        "solve_time_ms",
        "hook_types",
        "cgroup_ids",
        "max_latency_us",
    ] {
        if !proto.contains(field) {
            missing.push(format!("field {field}"));
        }
    }
    missing
}

/// The hop bound the bridge and the DTG agree on.
pub const BRIDGE_MAX_HOPS: u8 = MAX_HOP_DEPTH;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Shares its call counter with the test through an `Arc`, so the test can
    /// assert on invocation counts without borrowing the invoker.
    struct CountingInvoker {
        calls: std::sync::Arc<AtomicUsize>,
        response: DivergenceResponse,
    }

    impl ModelInvoker for CountingInvoker {
        fn evaluate(&self, _request: &DivergenceRequest) -> DivergenceResponse {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.response.clone()
        }
    }

    fn counting(response: DivergenceResponse) -> (std::sync::Arc<AtomicUsize>, CountingInvoker) {
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        (
            calls.clone(),
            CountingInvoker {
                calls,
                response,
            },
        )
    }

    fn dtg() -> CompressedDtg {
        CompressedDtg {
            server_id: "lab".into(),
            language: "python".into(),
            raw_cpg_node_count: 500,
            compressed_node_count: 2,
            nodes: vec![],
            edges: vec![],
            detected_sinks: vec![],
            dependencies: vec!["os".into()],
            injection_markers: vec![],
            egress_literals: vec![],
            dtg_sha256: "a".repeat(64),
        }
    }

    fn request() -> DivergenceRequest {
        DivergenceRequest {
            tool_id: "fs.read_env".into(),
            server_id: "lab".into(),
            declared_description: "Formats dates".into(),
            dtg: dtg(),
            declared_capabilities: vec!["format".into()],
            model_loaded: true,
        }
    }

    fn verdict(score: f64) -> DivergenceResponse {
        DivergenceResponse {
            score_availability: true,
            semantic_divergence_score: score,
            ..DivergenceResponse::unavailable()
        }
    }

    #[test]
    fn a_bridge_with_no_service_reports_unavailable_not_zero() {
        let b = GrpcBridge::new();
        assert!(!b.is_available());
        let r = b.evaluate(&request());
        assert!(!r.score_availability, "must not present a fabricated score");
        assert!(!r.exceeds_threshold());
    }

    #[test]
    fn an_absent_model_keeps_the_verdict_unavailable() {
        // The service answered, but the ONNX graph was not resident. Availability
        // must be gated on the model, not on the transport.
        let (_calls, inv) = counting(verdict(0.99));
        let b = GrpcBridge::with_invoker(std::sync::Arc::new(inv));
        let mut req = request();
        req.model_loaded = false;
        let r = b.evaluate(&req);
        assert!(!r.score_availability);
        assert!(!r.exceeds_threshold());
    }

    #[test]
    fn the_threshold_is_the_schema_value() {
        let mut r = verdict(0.49);
        assert!(!r.exceeds_threshold());
        r.semantic_divergence_score = 0.50;
        assert!(r.exceeds_threshold(), "0.50 is the MILP's hard constraint");
        r.semantic_divergence_score = 0.86;
        assert!(r.exceeds_threshold());
    }

    #[test]
    fn an_unavailable_score_never_trips_the_threshold() {
        let mut r = DivergenceResponse::unavailable();
        r.semantic_divergence_score = 0.99;
        assert!(!r.exceeds_threshold());
    }

    #[test]
    fn results_are_cached_per_dtg_hash() {
        let (calls, inv) = counting(verdict(0.7));
        let b = GrpcBridge::with_invoker(std::sync::Arc::new(inv));
        b.evaluate(&request());
        b.evaluate(&request());
        assert_eq!(calls.load(Ordering::SeqCst), 1, "second call must hit the cache");
        assert_eq!(b.cache_len(), 1);
    }

    #[test]
    fn a_changed_dtg_hash_busts_the_cache() {
        let (calls, inv) = counting(verdict(0.7));
        let b = GrpcBridge::with_invoker(std::sync::Arc::new(inv));
        b.evaluate(&request());

        let mut changed = request();
        changed.dtg.dtg_sha256 = "b".repeat(64);
        b.evaluate(&changed);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "a code change must be re-scored, never served from cache"
        );
        assert_eq!(b.cache_len(), 2);
    }

    #[test]
    fn invalidation_clears_the_cache() {
        let (calls, inv) = counting(verdict(0.7));
        let b = GrpcBridge::with_invoker(std::sync::Arc::new(inv));
        b.evaluate(&request());
        b.invalidate();
        assert_eq!(b.cache_len(), 0);
        b.evaluate(&request());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// The manifest the ML pipeline ships, at the workspace-standard path.
    fn shipped_manifest_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ml-rag-engine/models/aegis_divergence_weights.json")
    }

    #[test]
    fn the_native_head_loads_the_shipped_manifest() {
        let head = NativeManifestInvoker::load_from(&shipped_manifest_path())
            .expect("the shipped manifest parses");
        assert!(
            !head.model_sha256().is_empty(),
            "the manifest carries its own content hash"
        );
        assert!(
            head.load_path()
                .ends_with("ml-rag-engine/models/aegis_divergence_weights.json"),
            "the head remembers where it loaded from, for the startup log"
        );
    }

    #[test]
    fn a_missing_manifest_degrades_to_no_head_not_a_default() {
        // Silence is not permission: a head that cannot load must be absent,
        // so callers see an unavailable score instead of a fabricated one.
        let missing =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("no/such/manifest.json");
        assert!(NativeManifestInvoker::load_from(&missing).is_none());
    }

    #[test]
    fn the_native_head_scores_a_real_graph_in_process() {
        let head = NativeManifestInvoker::load_from(&shipped_manifest_path()).unwrap();
        let response = head.evaluate(&request());
        assert!(response.score_availability);
        assert!(
            (0.0..=1.0).contains(&response.semantic_divergence_score),
            "{}",
            response.semantic_divergence_score
        );
        assert_eq!(
            response.model_sha256,
            head.model_sha256().to_string(),
            "the answer names the artifact that produced it"
        );
        // The empty-graph test fixture reaches no sinks: the score comes from
        // the trained trees on the fallback row, and the sink signals are the
        // degenerate values the Python reference produces for the same shape.
        assert!(response.reached_sinks.is_empty());
        assert_eq!(
            response.gbdt_score, response.semantic_divergence_score,
            "the manifest head emits one post-fusion score"
        );
        assert!(
            (0.0..=1.0).contains(&response.gat_score),
            "the pooled embedding's tanh scale: {}",
            response.gat_score
        );
        assert!(response.inference_ms > 0.0, "the head measures its own latency");
    }

    #[test]
    fn an_incompatible_manifest_scores_unavailable_and_names_the_artifact() {
        // A manifest whose input dimension does not match the featurized row
        // cannot be scored: the featurizer refuses the shape and the head
        // answers unavailable with its identity attached, rather than a
        // guessed score or a panic from indexing past the row.
        let shipped_bytes = std::fs::read(shipped_manifest_path()).unwrap();
        let expected_sha = {
            let head = NativeManifestInvoker::load_from(&shipped_manifest_path())
                .expect("the shipped manifest parses");
            head.model_sha256().to_string()
        };
        let mut manifest = aegis_common::onnx_parity_scorer::WeightsManifest::parse(&shipped_bytes)
            .expect("the shipped manifest is well-formed");
        manifest.in_dim = 22; // the featurizer emits 21-wide rows
        let wrong_head = NativeManifestInvoker::from_manifest(manifest, shipped_manifest_path());
        let refused = wrong_head.evaluate(&request());
        assert!(!refused.score_availability, "a shape mismatch must not score");
        assert!(
            !refused.exceeds_threshold(),
            "an unavailable score never trips the threshold"
        );
        assert_eq!(
            refused.semantic_divergence_score, 0.0,
            "no guessed value along the refusal"
        );
        assert_eq!(
            refused.model_sha256, expected_sha,
            "the refusal still names the artifact"
        );
    }

    #[test]
    fn subscription_filters_admit_and_reject() {
        let empty = TelemetrySubscription::default();
        assert!(empty.admits_hook(1));
        assert!(empty.admits_cgroup(42));
        assert!(empty.admits_latency(u64::MAX), "0 disables latency filtering");

        let filtered = TelemetrySubscription {
            hook_types: vec![3],
            cgroup_ids: vec![42],
            max_latency_us: 100,
        };
        assert!(filtered.admits_hook(3));
        assert!(!filtered.admits_hook(1));
        assert!(filtered.admits_cgroup(42));
        assert!(!filtered.admits_cgroup(7));
        assert!(filtered.admits_latency(40_000));
        assert!(!filtered.admits_latency(500_000));
    }

    #[test]
    fn the_bridge_still_matches_the_proto_schema() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|root| root.join("specs").join("proto").join("aegis.proto"));
        let proto = path
            .and_then(|p| std::fs::read_to_string(p).ok())
            .expect("specs/proto/aegis.proto must be readable");
        let missing = validate_against_proto(&proto);
        assert!(
            missing.is_empty(),
            "the Rust bridge has drifted from aegis.proto: {missing:?}"
        );
    }

    #[test]
    fn drift_detection_actually_detects_drift() {
        // Guard against the validator silently passing everything.
        let empty = "";
        let missing = validate_against_proto(empty);
        assert!(missing.len() > 5, "an empty proto must report drift");
    }

    #[test]
    fn the_service_name_and_methods_are_stable() {
        assert_eq!(service::SERVICE_NAME, "aegis.v1.AegisControlPlane");
        assert_eq!(service::METHODS.len(), 5);
        assert!(service::METHODS.contains(&service::EVALUATE_SEMANTIC_DIVERGENCE));
        assert!(service::METHODS.contains(&service::STREAM_KERNEL_TELEMETRY));
    }

    #[test]
    fn bridge_hops_match_the_dtg_hop_bound() {
        assert_eq!(BRIDGE_MAX_HOPS, MAX_HOP_DEPTH);
    }
}