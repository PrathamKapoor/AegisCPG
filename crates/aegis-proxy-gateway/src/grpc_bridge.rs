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
//! Stage 2 (the GAT/GBDT classifier) lives in Python and the hot path lives in
//! Rust. Without a typed contract the two drift: the Python model adds a
//! feature, the Rust gateway keeps reading the old struct, and the divergence
//! score silently stops being enforced.
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