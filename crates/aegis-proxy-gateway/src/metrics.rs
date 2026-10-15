//! Runtime metrics, in JSON and Prometheus exposition formats.
//!
//! **Author:** Pratham Kapoor `<prathamkapoor027@gmail.com>`
//! **Repository:** <https://github.com/PrathamKapoor>
//!
//! ## WHAT
//! Fire-and-forget counters for every decision this gateway makes, exposed as
//! `GET /api/metrics` (JSON) and `GET /metrics` (Prometheus text format,
//! FR-5.3).
//!
//! ## WHY
//! The enforcement decisions are the product's output, and an unmeasured
//! control plane silently stops controlling: a mis-set flag that turns every
//! suppression into a pass-through is invisible until someone counts. The two
//! formats exist because they serve different consumers -- the portal polls the
//! JSON beside the graph, and Prometheus scrapes the line format -- and both
//! are rendered from **one** snapshot, so they can never disagree.
//!
//! ## HOW
//! `AtomicU64` with `Relaxed` ordering: these are independent counters, no
//! decision reads another, and a slightly inconsistent set under a concurrent
//! scrape is immaterial. The counter list is declared exactly once in
//! [`Metrics::SERIES`]; the snapshot and the Prometheus rendering both iterate
//! it, so a counter cannot exist in one format and be missing from the other.
//! Prometheus names carry the `_total` suffix that `rate()` requires.
//!
//! Complexity: `O(1)` per increment; `O(series)` per scrape.

use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! define_metrics {
    ($($field:ident => $prom:literal, $help:literal;)*) => {
        /// The gateway's runtime counters.
        #[derive(Debug, Default)]
        pub struct Metrics {
            $(pub $field: AtomicU64,)*
        }

        impl Metrics {
            /// The full series table: (JSON name, Prometheus name, help).
            ///
            /// Declared once so the JSON and Prometheus outputs are two
            /// renderings of one list rather than two lists that drift.
            pub const SERIES: &'static [(&'static str, &'static str, &'static str)] = &[
                $((stringify!($field), $prom, $help),)*
            ];
        }

        impl Metrics {
            /// Reads every counter once, keyed by JSON name.
            pub fn snapshot(&self) -> MetricSnapshot {
                let mut counters = BTreeMap::new();
                $(counters.insert(
                    stringify!($field).to_string(),
                    self.$field.load(Ordering::Relaxed),
                );)*
                MetricSnapshot { counters }
            }
        }
    };
}

define_metrics! {
    rpc_requests => "aegis_gateway_rpc_requests_total", "JSON-RPC envelopes processed.";
    rpc_allowed => "aegis_gateway_rpc_allowed_total", "Calls the edit automaton passed through unchanged.";
    rpc_suppressed => "aegis_gateway_rpc_suppressed_total", "Calls suppressed (unknown method, unattested or unpermitted tool).";
    rpc_modified => "aegis_gateway_rpc_modified_total", "Responses sanitised before returning.";
    step_ups_required => "aegis_gateway_step_ups_required_total", "Cross-domain flows held for a FIDO2 step-up.";
    tool_results => "aegis_gateway_tool_results_total", "Tool results labelled and sanitised.";
    taint_elevations => "aegis_gateway_taint_elevations_total", "Reads that raised a session's high-water mark.";
    approvals_recorded => "aegis_gateway_approvals_recorded_total", "Declassifications recorded after a verified assertion.";
    fido2_challenges => "aegis_gateway_fido2_challenges_total", "FIDO2 challenges issued.";
    fido2_verified => "aegis_gateway_fido2_verified_total", "FIDO2 assertions that verified.";
    fido2_rejected => "aegis_gateway_fido2_rejected_total", "FIDO2 assertions refused.";
    oauth_authorizations => "aegis_gateway_oauth_authorizations_total", "Authorization codes issued (PKCE, S256).";
    oauth_tokens => "aegis_gateway_oauth_tokens_total", "PASETO v3.public access tokens issued.";
    oauth_refusals => "aegis_gateway_oauth_refusals_total", "Authorization or token requests refused.";
    cypher_queries => "aegis_gateway_cypher_queries_total", "Cypher-subset queries executed.";
    cypher_rows => "aegis_gateway_cypher_rows_total", "Rows returned across all queries.";
    sse_connections => "aegis_gateway_sse_connections_total", "Telemetry streams opened.";
    portal_views => "aegis_gateway_portal_views_total", "Command center and approver page views.";
}

impl Metrics {
    /// A fresh, all-zero counter set.
    pub fn new() -> Self {
        Metrics::default()
    }

    /// Increments one counter by one.
    pub fn inc(&self, counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Adds `delta` to one counter.
    pub fn add(&self, counter: &AtomicU64, delta: u64) {
        counter.fetch_add(delta, Ordering::Relaxed);
    }
}

/// A point-in-time copy of every counter, serialised flat as the JSON body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MetricSnapshot {
    /// Counter name to value.
    #[serde(flatten)]
    pub counters: BTreeMap<String, u64>,
}

/// The gauges, computed from state at render time rather than counted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Gauges {
    /// Product name.
    pub product: String,
    /// Product version.
    pub version: String,
    /// Seconds since start.
    pub uptime_seconds: u64,
    /// Attested tools.
    pub tools: u64,
    /// Tracked sessions.
    pub sessions: u64,
    /// Loaded Data Transformation Graphs.
    pub graphs: u64,
    /// Open FIDO2 challenges.
    pub pending_challenges: u64,
}

/// The JSON response body: counters plus gauges in one flat object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MetricsResponse {
    /// The counters, flattened.
    #[serde(flatten)]
    pub counters: BTreeMap<String, u64>,
    /// Product name.
    pub product: String,
    /// Product version.
    pub version: String,
    /// Seconds since start.
    pub uptime_seconds: u64,
    /// Attested tools.
    pub tools: u64,
    /// Tracked sessions.
    pub sessions: u64,
    /// Loaded graphs.
    pub graphs: u64,
    /// Open FIDO2 challenges.
    pub pending_challenges: u64,
}

impl MetricsResponse {
    /// Joins a snapshot and the gauges.
    pub fn from_snapshot(snapshot: MetricSnapshot, gauges: &Gauges) -> Self {
        MetricsResponse {
            counters: snapshot.counters,
            product: gauges.product.clone(),
            version: gauges.version.clone(),
            uptime_seconds: gauges.uptime_seconds,
            tools: gauges.tools,
            sessions: gauges.sessions,
            graphs: gauges.graphs,
            pending_challenges: gauges.pending_challenges,
        }
    }
}

/// Escapes a Prometheus label value (quote and backslash only).
fn escape_label(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn push_counter(out: &mut String, prometheus_name: &str, help: &str, value: u64) {
    out.push_str("# HELP ");
    out.push_str(prometheus_name);
    out.push(' ');
    out.push_str(help);
    out.push_str("\n# TYPE ");
    out.push_str(prometheus_name);
    out.push_str(" counter\n");
    out.push_str(prometheus_name);
    out.push(' ');
    out.push_str(&value.to_string());
    out.push('\n');
}

fn push_gauge(out: &mut String, prometheus_name: &str, help: &str, value: u64) {
    out.push_str("# HELP ");
    out.push_str(prometheus_name);
    out.push(' ');
    out.push_str(help);
    out.push_str("\n# TYPE ");
    out.push_str(prometheus_name);
    out.push_str(" gauge\n");
    out.push_str(prometheus_name);
    out.push(' ');
    out.push_str(&value.to_string());
    out.push('\n');
}

impl Metrics {
    /// Renders the Prometheus text exposition format for one snapshot.
    ///
    /// Order is fixed (declaration order, then gauges), so two scrapes of the
    /// same state are byte-identical -- a property contract tests rely on.
    pub fn prometheus(&self, gauges: &Gauges) -> String {
        let snapshot = self.snapshot();
        let mut out = String::with_capacity(4096);

        out.push_str("# HELP aegis_gateway_build_info Build identity of the running gateway.\n");
        out.push_str("# TYPE aegis_gateway_build_info gauge\n");
        out.push_str(&format!(
            "aegis_gateway_build_info{{product=\"{}\",version=\"{}\"}} 1\n",
            escape_label(&gauges.product),
            escape_label(&gauges.version),
        ));

        for (json, prometheus_name, help) in Metrics::SERIES {
            let value = snapshot.counters.get(*json).copied().unwrap_or(0);
            push_counter(&mut out, prometheus_name, help, value);
        }

        push_gauge(
            &mut out,
            "aegis_gateway_uptime_seconds",
            "Seconds since the gateway started.",
            gauges.uptime_seconds,
        );
        push_gauge(
            &mut out,
            "aegis_gateway_tools",
            "Attested tools in the capability table.",
            gauges.tools,
        );
        push_gauge(
            &mut out,
            "aegis_gateway_sessions",
            "Sessions with a taint mark.",
            gauges.sessions,
        );
        push_gauge(
            &mut out,
            "aegis_gateway_graphs",
            "Data Transformation Graphs loaded for query.",
            gauges.graphs,
        );
        push_gauge(
            &mut out,
            "aegis_gateway_pending_challenges",
            "Open FIDO2 step-up challenges.",
            gauges.pending_challenges,
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauges() -> Gauges {
        Gauges {
            product: "AegisCPG".into(),
            version: "1.0.0".into(),
            uptime_seconds: 42,
            tools: 3,
            sessions: 2,
            graphs: 1,
            pending_challenges: 0,
        }
    }

    #[test]
    fn counters_increment_and_land_in_the_snapshot() {
        let metrics = Metrics::new();
        metrics.inc(&metrics.rpc_requests);
        metrics.inc(&metrics.rpc_requests);
        metrics.add(&metrics.cypher_rows, 7);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.counters["rpc_requests"], 2);
        assert_eq!(snapshot.counters["cypher_rows"], 7);
        assert_eq!(snapshot.counters["rpc_allowed"], 0);
        assert_eq!(snapshot.counters.len(), Metrics::SERIES.len());
    }

    #[test]
    fn every_series_appears_in_both_formats() {
        let metrics = Metrics::new();
        let snapshot = metrics.snapshot();
        let prometheus = metrics.prometheus(&gauges());
        for (json, prometheus_name, _) in Metrics::SERIES {
            assert!(
                snapshot.counters.contains_key(*json),
                "{json} missing from JSON"
            );
            assert!(
                prometheus.contains(&format!("\n{prometheus_name} ")),
                "{prometheus_name} missing from Prometheus output"
            );
            assert!(
                prometheus.contains(&format!("# TYPE {prometheus_name} counter")),
                "{prometheus_name} lacks a TYPE line"
            );
            assert!(
                prometheus.contains(&format!("# HELP {prometheus_name} ")),
                "{prometheus_name} lacks a HELP line"
            );
        }
    }

    #[test]
    fn counter_values_are_identical_across_the_two_renderings() {
        let metrics = Metrics::new();
        metrics.add(&metrics.rpc_requests, 11);
        metrics.add(&metrics.fido2_verified, 5);
        metrics.add(&metrics.oauth_tokens, 3);
        let snapshot = metrics.snapshot();
        let prometheus = metrics.prometheus(&gauges());

        for (json, prometheus_name, _) in Metrics::SERIES {
            let value = snapshot.counters[*json];
            if value == 0 {
                continue;
            }
            let line = prometheus
                .lines()
                .find(|line| line.starts_with(&format!("{prometheus_name} ")))
                .unwrap_or_else(|| panic!("{prometheus_name} line missing"));
            let rendered: u64 = line
                .split(' ')
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{line:?} is not name value"));
            assert_eq!(rendered, value, "{json} disagrees between formats");
        }
    }

    #[test]
    fn every_metric_line_parses_as_name_value() {
        let metrics = Metrics::new();
        metrics.inc(&metrics.rpc_requests);
        let text = metrics.prometheus(&gauges());
        let mut series = 0;
        for line in text.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split(' ');
            let name = parts.next().unwrap();
            assert!(
                name.starts_with("aegis_gateway_"),
                "{name} is not in the aegis namespace"
            );
            assert!(
                name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '{' || c == '}' || c == '"' || c == '=' || c == ',' || c == '.'),
                "{name} has characters outside the exposition alphabet"
            );
            let value = parts.next().expect("a value");
            assert!(
                value.parse::<u64>().is_ok(),
                "{line:?}: {value:?} is not a counter value"
            );
            assert!(parts.next().is_none(), "{line:?} has trailing tokens");
            series += 1;
        }
        assert_eq!(
            series,
            Metrics::SERIES.len() + 6,
            "18 counters + build_info + 5 gauges"
        );
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn rendering_is_byte_deterministic() {
        let metrics = Metrics::new();
        metrics.inc(&metrics.portal_views);
        let first = metrics.prometheus(&gauges());
        let second = metrics.prometheus(&gauges());
        assert_eq!(first, second);
    }

    #[test]
    fn build_info_carries_escaped_labels() {
        let metrics = Metrics::new();
        let mut g = gauges();
        g.version = "1.0.0\"x".into();
        let text = metrics.prometheus(&g);
        assert!(text.contains(r#"version="1.0.0\"x""#), "{text}");
    }

    #[test]
    fn the_json_response_flattens_counters_and_gauges() {
        let metrics = Metrics::new();
        metrics.inc(&metrics.rpc_suppressed);
        let response = MetricsResponse::from_snapshot(metrics.snapshot(), &gauges());
        let value = serde_json::to_value(&response).unwrap();
        assert_eq!(value["rpc_suppressed"], 1);
        assert_eq!(value["product"], "AegisCPG");
        assert_eq!(value["uptime_seconds"], 42);
        assert_eq!(value["graphs"], 1);
    }
}
