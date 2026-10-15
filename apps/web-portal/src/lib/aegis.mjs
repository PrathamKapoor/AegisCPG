/**
 * AegisCPG web portal -- metrics and DTG view models.
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *     Pure functions that turn the artefacts the Rust and Python stages emit
 *     into the shapes the dashboard renders. No React, no deck.gl, no fetch:
 *     everything here is a pure transformation of an already-parsed object.
 *
 * WHY THE LOGIC IS IN PLAIN JAVASCRIPT AND TESTED SEPARATELY
 *     The portal's numbers come from `benchmarks/results/summary_metrics.json`
 *     and `aegis-cli scan --json`. Every one of them is already asserted by the
 *     Python and Rust suites, so what the portal adds is presentation. Putting
 *     the transformation in plain ES modules means it can be tested with
 *     `node --test` on a host with no npm registry -- which is this one -- so
 *     the scorecard cannot silently show a field the producer renamed.
 *
 *     A `null` in a required field raises rather than rendering as `undefined`.
 *     A dashboard that quietly displays "—" for a metric that failed to load is
 *     worse than one that fails to render.
 *
 * HOW
 *     `readSummaryMetrics` validates the producer's schema version, every
 *     required section, and the invariants the producer asserts -- so a
 *     threshold failure in the benchmark harness surfaces as a red banner in the
 *     portal rather than as a stale green number.
 *
 * Complexity: O(1) in the size of the metrics document; O(V + E) for a graph.
 */

/** The producer's schema. A mismatch is a hard error, never a coercion. */
export const SUMMARY_SCHEMA = 'aegis.benchmarks.v1';

/** The scan document's schema, from `aegis-cli scan --json`. */
export const SCAN_SCHEMA = 'aegis.scan.v1';

/**
 * The Denning lattice the control plane enforces, ordered PUBLIC to SECRET.
 * Must match `workflows.workflows.LATTICE_ORDER` on the Python side and
 * `specs/lattice_rules.smt2`.
 */
export const LATTICE_ORDER = ['PUBLIC', 'INTERNAL', 'CONFIDENTIAL', 'SECRET'];

/** The lattice level a sink kind implies at minimum, for the 3D view's Y axis. */
export const SINK_LATTICE_FLOOR = {
  NetworkInbound: 1,
  FileRead: 1,
  FileWrite: 2,
  EnvAccess: 3,
  NetworkOutbound: 2,
  CommandExec: 3,
  UnboundedDynamic: 3,
};

/** Throws when a required field is absent or null. */
function requireField(object, field, context) {
  if (object === null || typeof object !== 'object') {
    throw new TypeError(`${context}: expected an object, got ${typeof object}`);
  }
  if (!(field in object) || object[field] === null || object[field] === undefined) {
    throw new TypeError(`${context}: required field '${field}' is missing`);
  }
  return object[field];
}

/**
 * Validate and index the benchmark summary.
 *
 * @param {object} summary parsed `benchmarks/results/summary_metrics.json`
 * @returns {object} the same document plus a `view` block the UI renders
 */
export function readSummaryMetrics(summary) {
  const schema = requireField(summary, 'schema', 'summary');
  if (schema !== SUMMARY_SCHEMA) {
    throw new TypeError(
      `summary schema '${schema}' is not the one this portal renders ` +
        `(expected '${SUMMARY_SCHEMA}'); regenerate with benchmarks/run_all_benchmarks.py`,
    );
  }

  const static_ = requireField(summary, 'static', 'summary');
  const chains = requireField(summary, 'chains', 'summary');
  const divergence = requireField(summary, 'divergence', 'summary');
  const rag = requireField(summary, 'rag', 'summary');
  const machine = requireField(summary, 'machine', 'summary');
  const failures = Array.isArray(summary.threshold_failures) ? summary.threshold_failures : [];

  const cards = [
    {
      id: 'static-recall',
      label: 'Polyglot static recall',
      value: `${static_.recall_count} / ${static_.vulnerable}`,
      unit: 'fixtures',
      detail: `${static_.fixtures} fixture servers scanned across the polyglot set`,
      severity: static_.recall_count >= static_.vulnerable ? 'good' : 'bad',
    },
    {
      id: 'static-fpr',
      label: 'False positives',
      value: `${static_.false_positive_count} / ${static_.benign}`,
      unit: 'benign',
      detail: `${static_.false_positive_percent.toFixed(1)} percent of known-clean servers flagged`,
      severity: static_.false_positive_count === 0 ? 'good' : 'bad',
    },
    {
      id: 'compression',
      label: 'DTG compression',
      value: `${(static_.min_compression_ratio * 100).toFixed(2)}`,
      unit: 'percent minimum',
      detail:
        `${static_.total_raw_nodes} raw CPG nodes reduced to ` +
        `${static_.total_compressed_nodes} retained data states`,
      severity:
        static_.min_compression_ratio >= 0.8 ? 'good' : 'bad',
    },
    {
      id: 'chain-block-rate',
      label: 'Compositional block rate',
      value: `${chains.blocked} / ${chains.malicious}`,
      unit: 'malicious chains',
      detail: `${chains.legitimate_allowed} of ${chains.legitimate} legitimate chains correctly allowed`,
      severity: chains.blocked === chains.malicious ? 'good' : 'bad',
    },
    {
      id: 'divergence-separation',
      label: 'Divergence separation',
      value: divergence.separation_margin.toFixed(4),
      unit: 'margin',
      detail:
        `benign max ${divergence.max_benign_score.toFixed(4)} vs ` +
        `malicious min ${divergence.min_malicious_score.toFixed(4)} at threshold ` +
        `${divergence.threshold.toFixed(2)}`,
      severity:
        divergence.max_benign_score < divergence.threshold &&
        divergence.min_malicious_score >= divergence.threshold
          ? 'good'
          : 'bad',
    },
    {
      id: 'rag-ablation',
      label: 'RAG ablation (nDCG@10)',
      value: rag.mean_hybrid_ndcg_at_10.toFixed(4),
      unit: 'hybrid',
      detail:
        `against ${rag.mean_dense_only_ndcg_at_10.toFixed(4)} dense-only over ` +
        `${rag.queries} queries; ${rag.dense_path} was the active encoder`,
      severity: rag.mean_hybrid_ndcg_at_10 > rag.mean_dense_only_ndcg_at_10 ? 'good' : 'bad',
    },
    {
      id: 'scoring-latency',
      label: 'Divergence scoring',
      value: summary.end_to_end_scoring_ms.toFixed(2),
      unit: 'ms minimum',
      detail:
        `${summary.end_to_end_scoring_median_ms.toFixed(2)} ms median under host ` +
        `contention; control plane, not the gateway hot path`,
      severity:
        summary.end_to_end_scoring_ms <= summary.thresholds.max_end_to_end_scoring_ms
          ? 'good'
          : 'bad',
    },
  ];

  return {
    schema,
    generatedUtc: requireField(summary, 'generated_utc', 'summary'),
    machine,
    thresholds: requireField(summary, 'thresholds', 'summary'),
    failures,
    healthy: failures.length === 0 && cards.every((card) => card.severity === 'good'),
    cards,
  };
}

/**
 * Convert one `CompressedDtg` into deck.gl-ready node and edge descriptors.
 *
 * The layout is a deterministic force-free arrangement on a cylinder: lattice
 * level is the Y axis (the property an auditor is actually looking for), the
 * source file position is the angle, and hop depth is the radius. Deterministic
 * rather than force-directed because a dashboard that re-shuffles on every
 * refresh cannot be compared to a screenshot in a bug report.
 *
 * @param {object} server one entry of `aegis-cli scan --json`'s `servers`
 * @returns {{nodes: object[], edges: object[], latticeOf: object}}
 */
export function buildGraphLayers(server) {
  const dtg = requireField(server, 'dtg', 'server');
  const nodes = requireField(dtg, 'nodes', 'dtg');
  const edges = requireField(dtg, 'edges', 'dtg');

  const sinkKindsByNode = new Map();
  for (const edge of edges) {
    if (edge.sink_kind) {
      sinkKindsByNode.set(edge.target, edge.sink_kind);
    }
  }

  const count = Math.max(1, nodes.length);
  const positions = new Map();
  nodes.forEach((node, index) => {
    const angle = (2 * Math.PI * index) / count;
    const rank = LATTICE_ORDER.indexOf(String(node.lattice).toUpperCase());
    if (rank < 0) {
      throw new TypeError(`node '${node.id}' carries the unknown lattice '${node.lattice}'`);
    }
    const radius = 18 + 6 * Number(node.hop_depth ?? 0);
    positions.set(node.id, {
      x: radius * Math.cos(angle),
      y: rank * 22,
      z: radius * Math.sin(angle),
    });
  });

  const layerNodes = nodes.map((node) => {
    const sinkKind = node.sink_kind ?? sinkKindsByNode.get(node.id) ?? null;
    const position = positions.get(node.id);
    return {
      id: node.id,
      label: node.label,
      nodeType: node.node_type,
      lattice: String(node.lattice).toUpperCase(),
      latticeRank: LATTICE_ORDER.indexOf(String(node.lattice).toUpperCase()),
      sinkKind,
      isHighRisk: sinkKind !== null && SINK_LATTICE_FLOOR[sinkKind] >= 3,
      isEntry: node.node_type === 'tool_input_param',
      filePath: node.file_path,
      line: node.line,
      position,
    };
  });

  const layerEdges = edges
    .filter((edge) => positions.has(edge.source) && positions.has(edge.target))
    .map((edge) => ({
      id: `${edge.source}->${edge.target}`,
      source: edge.source,
      target: edge.target,
      transformFn: edge.transform_fn,
      sinkKind: edge.sink_kind ?? null,
      hopDepth: edge.hop_depth,
      isSink: Boolean(edge.sink_kind),
    }));

  return {
    nodes: layerNodes,
    edges: layerEdges,
    serverId: requireField(dtg, 'server_id', 'dtg'),
    dtgSha256: requireField(dtg, 'dtg_sha256', 'dtg'),
    highRiskCount: layerNodes.filter((node) => node.isHighRisk).length,
    entryCount: layerNodes.filter((node) => node.isEntry).length,
  };
}

/**
 * Summarise an SMT verification report for the counterexample viewer.
 *
 * @param {object} report the `properties_verified` / `violations` shape the
 *   `aegis-smt-verifier` crate prints
 */
export function buildCounterexampleTable(report) {
  const violations = Array.isArray(report.violations) ? report.violations : [];
  return {
    safe: violations.length === 0 && Array.isArray(report.properties_verified),
    properties: (report.properties_verified ?? []).map((property) => ({
      property,
      status: 'VERIFIED',
    })),
    counterexamples: violations.map((violation) => ({
      property: violation.property,
      chain: violation.chain ?? [],
      source: violation.trace ? `${violation.trace.source_tool} (${violation.trace.source_lattice})` : '-',
      sink: violation.trace ? `${violation.trace.sink_tool} (${violation.trace.target_clearance})` : '-',
      steps: (violation.trace?.witness_steps ?? []).map((step) => ({
        index: step.step,
        action: step.action,
        serverId: step.server_id,
        lattice: step.lattice_level ?? '-',
      })),
    })),
  };
}

/**
 * The step-up approval modal's state machine.
 *
 * Kept here rather than inside the component so it can be tested: the rule that
 * matters is that the confirmation button is disabled until the operator has
 * read the transition, and that approving is impossible once the countdown
 * reaches zero. Both are properties of a transition function, not of markup.
 *
 * @param {object} request `{datumHash, sourceServer, targetServer,
 *   sourceLevel, targetLevel, ttlSeconds, issuedAt}`
 */
export function createApprovalModal(request) {
  const binding = [
    request.datumHash,
    request.sourceServer,
    request.targetServer,
    request.sourceLevel,
    request.targetLevel,
  ].join(' | ');

  let acknowledged = false;
  let resolved = null;

  return {
    binding,
    /** Seconds before the deadline, never negative. */
    secondsRemaining(nowMs) {
      const elapsed = (nowMs - request.issuedAt) / 1000;
      return Math.max(0, Math.ceil(request.ttlSeconds - elapsed));
    },
    /** The operator has read the transition. Idempotent. */
    acknowledge() {
      acknowledged = true;
      return this;
    },
    isAcknowledged() {
      return acknowledged;
    },
    canApprove(nowMs) {
      return acknowledged && resolved === null && this.secondsRemaining(nowMs) > 0;
    },
    /** Approve, or record the refusal and why. */
    approve(nowMs) {
      if (resolved !== null) {
        return resolved;
      }
      if (!acknowledged) {
        resolved = { outcome: 'DENIED', reason: 'not_acknowledged' };
      } else if (this.secondsRemaining(nowMs) <= 0) {
        resolved = { outcome: 'DENIED', reason: 'deadline_expired' };
      } else {
        resolved = { outcome: 'DECLASSIFIED_AND_ALLOWED', reason: 'step_up_verified', binding };
      }
      return resolved;
    },
    outcome() {
      return resolved;
    },
    isTerminal() {
      return resolved !== null;
    },
  };
}