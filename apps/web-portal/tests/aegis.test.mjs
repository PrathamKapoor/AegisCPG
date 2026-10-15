/**
 * AegisCPG web portal -- tests for the view-model layer.
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *     Exercises the pure transformations in `src/lib/aegis.mjs` against the
 *     *real* artefacts the toolchain produces: the benchmark summary this run
 *     wrote, and DTG documents from `aegis-cli scan --json`.
 *
 * WHY THESE RUN WITHOUT A PACKAGE INSTALL
 *     `node --test` is built into Node 18+, and this host has no npm registry
 *     access. Writing the tested layer as dependency-free ES modules means the
 *     portal's logic is verified here even though React, deck.gl and Next.js --
 *     which cannot be installed offline -- are not. What is not executed is
 *     stated as such rather than implied to be covered.
 *
 * HOW
 *     Fixtures are produced by the Rust and Python suites, so a green run means
 *     the portal renders numbers that the producers actually emitted. A field
 *     rename in either producer fails these tests rather than showing an empty
 *     cell.
 *
 * Run: node --test apps/web-portal/tests
 */

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import {
  LATTICE_ORDER,
  SCAN_SCHEMA,
  buildCounterexampleTable,
  buildGraphLayers,
  createApprovalModal,
  readSummaryMetrics,
} from '../src/lib/aegis.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..', '..', '..');
const SUMMARY_PATH = join(REPO, 'benchmarks', 'results', 'summary_metrics.json');
const LAB_DIR = join(REPO, 'benchmarks', 'fixtures', 'vulnerable-lab');

function findCli() {
  for (const profile of ['release', 'debug']) {
    const candidate = join(REPO, 'target', profile, process.platform === 'win32' ? 'aegis-cli.exe' : 'aegis-cli');
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

const CLI = findCli();
const summary = existsSync(SUMMARY_PATH) ? JSON.parse(readFileSync(SUMMARY_PATH, 'utf8')) : null;

function scan(fixture) {
  assert.ok(CLI, 'aegis-cli is not built; run `cargo build -p aegis-cli`');
  const stdout = execFileSync(CLI, ['scan', '--path', join(LAB_DIR, fixture), '--json'], {
    encoding: 'utf8',
    timeout: 120_000,
  });
  const document = JSON.parse(stdout);
  assert.equal(document.schema, SCAN_SCHEMA);
  return document.servers[0];
}

describe('readSummaryMetrics', () => {
  it('reads the document this repository actually produced', { skip: summary ? false : 'no summary_metrics.json; run the benchmark harness' }, () => {
    const view = readSummaryMetrics(summary);
    assert.equal(view.schema, 'aegis.benchmarks.v1');
    assert.ok(view.cards.length >= 7, `expected at least seven cards, got ${view.cards.length}`);
    for (const card of view.cards) {
      assert.ok(card.label.length > 0, 'a card has no label');
      assert.ok(card.value.length > 0, `card '${card.id}' has no value`);
      assert.ok(['good', 'bad'].includes(card.severity), `card '${card.id}' severity`);
    }
  });

  it('marks every card good when the producer reported no threshold failure', { skip: summary ? false : 'no summary' }, () => {
    const view = readSummaryMetrics(summary);
    assert.deepEqual(summary.threshold_failures, [], 'the producer itself reported failures');
    assert.equal(
      view.cards.filter((card) => card.severity === 'bad').length,
      0,
      'a card disagrees with the producer',
    );
    assert.equal(view.healthy, true);
  });

  it('rejects a document from a different producer version', () => {
    assert.throws(
      () => readSummaryMetrics({ ...summary, schema: 'aegis.benchmarks.v0' }),
      /not the one this portal renders/,
    );
  });

  it('raises on a missing required section rather than rendering undefined', () => {
    const broken = structuredClone(summary ?? { schema: 'aegis.benchmarks.v1' });
    delete broken.static;
    assert.throws(() => readSummaryMetrics(broken), /required field 'static' is missing/);
  });

  it('surfaces a producer threshold failure as an unhealthy view', () => {
    const failing = structuredClone(summary);
    failing.threshold_failures = ['compression below threshold'];
    const view = readSummaryMetrics(failing);
    assert.equal(view.healthy, false);
    assert.equal(view.failures.length, 1);
  });
});

describe('buildGraphLayers', () => {
  it('lifts sink kinds from edges onto the nodes they terminate', { skip: CLI ? false : 'aegis-cli not built' }, () => {
    // The wire format puts the sink on the edge; a builder that only reads
    // nodes reports no sinks at all, which is what "clean" looks like.
    const server = scan('py_filesystem_workspace.py');
    const layers = buildGraphLayers(server);
    assert.ok(layers.nodes.length > 0);
    assert.ok(layers.highRiskCount > 0, 'no high-risk node was recovered from the edges');
    assert.ok(
      layers.nodes.some((node) => node.sinkKind === 'CommandExec'),
      'CommandExec was not recovered from the edge',
    );
  });

  it('places lattice level on the Y axis and hop depth on the radius', { skip: CLI ? false : 'aegis-cli not built' }, () => {
    const layers = buildGraphLayers(scan('js_secrets_pii.js'));
    const ranks = new Set();
    for (const node of layers.nodes) {
      assert.equal(node.position.y, node.latticeRank * 22);
      assert.equal(node.latticeRank, LATTICE_ORDER.indexOf(node.lattice));
      assert.ok(Number.isFinite(node.position.x) && Number.isFinite(node.position.z));
      ranks.add(node.latticeRank);
    }
    assert.ok(ranks.size >= 2, `expected more than one lattice level, got ${[...ranks]}`);
  });

  it('is deterministic across calls', { skip: CLI ? false : 'aegis-cli not built' }, () => {
    const server = scan('js_malicious_tools.js');
    const first = buildGraphLayers(server);
    const second = buildGraphLayers(server);
    assert.deepEqual(first, second, 'the layout must be stable so two screenshots are comparable');
  });

  it('identifies the tool entry point', { skip: CLI ? false : 'aegis-cli not built' }, () => {
    const layers = buildGraphLayers(scan('py_wikipedia_streamable.py'));
    assert.equal(layers.entryCount, 1);
    assert.ok(layers.nodes.some((node) => node.isEntry && node.nodeType === 'tool_input_param'));
  });

  it('rejects an unknown lattice level rather than guessing a rank', () => {
    const broken = {
      dtg: {
        server_id: 'x',
        dtg_sha256: '0'.repeat(64),
        nodes: [{ id: 'n0', label: 'n', node_type: 'local_buffer', lattice: 'TOP_SECRET', file_path: 'a.py', line: 1 }],
        edges: [],
        detected_sinks: [],
      },
    };
    assert.throws(() => buildGraphLayers(broken), /unknown lattice/);
  });

  it('drops edges that reference nodes the graph does not contain', () => {
    const server = {
      dtg: {
        server_id: 'x',
        dtg_sha256: '0'.repeat(64),
        nodes: [{ id: 'n0', label: 'n', node_type: 'tool_input_param', lattice: 'PUBLIC', file_path: 'a.py', line: 1 }],
        edges: [{ source: 'n0', target: 'ghost', transform_fn: 'call', sink_kind: null, hop_depth: 1 }],
        detected_sinks: [],
      },
    };
    const layers = buildGraphLayers(server);
    assert.equal(layers.edges.length, 0, 'a dangling edge would hang the renderer on a missing node');
  });
});

describe('buildCounterexampleTable', () => {
  it('reports a clean report as safe with no counterexamples', () => {
    const table = buildCounterexampleTable({ properties_verified: ['P1', 'P2'], violations: [] });
    assert.equal(table.safe, true);
    assert.equal(table.counterexamples.length, 0);
    assert.equal(table.properties.length, 2);
  });

  it('renders a constructive witness trace step by step', () => {
    const table = buildCounterexampleTable({
      properties_verified: [],
      violations: [
        {
          property: 'Property 2: Data Confinement',
          chain: ['fs.read_env', 'slack.post'],
          trace: {
            source_tool: 'fs.read_env',
            source_lattice: 'SECRET',
            sink_tool: 'slack.post',
            target_clearance: 'PUBLIC',
            witness_steps: [
              { step: 1, action: 'read', server_id: 'fs', lattice_level: 'SECRET' },
              { step: 2, action: 'post', server_id: 'slack', lattice_level: 'PUBLIC' },
            ],
          },
        },
      ],
    });
    assert.equal(table.safe, false);
    assert.equal(table.counterexamples[0].steps.length, 2);
    assert.equal(table.counterexamples[0].steps[1].serverId, 'slack');
    assert.match(table.counterexamples[0].sink, /PUBLIC/);
  });
});

describe('createApprovalModal', () => {
  const request = {
    datumHash: 'a'.repeat(64),
    sourceServer: 'filesystem',
    targetServer: 'slack',
    sourceLevel: 'SECRET',
    targetLevel: 'PUBLIC',
    ttlSeconds: 300,
    issuedAt: 1_000_000,
  };

  it('cannot be approved before the operator acknowledges the transition', () => {
    const modal = createApprovalModal(request);
    assert.equal(modal.canApprove(request.issuedAt), false);
    const verdict = modal.approve(request.issuedAt);
    assert.equal(verdict.outcome, 'DENIED');
    assert.equal(verdict.reason, 'not_acknowledged');
  });

  it('allows after acknowledgement while the countdown is positive', () => {
    const modal = createApprovalModal(request).acknowledge();
    assert.equal(modal.canApprove(request.issuedAt + 1000), true);
    const verdict = modal.approve(request.issuedAt + 1000);
    assert.equal(verdict.outcome, 'DECLASSIFIED_AND_ALLOWED');
    assert.match(verdict.binding, /a{64}/);
  });

  it('denies once the deadline passes, even after acknowledgement', () => {
    const modal = createApprovalModal(request).acknowledge();
    const after = request.issuedAt + 301_000;
    assert.equal(modal.canApprove(after), false);
    assert.equal(modal.secondsRemaining(after), 0);
    assert.equal(modal.approve(after).reason, 'deadline_expired');
  });

  it('is exactly-once: a second approval returns the first verdict', () => {
    const modal = createApprovalModal(request).acknowledge();
    const first = modal.approve(request.issuedAt);
    const second = modal.approve(request.issuedAt);
    assert.deepEqual(second, first);
    assert.equal(modal.isTerminal(), true);
  });

  it('binds the verdict to the datum hash so a different payload does not match', () => {
    const other = createApprovalModal({ ...request, datumHash: 'b'.repeat(64) });
    assert.notEqual(other.binding, createApprovalModal(request).binding);
  });
});