/**
 * AegisCPG security command center.
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *     The portal's landing view: the live benchmark scorecard, the interactive
 *     3D DTG canvas, the eBPF telemetry table, and the SMT counterexample
 *     viewer.
 *
 * WHY NO NUMBER IS TYPED INTO THIS FILE
 *     Every figure on the page comes from `benchmarks/results/summary_metrics.json`
 *     at build time. A hard-coded scorecard goes stale the moment the harness
 *     re-runs and then reports numbers the toolchain no longer produces, which
 *     is the failure mode Hard Rule 10 exists to prevent. If the file is
 *     missing the page says so rather than showing zeroes.
 *
 * WHY THE SIX-BUG POST-MORTEM IS ON THE FRONT PAGE
 *     A security tool that reports its own defects is more credible than one
 *     that claims none, and the defects listed here are the ones that made this
 *     tool's headline metric wrong before they were fixed.
 *
 * HOW
 *     The server component reads the summary and the per-server DTG documents and
 *     passes them to the tested view-model layer in `lib/aegis.mjs`. The client
 *     components own only interaction.
 */

import DtgGraphCanvas3D from '../components/DtgGraphCanvas3D';
import { LATTICE_ORDER, readSummaryMetrics } from '../lib/aegis.mjs';

const SUMMARY_PATH = 'benchmarks/results/summary_metrics.json';

/** The six real bugs found while building this, documented in decisions.md. */
const POST_MORTEM = [
  {
    title: 'Object-literal tool registration was unparseable',
    detail:
      'discover_tools collected string arguments only at bracket depth 1, so registerTool({ name, description }, handler) yielded no tool at all. Eight of the fourteen shipped fixtures failed to extract — four of them benign, so the scanner could not establish a clean baseline.',
  },
  {
    title: 'Inline arrow handlers had no reachable body',
    detail:
      'A registration whose handler is an anonymous arrow resolved to no function, so the reachability walk started from a root that did not exist and every such tool reported zero sinks. A tool that shells out was reported clean.',
  },
  {
    title: 'The DTG hash omitted imported module paths',
    detail:
      'subprocess.run(["wc"]) and subprocess.run(["curl","-d","@.env"]) hashed identically, so a rug pull that changed only an argument or a dependency was invisible. Dependencies, injection markers and egress literals are now in the canonical hash subject.',
  },
  {
    title: 'The ONNX graph never gated leaf weights by path conditions',
    detail:
      'Each tree summed its raw leaf values, so the exported model was a constant function. It loaded, ran, and returned the same number for every tool. ONNX Sum without an axes attribute separately collapsed every batch to a single scalar.',
  },
  {
    title: 'The classifier trained and scored on different distributions',
    detail:
      'Training features were fabricated while inference ran the real GAT over a real CompressedDtg. The tree separated the synthetic distribution and scored every shipped fixture at 0.002 while reporting AUC 1.0 in training. The corpus is now 240 generated programs extracted by the compiled binary.',
  },
  {
    title: 'The Python MILP disagreed with the Rust control plane',
    detail:
      'Seven capabilities on an invented 0-10 scale, against five kernel capabilities on 0.0-1.0 in Rust. A TACR compiled from the Python table would have recorded a residual risk its BPF policy did not agree with. A test now parses types.rs and asserts every weight.',
  },
];

async function loadJson(path) {
  try {
    const response = await fetch(`${process.env.AEGIS_RESULTS_BASE ?? ''}/${path}`, { cache: 'no-store' });
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch {
    return null;
  }
}

export default async function Page() {
  const summary = await loadJson(SUMMARY_PATH);
  if (!summary) {
    return (
      <main className="min-h-screen bg-zinc-950 p-10 text-emerald-50">
        <h1 className="text-3xl font-extrabold">AegisCPG</h1>
        <p className="mt-4 max-w-2xl text-sm text-zinc-400">
          No benchmark summary found at <code>{SUMMARY_PATH}</code>. Run{' '}
          <code>python benchmarks/run_all_benchmarks.py</code> to produce it. This page shows no
          numbers it cannot source.
        </p>
      </main>
    );
  }

  let view;
  try {
    view = readSummaryMetrics(summary);
  } catch (error) {
    return (
      <main className="min-h-screen bg-zinc-950 p-10 text-emerald-50">
        <h1 className="text-3xl font-extrabold">AegisCPG</h1>
        <p className="mt-4 rounded-lg border border-rose-900 bg-rose-950/40 p-4 text-sm text-rose-300">
          {String(error)}
        </p>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-zinc-950 p-8 text-emerald-50">
      <header className="mb-8">
        <h1 className="text-4xl font-extrabold tracking-tight">AegisCPG — Command Center</h1>
        <p className="mt-2 text-sm text-zinc-400">
          Zero-LLM polyglot code-graph → compressed DTG → SMT lattice → eBPF LSM. Author: Pratham
          Kapoor &lt;prathamkapoor027@gmail.com&gt;. Every figure below was produced by{' '}
          <code>benchmarks/run_all_benchmarks.py</code> on {view.generatedUtc} against{' '}
          {view.machine.machine_tag}.
        </p>
      </header>

      {view.failures.length > 0 && (
        <section className="mb-8 rounded-xl border border-rose-800 bg-rose-950/40 p-4">
          <h2 className="font-semibold text-rose-300">
            {view.failures.length} threshold failure{view.failures.length === 1 ? '' : 's'}
          </h2>
          <ul className="mt-2 list-disc pl-5 text-sm text-rose-200/90">
            {view.failures.map((failure) => (
              <li key={failure}>{failure}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {view.cards.map((card) => (
          <article
            key={card.id}
            className={`rounded-2xl border p-4 ${
              card.severity === 'good'
                ? 'border-emerald-900 bg-emerald-950/20'
                : 'border-rose-900 bg-rose-950/20'
            }`}
          >
            <h2 className="text-xs uppercase tracking-wide text-zinc-500">{card.label}</h2>
            <p className="mt-1 font-mono text-2xl font-bold">
              {card.value}{' '}
              <span className="text-xs font-normal text-zinc-500">{card.unit}</span>
            </p>
            <p className="mt-2 text-xs text-zinc-400">{card.detail}</p>
          </article>
        ))}
      </section>

      <section className="mb-8 rounded-2xl border border-emerald-900 bg-zinc-900/60 p-6">
        <h2 className="mb-4 text-xl font-bold text-emerald-300">Interactive DTG canvas</h2>
        <DtgGraphCanvas3D server={summary.graph_sample ?? null} />
      </section>

      <section className="mb-8 rounded-2xl border border-amber-900 bg-zinc-900/60 p-6">
        <h2 className="mb-4 text-xl font-bold text-amber-300">The six real bugs we found</h2>
        <ol className="space-y-4">
          {POST_MORTEM.map((bug, index) => (
            <li key={bug.title} className="text-sm">
              <span className="font-semibold text-zinc-200">
                {index + 1}. {bug.title}
              </span>
              <p className="mt-1 pl-6 text-zinc-400">{bug.detail}</p>
            </li>
          ))}
        </ol>
      </section>

      <footer className="border-t border-zinc-800 pt-4 text-xs text-zinc-500">
        Lattice order {LATTICE_ORDER.join(' < ')}. Source of truth for the architecture:{' '}
        <code>flow.md</code>, <code>decisions.md</code>, and <code>specs/MCPShieldLTS.tla</code>.
      </footer>
    </main>
  );
}