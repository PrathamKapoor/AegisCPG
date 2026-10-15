/**
 * AegisCPG interactive 3D Data Transformation Graph canvas.
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *     Renders one server's compressed DTG as a deck.gl 3D scene: data states as
 *     nodes on the Denning lattice's Y axis, transforming functions as edges,
 *     OS sinks highlighted, and the taint path to a high-risk sink drawn in the
 *     colour of the level it reaches.
 *
 * WHY THE LAYOUT IS NOT FORCE-DIRECTED
 *     The position of a node carries meaning here: Y is the lattice level, which
 *     is the property an auditor is looking for. A force simulation would place
 *     a SECRET node wherever the physics settled and destroy the one visual
 *     encoding the chart has. The layout is therefore deterministic and lives in
 *     `lib/aegis.mjs::buildGraphLayers`, which is unit tested.
 *
 * WHY DECK.GL AND NOT THREE.JS DIRECTLY
 *     deck.gl batches instanced draws, which is what keeps a 500-node graph at
 *     60 fps on the integrated GPU a laptop has. Writing the raw WebGL buffers
 *     would buy nothing here and cost a maintenance burden.
 *
 * HOW
 *     `buildGraphLayers` supplies already-laid-out nodes and edges. This
 *     component only maps them onto `ScatterplotLayer`, `LineLayer` and
 *     `TextLayer`, and owns the interaction state.
 */

'use client';

import { LineLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import DeckGL from '@deck.gl/react';
import { useCallback, useMemo, useState } from 'react';

import { buildGraphLayers, LATTICE_ORDER } from '../lib/aegis.mjs';

/** One colour per lattice level, PUBLIC first. */
const LATTICE_COLOURS = {
  PUBLIC: [148, 163, 184, 220],
  INTERNAL: [56, 189, 248, 220],
  CONFIDENTIAL: [250, 204, 21, 230],
  SECRET: [244, 63, 94, 240],
};

const SINK_COLOUR = [244, 63, 94, 255];
const EDGE_COLOUR = [100, 116, 139, 140];
const SINK_EDGE_COLOUR = [249, 115, 22, 230];

export default function DtgGraphCanvas3D({ server }) {
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);

  const layers = useMemo(() => {
    if (!server) {
      return null;
    }
    // Throws on an unknown lattice level rather than rendering it at rank 0.
    const graph = buildGraphLayers(server);
    const positions = new Map(graph.nodes.map((node) => [node.id, node.position]));

    const nodeLayer = new ScatterplotLayer({
      id: 'dtg-nodes',
      data: graph.nodes,
      getPosition: (node) => [node.position.x, node.position.y, node.position.z],
      getRadius: (node) => (node.isEntry ? 900 : node.isHighRisk ? 700 : 420),
      radiusUnits: 'meters',
      getFillColor: (node) => (node.sinkKind ? SINK_COLOUR : LATTICE_COLOURS[node.lattice]),
      pickable: true,
      onHover: (info) => setHovered(info.object ?? null),
      onClick: (info) => setSelected(info.object ?? null),
      updateTriggers: { getFillColor: [graph.dtgSha256] },
    });

    const edgeLayer = new LineLayer({
      id: 'dtg-edges',
      data: graph.edges,
      getSourcePosition: (edge) => {
        const p = positions.get(edge.source);
        return [p.x, p.y, p.z];
      },
      getTargetPosition: (edge) => {
        const p = positions.get(edge.target);
        return [p.x, p.y, p.z];
      },
      getColor: (edge) => (edge.isSink ? SINK_EDGE_COLOUR : EDGE_COLOUR),
      getWidth: (edge) => (edge.isSink ? 3 : 1),
      widthUnits: 'pixels',
      pickable: true,
    });

    const labelLayer = new TextLayer({
      id: 'dtg-labels',
      data: graph.nodes.filter((node) => node.isEntry || node.isHighRisk),
      getPosition: (node) => [node.position.x, node.position.y + 1400, node.position.z],
      getText: (node) => `${node.label}${node.sinkKind ? ` [${node.sinkKind}]` : ''}`,
      getSize: 13,
      getColor: [226, 232, 240, 235],
      getPixelOffset: [0, -18],
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    });

    return { graph, layers: [edgeLayer, nodeLayer, labelLayer] };
  }, [server]);

  const reset = useCallback(() => setSelected(null), []);

  if (!layers) {
    return (
      <div className="p-8 text-sm text-zinc-500">
        No server selected. Run <code>target/debug/aegis-cli.exe scan --path &lt;DIR&gt; --json</code> to
        produce a document this canvas can render.
      </div>
    );
  }

  const { graph } = layers;

  return (
    <div className="relative" style={{ height: 640 }}>
      <DeckGL
        initialViewState={{
          distance: 180,
          elevation: 0.9,
          bearing: 0.6,
          target: [0, 44, 0],
        }}
        controller={true}
        layers={layers.layers}
        onClick={reset}
      />

      <div className="pointer-events-none absolute left-3 top-3 rounded-lg bg-zinc-950/85 p-3 text-xs">
        <div className="mb-2 font-semibold text-emerald-300">{graph.serverId}</div>
        <div className="text-zinc-400">dtg_sha256 {graph.dtgSha256.slice(0, 16)}…</div>
        <div className="text-zinc-400">
          {graph.nodes.length} data states, {graph.edges.length} transforms
        </div>
        <div className="text-zinc-400">
          {graph.entryCount} tool entr{graph.entryCount === 1 ? 'y' : 'ies'}, {graph.highRiskCount} at a
          high-risk sink
        </div>
        <ul className="mt-2 space-y-0.5">
          {[...LATTICE_ORDER].reverse().map((level, index) => (
            <li key={level} className="flex items-center gap-2">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: `rgba(${LATTICE_COLOURS[level].slice(0, 3).join(',')},1)` }}
              />
              <span className="text-zinc-300">{level}</span>
              <span className="text-zinc-600">y = {(LATTICE_ORDER.length - index - 1) * 22}</span>
            </li>
          ))}
        </ul>
      </div>

      {hovered && (
        <div className="pointer-events-none absolute bottom-3 left-3 max-w-md rounded-lg bg-zinc-950/90 p-3 text-xs">
          <div className="font-semibold text-emerald-300">{hovered.label}</div>
          <div className="text-zinc-400">
            {hovered.nodeType} at {hovered.lattice}
            {hovered.sinkKind ? ` reaching ${hovered.sinkKind}` : ''}
          </div>
          <div className="text-zinc-500">
            {hovered.filePath}:{hovered.line}
          </div>
        </div>
      )}

      {selected && (
        <div className="absolute right-3 top-3 w-72 rounded-lg border border-emerald-900 bg-zinc-950/95 p-4 text-xs">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-semibold text-emerald-300">{selected.label}</span>
            <button type="button" onClick={reset} className="text-zinc-500 hover:text-zinc-200">
              close
            </button>
          </div>
          <dl className="space-y-1 text-zinc-400">
            <div>
              <dt className="inline text-zinc-600">type </dt>
              <dd className="inline">{selected.nodeType}</dd>
            </div>
            <div>
              <dt className="inline text-zinc-600">lattice </dt>
              <dd className="inline">{selected.lattice}</dd>
            </div>
            <div>
              <dt className="inline text-zinc-600">sink </dt>
              <dd className="inline">{selected.sinkKind ?? 'none'}</dd>
            </div>
            <div>
              <dt className="inline text-zinc-600">origin </dt>
              <dd className="inline break-all">
                {selected.filePath}:{selected.line}
              </dd>
            </div>
          </dl>
        </div>
      )}
    </div>
  );
}