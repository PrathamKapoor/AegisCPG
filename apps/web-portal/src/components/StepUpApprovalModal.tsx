/**
 * AegisCPG step-up approval modal.
 *
 * Author: Pratham Kapoor <prathamkapoor027@gmail.com>
 * https://github.com/PrathamKapoor
 *
 * WHAT
 *     The FIDO2 approval surface for a cross-domain declassification: shows the
 *     exact transition being authorised, requires the operator to acknowledge it
 *     before the button enables, counts down the deadline, and submits the
 *     assertion to the ML service's `/ml/milp/synthesize` companion endpoint.
 *
 * WHY THE STATE MACHINE IS NOT IN THIS FILE
 *     The rules that matter -- approve is impossible before acknowledgement, and
 *     impossible after the deadline, and a second approval returns the first
 *     verdict -- are properties of a transition function, not of markup. They
 *     live in `lib/aegis.mjs::createApprovalModal` and are unit tested against
 *     the real clock semantics. This component renders that state and posts it.
 *
 * WHY THE DATUM HASH IS SHOWN, NOT THE DATUM
 *     The operator cannot verify SECRET content, but they can verify which
 *     payload they are releasing. The hash is what the assertion is bound to, so
 *     showing it makes the binding legible instead of implicit.
 *
 * HOW
 *     `onResolve` receives the modal's verdict. The caller is responsible for
 *     the actual WebAuthn ceremony; this component never handles a private key.
 */

'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

import { createApprovalModal } from '../lib/aegis.mjs';

const LEVEL_COLOURS = {
  PUBLIC: 'text-slate-300',
  INTERNAL: 'text-sky-300',
  CONFIDENTIAL: 'text-yellow-300',
  SECRET: 'text-rose-400',
};

export default function StepUpApprovalModal({ request, onResolve, onCancel }) {
  const modal = useMemo(() => createApprovalModal(request), [request]);
  const [nowMs, setNowMs] = useState(() => request.issuedAt);
  const [verdict, setVerdict] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    timer.current = setInterval(() => setNowMs(Date.now()), 250);
    return () => clearInterval(timer.current);
  }, []);

  const remaining = modal.secondsRemaining(nowMs);
  const canApprove = modal.canApprove(nowMs);

  function approve() {
    const outcome = modal.approve(Date.now());
    setVerdict(outcome);
    onResolve?.(outcome);
  }

  const expired = remaining <= 0 && verdict === null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Cross-domain declassification approval"
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/80 p-6"
    >
      <div className="w-full max-w-lg rounded-2xl border border-rose-900/60 bg-zinc-900 p-6 shadow-2xl">
        <header className="mb-4">
          <h2 className="text-lg font-bold text-rose-300">Cross-domain release</h2>
          <p className="mt-1 text-xs text-zinc-400">
            This transition lowers the classification of a datum you are not cleared to see. It
            requires a hardware-backed assertion.
          </p>
        </header>

        <dl className="mb-5 space-y-2 rounded-xl bg-zinc-950/70 p-4 text-sm">
          <div className="flex justify-between gap-4">
            <dt className="text-zinc-500">source</dt>
            <dd className="font-mono">
              {request.sourceServer}{' '}
              <span className={LEVEL_COLOURS[request.sourceLevel]}>{request.sourceLevel}</span>
            </dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-zinc-500">target</dt>
            <dd className="font-mono">
              {request.targetServer}{' '}
              <span className={LEVEL_COLOURS[request.targetLevel]}>{request.targetLevel}</span>
            </dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-zinc-500">datum</dt>
            <dd className="font-mono text-xs text-emerald-300" title={request.datumHash}>
              sha256:{request.datumHash.slice(0, 24)}…
            </dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-zinc-500">expires in</dt>
            <dd className={`font-mono ${expired ? 'text-rose-400' : 'text-zinc-300'}`}>
              {expired ? 'expired' : `${remaining}s`}
            </dd>
          </div>
        </dl>

        <p className="mb-4 text-xs text-amber-300/90">
          This action is recorded in the hash-chained audit log with your credential id. A timeout
          denies the release; it never allows it.
        </p>

        {verdict ? (
          <div
            className={`rounded-xl border p-4 text-sm ${
              verdict.outcome === 'DECLASSIFIED_AND_ALLOWED'
                ? 'border-emerald-800 bg-emerald-950/40 text-emerald-300'
                : 'border-rose-800 bg-rose-950/40 text-rose-300'
            }`}
          >
            <div className="font-semibold">{verdict.outcome}</div>
            <div className="mt-1 font-mono text-xs opacity-80">{verdict.reason}</div>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <label className="flex flex-1 items-start gap-2 text-xs text-zinc-400">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={modal.isAcknowledged()}
                onChange={() => modal.acknowledge()}
                disabled={expired}
              />
              I have read the source, target and datum hash above and authorise this release.
            </label>
          </div>
        )}

        <footer className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={() => {
              onCancel?.();
            }}
            className="rounded-lg px-4 py-2 text-sm text-zinc-400 hover:bg-zinc-800"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={approve}
            disabled={!canApprove}
            className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-600"
          >
            Approve with passkey
          </button>
        </footer>
      </div>
    </div>
  );
}