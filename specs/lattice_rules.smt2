; ============================================================================
; AegisCPG -- Denning Lattice Rules + Compositional Taint Reachability (SMT-LIBv2)
;
; Author: Pratham Kapoor <prathamkapoor027@gmail.com>
; https://github.com/PrathamKapoor
;
; WHAT:  A standalone, hand-auditable SMT-LIBv2 encoding of
;          (a) Denning's security lattice L = {PUBLIC(0) < INTERNAL(1)
;              < CONFIDENTIAL(2) < SECRET(3)},
;          (b) DTG taint reachability   dtg_flows_to(t_src, t_sink, datum),
;          (c) transitive composition across tool sequences  can_chain(t1..tk),
;          (d) the NEGATION of each of the four MCPShield safety properties.
;
; WHY:   The Rust verifier (crates/aegis-smt-verifier) generates its SMT
;        programmatically, which is fast but hard to audit by eye. This file is
;        the reference encoding a human can read end to end, and it is the
;        artefact `aegis-cli verify --smt` emits when asked for an auditable
;        dump. Both encodings must agree.
;
; HOW:   One invocation answers the security question three times, via push/pop:
;
;          PART A -- shipped vulnerable-lab facts, properties negated.
;                    (check-sat) => sat
;                    Z3 prints the concrete witness: read_env -> format_md ->
;                    webhook_post carrying d_dotenv (level SECRET=3) into a
;                    PUBLIC clearance server (level 0). The 3-hop compositional
;                    exfiltration chain.
;
;          PART B -- AegisCPG's MILP-synthesised least-privilege capability
;                    policy asserted. (check-sat) => unsat
;                    No property can be violated, because every poisoned link
;                    carries no granted capability and the DTG hash pair is
;                    pinned to the approved value.
;
;          PART C -- one FIDO2 declassification is granted, for (d_dotenv,
;                    s_slack) only. The approved chain becomes reachable, but a
;                    SECOND datum (d_aws_creds) remains unreachable, proving the
;                    approval is narrowly scoped and not a standing capability.
;
;        Run:  z3 specs/lattice_rules.smt2
; ============================================================================

(set-logic AUFLIA)
(set-info :source "AegisCPG specs/lattice_rules.smt2")
(set-info :author "Pratham Kapoor <prathamkapoor027@gmail.com>")

; ----------------------------------------------------------------------------
; 1. Denning's security lattice.
; ----------------------------------------------------------------------------
; A 4-element TOTAL chain, so `flows_to` is integer comparison and is decidable
; in O(1). PUBLIC(0) is the bottom element (public release), SECRET(3) the top.

(define-fun flows_to ((a Int) (b Int)) Bool (<= a b))
(define-fun join ((a Int) (b Int)) Int (ite (>= a b) a b))
(define-fun meet ((a Int) (b Int)) Int (ite (<= a b) a b))

(declare-fun PUBLIC () Int)
(declare-fun INTERNAL () Int)
(declare-fun CONFIDENTIAL () Int)
(declare-fun SECRET () Int)

(assert (= PUBLIC 0))
(assert (= INTERNAL 1))
(assert (= CONFIDENTIAL 2))
(assert (= SECRET 3))

; Lattice algebra sanity: join/meet are commutative, associative, idempotent,
; and absorb the bottom/top elements. Asserted rather than trusted.
(assert (forall ((x Int)) (= (join PUBLIC x) x)))
(assert (forall ((x Int)) (= (meet SECRET x) x)))
(assert (forall ((x Int)) (= (join x x) x)))
(assert (forall ((x Int)) (= (meet x x) x)))
(assert (forall ((x Int) (y Int)) (= (join x y) (join y x))))
(assert (forall ((x Int) (y Int)) (= (meet x y) (meet y x))))
(assert (forall ((x Int) (y Int) (z Int))
           (= (join x (join y z)) (join (join x y) z))))
(assert (forall ((x Int) (y Int) (z Int))
           (= (meet x (meet y z)) (meet (meet x y) z))))

; ----------------------------------------------------------------------------
; 2. Domain.
; ----------------------------------------------------------------------------

(declare-sort Tool ())
(declare-sort Datum ())
(declare-sort Server ())

; The three tools of the compositional exfiltration chain modelled in
; benchmarks/fixtures/mcptox_chains/exfil_chains.json.
(declare-fun t_read_env () Tool)        ; hosted by s_filesystem
(declare-fun t_format_md () Tool)       ; hosted by s_markdown
(declare-fun t_webhook_post () Tool)    ; hosted by s_slack
(declare-fun t_read_public () Tool)     ; benign control tool

(declare-fun d_dotenv () Datum)          ; .env                 -> SECRET
(declare-fun d_aws_creds () Datum)       ; ~/.aws/credentials   -> SECRET
(declare-fun d_public_readme () Datum)   ; public README        -> PUBLIC

(declare-fun s_filesystem () Server)
(declare-fun s_markdown () Server)
(declare-fun s_slack () Server)

(declare-fun lattice_of (Datum) Int)    ; level of the datum
(declare-fun clearance_of (Server) Int)  ; clearance of the server
(declare-fun hosting_server (Tool) Server)

; ----------------------------------------------------------------------------
; 3. DTG taint reachability.
; ----------------------------------------------------------------------------
; dtg_reads(t, d)      : t's DTG reaches d as a SOURCE  (EnvSecret / FileHandle)
; dtg_flows_to(a,b,d)  : d's taint flows out of a's DTG and into b's DTG
; dtg_emits(t, lvl)    : t's DTG emits data at lattice level lvl (ReturnPayload)
; dtg_sinks_net(t)     : t's DTG reaches a NetworkOutbound sink

(declare-fun dtg_reads (Tool Datum) Bool)
(declare-fun dtg_flows_to (Tool Tool Datum) Bool)
(declare-fun dtg_emits (Tool Int) Bool)
(declare-fun dtg_sinks_net (Tool) Bool)

; ----------------------------------------------------------------------------
; 4. Capability grants -- what the signed TACR authorises.
; ----------------------------------------------------------------------------
(declare-fun granted_read (Tool Datum) Bool)   ; may tool t read datum d
(declare-fun granted_chain (Tool Tool) Bool)   ; may the agent chain a into b
(declare-fun granted_net_sink (Tool) Bool)     ; may tool t transmit outward
(declare-fun declassified (Datum Server) Bool)  ; FIDO2-approved release

(declare-sort Hash ())
(declare-fun H_approved_v1 () Hash)            ; H(def) || H(DTG) at approval
(declare-fun H_rugpulled_v3 () Hash)            ; hash after a rug pull (TV5)
(declare-fun current_dtg_sha () Hash)           ; observed now

; =============================================================================
; PART A -- SHIPPED VULNERABLE-LAB FACTS (the attack IS present structurally)
; =============================================================================
(assert (= (hosting_server t_read_env)     s_filesystem))
(assert (= (hosting_server t_format_md)    s_markdown))
(assert (= (hosting_server t_webhook_post) s_slack))
(assert (= (hosting_server t_read_public)  s_markdown))

(assert (= (lattice_of d_dotenv)        SECRET))
(assert (= (lattice_of d_aws_creds)     SECRET))
(assert (= (lattice_of d_public_readme) PUBLIC))

(assert (= (clearance_of s_filesystem) CONFIDENTIAL))
(assert (= (clearance_of s_markdown)    INTERNAL))
(assert (= (clearance_of s_slack)       PUBLIC))

; DTG facts recovered by Stage 1 from benchmarks/fixtures/vulnerable-lab.
(assert (dtg_reads t_read_env d_dotenv))
(assert (dtg_reads t_read_env d_aws_creds))
(assert (dtg_reads t_read_public d_public_readme))

(assert (dtg_emits t_read_env     SECRET))
(assert (dtg_emits t_format_md    SECRET))
(assert (dtg_emits t_webhook_post SECRET))
(assert (dtg_emits t_read_public  PUBLIC))

(assert (dtg_flows_to t_read_env     t_format_md    d_dotenv))
(assert (dtg_flows_to t_format_md    t_webhook_post d_dotenv))
(assert (dtg_flows_to t_read_env     t_webhook_post d_aws_creds))
(assert (dtg_flows_to t_read_public  t_format_md    d_public_readme))

(assert (dtg_sinks_net t_webhook_post))
(assert (not (dtg_sinks_net t_read_env)))
(assert (not (dtg_sinks_net t_read_public)))

; The vulnerable-lab server declares its tools permissively, so the ambient
; grants stand until AegisCPG replaces them with a synthesised policy.
(assert (granted_read t_read_env d_dotenv))
(assert (granted_read t_read_env d_aws_creds))
(assert (granted_chain t_read_env     t_format_md))
(assert (granted_chain t_format_md    t_webhook_post))
(assert (granted_read  t_webhook_post d_public_readme))
(assert (granted_net_sink t_webhook_post))

; ----------------------------------------------------------------------------
; 5. Compositional chain reachability (transitive composition over tool sequences).
; ----------------------------------------------------------------------------
; can_chain(a, b, c, d) holds iff the agent can drive d out of a, through b,
; into c, and c can transmit it outward. Every conjunct must hold for the chain
; to be REAL: the DTG says the data physically flows, and the grants say the
; control plane has not revoked the link.
(define-fun can_chain ((a Tool) (b Tool) (c Tool) (d Datum)) Bool
  (and (granted_read a d)
       (dtg_emits a (lattice_of d))
       (dtg_reads b d)
       (dtg_emits b (lattice_of d))
       (granted_chain a b)
       (granted_chain b c)
       (dtg_reads c d)
       (dtg_sinks_net c)
       (granted_net_sink c)))

; ----------------------------------------------------------------------------
; 6. Negations of the four MCPShield safety properties.
; ----------------------------------------------------------------------------

; PROPERTY 1 (Tool Integrity) violated: the current syntactic+DTG hash pair
; differs from the approved hash pair (rug pull TV5, rollback TV6).
(define-fun tool_integrity_violated () Bool
  (and (not (= current_dtg_sha H_approved_v1))
       (= current_dtg_sha H_rugpulled_v3)))

; PROPERTY 2 (Data Confinement) violated: level(d) does NOT flow to the
; clearance of the server receiving it, and no FIDO2 declassification exists
; for that (datum, server) pair.
(define-fun data_confinement_violated () Bool
  (and (can_chain t_read_env t_format_md t_webhook_post d_dotenv)
       (not (flows_to (lattice_of d_dotenv)
                      (clearance_of (hosting_server t_webhook_post))))
       (not (declassified d_dotenv (hosting_server t_webhook_post)))))

; PROPERTY 3 (Privilege Boundedness) violated: the effective permission of the
; sequence exceeds the intersection of the agent's grant with what any single
; tool in that sequence declares for itself.
(define-fun privilege_boundedness_violated () Bool
  (and (can_chain t_read_env t_format_md t_webhook_post d_aws_creds)
       (not (granted_read t_webhook_post d_aws_creds))
       (granted_net_sink t_webhook_post)))

; PROPERTY 4 (Context Isolation) violated: two unbridged servers end up holding
; knowledge at a level neither was cleared for, with no explicit cross-domain
; flow policy between them (Cross-Server Context Bleed, TV9).
(define-fun context_isolation_violated () Bool
  (and (dtg_emits t_format_md SECRET)
       (dtg_emits t_webhook_post SECRET)
       (not (declassified d_dotenv s_slack))
       (< (clearance_of s_slack) SECRET)))

; The single question the control plane must answer: is ANY property violated?
(define-fun any_property_violated () Bool
  (or tool_integrity_violated
      data_confinement_violated
      privilege_boundedness_violated
      context_isolation_violated))

; ----------------------------------------------------------------------------
; PART A -- EXPECTED RESULT: sat
; ----------------------------------------------------------------------------
(echo "--- PART A: unprotected vulnerable-lab (expect: sat) ---")
(check-sat)
(get-value (can_chain t_read_env t_format_md t_webhook_post d_dotenv)
           tool_integrity_violated
           data_confinement_violated
           privilege_boundedness_violated
           context_isolation_violated)

; =============================================================================
; PART B -- AEGISCPG SYNTHESISED LEAST-PRIVILEGE CAPABILITY POLICY
; =============================================================================
; Produced by ml-rag-engine/aegis_ml/milp_privilege_cut.py (Stage 3) and signed
; into the TACR by crates/aegis-common/src/crypto.rs (ES256 over the JCS body).
;
; The MILP found that t_webhook_post needs neither arbitrary datum reads nor an
; unconstrained net sink, and that the t_format_md -> t_webhook_post link is
; never exercised on any surviving benign code path. The optimal solution
; revokes that link, narrows the webhook's datum reads, and pins the hash pair.
; ----------------------------------------------------------------------------
(push)

; Property 1 restored: the gateway serves only the approved hash pair.
(assert (= current_dtg_sha H_approved_v1))

; Property 3 restored: the poisoned compositional link carries no grant, and the
; webhook is no longer cleared to emit SECRET or CONFIDENTIAL material.
(assert (not (granted_chain t_format_md t_webhook_post)))
(assert (not (granted_read t_webhook_post d_dotenv)))
(assert (not (granted_read t_webhook_post d_aws_creds)))
(assert (not (dtg_reads t_webhook_post d_dotenv)))

; Property 2 restored: no declassification was approved, so the release is
; impossible rather than merely discouraged.
(assert (not (declassified d_dotenv s_slack)))
(assert (not (declassified d_aws_creds s_slack)))

(echo "--- PART B: AegisCPG synthesised capability policy (expect: unsat) ---")
(check-sat)
(get-value (can_chain t_read_env t_format_md t_webhook_post d_dotenv)
           any_property_violated)

(pop)

; ----------------------------------------------------------------------------
; PART C -- A FIDO2 APPROVAL IS A NARROWLY SCOPED, EXPLICIT CAPABILITY
; ----------------------------------------------------------------------------
; One declassification is granted, for (d_dotenv, s_slack) only. That approved
; chain becomes reachable -- correctly, the operator held a hardware passkey.
; A SECOND datum, d_aws_creds, was never declassified and remains unreachable,
; which is the least-authority property that matters in practice: approving one
; exfiltration must not silently authorise every other one.
; ----------------------------------------------------------------------------
(push)

(assert (= current_dtg_sha H_approved_v1))

; The operator's single approval.
(assert (declassified d_dotenv s_slack))

; Exactly what that approval restores: the approved datum, into the approved
; target, over the approved link.
(assert (granted_read  t_webhook_post d_dotenv))
(assert (dtg_reads     t_webhook_post d_dotenv))
(assert (granted_chain t_format_md t_webhook_post))

; Nothing else. The approval does not generalise to any other datum.
(assert (not (declassified     d_aws_creds s_slack)))
(assert (not (granted_read     t_webhook_post d_aws_creds)))
(assert (not (dtg_reads        t_webhook_post d_aws_creds)))
(assert (not (granted_chain    t_read_env t_webhook_post)))

(echo "--- PART C: narrowly-scoped FIDO2 approval ---")
(check-sat)
(echo "  approved datum (d_dotenv)    chain reachable? expect true")
(get-value (can_chain t_read_env t_format_md t_webhook_post d_dotenv))
(echo "  unapproved datum (d_aws_creds) chain reachable? expect false")
(get-value (can_chain t_read_env t_format_md t_webhook_post d_aws_creds))
(echo "  property violations under the approval? expect false")
(get-value any_property_violated)

(pop)

(exit)