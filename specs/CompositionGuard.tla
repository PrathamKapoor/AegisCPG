--------------------------- MODULE CompositionGuard ---------------------------
(*****************************************************************************)
(* Author:  Pratham Kapoor <prathamkapoor027@gmail.com>                       *)
(* Repo:    https://github.com/PrathamKapoor                                 *)
(*                                                                            *)
(* WHAT: The single-datum composition guard of AegisCPG as a tiny, runnable    *)
(*       TLA+ module: pure operators over a direct (level, clearance,          *)
(*       declassified, crossServer) tuple, with no state and no transitions.   *)
(*                                                                            *)
(* WHY:  The verifier exists in three independent readings that must agree:    *)
(*       the pure-Rust decision procedure (src/verifier.rs), the Z3 encoding   *)
(*       (src/smt_encoder.rs), and this specification. The two Rust readings   *)
(*       share the bounded chain enumeration by construction; this module      *)
(*       shares nothing with either, so it is the one place where the guard    *)
(*       *rule* itself -- not its implementation -- is written down. An        *)
(*       earlier revision of the encoder folded an extra conjunct             *)
(*       (`clearance > 0`) into the Data Confinement clause, which silently    *)
(*       excused every downgrade into a PUBLIC server; a truth-table           *)
(*       comparison of all three (tests/algebraic_properties.rs) is what       *)
(*       catches that class of drift.                                          *)
(*                                                                            *)
(* HOW:  Extends Naturals only. `CanFlow` is the Denning order on              *)
(*       0..3; `CompositionGuardSatisfied` is the gateway's release rule       *)
(*       (src/verifier.rs::information_flow_allowed); the two negated          *)
(*       properties are the shapes the SMT encoder emits as witness clauses;   *)
(*       `SessionSafe` is the composite the script's `any_property_violated`   *)
(*       must negate exactly. Law `GuardEquivalence` states the identity the   *)
(*       three readings are checked against. Evaluated by the native TLA+      *)
(*       engine (src/tla_eval.rs); TLC can load the module as well, since it   *)
(*       needs no configuration beyond the operators themselves.               *)
(*****************************************************************************)

EXTENDS Naturals

-------------------------------------------------------------------------------
(* Denning security lattice L = {PUBLIC(0) < INTERNAL(1) < CONFIDENTIAL(2) <  *)
(* SECRET(3)}; the order is a total chain, so `CanFlow` is `<=`.               *)
(******************************************************************************)
PublicLevel       == 0
InternalLevel     == 1
ConfidentialLevel == 2
SecretLevel       == 3

CanFlow(level, clearance) ==
    level <= clearance

-------------------------------------------------------------------------------
(* The release rule: a datum may leave for a destination when the destination   *)
(* is cleared for it, or when the operator approved this exact release with a   *)
(* hardware (FIDO2) declassification. Identical to the Rust function            *)
(* `information_flow_allowed(level, clearance, declassified)`.                  *)
(******************************************************************************)
CompositionGuardSatisfied(level, clearance, declassified) ==
    declassified \/ CanFlow(level, clearance)

-------------------------------------------------------------------------------
(* PROPERTY 2 -- Data Confinement, negated. `true` exactly when the qualifier   *)
(* test of the SMT encoder emits a witness clause for a (source, sink) pair.    *)
(******************************************************************************)
DataConfinementClause(level, clearance, declassified) ==
    ~declassified /\ ~CanFlow(level, clearance)

DataConfinementViolated(level, clearance, declassified) ==
    DataConfinementClause(level, clearance, declassified)

-------------------------------------------------------------------------------
(* PROPERTY 4 -- Context Isolation, negated. A datum at or above              *)
(* CONFIDENTIAL reaching a *different* server cleared below its level, with no *)
(* declassification, is the Cross-Server Context Bleed (TV9).                   *)
(******************************************************************************)
RequiresStepUp(level) ==
    level >= ConfidentialLevel

ContextIsolationViolated(level, clearance, declassified, crossServer) ==
    /\ ~declassified
    /\ crossServer
    /\ RequiresStepUp(level)
    /\ ~CanFlow(level, clearance)

-------------------------------------------------------------------------------
(* The composite a session must not enter: some negated property holds.        *)
(* `ContextIsolationViolated => DataConfinementViolated` for every tuple, so   *)
(* this is the disjunction the encoder's `any_property_violated` must match.    *)
(******************************************************************************)
SessionSafe(level, clearance, declassified, crossServer) ==
    ~(DataConfinementViolated(level, clearance, declassified)
        \/ ContextIsolationViolated(level, clearance, declassified, crossServer))

-------------------------------------------------------------------------------
(* The equivalence the three readings are tested against: the session is safe   *)
(* from a guard perspective exactly when the release rule is satisfied.         *)
(* Holds for every (level, clearance, declassified) -- including                *)
(* INTERNAL -> PUBLIC, where the old `clearance > 0` guard split the readings. *)
(******************************************************************************)
GuardEquivalence(level, clearance, declassified) ==
    CompositionGuardSatisfied(level, clearance, declassified)
        = SessionSafe(level, clearance, declassified, TRUE)

=============================================================================
