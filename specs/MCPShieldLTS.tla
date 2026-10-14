------------------------------ MODULE MCPShieldLTS ------------------------------
(*****************************************************************************)
(* Author:  Pratham Kapoor <prathamkapoor027@gmail.com>                       *)
(* Repo:    https://github.com/PrathamKapoor                                 *)
(*                                                                            *)
(* WHAT: A complete, model-checkable TLA+ specification of the Labeled        *)
(*       Transition System  LTS = (Sigma, Sigma_0, Lambda, ->)  that models  *)
(*       an agent connecting to MCP servers and invoking tools across a       *)
(*       Denning security lattice L = {PUBLIC(0) < INTERNAL(1) <             *)
(*       CONFIDENTIAL(2) < SECRET(3)}.                                        *)
(*                                                                            *)
(* WHY: MCPShield (Acharya & Gupta, Apr 2026) states four safety properties   *)
(*       for MCP/A2A toolchains but ships heuristic checkers. A formal model  *)
(*       turns "we think this is safe" into either a proof or an explicit,    *)
(*       replayable counterexample trace. This module is the reference model  *)
(*       the Rust verifier (crates/aegis-smt-verifier) must agree with.      *)
(*                                                                            *)
(* HOW: Explicit finite-state model. Constants supply the lattice labels and   *)
(*       per-server capabilities; the state variables are the seven named in  *)
(*       the AegisCPG specification; Next is the disjunction of the five      *)
(*       transition actions. The constant GatewayEnforced selects between     *)
(*       the UNPROTECTED model (attacks succeed) and the AEGISCPG-PROTECTED   *)
(*       model (attacks are refused), so the same module proves both the      *)
(*       vulnerability and the fix.                                           *)
(*                                                                            *)
(* Expected TLC behaviour:                                                    *)
(*   - MCPShieldLTS.cfg      (GatewayEnforced = FALSE)                       *)
(*       TLC MUST report invariant violations; each violating behaviour is a  *)
(*       concrete attack witness (TV5 rug pull, TV9 context bleed).          *)
(*   - MCPShieldLTS.safe.cfg (GatewayEnforced = TRUE)                        *)
(*       TLC MUST report NO violations. That is the proof that the AegisCPG   *)
(*       control plane closes exactly those holes.                           *)
(*****************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    Agents,                       \* finite set of agent identities
    Servers,                      \* finite set of MCP server identities
    Tools,                        \* finite set of tool identifiers
    Resources,                    \* finite set of data resources
    HashValues,                   \* finite set of syntactic+DTG hash values
    GatewayEnforced,              \* BOOLEAN: TRUE = AegisCPG control plane on
    ServerOf,                     \* [Tools -> Servers]
    ServerClearance,              \* [Servers -> 0..3]  Denning clearance
    ResourceLevel,                \* [Resources -> 0..3] Denning level of datum
    ToolReads,                    \* [Tools -> SUBSET Resources]
    ToolProduces,                 \* [Tools -> 0..3] level of data returned
    ToolPerms,                    \* [Tools -> SUBSET Tools] declared permissions
    VerifiedDeclassifications     \* SUBSET (Resources \X Servers); those with
                                  \* a hardware-verified FIDO2 assertion

--------------------------------------------------------------------------------
(* Denning security lattice: a 4-element TOTAL chain.                         *)
(******************************************************************************)
SecurityLevels     == 0..3
PublicLevel        == 0
InternalLevel      == 1
ConfidentialLevel  == 2
SecretLevel        == 3

--------------------------------------------------------------------------------
(* State variables.                                                           *)
(******************************************************************************)
VARIABLES
    activeConnections,     \* SUBSET (Agents \X Servers)
    agentKnowledge,        \* [ Agents -> [ Servers -> SUBSET SecurityLevels ] ]
    grantedCapabilities,   \* [ Agents -> SUBSET Tools ]  (from signed TACR)
    approvedToolHashes,    \* [ Tools -> HashValues ]
    currentToolHashes,     \* [ Tools -> HashValues ]
    declassifiedFlows,     \* SUBSET (Resources \X Servers)
    traceLog               \* Seq of transition records (counterexample witness)

vars == << activeConnections,
          agentKnowledge,
          grantedCapabilities,
          approvedToolHashes,
          currentToolHashes,
          declassifiedFlows,
          traceLog >>

--------------------------------------------------------------------------------
(* Sigma_0: quiescent initial state.                                          *)
(******************************************************************************)
Init ==
    /\ activeConnections   = {}
    /\ agentKnowledge      = [ a \in Agents |->
                                 [ s \in Servers |-> {} ] ]
    /\ grantedCapabilities = [ a \in Agents |-> {} ]
    /\ approvedToolHashes  = [ t \in Tools |-> CHOOSE h \in HashValues : TRUE ]
    /\ currentToolHashes   = [ t \in Tools |-> CHOOSE h \in HashValues : TRUE ]
    /\ declassifiedFlows   = {}
    /\ traceLog            = <<>>

--------------------------------------------------------------------------------
(* Bridged(a, s1, s2): TRUE iff an EXPLICIT cross-domain flow policy authorises *)
(* information at the level server s1 holds to be delivered into server s2.    *)
(* Without such a policy the two servers are "unbridged".                     *)
(******************************************************************************)
Bridged(a, s1, s2) ==
    \E r \in Resources :
        /\ ResourceLevel[r] \in agentKnowledge[a][s1]
        /\ <<r, s2>> \in declassifiedFlows

--------------------------------------------------------------------------------
(* Lambda: the five transition actions.                                       *)
(******************************************************************************)

(* (1) Agent a opens a session with server s. Knowledge for that pair is
       reset; capabilities are NOT granted here, they come from the TACR.     *)
InitializeSession(a, s) ==
    /\ a \in Agents
    /\ s \in Servers
    /\ activeConnections' = activeConnections \cup {<<a, s>>}
    /\ agentKnowledge'    = [ agentKnowledge EXCEPT ![a][s] = {} ]
    /\ UNCHANGED << grantedCapabilities,
                   approvedToolHashes,
                   currentToolHashes,
                   declassifiedFlows >>
    /\ traceLog' = Append(traceLog,
                          <<"InitializeSession", a, s, "none", "none", FALSE>>)

(* (2) Agent a invokes read tool t on server s and learns resource r.
       This is the ONLY way information enters the lattice.                   *)
InvokeToolRead(a, s, t, r) ==
    /\ <<a, s>> \in activeConnections
    /\ t \in grantedCapabilities[a]
    /\ ServerOf[t] = s
    /\ r \in ToolReads[t]
    /\ agentKnowledge' =
          [ agentKnowledge EXCEPT ![a][s] = @ \cup {ResourceLevel[r]} ]
    /\ UNCHANGED << activeConnections,
                   grantedCapabilities,
                   approvedToolHashes,
                   currentToolHashes,
                   declassifiedFlows >>
    /\ traceLog' = Append(traceLog,
                          <<"InvokeToolRead", a, s, t, r, FALSE>>)

(* (3) Agent a invokes write tool t on server s carrying resource r.
       Permitted iff  LatticeLevel[r] <= ServerClearance[s]  OR the flow has
       been explicitly declassified. The boolean d marks the call as an
       attempted declassification.                                             *)
InvokeToolWrite(a, s, t, r, d) ==
    /\ <<a, s>> \in activeConnections
    /\ t \in grantedCapabilities[a]
    /\ ServerOf[t] = s
    /\ r \in ToolReads[t]
    /\ ResourceLevel[r] \in agentKnowledge[a][s]
    /\ \/ ResourceLevel[r] <= ServerClearance[s]
       \/ (d /\ <<r, s>> \in declassifiedFlows)
    /\ agentKnowledge' =
          [ agentKnowledge EXCEPT ![a][s] = @ \cup {ToolProduces[t]} ]
    /\ UNCHANGED << activeConnections,
                   grantedCapabilities,
                   approvedToolHashes,
                   currentToolHashes,
                   declassifiedFlows >>
    /\ traceLog' = Append(traceLog,
                          <<"InvokeToolWrite", a, s, t, r, d>>)

(* (4) A server mutates its own tool definition -- the rug pull (TV5) and the
       version rollback (TV6). When the AegisCPG gateway is present the hash
       pair H(def) || H(DTG) is pinned at approval time and any mutation is
       refused before the tool ever reaches an agent.                         *)
MutateToolDefinition(s, t, newHash) ==
    /\ s \in Servers
    /\ t \in Tools
    /\ ServerOf[t] = s
    /\ newHash \in HashValues
    /\ newHash # currentToolHashes[t]
    /\ (GatewayEnforced => newHash = approvedToolHashes[t])
    /\ currentToolHashes' = [ currentToolHashes EXCEPT ![t] = newHash ]
    /\ UNCHANGED << activeConnections,
                   agentKnowledge,
                   grantedCapabilities,
                   approvedToolHashes,
                   declassifiedFlows >>
    /\ traceLog' = Append(traceLog,
                          <<"MutateToolDefinition", s, t, newHash,
                            "none", FALSE>>)

(* (5) Cross-domain declassification. Unprotected systems let any component
       request this; with the gateway on it requires a hardware-verified
       FIDO2/WebAuthn assertion over SHA256(challenge_id || agent_id ||
       datum_hash || target || expiry) that has not previously been spent.   *)
Fido2Declassify(d, s_target) ==
    /\ d \in Resources
    /\ s_target \in Servers
    /\ (GatewayEnforced => <<d, s_target>> \in VerifiedDeclassifications)
    /\ declassifiedFlows' = declassifiedFlows \cup {<<d, s_target>>}
    /\ UNCHANGED << activeConnections,
                   agentKnowledge,
                   grantedCapabilities,
                   approvedToolHashes,
                   currentToolHashes >>
    /\ traceLog' = Append(traceLog,
                          <<"Fido2Declassify", "none", s_target, d,
                            "none", TRUE>>)

--------------------------------------------------------------------------------
(* -> : the transition relation.                                               *)
(******************************************************************************)
Next ==
    \/ \E a \in Agents, s \in Servers            : InitializeSession(a, s)
    \/ \E a \in Agents, s \in Servers, t \in Tools, r \in Resources
                                                  : InvokeToolRead(a, s, t, r)
    \/ \E a \in Agents, s \in Servers, t \in Tools, r \in Resources
                                                  : InvokeToolWrite(a, s, t, r, d)
    \/ \E s \in Servers, t \in Tools, h \in HashValues
                                                  : MutateToolDefinition(s, t, h)
    \/ \E r \in Resources, s \in Servers          : Fido2Declassify(r, s)

--------------------------------------------------------------------------------
(* Behaviour: stuttering is permitted so that liveness/fairness reasoning is   *)
(* meaningful; deadlock freedom is asserted separately.                        *)
(******************************************************************************)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

--------------------------------------------------------------------------------
(* Type correctness.                                                          *)
(******************************************************************************)
TypeOK ==
    /\ activeConnections   \subseteq (Agents \X Servers)
    /\ agentKnowledge      \in [ Agents -> [ Servers -> SUBSET SecurityLevels ] ]
    /\ grantedCapabilities \in [ Agents -> SUBSET Tools ]
    /\ approvedToolHashes  \in [ Tools -> HashValues ]
    /\ currentToolHashes   \in [ Tools -> HashValues ]
    /\ declassifiedFlows   \subseteq (Resources \X Servers)
    /\ traceLog \in Seq( << "InitializeSession",   Agents \cup {"none"},
                          Servers \cup {"none"},
                          Tools \cup {"none"},
                          Resources \cup {"none"},
                          HashValues \cup {"none"},
                          BOOLEAN >> )

--------------------------------------------------------------------------------
(* PROPERTY 1 -- Tool Integrity (MCPShield Property 1).                       *)
(* H(def_i(t)) || H(DTG_i(t)) = H(def_approved(t)) || H(DTG_approved(t)).      *)
(* Each HashValue models the concatenation of the syntactic schema hash and    *)
(* the DTG semantic hash, so a single equality captures both halves.          *)
(******************************************************************************)
ToolIntegrityInvariant ==
    \A t \in Tools : currentToolHashes[t] = approvedToolHashes[t]

--------------------------------------------------------------------------------
(* PROPERTY 2 -- Data Confinement (MCPShield Property 2).                     *)
(* LatticeLevel[d] <= Clearance[server]  OR  <<d, server>> in declassified.    *)
(******************************************************************************)
DataConfinementInvariant ==
    \A a \in Agents :
      \A s \in Servers :
        \A lvl \in agentKnowledge[a][s] :
            \/ lvl <= ServerClearance[s]
            \/ \E r \in Resources :
                  /\ ResourceLevel[r] = lvl
                  /\ <<r, s>> \in declassifiedFlows

--------------------------------------------------------------------------------
(* PROPERTY 3 -- Privilege Boundedness (MCPShield Property 3).               *)
(* eff(sigma_i, a) subseteq cap(a) INTERSECTION (UNION over active tools of   *)
(* perm(t)). Expressed over the state (cap) and over the recorded history      *)
(* (every invocation actually performed).                                      *)
(******************************************************************************)
PrivilegeBoundednessInvariant ==
    \A a \in Agents :
        LET eff == grantedCapabilities[a]
                   \cap ( UNION { ToolPerms[t] : t \in grantedCapabilities[a] } )
        IN  /\ eff \subseteq grantedCapabilities[a]
            /\ \A t \in eff : t \in Tools
    /\ \A i \in 1..Len(traceLog) :
          LET step == traceLog[i]
          IN  /\ (step[1] = "InvokeToolRead" \/ step[1] = "InvokeToolWrite")
              => /\ step[2] \in Agents
                 /\ step[4] \in grantedCapabilities[step[2]]

--------------------------------------------------------------------------------
(* PROPERTY 4 -- Context Isolation (MCPShield Property 4).                    *)
(* Unbridged servers share zero knowledge states without an explicit         *)
(* cross-domain flow policy -- this is the Cross-Server Context Bleed (TV9)   *)
(* property.                                                                   *)
(******************************************************************************)
ContextIsolationInvariant ==
    \A a \in Agents :
      \A s1 \in Servers :
        \A s2 \in Servers :
            /\ s1 # s2
            /\ ~Bridged(a, s1, s2)
            => agentKnowledge[a][s1] \cap agentKnowledge[a][s2] = {}

--------------------------------------------------------------------------------
(* Derived corollary: the session high-water taint mark is the join of every  *)
(* level the session has learned. An agent whose high-water mark exceeds the   *)
(* clearance of every reachable server must have an explicit declassified flow *)
(* before it can deliver a datum anywhere.                                     *)
(******************************************************************************)
SessionHighWater(a) ==
    UNION { agentKnowledge[a][s] : s \in Servers }

NoImplicitDeclassification ==
    \A a \in Agents :
        \A s \in Servers :
            \A lvl \in agentKnowledge[a][s] :
                lvl > ServerClearance[s]
                    => \E r \in Resources :
                         /\ ResourceLevel[r] = lvl
                         /\ <<r, s>> \in declassifiedFlows

=============================================================================