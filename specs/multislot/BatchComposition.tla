--------------------------- MODULE BatchComposition ---------------------------
(*****************************************************************************)
(* S2.a — TLA+ spec for the Bug C closure scheduler.                          *)
(*                                                                            *)
(* Companion to the Allium spec specs/scheduler/batch_composition.allium.    *)
(* Models the slot state machine and the two gates that, together, close     *)
(* Bug C in the current shared-pool KV-cache world:                          *)
(*                                                                            *)
(*   - PrefillSerialisationGate (v1's cherry-pick, batch_pending_prompt):    *)
(*       at most ONE slot in LOAD_PROMPT contributes prefill tokens to any  *)
(*       given tick. Choose the slot with n_prompt_tokens_processed > 0     *)
(*       first (continuation), else the lowest-index LOAD_PROMPT slot.      *)
(*       This is a perf policy — parallel multi-slot prefill is ~4.8x       *)
(*       slower per-sequence on this geometry — and a precondition for      *)
(*       the structural Bug C fix below.                                     *)
(*                                                                            *)
(*   - DecodeHoldGate (Bug C fix, add_sampled_tokens):                       *)
(*       if ANY slot is in LOAD_PROMPT, PROCESSING slots may NOT contribute *)
(*       decode tokens this tick. The decode is held until all prefills    *)
(*       drain. This is the missing half of v1; with v1 alone, slot 0's    *)
(*       decode (pos=210) mixes with slot 1's first prefill chunk (pos=0)   *)
(*       in one llama_decode, exercising the mul_mat GEMV-vs-GEMM            *)
(*       accumulation-order divergence that produces Bug C's stochastic    *)
(*       NPC failures.                                                       *)
(*                                                                            *)
(* The load-bearing safety property is MixedBatchProhibition: no batch     *)
(* contains both a prefill token and a decode token. This is what the      *)
(* downstream kernel paths require to remain byte-identical across slots.  *)
(*                                                                            *)
(* The matching liveness property is EventualProgress: every slot in       *)
(* LOAD_PROMPT eventually reaches PROCESSING; every PROCESSING slot       *)
(* eventually reaches IDLE (modulo external arrival of decode work).      *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic):                          *)
(*   ik_llama.cpp/examples/server/server-context.cpp:3194  add_sampled_tokens *)
(*   ik_llama.cpp/examples/server/server-context.cpp:3205-3209  DecodeHoldGate *)
(*   ik_llama.cpp/examples/server/server-context.cpp:3626  batch_pending_prompt *)
(*   ik_llama.cpp/examples/server/server-context.cpp:3644-3681  PrefillSerial. *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Slots,            \* set of slot ids (e.g. {s0, s1, s2})
    MaxStep,          \* bound on global tick counter for finite MC
    MaxPromptLen,     \* upper bound on a prompt's token count
    ChunkSize,        \* prefill chunk per tick (e.g. 512 in production)
    DecodeHoldGateOn  \* TRUE = Bug C closure active; FALSE = control

VARIABLES
    slot_state,       \* [Slots -> {"IDLE", "PROCESSING"}]
    slot_command,     \* [Slots -> {"NONE", "LOAD_PROMPT", "RELEASE"}]
    n_prompt_total,   \* [Slots -> Nat] — total prompt length to prefill
    n_prompt_done,    \* [Slots -> Nat] — tokens prefilled so far
    pending_decode,   \* [Slots -> BOOLEAN] — has a sampled token to add
    batch_prefill,    \* SUBSET Slots — slots whose prefill is in this tick's batch
    batch_decodes,    \* SUBSET Slots — slots contributing decode tokens this tick
    step_count        \* Nat — global tick counter

vars == <<slot_state, slot_command, n_prompt_total, n_prompt_done,
          pending_decode, batch_prefill, batch_decodes, step_count>>

----------------------------------------------------------------------------
(* Types and helpers.                                                        *)
----------------------------------------------------------------------------

SlotState == {"IDLE", "PROCESSING"}
SlotCmd == {"NONE", "LOAD_PROMPT", "RELEASE"}

TypeOK ==
    /\ slot_state \in [Slots -> SlotState]
    /\ slot_command \in [Slots -> SlotCmd]
    /\ n_prompt_total \in [Slots -> 0..MaxPromptLen]
    /\ n_prompt_done \in [Slots -> 0..MaxPromptLen]
    /\ pending_decode \in [Slots -> BOOLEAN]
    /\ batch_prefill \subseteq Slots
    /\ batch_decodes \subseteq Slots
    /\ step_count \in 0..MaxStep
    /\ \A s \in Slots: n_prompt_done[s] <= n_prompt_total[s]

\* A slot is "in LOAD_PROMPT" iff command says so AND prompt isn't fully prefilled.
\* Matches server_slot.command == SLOT_COMMAND_LOAD_PROMPT in source.
IsLoadingPrompt(s) ==
    /\ slot_command[s] = "LOAD_PROMPT"
    /\ n_prompt_done[s] < n_prompt_total[s]

\* A slot is "in continuation" iff it's mid-prefill (some tokens already done).
\* Matches the `n_prompt_tokens_processed > 0` predicate in active_pp_slot_id
\* selection in batch_pending_prompt.
IsPrefillContinuation(s) ==
    /\ IsLoadingPrompt(s)
    /\ n_prompt_done[s] > 0

\* The set of slots currently in LOAD_PROMPT.
LoadingPromptSlots == { s \in Slots : IsLoadingPrompt(s) }

\* The set of slots eligible to contribute a decode this tick.
\* Decode eligibility requires PROCESSING + pending_decode + (gate permits).
DecodeEligible(s) ==
    /\ slot_state[s] = "PROCESSING"
    /\ pending_decode[s]
    /\ (~DecodeHoldGateOn \/ LoadingPromptSlots = {})

DecodeEligibleSlots == { s \in Slots : DecodeEligible(s) }

----------------------------------------------------------------------------
(* The PrefillSerialisationGate.                                             *)
(*                                                                           *)
(* Implements v1's active_pp_slot_id selection: continuation first, else    *)
(* lowest-index LOAD_PROMPT slot. Returns a singleton set (or empty if no  *)
(* slot is LOAD_PROMPT).                                                     *)
(*                                                                           *)
(* The "lowest-index" tie-break uses CHOOSE on Slots; for symmetry-          *)
(* reduction-friendliness we don't depend on a particular tie-break — the *)
(* invariant only requires Cardinality <= 1, not stability.                *)
(*****************************************************************************)
PrefillContributor ==
    LET continuations == { s \in Slots : IsPrefillContinuation(s) } IN
    IF continuations # {}
    THEN { CHOOSE s \in continuations : TRUE }
    ELSE IF LoadingPromptSlots # {}
         THEN { CHOOSE s \in LoadingPromptSlots : TRUE }
         ELSE {}

----------------------------------------------------------------------------
(* Init.                                                                     *)
----------------------------------------------------------------------------
Init ==
    /\ slot_state = [s \in Slots |-> "IDLE"]
    /\ slot_command = [s \in Slots |-> "NONE"]
    /\ n_prompt_total = [s \in Slots |-> 0]
    /\ n_prompt_done = [s \in Slots |-> 0]
    /\ pending_decode = [s \in Slots |-> FALSE]
    /\ batch_prefill = {}
    /\ batch_decodes = {}
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: ArrivePrompt(s, n).                                               *)
(*                                                                           *)
(* External request arrives for slot s. Slot transitions IDLE -> IDLE +    *)
(* LOAD_PROMPT, with n_prompt_total[s] := n. Matches the server's launch  *)
(* path where slot.state stays IDLE and slot.command becomes LOAD_PROMPT.  *)
(* The slot remains IDLE state-wise until prefill completes, then          *)
(* transitions to PROCESSING (modelled in CompletePrefill).                 *)
(*****************************************************************************)
ArrivePrompt(s, n) ==
    /\ slot_state[s] = "IDLE"
    /\ slot_command[s] = "NONE"
    /\ n \in 1..MaxPromptLen
    /\ slot_command' = [slot_command EXCEPT ![s] = "LOAD_PROMPT"]
    /\ n_prompt_total' = [n_prompt_total EXCEPT ![s] = n]
    /\ batch_prefill' = {}    \* batch is dispatched-and-consumed atomically in Tick
    /\ batch_decodes' = {}
    /\ UNCHANGED <<slot_state, n_prompt_done, pending_decode, step_count>>

----------------------------------------------------------------------------
(* Action: Tick — assemble and dispatch one llama_decode batch.              *)
(*                                                                           *)
(* Atomic at this abstraction level. The batch is computed in one step:    *)
(*   - batch_prefill := PrefillContributor (per PrefillSerialisationGate)  *)
(*   - batch_decodes := DecodeEligibleSlots (per DecodeHoldGate)            *)
(*   - The prefill slot's n_prompt_done advances by min(ChunkSize, remaining). *)
(*   - The decode slots have their pending_decode cleared.                  *)
(*   - step_count advances.                                                  *)
(*                                                                           *)
(* When DecodeHoldGateOn = FALSE, the gate is absent and decodes can mix   *)
(* with a prefill in the same tick — this is the pre-Bug-C-closure         *)
(* behaviour, used by the negative-test config to confirm the spec         *)
(* actually binds.                                                           *)
(*****************************************************************************)
Tick ==
    /\ step_count < MaxStep
    \* Something to do this tick:
    /\ PrefillContributor # {} \/ DecodeEligibleSlots # {}
    /\ batch_prefill' = PrefillContributor
    /\ batch_decodes' =
         IF DecodeHoldGateOn /\ LoadingPromptSlots # {}
         THEN {}
         ELSE DecodeEligibleSlots
    /\ n_prompt_done' = [s \in Slots |->
                          IF s \in PrefillContributor
                          THEN LET remaining == n_prompt_total[s] - n_prompt_done[s]
                                   chunk == IF remaining < ChunkSize
                                            THEN remaining
                                            ELSE ChunkSize
                               IN n_prompt_done[s] + chunk
                          ELSE n_prompt_done[s]]
    /\ pending_decode' = [s \in Slots |->
                            IF DecodeHoldGateOn /\ LoadingPromptSlots # {}
                            THEN pending_decode[s]
                            ELSE IF s \in DecodeEligibleSlots
                                 THEN FALSE
                                 ELSE pending_decode[s]]
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<slot_state, slot_command, n_prompt_total>>

----------------------------------------------------------------------------
(* Action: CompletePrefill(s).                                               *)
(*                                                                           *)
(* When a slot's prefill has fully drained (n_prompt_done = n_prompt_total),*)
(* the command transitions LOAD_PROMPT -> NONE and the state IDLE ->       *)
(* PROCESSING. The next Tick may now schedule its first decode.             *)
(*****************************************************************************)
CompletePrefill(s) ==
    /\ slot_command[s] = "LOAD_PROMPT"
    /\ n_prompt_done[s] = n_prompt_total[s]
    /\ n_prompt_total[s] > 0
    /\ slot_state' = [slot_state EXCEPT ![s] = "PROCESSING"]
    /\ slot_command' = [slot_command EXCEPT ![s] = "NONE"]
    /\ pending_decode' = [pending_decode EXCEPT ![s] = TRUE]
    /\ batch_prefill' = {}
    /\ batch_decodes' = {}
    /\ UNCHANGED <<n_prompt_total, n_prompt_done, step_count>>

----------------------------------------------------------------------------
(* Action: ProduceSample(s).                                                 *)
(*                                                                           *)
(* After a successful Tick that consumed slot s's decode contribution, the *)
(* sampler produces the next decode token. This re-arms pending_decode[s]. *)
(*****************************************************************************)
ProduceSample(s) ==
    /\ slot_state[s] = "PROCESSING"
    /\ ~pending_decode[s]
    /\ pending_decode' = [pending_decode EXCEPT ![s] = TRUE]
    /\ batch_prefill' = {}
    /\ batch_decodes' = {}
    /\ UNCHANGED <<slot_state, slot_command, n_prompt_total, n_prompt_done,
                   step_count>>

----------------------------------------------------------------------------
(* Action: Release(s).                                                       *)
(*                                                                           *)
(* External signal: client disconnect or generation complete. Slot         *)
(* transitions PROCESSING -> IDLE; command -> NONE. Required for           *)
(* EventualProgress liveness.                                                *)
(*****************************************************************************)
Release(s) ==
    /\ slot_state[s] = "PROCESSING"
    /\ slot_state' = [slot_state EXCEPT ![s] = "IDLE"]
    /\ slot_command' = [slot_command EXCEPT ![s] = "NONE"]
    /\ n_prompt_total' = [n_prompt_total EXCEPT ![s] = 0]
    /\ n_prompt_done' = [n_prompt_done EXCEPT ![s] = 0]
    /\ pending_decode' = [pending_decode EXCEPT ![s] = FALSE]
    /\ batch_prefill' = {}
    /\ batch_decodes' = {}
    /\ UNCHANGED <<step_count>>

----------------------------------------------------------------------------
(* Next.                                                                     *)
----------------------------------------------------------------------------
Next ==
    \/ \E s \in Slots, n \in 1..MaxPromptLen: ArrivePrompt(s, n)
    \/ Tick
    \/ \E s \in Slots: CompletePrefill(s)
    \/ \E s \in Slots: ProduceSample(s)
    \/ \E s \in Slots: Release(s)

Fairness ==
    /\ WF_vars(Tick)
    /\ \A s \in Slots: WF_vars(CompletePrefill(s))

Spec == Init /\ [][Next]_vars /\ Fairness

----------------------------------------------------------------------------
(* Safety invariants.                                                        *)
----------------------------------------------------------------------------

\* BatchCompositionInvariant — every dispatched batch is either pure-prefill
\* (one slot's tokens, possibly chunked) OR pure-decode (n_tokens in
\* {0, 1, ..., n_active_PROCESSING}). NEVER mixed.
BatchCompositionInvariant ==
    \/ batch_prefill = {} \/ batch_decodes = {}

\* MixedBatchProhibition — the same property at the per-token level. If
\* any prefill token and any decode token both appear in the dispatched
\* batch, the invariant fails. Lifted form of the Allium contract from
\* specs/scheduler/batch_composition.allium.
MixedBatchProhibition ==
    ~(batch_prefill # {} /\ batch_decodes # {})

\* AtMostOnePrefillSlotPerBatch — implements the v1 PrefillSerialisationGate.
\* The PrefillContributor function returns a singleton or empty; this
\* lifts that to a state invariant on the batch composition.
AtMostOnePrefillSlotPerBatch ==
    Cardinality(batch_prefill) <= 1

\* DecodeHoldImpliedByPendingPrefill — when the gate is enabled, the batch
\* is decode-empty whenever any slot is LOAD_PROMPT. The contrapositive of
\* the DecodeHoldGate's early-return.
DecodeHoldImpliedByPendingPrefill ==
    (DecodeHoldGateOn /\ LoadingPromptSlots # {}) => batch_decodes = {}

\* PrefillContinuationPriority — if any slot is mid-prefill, that's the slot
\* whose prefill is in this tick's batch. Mirrors active_pp_slot_id's
\* continuation-first selection.
PrefillContinuationPriority ==
    LET continuations == { s \in Slots : IsPrefillContinuation(s) } IN
    (continuations # {} /\ batch_prefill # {})
        => batch_prefill \subseteq continuations

\* PromptDoneBoundedByTotal — book-keeping safety; redundant with TypeOK
\* but cheap to assert.
PromptDoneBoundedByTotal ==
    \A s \in Slots: n_prompt_done[s] <= n_prompt_total[s]

\* SlotStateConsistency — IDLE slots have no pending decode and no
\* in-progress prefill (n_prompt_done = 0 modulo a not-yet-started
\* LOAD_PROMPT). Catches stale state on transitions.
SlotStateConsistency ==
    \A s \in Slots:
        slot_state[s] = "IDLE" =>
            (slot_command[s] \in {"NONE", "LOAD_PROMPT"}
             /\ (~pending_decode[s]))

\* Bounded counter for finite MC.
BoundedStep == step_count <= MaxStep

----------------------------------------------------------------------------
(* Liveness properties.                                                      *)
----------------------------------------------------------------------------

\* EventualProgress — every LOAD_PROMPT eventually completes prefill.
\* Under weak fairness on Tick + CompletePrefill, the prefill drains and
\* the slot transitions to PROCESSING.
EventualProgress ==
    \A s \in Slots:
        (slot_command[s] = "LOAD_PROMPT" /\ n_prompt_total[s] > 0)
            ~> (slot_state[s] = "PROCESSING")

\* PrefillEventuallyDrains — every started prefill eventually fills
\* completely. Slightly weaker than EventualProgress; useful for
\* diagnosing if liveness fails.
PrefillEventuallyDrains ==
    \A s \in Slots:
        (slot_command[s] = "LOAD_PROMPT" /\ n_prompt_total[s] > 0)
            ~> (n_prompt_done[s] = n_prompt_total[s])

============================================================================
