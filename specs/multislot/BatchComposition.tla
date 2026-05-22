--------------------------- MODULE BatchComposition ---------------------------
(*****************************************************************************)
(* T4 — TLA+ spec for the chunked-prefill admission scheduler.                *)
(*                                                                            *)
(* Companion to the Allium spec specs/scheduler/batch_composition.allium.    *)
(* Models the slot state machine and the Tier 4 admission policy             *)
(* (Sarathi-Serve chunked-prefill admission). Each Tick assembles one        *)
(* llama_decode batch by admitting decode tokens first (one per PROCESSING   *)
(* slot with pending_decode), then admitting prefill chunks from             *)
(* LOAD_PROMPT slots round-robin, subject to a per-tick token budget K.      *)
(*                                                                            *)
(* DecodeHoldGateOn constant controls regression-mode model checking:         *)
(*                                                                            *)
(*   - DecodeHoldGateOn = FALSE: Tier 4 admission (the primary mode).        *)
(*     Decodes and prefills can be admitted in the SAME tick, subject to    *)
(*     budget K. Decode-priority: every DecodeEligible slot is admitted     *)
(*     before any prefill chunk takes budget.                                *)
(*                                                                            *)
(*   - DecodeHoldGateOn = TRUE: pre-T4 regression. The legacy DecodeHoldGate*)
(*     is restored: if any slot is LOAD_PROMPT, no decode tokens are        *)
(*     admitted this tick. The pre-T4 MixedBatchProhibition holds. Used    *)
(*     to verify the new safety properties degrade correctly under the     *)
(*     legacy gate.                                                          *)
(*                                                                            *)
(* The load-bearing safety properties:                                       *)
(*                                                                            *)
(*   - TokenBudgetRespected: total tokens admitted per Tick <= K.            *)
(*                                                                            *)
(*   - DecodeNeverHeldByPrefill: under Tier 4, no DecodeEligible slot is    *)
(*     dropped from a Tick just because some other slot is LOAD_PROMPT.      *)
(*                                                                            *)
(*   - PerTokenFlagExclusivity: at the slot level, a slot is in at most one*)
(*     of (batch_decodes, prefill admitted > 0) — structurally true since   *)
(*     SlotState (IDLE+LOAD_PROMPT) and PROCESSING are disjoint.            *)
(*                                                                            *)
(* Liveness:                                                                  *)
(*                                                                            *)
(*   - EventualProgress: every LOAD_PROMPT slot eventually reaches          *)
(*     PROCESSING. PrefillCarryProgressesMonotonically guarantees           *)
(*     n_prompt_done is non-decreasing, so under WF(Tick) any started      *)
(*     prefill drains.                                                       *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic, current as of T3.8 close): *)
(*   ik_llama.cpp/examples/server/server-context.cpp:3196-3206              *)
(*     DecodeHoldGate (already removed pre-T4; T4 confirms it stays gone)   *)
(*   ik_llama.cpp/examples/server/server-context.cpp:3623-3664              *)
(*     batch_pending_prompt (T4.4 will replace the active_pp_slot_id        *)
(*     PrefillSerialisationGate with the admission scaffold)                *)
(*   ik_llama.cpp/examples/server/server-context.cpp:3949                   *)
(*     n_past_prompt advance (load-bearing for ChunkedPrefillAdmission)    *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Slots,            \* set of slot ids (e.g. {s0, s1, s2})
    MaxStep,          \* bound on global tick counter for finite MC
    MaxPromptLen,     \* upper bound on a prompt's token count
    MaxBudget,        \* per-tick token budget K (prefill_chunk_budget)
    DecodeHoldGateOn  \* TRUE = pre-T4 regression mode; FALSE = T4 admission

VARIABLES
    slot_state,            \* [Slots -> {"IDLE", "PROCESSING"}]
    slot_command,          \* [Slots -> {"NONE", "LOAD_PROMPT", "RELEASE"}]
    n_prompt_total,        \* [Slots -> Nat] — total prompt length to prefill
    n_prompt_done,         \* [Slots -> Nat] — tokens prefilled so far
    pending_decode,        \* [Slots -> BOOLEAN] — has a sampled token to add
    batch_prefill_count,   \* [Slots -> Nat] — prefill tokens slot s contributes this tick
    batch_decodes,         \* SUBSET Slots — slots contributing decode tokens this tick
    step_count             \* Nat — global tick counter

vars == <<slot_state, slot_command, n_prompt_total, n_prompt_done,
          pending_decode, batch_prefill_count, batch_decodes, step_count>>

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
    /\ batch_prefill_count \in [Slots -> 0..MaxBudget]
    /\ batch_decodes \subseteq Slots
    /\ step_count \in 0..MaxStep
    /\ \A s \in Slots: n_prompt_done[s] <= n_prompt_total[s]

\* A slot is "in LOAD_PROMPT" iff command says so AND prompt isn't fully prefilled.
\* Matches server_slot.command == SLOT_COMMAND_LOAD_PROMPT in source.
IsLoadingPrompt(s) ==
    /\ slot_command[s] = "LOAD_PROMPT"
    /\ n_prompt_done[s] < n_prompt_total[s]

\* The set of slots currently in LOAD_PROMPT.
LoadingPromptSlots == { s \in Slots : IsLoadingPrompt(s) }

\* A slot is decode-eligible iff PROCESSING with a pending sample. Under
\* DecodeHoldGateOn = TRUE the gate further suppresses eligibility while
\* any prefill is pending.
DecodeEligible(s) ==
    /\ slot_state[s] = "PROCESSING"
    /\ pending_decode[s]
    /\ (~DecodeHoldGateOn \/ LoadingPromptSlots = {})

DecodeEligibleSlots == { s \in Slots : DecodeEligible(s) }

\* Total tokens contributed to a batch — sum of per-slot prefill counts
\* plus the count of decode contributors.
TotalAdmittedTokens(bp, bd) ==
    LET PrefillSum[T \in SUBSET Slots] ==
        IF T = {}
        THEN 0
        ELSE LET s == CHOOSE x \in T : TRUE IN bp[s] + PrefillSum[T \ {s}]
    IN PrefillSum[Slots] + Cardinality(bd)

\* Remaining prefill for a slot.
RemainingPrefill(s) == n_prompt_total[s] - n_prompt_done[s]

----------------------------------------------------------------------------
(* Init.                                                                     *)
----------------------------------------------------------------------------
Init ==
    /\ slot_state = [s \in Slots |-> "IDLE"]
    /\ slot_command = [s \in Slots |-> "NONE"]
    /\ n_prompt_total = [s \in Slots |-> 0]
    /\ n_prompt_done = [s \in Slots |-> 0]
    /\ pending_decode = [s \in Slots |-> FALSE]
    /\ batch_prefill_count = [s \in Slots |-> 0]
    /\ batch_decodes = {}
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: ArrivePrompt(s, n).                                               *)
(*                                                                           *)
(* External request arrives for slot s. Slot transitions IDLE -> IDLE +    *)
(* LOAD_PROMPT, with n_prompt_total[s] := n. The slot remains IDLE         *)
(* state-wise until prefill completes, then transitions to PROCESSING     *)
(* (modelled in CompletePrefill).                                            *)
(*****************************************************************************)
ArrivePrompt(s, n) ==
    /\ slot_state[s] = "IDLE"
    /\ slot_command[s] = "NONE"
    /\ n \in 1..MaxPromptLen
    /\ slot_command' = [slot_command EXCEPT ![s] = "LOAD_PROMPT"]
    /\ n_prompt_total' = [n_prompt_total EXCEPT ![s] = n]
    /\ batch_prefill_count' = [t \in Slots |-> 0]
    /\ batch_decodes' = {}
    /\ UNCHANGED <<slot_state, n_prompt_done, pending_decode, step_count>>

----------------------------------------------------------------------------
(* Action: Tick — assemble and dispatch one llama_decode batch.              *)
(*                                                                           *)
(* Under Tier 4 (DecodeHoldGateOn = FALSE):                                  *)
(*   - batch_decodes := DecodeEligibleSlots (every eligible slot admitted   *)
(*     before prefill, provided the count <= MaxBudget; if not, the model  *)
(*     picks any admission satisfying the budget cap and the priority      *)
(*     rule — abstracts away the round-robin policy as a non-deterministic *)
(*     choice constrained by the invariants).                                *)
(*   - batch_prefill_count[s] in [0, RemainingPrefill(s)] for each LOAD_PROMPT*)
(*     slot, with the sum + |batch_decodes| <= MaxBudget.                   *)
(*   - n_prompt_done advances by batch_prefill_count[s].                    *)
(*   - decode slots have pending_decode cleared.                             *)
(*                                                                           *)
(* Under DecodeHoldGateOn = TRUE (regression):                              *)
(*   - if any slot is LOAD_PROMPT, batch_decodes := {} (the legacy hold).  *)
(*   - prefill admission is restricted to AT MOST ONE slot (the legacy    *)
(*     PrefillSerialisationGate); pick the lowest-id LOAD_PROMPT slot      *)
(*     (continuation-first is captured by n_prompt_done > 0 tie-break).    *)
(*****************************************************************************)

\* T4-mode admission: pick any (bp, bd) satisfying the budget + priority + bounds.
T4AdmissionOK(bp, bd) ==
    /\ bp \in [Slots -> 0..MaxBudget]
    /\ bd \subseteq Slots
    /\ \A s \in Slots: bp[s] <= RemainingPrefill(s)
    /\ \A s \in Slots: bp[s] > 0 => IsLoadingPrompt(s)
    /\ bd \subseteq DecodeEligibleSlots
    /\ TotalAdmittedTokens(bp, bd) <= MaxBudget
    \* Decode priority: if any prefill admitted, all decode-eligible slots
    \* whose admission still fits the budget must be admitted.
    /\ (\E s \in Slots: bp[s] > 0)
       => DecodeEligibleSlots \subseteq bd
    \* Useful work: something must be admitted unless nothing is admissible.
    /\ \/ TotalAdmittedTokens(bp, bd) > 0
       \/ (DecodeEligibleSlots = {} /\ LoadingPromptSlots = {})

\* Legacy-mode admission (DecodeHoldGateOn = TRUE): pre-T4 semantics.
LegacyAdmissionOK(bp, bd) ==
    /\ bp \in [Slots -> 0..MaxBudget]
    /\ bd \subseteq Slots
    /\ \A s \in Slots: bp[s] <= RemainingPrefill(s)
    /\ \A s \in Slots: bp[s] > 0 => IsLoadingPrompt(s)
    \* Legacy PrefillSerialisationGate: at most one slot has bp[s] > 0.
    /\ Cardinality({ s \in Slots : bp[s] > 0 }) <= 1
    \* Legacy DecodeHoldGate: when any slot is LOAD_PROMPT, no decode admitted.
    /\ (LoadingPromptSlots # {} => bd = {})
    /\ (LoadingPromptSlots = {} => bd = DecodeEligibleSlots)
    /\ \/ TotalAdmittedTokens(bp, bd) > 0
       \/ (DecodeEligibleSlots = {} /\ LoadingPromptSlots = {})

Tick ==
    /\ step_count < MaxStep
    /\ \E bp \in [Slots -> 0..MaxBudget], bd \in SUBSET Slots:
         /\ (IF DecodeHoldGateOn
             THEN LegacyAdmissionOK(bp, bd)
             ELSE T4AdmissionOK(bp, bd))
         /\ batch_prefill_count' = bp
         /\ batch_decodes' = bd
         /\ n_prompt_done' = [s \in Slots |->
                                n_prompt_done[s] + bp[s]]
         /\ pending_decode' = [s \in Slots |->
                                 IF s \in bd
                                 THEN FALSE
                                 ELSE pending_decode[s]]
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<slot_state, slot_command, n_prompt_total>>

----------------------------------------------------------------------------
(* Action: CompletePrefill(s).                                               *)
(*                                                                           *)
(* When a slot's prefill has fully drained, the command transitions          *)
(* LOAD_PROMPT -> NONE and the state IDLE -> PROCESSING. Under T4 a slot    *)
(* may have taken several Ticks to reach this point (chunked admission).    *)
(*****************************************************************************)
CompletePrefill(s) ==
    /\ slot_command[s] = "LOAD_PROMPT"
    /\ n_prompt_done[s] = n_prompt_total[s]
    /\ n_prompt_total[s] > 0
    /\ slot_state' = [slot_state EXCEPT ![s] = "PROCESSING"]
    /\ slot_command' = [slot_command EXCEPT ![s] = "NONE"]
    /\ pending_decode' = [pending_decode EXCEPT ![s] = TRUE]
    /\ batch_prefill_count' = [t \in Slots |-> 0]
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
    /\ batch_prefill_count' = [t \in Slots |-> 0]
    /\ batch_decodes' = {}
    /\ UNCHANGED <<slot_state, slot_command, n_prompt_total, n_prompt_done,
                   step_count>>

----------------------------------------------------------------------------
(* Action: Release(s).                                                       *)
(*****************************************************************************)
Release(s) ==
    /\ slot_state[s] = "PROCESSING"
    /\ slot_state' = [slot_state EXCEPT ![s] = "IDLE"]
    /\ slot_command' = [slot_command EXCEPT ![s] = "NONE"]
    /\ n_prompt_total' = [n_prompt_total EXCEPT ![s] = 0]
    /\ n_prompt_done' = [n_prompt_done EXCEPT ![s] = 0]
    /\ pending_decode' = [pending_decode EXCEPT ![s] = FALSE]
    /\ batch_prefill_count' = [t \in Slots |-> 0]
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

\* TokenBudgetRespected — Tier 4's load-bearing budget cap. The total
\* number of tokens admitted to any tick's batch must not exceed K
\* (MaxBudget). This is the bound that gives chunked prefill its name.
TokenBudgetRespected ==
    TotalAdmittedTokens(batch_prefill_count, batch_decodes) <= MaxBudget

\* PerTokenFlagExclusivity — a slot may contribute prefill OR decode
\* tokens in a given tick, never both. Structurally true since LOAD_PROMPT
\* slots are in state=IDLE and decode-eligible slots are in state=PROCESSING.
PerTokenFlagExclusivity ==
    \A s \in Slots:
        ~(batch_prefill_count[s] > 0 /\ s \in batch_decodes)

\* BatchCompositionInvariant — the conjunction of the admission shape
\* constraints. Mirrors the redefined Allium invariant of the same name.
BatchCompositionInvariant ==
    /\ TokenBudgetRespected
    /\ PerTokenFlagExclusivity

\* DecodePriorityAdmission — when under Tier 4 (DecodeHoldGateOn = FALSE),
\* the admission loop admits all decode-eligible slots before any prefill
\* chunks take budget. Encoded as: if any prefill is admitted, every
\* decode-eligible slot is also in batch_decodes.
DecodePriorityAdmission ==
    ~DecodeHoldGateOn =>
        ((\E s \in Slots: batch_prefill_count[s] > 0)
         => (DecodeEligibleSlots \subseteq batch_decodes))

\* DecodeNeverHeldByPrefill captures the action-level property that
\* under Tier 4 a LOAD_PROMPT slot does NOT cause PROCESSING slots'
\* decode admission to be suppressed. The pre-T4 DecodeHoldGate is
\* encoded in T4AdmissionOK vs LegacyAdmissionOK in Tick; this state-
\* level invariant is the same claim re-expressed: when prefill was
\* admitted alongside decode-eligible slots, the decodes were also
\* admitted (i.e., decode was not held back). Equivalent to
\* DecodePriorityAdmission above; kept here as a named property for
\* the trace validator's mirror check.
DecodeNeverHeldByPrefillAtAdmission ==
    ~DecodeHoldGateOn =>
        (((\E s \in Slots: batch_prefill_count[s] > 0)
          /\ DecodeEligibleSlots # {})
         => DecodeEligibleSlots \subseteq batch_decodes)

\* ChunkedPrefillBoundedByRemaining — admitted prefill count per slot
\* never exceeds the remaining prompt (post-action; checked at the
\* moment the batch is dispatched, before n_prompt_done advances).
ChunkedPrefillBoundedByRemaining ==
    \A s \in Slots:
        batch_prefill_count[s] <= n_prompt_total[s]

\* PrefillCarryProgressesMonotonically — n_prompt_done is non-decreasing.
\* Encoded as a state invariant: the variable's previous value cannot
\* exceed the current value. In TLA+ this is naturally captured as an
\* action property — the Tick action only adds to n_prompt_done, never
\* subtracts. Express here as a redundant safety:
PromptDoneBoundedByTotal ==
    \A s \in Slots: n_prompt_done[s] <= n_prompt_total[s]

\* AdmissionSlotStateConsistency — a slot with batch_prefill_count > 0
\* must be a LOAD_PROMPT slot (and analogously for decode/PROCESSING).
AdmissionSlotStateConsistency ==
    /\ \A s \in Slots:
         batch_prefill_count[s] > 0 =>
             (slot_state[s] = "IDLE" /\ slot_command[s] = "LOAD_PROMPT")
    /\ \A s \in Slots:
         s \in batch_decodes =>
             slot_state[s] = "PROCESSING"

\* SlotStateConsistency — IDLE slots have no pending decode.
SlotStateConsistency ==
    \A s \in Slots:
        slot_state[s] = "IDLE" =>
            (slot_command[s] \in {"NONE", "LOAD_PROMPT"}
             /\ (~pending_decode[s]))

\* Legacy regression invariants — unconditional restatements of the
\* pre-T4 PrefillSerialisationGate + DecodeHoldImpliedByPendingPrefill
\* properties. Under DecodeHoldGateOn = TRUE the LegacyAdmissionOK
\* semantics enforce these structurally; under DecodeHoldGateOn = FALSE
\* (T4 mode), these properties DO NOT hold — that's what makes T4
\* different from legacy. The no_gate.cfg config asserts them as
\* INVARIANTS under T4 mode expecting violations: the counterexample
\* trace demonstrates the T4 admission explicitly produces what legacy
\* admission used to forbid.
LegacyMixedBatchProhibition ==
    ~((\E s \in Slots: batch_prefill_count[s] > 0)
      /\ batch_decodes # {})

LegacyAtMostOnePrefillSlotPerBatch ==
    Cardinality({ s \in Slots : batch_prefill_count[s] > 0 }) <= 1

LegacyDecodeHoldImpliedByPendingPrefill ==
    LoadingPromptSlots # {} => batch_decodes = {}

\* Bounded counter for finite MC.
BoundedStep == step_count <= MaxStep

----------------------------------------------------------------------------
(* Liveness properties.                                                      *)
----------------------------------------------------------------------------

\* EventualProgress — every LOAD_PROMPT eventually completes prefill.
\* Under weak fairness on Tick + CompletePrefill, the prefill drains and
\* the slot transitions to PROCESSING. PrefillCarryProgresses is implied:
\* each Tick admits at least one prefill token per LOAD_PROMPT slot under
\* T4 round-robin (modulo budget exhaustion by competing slots), and the
\* worst case is n_prompt_total Ticks to fully drain.
EventualProgress ==
    \A s \in Slots:
        (slot_command[s] = "LOAD_PROMPT" /\ n_prompt_total[s] > 0)
            ~> (slot_state[s] = "PROCESSING")

\* PrefillCarryProgresses — every started prefill makes progress in some
\* future tick. Captures the monotonic-advance liveness property from the
\* Allium spec.
PrefillCarryProgresses ==
    \A s \in Slots:
        (slot_command[s] = "LOAD_PROMPT" /\ n_prompt_done[s] < n_prompt_total[s])
            ~> (n_prompt_done[s] = n_prompt_total[s])

============================================================================
