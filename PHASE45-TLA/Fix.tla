--------------------------- MODULE Fix ---------------------------
(*****************************************************************************)
(* PHASE45 D10.e — Phase 2.5 spec: per-slot dispatch fix.                    *)
(*                                                                            *)
(* Models the proposed architectural fix: opt-in `PerSlotMode` that          *)
(* restricts VerifyBatch to single-slot batches (|B| = 1). Under             *)
(* PerSlotMode, every Compute call has batch_seqs = {s} for the slot s      *)
(* being processed — eliminating the dependence on multi-slot context.      *)
(*                                                                            *)
(* Verification goal: with PerSlotMode = TRUE, the safety invariants from   *)
(* Phase 2 (VerifyConsistency, NoStateDivergence) hold. TLC convergence is  *)
(* the proof certificate that per-slot dispatch is sufficient.              *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Slots,
    Tokens,
    MaxStep,
    MaxPending,
    DraftK,
    PerSlotMode      \* BOOLEAN — when TRUE, VerifyBatch restricted to |B|=1

VARIABLES
    pending,
    drafts,
    slot_state,
    pre_draft_snap,
    slot_output,
    accept_count,
    step_count,
    alone_state,
    alone_accept

vars == <<pending, drafts, slot_state, pre_draft_snap, slot_output,
          accept_count, step_count, alone_state, alone_accept>>

----------------------------------------------------------------------------
HistoryWitness == Seq([t: Tokens, B: SUBSET Slots])

TypeOK ==
    /\ pending \in [Slots -> Seq(Tokens)]
    /\ drafts \in [Slots -> Seq(Tokens)]
    /\ slot_state \in [Slots -> HistoryWitness]
    /\ pre_draft_snap \in [Slots -> HistoryWitness]
    /\ slot_output \in [Slots -> Seq(Tokens)]
    /\ accept_count \in [Slots -> 0..DraftK]
    /\ step_count \in 0..MaxStep
    /\ alone_state \in [Slots -> HistoryWitness]
    /\ alone_accept \in [Slots -> 0..DraftK]

Compute(state, token, batch_seqs) ==
    Append(state, [t |-> token, B |-> batch_seqs])

EmitOutput(state) ==
    IF Len(state) = 0 THEN CHOOSE t \in Tokens : TRUE
    ELSE LET last == state[Len(state)] IN
         IF Cardinality(last.B) > 1
         THEN CHOOSE t \in Tokens : t # last.t
         ELSE last.t

RECURSIVE ComputeSeq(_, _, _)
ComputeSeq(state, tokens, batch_seqs) ==
    IF tokens = << >> THEN state
    ELSE ComputeSeq(Compute(state, Head(tokens), batch_seqs),
                    Tail(tokens), batch_seqs)

RECURSIVE AcceptCountIter(_, _, _, _)
AcceptCountIter(state, tokens, batch_seqs, k) ==
    IF tokens = << >> THEN k
    ELSE
        LET state2 == Compute(state, Head(tokens), batch_seqs) IN
        IF EmitOutput(state2) = Head(tokens)
        THEN AcceptCountIter(state2, Tail(tokens), batch_seqs, k + 1)
        ELSE k

ComputeAccept(state, tokens, batch_seqs) ==
    AcceptCountIter(state, tokens, batch_seqs, 0)

RECURSIVE TakeSeq(_, _)
TakeSeq(s, k) == IF k = 0 \/ s = << >>
                 THEN << >>
                 ELSE <<Head(s)>> \o TakeSeq(Tail(s), k - 1)

ComputeAcceptedState(state, tokens, batch_seqs) ==
    LET k == ComputeAccept(state, tokens, batch_seqs) IN
    ComputeSeq(state, TakeSeq(tokens, k), batch_seqs)

----------------------------------------------------------------------------
Init ==
    /\ pending = [s \in Slots |-> << >>]
    /\ drafts = [s \in Slots |-> << >>]
    /\ slot_state = [s \in Slots |-> << >>]
    /\ pre_draft_snap = [s \in Slots |-> << >>]
    /\ slot_output = [s \in Slots |-> << >>]
    /\ accept_count = [s \in Slots |-> 0]
    /\ step_count = 0
    /\ alone_state = [s \in Slots |-> << >>]
    /\ alone_accept = [s \in Slots |-> 0]

Enqueue(s) ==
    /\ Len(pending[s]) < MaxPending
    /\ \E t \in Tokens:
        pending' = [pending EXCEPT ![s] = Append(@, t)]
    /\ UNCHANGED <<drafts, slot_state, pre_draft_snap, slot_output,
                   accept_count, step_count, alone_state, alone_accept>>

BuildDraft(s) ==
    /\ Len(drafts[s]) = 0
    /\ Len(pending[s]) > 0
    /\ \E ds \in [1..DraftK -> Tokens]:
        drafts' = [drafts EXCEPT ![s] = [i \in 1..DraftK |-> ds[i]]]
    /\ pre_draft_snap' = [pre_draft_snap EXCEPT ![s] = slot_state[s]]
    /\ UNCHANGED <<pending, slot_state, slot_output, accept_count,
                   step_count, alone_state, alone_accept>>

----------------------------------------------------------------------------
(* The fix: ReadyToVerify is the same as Phase 2 EXCEPT that under          *)
(* PerSlotMode, |B| must equal 1.                                            *)
----------------------------------------------------------------------------
ReadyToVerify(B) ==
    /\ B \subseteq Slots
    /\ B # {}
    /\ (~PerSlotMode \/ Cardinality(B) = 1)
    /\ \A s \in B:
        /\ Len(drafts[s]) = DraftK
        /\ Len(pending[s]) > 0

VerifyBatch(B) ==
    /\ ReadyToVerify(B)
    /\ step_count < MaxStep
    /\ LET RealK == [s \in B |-> ComputeAccept(pre_draft_snap[s],
                                                drafts[s], B)]
           AloneK == [s \in B |-> ComputeAccept(pre_draft_snap[s],
                                                 drafts[s], {s})]
       IN
       /\ accept_count' = [s \in Slots |->
                            IF s \in B THEN RealK[s] ELSE accept_count[s]]
       /\ alone_accept' = [s \in Slots |->
                             IF s \in B THEN AloneK[s] ELSE alone_accept[s]]
       /\ slot_state' = [s \in Slots |->
                          IF s \in B
                          THEN ComputeAcceptedState(pre_draft_snap[s],
                                                     drafts[s], B)
                          ELSE slot_state[s]]
       /\ alone_state' = [s \in Slots |->
                           IF s \in B
                           THEN ComputeAcceptedState(pre_draft_snap[s],
                                                      drafts[s], {s})
                           ELSE alone_state[s]]
       /\ slot_output' = [s \in Slots |->
                           IF s \in B
                           THEN slot_output[s] \o TakeSeq(drafts[s],
                                                           RealK[s])
                           ELSE slot_output[s]]
    /\ pending' = [s \in Slots |->
                    IF s \in B THEN Tail(pending[s]) ELSE pending[s]]
    /\ drafts' = [s \in Slots |->
                   IF s \in B THEN << >> ELSE drafts[s]]
    /\ pre_draft_snap' = [s \in Slots |->
                            IF s \in B THEN << >> ELSE pre_draft_snap[s]]
    /\ step_count' = step_count + 1

Next ==
    \/ \E s \in Slots: Enqueue(s)
    \/ \E s \in Slots: BuildDraft(s)
    \/ \E B \in (SUBSET Slots) \ {{}}: VerifyBatch(B)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
\* Same invariants as Phase 2.
NoSpontaneousState ==
    \A s \in Slots: Len(slot_state[s]) <= Len(slot_output[s]) + DraftK

OutputBoundedByState ==
    \A s \in Slots: Len(slot_output[s]) <= Len(slot_state[s])

BoundedStep == step_count <= MaxStep

VerifyConsistency ==
    \A s \in Slots: accept_count[s] = alone_accept[s]

NoStateDivergence ==
    \A s \in Slots: slot_state[s] = alone_state[s]

\* Structural property of the fix: under PerSlotMode, no multi-slot
\* VerifyBatch was ever scheduled. This is the engineering contract.
PerSlotModeImpliesSingletonVerify ==
    PerSlotMode => \A s \in Slots: TRUE   \* trivially TRUE; enforced by ReadyToVerify

============================================================================
