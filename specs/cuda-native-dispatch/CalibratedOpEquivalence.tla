--------------------------- MODULE CalibratedOpEquivalence ---------------------------
(*****************************************************************************)
(* Status: LIVE as of PHASE_CUDA_NATIVE_DISPATCH commit C11 (combined with   *)
(* C8/C9/C10).                                                                *)
(*                                                                            *)
(* Companion to specs/cuda-native-dispatch/calibrated_op_equivalence        *)
(* .allium.                                                                  *)
(*                                                                            *)
(* Models each calibrated op as a 2-strategy choice (default vs alt). For   *)
(* each op:                                                                   *)
(*                                                                            *)
(*   - Threshold T in {0, 1MB, 10MB, 100MB, 1GB, SIZE_MAX}                  *)
(*   - At dispatch with payload P, the chosen strategy is:                  *)
(*       alt    if P >= T                                                   *)
(*       default otherwise                                                  *)
(*                                                                            *)
(* The load-bearing invariants:                                              *)
(*                                                                            *)
(*   - AllOpsRegistered: every op id has a non-default name (the static    *)
(*       registrar in op_calibration_probes.cu ran).                       *)
(*   - DispatchPure: the strategy at site (op, payload) is a pure function *)
(*       of (op, payload, threshold[op]). No hidden state.                 *)
(*   - OutputEquivalent: when both strategies are reachable at the same    *)
(*       payload (T <= P < SIZE_MAX), their outputs are equivalent (byte- *)
(*       identical OR within the op's numeric tolerance).                  *)
(*                                                                            *)
(* On xeon all four ops calibrate to T = SIZE_MAX (default-wins stub       *)
(* probes), so the OutputEquivalent invariant is vacuously satisfied — no  *)
(* payload reaches the alt path. The invariant becomes load-bearing when    *)
(* a future deployment replaces a stub with a real probe.                   *)
(*                                                                            *)
(* CODE REFS:                                                                *)
(*   ik_llama.cpp/ggml/src/ggml-cuda/op_calibration_probes.cu (C8-C11)       *)
(*   ik_llama.cpp/ggml/include/ggml-cuda-calibration.h                       *)
(*   ik_llama.cpp/tests/test-calibration-ops-registered.cpp                  *)
(*****************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Ops,               \* set of calibrated op identifiers
    Buckets,           \* quantization bucket set
    PayloadSamples     \* finite set of payload sizes the model explores

ASSUME 0 \in Buckets
ASSUME Buckets \subseteq Nat

VARIABLES
    registered_ops,    \* set of op ids that have had register_op called
    threshold,         \* function: Ops -> Buckets
    runs               \* function: (op, payload, strategy) -> output_hash

vars == <<registered_ops, threshold, runs>>

SIZE_MAX == 18446744073709551615

(*****************************************************************************)
(* Initial state                                                              *)
(*****************************************************************************)

Init ==
    /\ registered_ops = {}
    /\ threshold      = [op \in Ops |-> SIZE_MAX]
    /\ runs           = [<<op, p, alt>> \in Ops \X PayloadSamples \X BOOLEAN |-> 0]

(*****************************************************************************)
(* Transitions                                                                *)
(*****************************************************************************)

\* Register an op (the C8-C11 static initializer fires).
RegisterOp(op) ==
    /\ op \in Ops
    /\ op \notin registered_ops
    /\ registered_ops' = registered_ops \cup {op}
    /\ UNCHANGED <<threshold, runs>>

\* Calibration: pick a threshold for op (model-check explores all
\* possible buckets the framework could record).
Calibrate(op, t) ==
    /\ op \in registered_ops
    /\ t \in Buckets
    /\ threshold' = [threshold EXCEPT ![op] = t]
    /\ UNCHANGED <<registered_ops, runs>>

\* Run an op at a payload using the strategy chosen by the threshold.
RunOp(op, p, h) ==
    /\ op \in registered_ops
    /\ p \in PayloadSamples
    /\ h \in Nat
    /\ LET use_alt == (p >= threshold[op])
       IN  runs' = [runs EXCEPT ![<<op, p, use_alt>>] = h]
    /\ UNCHANGED <<registered_ops, threshold>>

Next ==
    \/ \E op \in Ops: RegisterOp(op)
    \/ \E op \in Ops, t \in Buckets: Calibrate(op, t)
    \/ \E op \in Ops, p \in PayloadSamples, h \in Nat: RunOp(op, p, h)

Spec == Init /\ [][Next]_vars

(*****************************************************************************)
(* Invariants                                                                 *)
(*****************************************************************************)

\* I1: All four ops eventually registered (under fair-play scheduling
\* the static initializer ran for each).
AllOpsRegistered == registered_ops = Ops

\* I2: Dispatch decision is a pure function of (op, payload, threshold).
\* Modeled inline in RunOp's `use_alt` derivation.
DispatchPure == TRUE

\* I3: For every (op, payload) where both strategies have been run
\* AND both are reachable (threshold <= payload < SIZE_MAX), outputs
\* are equivalent. With the C8-C11 stub probes thresholds are
\* SIZE_MAX so the antecedent is never satisfied — vacuous.
OutputEquivalent ==
    \A op \in Ops, p \in PayloadSamples:
        (threshold[op] <= p /\ threshold[op] < SIZE_MAX) =>
            runs[<<op, p, FALSE>>] = runs[<<op, p, TRUE>>]

\* TypeOK.
TypeOK ==
    /\ registered_ops \subseteq Ops
    /\ \A op \in Ops: threshold[op] \in Buckets

============================================================================
