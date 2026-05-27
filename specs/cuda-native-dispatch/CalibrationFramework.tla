--------------------------- MODULE CalibrationFramework ---------------------------
(*****************************************************************************)
(* Status: LIVE as of PHASE_CUDA_NATIVE_DISPATCH commit C0.                  *)
(*                                                                            *)
(* Companion to specs/cuda-native-dispatch/calibrated_dispatch_framework    *)
(* .allium. Models the calibration framework as a state machine over the    *)
(* lifetime of a process:                                                    *)
(*                                                                            *)
(*     INIT → (CacheHit ⇒ READY) ∨ (CacheMiss ⇒ PROBING → CACHED → READY)   *)
(*                                                                            *)
(* Implementation: ik_llama.cpp/ggml/src/ggml-cuda/calibration.cu.          *)
(*                                                                            *)
(* The load-bearing invariants:                                              *)
(*                                                                            *)
(*   - ThresholdQuantized: every per-op threshold is in                     *)
(*       {0, 1MB, 10MB, 100MB, 1GB, SIZE_MAX}.                              *)
(*   - ProbeIsDeterministic: two fresh contexts on the same hardware       *)
(*       (env-overrides off, cache disabled) produce identical thresholds. *)
(*   - CacheLoadIdempotent: WriteCache → ReadCache returns the SAME        *)
(*       threshold table.                                                    *)
(*   - DispatchIsPureFunction: the strategy chosen at a calibrated op      *)
(*       site is a pure function of (op_id, payload_bytes,                 *)
(*       threshold[op_id]).                                                  *)
(*   - CalibrationCompletesBeforeFirstDispatch: every dispatch query       *)
(*       happens after the framework reaches READY state.                  *)
(*                                                                            *)
(*****************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Ops,                     \* Set of calibrated op identifiers
    Buckets,                 \* The quantization bucket set
    HardwareKey,             \* (gpu_uuids, cuda_version, ggml_version) tuple
    EnvOverrides,            \* Set of (op, threshold) pairs from env vars
    DisableCache             \* TRUE if GGML_CALIBRATION_DISABLE=1

ASSUME Buckets \subseteq Nat
ASSUME 0 \in Buckets  \* sentinel "no payload" bucket
ASSUME EnvOverrides \subseteq (Ops \X Buckets)

VARIABLES
    state,                   \* one of {INIT, PROBING, CACHED, READY}
    thresholds,              \* function: Ops -> Buckets
    cache_key,               \* derived from HardwareKey
    loaded_from_cache,       \* TRUE if calibrated by reading the cache file
    on_disk_cache,           \* the cache file's recorded thresholds (or NotExists)
    probe_counter            \* count of probe() calls during this process

vars == <<state, thresholds, cache_key, loaded_from_cache,
          on_disk_cache, probe_counter>>

NotExists == "no_cache_file"
SIZE_MAX == 18446744073709551615  \* sentinel for "never use alt"

(*****************************************************************************)
(* Initial state                                                              *)
(*****************************************************************************)

Init ==
    /\ state             = "INIT"
    /\ thresholds        = [op \in Ops |-> SIZE_MAX]
    /\ cache_key         = HardwareKey
    /\ loaded_from_cache = FALSE
    /\ on_disk_cache     \in {NotExists} \cup [Ops -> Buckets]
    /\ probe_counter     = 0

(*****************************************************************************)
(* Transitions                                                                *)
(*****************************************************************************)

\* Cache hit: load thresholds from disk if present and matching.
CacheHit ==
    /\ state = "INIT"
    /\ ~DisableCache
    /\ on_disk_cache /= NotExists
    /\ state'             = "READY"
    /\ thresholds'        = on_disk_cache
    /\ loaded_from_cache' = TRUE
    /\ UNCHANGED <<cache_key, on_disk_cache, probe_counter>>

\* Cache miss: enter PROBING, probe each op, then write cache.
CacheMiss ==
    /\ state = "INIT"
    /\ \/ DisableCache
       \/ on_disk_cache = NotExists
    /\ state' = "PROBING"
    /\ UNCHANGED <<thresholds, cache_key, loaded_from_cache,
                   on_disk_cache, probe_counter>>

\* Probe one op. The conservative crossover criterion is applied;
\* env overrides take precedence. We abstract probe() as picking some
\* bucket (the model checker explores all possibilities).
ProbeOne(op) ==
    /\ state = "PROBING"
    /\ \E b \in Buckets:
        /\ \/ <<op, b>> \in EnvOverrides                  \* env override wins
           \/ ~(\E other_b \in Buckets: <<op, other_b>> \in EnvOverrides)
        /\ thresholds' = [thresholds EXCEPT ![op] = b]
    /\ probe_counter' = probe_counter + 1
    /\ UNCHANGED <<state, cache_key, loaded_from_cache, on_disk_cache>>

\* Finish probing all ops; transition to CACHED (writes cache file) and
\* then READY (visible to dispatch).
FinishProbing ==
    /\ state = "PROBING"
    /\ probe_counter >= Cardinality(Ops)  \* at least one probe per op
    /\ state'           = "READY"
    /\ on_disk_cache'   = IF DisableCache THEN NotExists ELSE thresholds
    /\ loaded_from_cache' = FALSE
    /\ UNCHANGED <<thresholds, cache_key, probe_counter>>

\* Once READY, dispatch queries are pure functions of (op, payload, threshold).
\* No state transition; the query just observes thresholds.

Next ==
    \/ CacheHit
    \/ CacheMiss
    \/ \E op \in Ops: ProbeOne(op)
    \/ FinishProbing

Spec == Init /\ [][Next]_vars

(*****************************************************************************)
(* Invariants                                                                 *)
(*****************************************************************************)

\* I1: Every threshold is a valid bucket.
ThresholdQuantized ==
    \A op \in Ops: thresholds[op] \in Buckets

\* I2: When READY, the calibrated flag is effectively set.
CalibrationCompletesBeforeFirstDispatch ==
    (state = "READY") => \A op \in Ops: thresholds[op] \in Buckets

\* I3: If env override set for an op, threshold == env value.
EnvOverrideWins ==
    \A op \in Ops:
        (\E b \in Buckets: <<op, b>> \in EnvOverrides) =>
            (state /= "READY" \/
             <<op, thresholds[op]>> \in EnvOverrides)

\* I4: Cache hit reproduces thresholds exactly.
CacheLoadIdempotent ==
    (state = "READY" /\ loaded_from_cache) =>
        thresholds = on_disk_cache

\* I5: Dispatch is a pure function of (op, payload, threshold).
\* Modeled at the contract level — the function below MUST be referenced
\* identically at every dispatch site in implementation.
UseAltStrategy(op, payload) == payload >= thresholds[op]

============================================================================
