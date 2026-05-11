---- MODULE DFlashCycle_TTrace_1778528154 ----
EXTENDS Sequences, TLCExt, Toolbox, DFlashCycle, Naturals, TLC

_expression ==
    LET DFlashCycle_TEExpression == INSTANCE DFlashCycle_TEExpression
    IN DFlashCycle_TEExpression!expression
----

_trace ==
    LET DFlashCycle_TETrace == INSTANCE DFlashCycle_TETrace
    IN DFlashCycle_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        draft_kv_injected_n_cells = (0)
        /\
        accept_history = (<<>>)
        /\
        target_kv_n_cells = (1)
        /\
        draft_kv_self_n_cells = (0)
        /\
        in_flight_block = ([present |-> TRUE, anchor_pos |-> 1, n_tokens |-> 3])
        /\
        pc = ("verify")
        /\
        last_n_accepted = (0)
        /\
        step = (0)
        /\
        verify_seq_lens_seen = (0)
        /\
        last_bonus_pos = (0)
        /\
        n_rejected_prev = (0)
        /\
        verify_effective_seen = (0)
        /\
        anchor_pos = (0)
    )
----

_init ==
    /\ accept_history = _TETrace[1].accept_history
    /\ last_bonus_pos = _TETrace[1].last_bonus_pos
    /\ pc = _TETrace[1].pc
    /\ anchor_pos = _TETrace[1].anchor_pos
    /\ draft_kv_injected_n_cells = _TETrace[1].draft_kv_injected_n_cells
    /\ in_flight_block = _TETrace[1].in_flight_block
    /\ n_rejected_prev = _TETrace[1].n_rejected_prev
    /\ step = _TETrace[1].step
    /\ verify_effective_seen = _TETrace[1].verify_effective_seen
    /\ target_kv_n_cells = _TETrace[1].target_kv_n_cells
    /\ last_n_accepted = _TETrace[1].last_n_accepted
    /\ verify_seq_lens_seen = _TETrace[1].verify_seq_lens_seen
    /\ draft_kv_self_n_cells = _TETrace[1].draft_kv_self_n_cells
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ accept_history  = _TETrace[i].accept_history
        /\ accept_history' = _TETrace[j].accept_history
        /\ last_bonus_pos  = _TETrace[i].last_bonus_pos
        /\ last_bonus_pos' = _TETrace[j].last_bonus_pos
        /\ pc  = _TETrace[i].pc
        /\ pc' = _TETrace[j].pc
        /\ anchor_pos  = _TETrace[i].anchor_pos
        /\ anchor_pos' = _TETrace[j].anchor_pos
        /\ draft_kv_injected_n_cells  = _TETrace[i].draft_kv_injected_n_cells
        /\ draft_kv_injected_n_cells' = _TETrace[j].draft_kv_injected_n_cells
        /\ in_flight_block  = _TETrace[i].in_flight_block
        /\ in_flight_block' = _TETrace[j].in_flight_block
        /\ n_rejected_prev  = _TETrace[i].n_rejected_prev
        /\ n_rejected_prev' = _TETrace[j].n_rejected_prev
        /\ step  = _TETrace[i].step
        /\ step' = _TETrace[j].step
        /\ verify_effective_seen  = _TETrace[i].verify_effective_seen
        /\ verify_effective_seen' = _TETrace[j].verify_effective_seen
        /\ target_kv_n_cells  = _TETrace[i].target_kv_n_cells
        /\ target_kv_n_cells' = _TETrace[j].target_kv_n_cells
        /\ last_n_accepted  = _TETrace[i].last_n_accepted
        /\ last_n_accepted' = _TETrace[j].last_n_accepted
        /\ verify_seq_lens_seen  = _TETrace[i].verify_seq_lens_seen
        /\ verify_seq_lens_seen' = _TETrace[j].verify_seq_lens_seen
        /\ draft_kv_self_n_cells  = _TETrace[i].draft_kv_self_n_cells
        /\ draft_kv_self_n_cells' = _TETrace[j].draft_kv_self_n_cells

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("DFlashCycle_TTrace_1778528154.json", _TETrace)

=============================================================================

 Note that you can extract this module `DFlashCycle_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `DFlashCycle_TEExpression.tla` file takes precedence 
  over the module `DFlashCycle_TEExpression` below).

---- MODULE DFlashCycle_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, DFlashCycle, Naturals, TLC

expression == 
    [
        \* To hide variables of the `DFlashCycle` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        accept_history |-> accept_history
        ,last_bonus_pos |-> last_bonus_pos
        ,pc |-> pc
        ,anchor_pos |-> anchor_pos
        ,draft_kv_injected_n_cells |-> draft_kv_injected_n_cells
        ,in_flight_block |-> in_flight_block
        ,n_rejected_prev |-> n_rejected_prev
        ,step |-> step
        ,verify_effective_seen |-> verify_effective_seen
        ,target_kv_n_cells |-> target_kv_n_cells
        ,last_n_accepted |-> last_n_accepted
        ,verify_seq_lens_seen |-> verify_seq_lens_seen
        ,draft_kv_self_n_cells |-> draft_kv_self_n_cells
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_accept_historyUnchanged |-> accept_history = accept_history'
        
        \* Format the `accept_history` variable as Json value.
        \* ,_accept_historyJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(accept_history)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_accept_historyModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].accept_history # _TETrace[s-1].accept_history
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE DFlashCycle_TETrace ----
\*EXTENDS IOUtils, DFlashCycle, TLC
\*
\*trace == IODeserialize("DFlashCycle_TTrace_1778528154.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE DFlashCycle_TETrace ----
EXTENDS DFlashCycle, TLC

trace == 
    <<
    ([draft_kv_injected_n_cells |-> 0,accept_history |-> <<>>,target_kv_n_cells |-> 1,draft_kv_self_n_cells |-> 0,in_flight_block |-> [present |-> FALSE, anchor_pos |-> 0, n_tokens |-> 0],pc |-> "draft",last_n_accepted |-> 0,step |-> 0,verify_seq_lens_seen |-> 0,last_bonus_pos |-> 0,n_rejected_prev |-> 0,verify_effective_seen |-> 0,anchor_pos |-> 0]),
    ([draft_kv_injected_n_cells |-> 0,accept_history |-> <<>>,target_kv_n_cells |-> 1,draft_kv_self_n_cells |-> 0,in_flight_block |-> [present |-> TRUE, anchor_pos |-> 1, n_tokens |-> 3],pc |-> "verify",last_n_accepted |-> 0,step |-> 0,verify_seq_lens_seen |-> 0,last_bonus_pos |-> 0,n_rejected_prev |-> 0,verify_effective_seen |-> 0,anchor_pos |-> 0])
    >>
----


=============================================================================

---- CONFIG DFlashCycle_TTrace_1778528154 ----
CONSTANTS
    MaxStep = 4
    BlockSize = 3
    BugAFamilyActive = FALSE
    BugCFamilyActive = TRUE

INVARIANT
    _inv

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Mon May 11 19:35:55 UTC 2026