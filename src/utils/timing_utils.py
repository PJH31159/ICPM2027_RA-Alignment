"""Sequential test-only replay with a trained agent to measure per-case repair latency.

Why: in the official run (multiprocessing, 15 workers) per-case times measured inside a worker include
CPU sharing. Here the saved agent (Q-table) processes only the test cases, one by one in a single
process, which gives per-case repair times without multiprocessing effects. Accuracy is not re-measured.

Guarantees:
    - RP-BEAR core (rpbear/) and sim.py are not changed. Test cases are processed with
      sim._process_case_with_memory(train_mode=False, case_type="test"): no Q-table update,
      unchanged action selection.
    - Setup order is the same as sim.run_rpbear: seed -> data load -> NBG -> val/test split.

Repair memory: the official test phase continues with the repair memory filled during training. The
memory is not saved, so it is rebuilt from the training rows of the official repaired log.

Limitation (byte-identity): random actions (epsilon-greedy) during the official test depend on the
per-worker random streams, whose state was not saved, so RL-path results of a replay are not guaranteed
to equal the official ones. validate_test_only_equivalence() instead shows, under sequential execution,
that a test-only replay starting from the same state (Q-table, exploration rate, memory, RNG state)
is byte-identical to the original test phase.
"""

import copy
import random
import time

import numpy as np


def prepare_rpbear_inputs(data_name, input_paths, val_ratio, seed, nbgs_override=None):
    """Prepare the inputs in the same order as sim.run_rpbear (seed -> load -> NBG -> split)."""
    from rpbear.func_rpbear_final import discover_NBG
    from sim import _load_and_prepare_data, _prepare_case_split

    random.seed(seed)
    np.random.seed(seed)
    clean, anomaly, gt_traces, actset, filtered_actset, act_freq = _load_and_prepare_data(data_name, input_paths)
    nbgs = discover_NBG(clean, vis=False) if nbgs_override is None else list(nbgs_override)
    anomaly_nolabel, case_to_trace, val_ids, test_ids = _prepare_case_split(anomaly, val_ratio)
    return {"anomaly_nolabel": anomaly_nolabel, "case_to_trace": case_to_trace, "val_ids": val_ids,
            "test_ids": test_ids, "actset": actset, "filtered_actset": filtered_actset, "act_freq": act_freq,
            "nbgs": nbgs, "gt_traces": gt_traces}


def _case_func(prep, agent, repair_memory, alpha, num_epi, max_step, train_mode, case_type):
    from functools import partial

    from sim import _process_case_with_memory

    return partial(_process_case_with_memory, case_to_trace=prep["case_to_trace"], repair_memory=repair_memory,
                   p_anomaly_nolabel=prep["anomaly_nolabel"], p_filtered_actset=prep["filtered_actset"],
                   p_act_freq=prep["act_freq"], p_actset=prep["actset"], p_NBGs=prep["nbgs"], p_num_epi=num_epi,
                   alpha=alpha, agent=agent, train_mode=train_mode, case_type=case_type, max_step=max_step)


def replay_test_phase(prep, agent, repair_memory, alpha, num_epi, max_step, rng_state=None, seed=None):
    """Process the test cases one by one in one process. Returns (results list, timing rows, phase wall ms).

    rng_state: (random state, numpy state) to start from; otherwise the RNGs are seeded.
    """
    from sim import _timing_row

    if rng_state is not None:
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
    elif seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    func = _case_func(prep, agent, repair_memory, alpha, num_epi, max_step, False, "test")
    results, timing = [], []
    t0 = time.perf_counter()
    for case_id in prep["test_ids"]:
        cid, result, is_run, elapsed_ms = func(case_id)
        results.append(result)
        timing.append(_timing_row(cid, "test", is_run, elapsed_ms))
    return results, timing, (time.perf_counter() - t0) * 1000


def rebuild_repair_memory(official_repaired_csv, case_to_trace, val_ids):
    """Rebuild the repair memory {trace: result df} from the training rows of the official repaired log.

    Returns (memory, info). If a trace occurs several times in training, the result of its first case in the
    file is used, and info["conflicting_traces"] records whether results differed between those cases.
    """
    import pandas as pd

    df = pd.read_csv(official_repaired_csv)
    df["case:concept:name"] = df["case:concept:name"].astype(str)
    val = df[df["type"] == "val"]
    memory, seen_payload, conflicts = {}, {}, 0
    val_set = set(map(str, val_ids))
    for cid, rows in val.groupby("case:concept:name", sort=False):
        if cid not in val_set:
            continue
        trace = case_to_trace[cid]
        rows = rows.reset_index(drop=True)
        payload = tuple(rows["concept:name"])
        if trace in memory:
            conflicts += payload != seen_payload[trace]
            continue
        memory[trace] = rows
        seen_payload[trace] = payload
    info = {"validation_cases": int(val["case:concept:name"].nunique()), "memory_traces": len(memory),
            "conflicting_traces": int(conflicts),
            "note": "rebuilt from official repaired log validation rows (first occurrence per trace)"}
    return memory, info


def test_result_frame(results):
    import pandas as pd

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def summarize(timing_rows, phase_wall_ms):
    """Summary over all / RL / Mem test cases plus the auxiliary metric (test phase wall-clock / test cases)."""
    from sim import _timing_summary

    out = _timing_summary(timing_rows, "test")
    n = len(timing_rows)
    out["test_phase_wall_ms"] = round(phase_wall_ms, 4)
    out["test_phase_wall_ms_per_case"] = round(phase_wall_ms / n, 4) if n else None
    out["rl_share"] = round(sum(r["path"] == "RL" for r in timing_rows) / n, 4) if n else None
    return out


def replay_alignment_timing(exp061_script, pnml_path, test_ids, traces=None):
    """Method B (alignment): sequential alignment + repair of the test cases with the saved official Petri net.

    Uses load_case_traces / build_event_log / align_case / repair_from_alignment of the Method B script,
    i.e. the same code path as the official run. The timed section (perf_counter) is the same as in the
    official run: after building the trace object, align_case (PM4Py alignment) + repair_from_alignment.
    Data loading, discovery, Petri-net loading (PNML), CSV writing and evaluation are excluded.
    Returns (rows: case_id, repaired_trace, repair_time_ms, alignment_cost, failed), net info, test phase wall ms.
    """
    import importlib.util

    import pm4py

    spec = importlib.util.spec_from_file_location("exp061_timing", exp061_script)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    if traces is None:  # Small default input path; other datasets pass traces explicitly
        traces, _labels, _clean, _anom = m.load_case_traces()
    net, im, fm = pm4py.read_pnml(str(pnml_path))
    net_info = {"n_places": len(net.places), "n_transitions": len(net.transitions), "n_arcs": len(net.arcs),
                "alignment_variant": str(m.ALIGNMENT_VARIANT), "pm4py": pm4py.__version__}
    rows = []
    t_phase = time.perf_counter()
    for cid in test_ids:
        trace_obj = m.build_event_log([cid], traces)[0]
        t0 = time.perf_counter()
        try:
            res = m.align_case(trace_obj, net, im, fm)
            repaired, _moves = m.repair_from_alignment(res["alignment"])
            failed, cost = False, res["cost"]
        except Exception:  # noqa: BLE001  (same fallback as the official run: keep the original trace)
            repaired, failed, cost = list(traces[cid]), True, None
        elapsed = (time.perf_counter() - t0) * 1000
        rows.append({"case_id": str(cid), "repaired_trace": ">>".join(repaired), "repair_time_ms": round(elapsed, 4),
                     "alignment_cost": cost, "failed": failed})
    return rows, net_info, (time.perf_counter() - t_phase) * 1000


def summarize_values(values):
    """n, mean, median, std (sample), min, max (ms)."""
    import statistics

    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {"n": len(values), "mean": round(statistics.fmean(values), 4), "median": round(statistics.median(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else None,
            "min": round(min(values), 4), "max": round(max(values), 4)}


def validate_test_only_equivalence(data_name, input_paths, num_epi, alpha, val_ratio, seed, max_step,
                                   nbgs_override=None):
    """Check under sequential execution that a test-only replay is byte-identical to the original test phase.

    1) run training -> (save state) -> test in the same order as run_rpbear(use_multiprocessing=False) (reference)
    2) replay only the test phase with replay_test_phase from the saved state (agent deep copy, memory copy, RNG state)
    Returns a dict (reference DataFrames, replay DataFrame, timing).
    """
    from rpbear.func_rpbear_final import QLearningAgent
    from sim import _run_phase

    prep = prepare_rpbear_inputs(data_name, input_paths, val_ratio, seed, nbgs_override)
    agent = QLearningAgent(filtered_actset=prep["filtered_actset"], act_freq=prep["act_freq"], learning_rate=0.1,
                           discount_factor=0.9, exploration_rate=0.5, exploration_decay_rate=0.0001)
    agent.q_table = {}
    memory = {}
    common = dict(case_to_trace=prep["case_to_trace"], repair_memory=memory, anomaly_nolabel=prep["anomaly_nolabel"],
                  filtered_actset=prep["filtered_actset"], act_freq=prep["act_freq"], actset=prep["actset"],
                  NBGs=prep["nbgs"], num_epi=num_epi, alpha=alpha, agent=agent, max_step=max_step,
                  gt_traces=prep["gt_traces"], use_multiprocessing=False, num_cores=1, seed=seed, verbose=False)
    val_res, *_ = _run_phase(caseids=prep["val_ids"], case_type="val", train_mode=True, **common)

    snapshot = (copy.deepcopy(agent), {k: v.copy() for k, v in memory.items()},
                (random.getstate(), np.random.get_state()))

    test_ref, *_ = _run_phase(caseids=prep["test_ids"], case_type="test", train_mode=False, **common)

    agent2, memory2, rng = snapshot
    replay, timing, wall = replay_test_phase(prep, agent2, memory2, alpha, num_epi, max_step, rng_state=rng)
    return {"val": test_result_frame(val_res), "test_reference": test_result_frame(test_ref),
            "test_replay": test_result_frame(replay), "timing": timing, "wall_ms": wall}
