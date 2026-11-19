"""Method B (Petri net + alignment repair) for every dataset.

Re-uses align_case (PM4Py VERSION_STATE_EQUATION_A_STAR) and repair_from_alignment from
src/experiments/p6_small_repaired_log_comparison/61_method_b_alignment_repair.py, loaded by path.
Repair rule: keep synchronous moves, delete log moves, insert visible model moves, skip tau moves.
Per-case time covers align_case + repair_from_alignment only (perf_counter).
"""

import importlib.util
import time

from utils.validation_utils import PROJECT_ROOT

EXP061_SCRIPT = PROJECT_ROOT / "src/experiments/p6_small_repaired_log_comparison/61_method_b_alignment_repair.py"


def load_exp061():
    spec = importlib.util.spec_from_file_location("exp061_alignment", EXP061_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def alignment_repair(traces, labels, ids_by_type, net, im, fm, language_set=None):
    """ids_by_type: [("test", ids), ("val", ids)]. Returns (rows_log, rows_case)."""
    from utils.data_utils import build_event_log

    m = load_exp061()
    rows_log, rows_case = [], []
    for case_type, ids in ids_by_type:
        for cid in ids:
            original = list(traces[cid])
            trace_obj = build_event_log([cid], traces)[0]
            t0 = time.perf_counter()
            try:
                res = m.align_case(trace_obj, net, im, fm)
                repaired, moves = m.repair_from_alignment(res["alignment"])
                failed, error = False, ""
                diag = {"alignment_cost": res["cost"], "fitness": res["fitness"],
                        "visited_states": res.get("visited_states"), "queued_states": res.get("queued_states")}
            except Exception as e:  # noqa: BLE001  (on failure keep the original trace and set the failure flag)
                repaired, failed, error = list(original), True, repr(e)
                moves = {"n_sync_moves": None, "n_log_moves": None, "n_model_moves_visible": None, "n_tau_moves": None}
                diag = {"alignment_cost": None, "fitness": None, "visited_states": None, "queued_states": None}
            case_ms = (time.perf_counter() - t0) * 1000
            for i, act in enumerate(repaired, start=1):
                rows_log.append({"case:concept:name": cid, "concept:name": act, "order": i, "type": case_type,
                                 "alignment_failed": failed})
            rows_case.append({"case:concept:name": cid, "type": case_type, "anomaly_label": labels[cid],
                              "original_trace": ">>".join(map(str, original)),
                              "repaired_trace": ">>".join(map(str, repaired)), "alignment_failed": failed,
                              "error": error,
                              "model_conformant": (None if language_set is None else tuple(repaired) in language_set)
                              if not failed else False,
                              **diag, **moves, "align_ms": round(case_ms, 4)})
    return rows_log, rows_case
