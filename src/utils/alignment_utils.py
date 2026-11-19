"""
alignment_utils — Method B (Petri Net + Alignment repair)를 모든 dataset에 쓰기 위한 공통 함수 (2026-10-13)

EXP-061(Small)의 alignment / repair 코드를 그대로 재사용한다 (D032):
    src/experiments/p6_small_repaired_log_comparison/61_method_b_alignment_repair.py 의
    align_case (PM4Py VERSION_STATE_EQUATION_A_STAR) 와 repair_from_alignment
    (sync 유지 / log move 삭제 / visible model move 삽입 / tau 제외).
61 script는 수정하지 않는다. dataset별 경로와 split만 pipeline에서 넘긴다.
case별 시간은 perf_counter로 align_case + repair_from_alignment 구간만 잰다 (Small timing 정의와 같음).
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
    """ids_by_type: [("test", ids), ("val", ids)] (EXP-061과 같은 순서). return: (rows_log, rows_case)."""
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
            except Exception as e:  # noqa: BLE001  (EXP-061과 같은 fallback: 원래 trace 보존 + failure flag)
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
