"""
timing_utils — 학습된 agent로 Test phase만 순차 재실행해 per-case repair latency를 잰다 (2026-10-12)

왜:
    공식 run(multiprocessing, 15 workers)에서 worker 안에서 잰 case 시간은 CPU를 나눠 쓰는 영향이 섞인다.
    이 모듈은 저장된 agent(Q-table)로 Test case만 한 프로세스에서 순서대로 처리해,
    multiprocessing 영향이 없는 case별 수리 시간을 잰다. accuracy를 다시 재는 실험이 아니다.

지키는 것:
    - RP-BEAR core(rpbear/)와 sim.py 로직을 바꾸지 않는다. Test 처리는 sim._process_case_with_memory
      (train_mode=False, case_type="test")를 그대로 호출한다 → Q-table 업데이트 없음, 기존 action-selection 그대로.
    - setup 순서는 sim.run_rpbear와 같다: seed → 데이터 로드 → NBG → val/test split.

repair memory:
    공식 Test는 validation에서 채워진 repair memory를 이어서 쓴다. 공식 run은 memory를 저장하지 않으므로
    공식 repaired log의 validation 행으로 다시 만든다 (trace당 validation 결과).

한계 (byte-identical에 대해):
    공식 Test의 무작위 행동(epsilon-greedy)은 worker별 난수 흐름에 달려 있고 그 상태는 저장되지 않았다.
    그래서 공식 agent 재실행의 RL path 결과는 공식 결과와 같다고 보장할 수 없다. 대신
    validate_test_only_equivalence()로, 같은 상태(Q-table, 탐색률, memory, 난수 상태)에서 시작하면
    Test-only 재실행이 원래 Test 결과와 byte-identical임을 순차 조건에서 확인한다.
"""

import copy
import random
import time

import numpy as np


def prepare_rpbear_inputs(data_name, input_paths, val_ratio, seed, nbgs_override=None):
    """sim.run_rpbear와 같은 순서로 입력을 준비한다 (seed → load → NBG → split)."""
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
    """Test case를 순서대로 한 프로세스에서 처리한다. return: (results list, timing rows, phase wall ms).

    rng_state: (random state, numpy state)를 주면 그 상태에서 시작, 아니면 seed로 초기화한다.
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
    """공식 repaired log의 validation 행으로 repair memory {trace: result df}를 다시 만든다.

    return: (memory, info). 같은 trace가 validation에 여러 번 있으면 파일에서 처음 나온 case의 결과를 쓰고,
    그 trace의 validation 결과가 case마다 달랐는지(info["conflicting_traces"])를 기록한다.
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
    """Test 전체 / RL / Mem 요약 + 보조 지표(논문 산식: test phase wall-clock / test cases)."""
    from sim import _timing_summary

    out = _timing_summary(timing_rows, "test")
    n = len(timing_rows)
    out["test_phase_wall_ms"] = round(phase_wall_ms, 4)
    out["test_phase_wall_ms_per_case"] = round(phase_wall_ms / n, 4) if n else None
    out["rl_share"] = round(sum(r["path"] == "RL" for r in timing_rows) / n, 4) if n else None
    return out


def replay_alignment_timing(exp061_script, pnml_path, test_ids, traces=None):
    """Method B (Alignment): 저장된 공식 Petri Net으로 Test case만 순차 alignment + repair, case별 시간을 잰다.

    EXP-061 script의 load_case_traces / build_event_log / align_case / repair_from_alignment를 그대로 import해서
    공식 run과 같은 코드 경로를 쓴다. 측정 구간(perf_counter)은 공식 run과 같다:
    trace 객체를 만든 뒤 align_case(PM4Py alignment) + repair_from_alignment까지.
    데이터 로딩 / Process Discovery / Petri Net 생성(PNML 읽기) / CSV 저장 / 평가는 제외한다.
    return: (rows: case_id, repaired_trace, repair_time_ms, alignment_cost, failed), net 정보, test phase wall ms
    """
    import importlib.util

    import pm4py

    spec = importlib.util.spec_from_file_location("exp061_timing", exp061_script)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    if traces is None:  # Small (EXP-061 data/raw 경로). 다른 dataset은 호출하는 쪽에서 traces를 넘긴다
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
        except Exception:  # noqa: BLE001  (공식 run과 같은 fallback: 원래 trace 보존)
            repaired, failed, cost = list(traces[cid]), True, None
        elapsed = (time.perf_counter() - t0) * 1000
        rows.append({"case_id": str(cid), "repaired_trace": ">>".join(repaired), "repair_time_ms": round(elapsed, 4),
                     "alignment_cost": cost, "failed": failed})
    return rows, net_info, (time.perf_counter() - t_phase) * 1000


def summarize_values(values):
    """n, mean, median, std(표본), min, max (ms)."""
    import statistics

    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {"n": len(values), "mean": round(statistics.fmean(values), 4), "median": round(statistics.median(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else None,
            "min": round(min(values), 4), "max": round(max(values), 4)}


def validate_test_only_equivalence(data_name, input_paths, num_epi, alpha, val_ratio, seed, max_step,
                                   nbgs_override=None):
    """순차 조건에서 Test-only 재실행이 원래 Test 결과와 byte-identical인지 확인한다.

    1) run_rpbear(use_multiprocessing=False)와 같은 순서로 validation → (상태 저장) → test를 실행 (reference)
    2) 저장한 상태(agent deep copy, memory copy, 난수 상태)에서 replay_test_phase로 Test만 다시 실행
    return: dict (reference DataFrame들, replay DataFrame, timing)
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
