"""
sim.py  —  RP-BEAR execution engine / wrapper

역할 구분:
    sim.py        = RP-BEAR를 한 번 실행하는 엔진(wrapper). 데이터 로드, NBG, val/test
                    split, Q-learning 학습/추론을 수행하고 repaired log를 돌려준다.
    simulation.py = 여러 dataset에 같은 연구 pipeline(discover → generate → nbg →
                    smoke → run → evaluate)을 적용하는 orchestrator. RP-BEAR 실행이
                    필요할 때 이 파일의 run_rpbear()를 호출한다.

Method A(RP-BEAR) 실행 로직을 모아둔 모듈이다.

`main.py`는 이 파일의 `run_rpbear(...)`만 호출한다. RP-BEAR의 핵심
알고리즘(NBG 생성, ADM, QLearningAgent, P_BEAR_RL)은
`src/rpbear/func_rpbear_final.py`를 reference implementation으로
그대로 import해서 사용하며, 이 파일에서는 수정하지 않는다.

이 wrapper의 목적은 새로운 RP-BEAR를 만드는 것이 아니라,
`src/rpbear/main_rpbear_final.py`의 baseline behavior를
ICPM2027 프로젝트 경로/구조 안에서 최대한 동일하게 재현하는 것이다.
"""

from __future__ import annotations

import os
import pickle
import random
import time
import warnings
from functools import partial
from multiprocessing import Manager, Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from rpbear.func_rpbear_final import (
    QLearningAgent,
    discover_NBG,
    process_case_id_wrapper,
)

warnings.filterwarnings(action="ignore")


# ============================================================
# 1. Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 2. Case 단위 처리 (main_rpbear_final.py의 process_case_with_memory 이식)
# ============================================================
#
# multiprocessing.Pool로 워커 프로세스에 전달되려면 이 함수가
# 모듈 최상위(top-level)에 있어야 pickle이 가능하다. 로직 자체는
# 원본 main_rpbear_final.py의 process_case_with_memory와 동일하다.

def _process_case_with_memory(
    case_id,
    case_to_trace,
    repair_memory,
    p_anomaly_nolabel,
    p_filtered_actset,
    p_act_freq,
    p_actset,
    p_NBGs,
    p_num_epi,
    alpha,
    agent,
    train_mode,
    case_type,
    max_step,
):
    # timing instrumentation: case 하나를 처리하는 데 걸린 시간 (worker 안에서 측정).
    # time.perf_counter()는 난수를 쓰지 않으므로 수리 결과에 영향이 없다.
    t0 = time.perf_counter()
    trace_str = case_to_trace[case_id]

    if trace_str in repair_memory:
        # 이미 수리된 trace라면 메모리에서 불러오기 (RL 학습 생략)
        result = repair_memory[trace_str].copy()
        result["type"] = case_type
        result["case:concept:name"] = case_id
        return case_id, result, 0, (time.perf_counter() - t0) * 1000  # 0: RL 안 돌았음 (메모리 사용)

    # 메모리에 없는 새로운 trace면 RL 수리 진행
    result = process_case_id_wrapper(
        case_id_for_processing=case_id,
        p_anomaly_nolabel=p_anomaly_nolabel,
        p_filtered_actset=p_filtered_actset,
        p_act_freq=p_act_freq,
        p_actset=p_actset,
        p_NBGs=p_NBGs,
        p_num_epi=p_num_epi,
        alpha=alpha,
        agent=agent,
        train_mode=train_mode,
        case_type=case_type,
        max_step=max_step,
    )
    repair_memory[trace_str] = result.copy()
    return case_id, result, 1, (time.perf_counter() - t0) * 1000  # 1: 새롭게 RL/추론 돌았음


# ============================================================
# 3. Worker 초기화 (multiprocessing=True일 때 seed 재현용)
# ============================================================
#
# 원본 main_rpbear_final.py는 random.seed(seed)/np.random.seed(seed)를
# 모듈 최상위(스크립트 레벨)에서 호출한다. Windows spawn 방식에서는
# 워커 프로세스가 그 스크립트를 다시 import하면서 이 seed 설정도
# 함께 재실행되므로, 결과적으로 "모든 워커가 동일한 seed로 시작"하는
# 효과가 있었다. sim.py에서는 seed 설정을 run_rpbear() 함수 안으로
# 옮겼기 때문에(부모 프로세스에서만 실행됨), 워커 프로세스가 그대로
# 두면 OS entropy로 시드가 결정되어 원본과 다른 무작위성을 갖는다.
# 이를 막기 위해 Pool의 initializer로 각 워커에서도 동일하게
# seed를 다시 걸어준다.

def _worker_init(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 4. 데이터 로드 / clean-anomaly 분리 (경로만 ICPM2027 구조로 수정)
# ============================================================

def _load_and_prepare_data(data_name: str, input_paths: dict | None = None):
    # input_paths=None: 기존과 같이 data/raw/{data_name}_1.00.csv, data/raw/{data_name}.csv
    # input_paths 지정: {"anomaly_log": Path, "ground_truth_log": Path} 파일을 그대로 사용
    if input_paths is None:
        anomalous_data_path = PROJECT_ROOT / "data" / "raw" / f"{data_name}_1.00.csv"
        normal_data_path = PROJECT_ROOT / "data" / "raw" / f"{data_name}.csv"
    else:
        anomalous_data_path = Path(input_paths["anomaly_log"])
        normal_data_path = Path(input_paths["ground_truth_log"])

    dtype_data = {"Case": "category", "Activity": "category", "type_res_trace": "category"}
    data = pd.read_csv(
        anomalous_data_path,
        usecols=["Case", "Activity", "Timestamp", "type_res_trace"],
        dtype=dtype_data,
    )

    data = data.rename(
        columns={
            "Case": "case:concept:name",
            "Activity": "concept:name",
            "Timestamp": "time:timestamp",
            "type_res_trace": "label",
        },
        copy=False,
    )

    # 실시간 성능 검증을 위한 원본(정상) 데이터 Ground Truth 로드
    gt_traces = {}
    try:
        normal_data = pd.read_csv(normal_data_path, usecols=["Case", "Activity"])
        normal_data = normal_data.rename(
            columns={"Case": "case:concept:name", "Activity": "concept:name"}, copy=False
        )
        normal_data_no_se = normal_data[~normal_data["concept:name"].isin(["Start", "End"])]
        gt_traces = (
            normal_data_no_se.groupby("case:concept:name", observed=True)["concept:name"]
            .apply(lambda x: ">>".join(x))
            .to_dict()
        )
    except Exception as e:
        print(f"[경고] 정상 데이터({normal_data_path})를 불러오지 못했습니다. 실시간 정확도 계산이 생략됩니다. 오류: {e}")

    # Add 'Start' and 'End' activity
    data["order"] = data.groupby("case:concept:name", observed=True).cumcount() + 1

    start_df_orig = data[data["order"] == 1].copy()
    start_df_orig.loc[:, "order"] = 0
    start_df_orig.loc[:, "concept:name"] = "Start"

    end_indices = data.groupby("case:concept:name", observed=True)["order"].idxmax()
    end_df_orig = data.loc[end_indices].copy()
    end_df_orig.loc[:, "order"] = end_df_orig["order"] + 1
    end_df_orig.loc[:, "concept:name"] = "End"

    data2 = (
        pd.concat([data, start_df_orig, end_df_orig], ignore_index=True)
        .sort_values(by=["case:concept:name", "order"], kind="stable")
        .reset_index(drop=True)
    )

    clean = data2[data2.label.isna()].reset_index(drop=True)
    anomaly = data2.reset_index(drop=True)

    actset = clean["concept:name"].unique()
    filtered_actset = [item for item in actset if item not in ["Start", "End"]]

    act_freq = clean["concept:name"].value_counts()
    if not act_freq.empty:
        act_freq = act_freq.drop(index=["Start", "End"], errors="ignore")

    if not act_freq.empty:
        act_freq = act_freq.reset_index()
        act_freq.columns = ["concept:name", "count"] if "index" not in act_freq.columns else [act_freq.columns[0], "count"]
        act_freq["prob"] = act_freq["count"] / sum(act_freq["count"])
        act_freq = act_freq.set_index("concept:name").reindex(filtered_actset).reset_index()
    else:
        act_freq = pd.DataFrame(columns=["concept:name", "count", "prob"])
        if filtered_actset:
            act_freq["concept:name"] = filtered_actset
            act_freq = act_freq.fillna(0)

    return clean, anomaly, gt_traces, actset, filtered_actset, act_freq


# ============================================================
# 5. Anomaly case를 val/test로 분할 (원본과 동일 로직)
# ============================================================

def _prepare_case_split(anomaly: pd.DataFrame, val_ratio: float):
    anomaly_caseids = anomaly.loc[anomaly["label"].notna(), "case:concept:name"].unique().tolist()

    anomaly_only = anomaly.loc[anomaly["case:concept:name"].isin(anomaly_caseids)].reset_index(drop=True)
    anomaly_nolabel = anomaly_only[["case:concept:name", "concept:name", "order"]]

    case_to_trace = (
        anomaly_nolabel.groupby("case:concept:name", observed=True)["concept:name"]
        .apply(lambda x: ">>".join(x))
        .to_dict()
    )

    all_caseids = anomaly_caseids.copy()
    random.shuffle(all_caseids)

    n_total = len(all_caseids)
    n_val = max(1, int(n_total * val_ratio))
    if n_total > 1 and n_val >= n_total:
        n_val = n_total - 1

    val_caseids = all_caseids[:n_val]
    test_caseids = all_caseids[n_val:]

    return anomaly_nolabel, case_to_trace, val_caseids, test_caseids


# ============================================================
# 6. Validation / Test 진행 상황 출력 (원본 print 포맷 재현)
# ============================================================

def _report_progress(result, case_id, gt_traces, idx, total, success_count, case_type, is_run, verbose):
    repaired_trace = ">>".join(
        result[~result["concept:name"].isin(["Start", "End"])]["concept:name"].tolist()
    )
    gt_trace = gt_traces.get(case_id, "")

    is_correct = repaired_trace == gt_trace
    if is_correct:
        success_count += 1

    if verbose:
        if case_type == "val":
            method_label = "RL " if is_run else "Mem"
        else:
            method_label = "Greedy" if is_run else "Mem   "

        status = "성공(O)" if is_correct else "실패(X)"
        acc_rate = (success_count / idx) * 100

        print(
            f"[{case_type.capitalize()} {idx:4d}/{total}] Case: {str(case_id):12s} | "
            f"방식: {method_label} | 수리: {status} | 실시간 p.anomaly: {acc_rate:6.2f}% "
            f"({success_count}/{idx})"
        )

    return success_count


# ============================================================
# 7. Validation 또는 Test 한 phase 실행
# ============================================================

def _run_phase(
    caseids,
    case_type,
    train_mode,
    case_to_trace,
    repair_memory,
    anomaly_nolabel,
    filtered_actset,
    act_freq,
    actset,
    NBGs,
    num_epi,
    alpha,
    agent,
    max_step,
    gt_traces,
    use_multiprocessing,
    num_cores,
    seed,
    verbose,
):
    func = partial(
        _process_case_with_memory,
        case_to_trace=case_to_trace,
        repair_memory=repair_memory,
        p_anomaly_nolabel=anomaly_nolabel,
        p_filtered_actset=filtered_actset,
        p_act_freq=act_freq,
        p_actset=actset,
        p_NBGs=NBGs,
        p_num_epi=num_epi,
        alpha=alpha,
        agent=agent,
        train_mode=train_mode,
        case_type=case_type,
        max_step=max_step,
    )

    results = []
    case_timing = []  # timing instrumentation: case별 처리 시간 (phase, path = RL / Mem)
    rl_count = 0
    success_count = 0
    start_time = time.time()

    if use_multiprocessing:
        with Pool(processes=num_cores, initializer=_worker_init, initargs=(seed,)) as pool:
            for idx, (case_id, result, is_run, elapsed_ms) in enumerate(pool.imap_unordered(func, caseids), start=1):
                results.append(result)
                case_timing.append(_timing_row(case_id, case_type, is_run, elapsed_ms))
                rl_count += is_run
                success_count = _report_progress(
                    result, case_id, gt_traces, idx, len(caseids), success_count, case_type, is_run, verbose
                )
    else:
        # Debug / smoke-test 경로: 단일 프로세스, case 순서대로 순차 처리.
        # Pool.imap_unordered와 처리 순서가 다르므로 Q-table 업데이트
        # 순서, 그리고 alpha/exploration 관련 결과가 use_multiprocessing=True와
        # 정확히 일치하지 않을 수 있다.
        for idx, case_id in enumerate(caseids, start=1):
            case_id_out, result, is_run, elapsed_ms = func(case_id)
            results.append(result)
            case_timing.append(_timing_row(case_id_out, case_type, is_run, elapsed_ms))
            rl_count += is_run
            success_count = _report_progress(
                result, case_id_out, gt_traces, idx, len(caseids), success_count, case_type, is_run, verbose
            )

    total_ms = (time.time() - start_time) * 1000
    per_case_ms = total_ms / len(caseids) if len(caseids) > 0 else 0.0

    return results, rl_count, success_count, total_ms, per_case_ms, case_timing


def _timing_row(case_id, case_type, is_run, elapsed_ms):
    """case timing CSV의 한 행. path: RL = 새로 RL 학습/추론, Mem = repair memory 재사용."""
    return {"case_id": str(case_id), "phase": case_type, "path": "RL" if is_run else "Mem",
            "repair_time_ms": round(float(elapsed_ms), 4)}


def _timing_summary(case_timing, phase):
    """phase 하나의 case 처리 시간 요약 (all / RL / Mem 각각 n, mean, median, std, min, max; ms).

    std는 표본 표준편차(ddof=1). case가 1개 이하이면 std는 None.
    """
    import statistics

    out = {}
    rows = [r for r in case_timing if r["phase"] == phase]
    for path in ("all", "RL", "Mem"):
        vals = [r["repair_time_ms"] for r in rows if path == "all" or r["path"] == path]
        key = f"{phase}_case_ms_{path.lower()}"
        out[f"{key}_n"] = len(vals)
        if vals:
            out[f"{key}_mean"] = round(statistics.fmean(vals), 4)
            out[f"{key}_median"] = round(statistics.median(vals), 4)
            out[f"{key}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
            out[f"{key}_min"] = round(min(vals), 4)
            out[f"{key}_max"] = round(max(vals), 4)
        else:
            for s in ("mean", "median", "std", "min", "max"):
                out[f"{key}_{s}"] = None
    return out


# ============================================================
# 8. Public Entry Point - Method A
# ============================================================

def run_rpbear(
    data_name: str,
    num_epi: int = 100,
    alpha: float = 0.0,
    val_ratio: float = 0.5,
    seed: int = 42,
    max_step: int = 8,
    num_cores: int | None = None,
    use_multiprocessing: bool = True,
    save_outputs: bool = True,
    verbose: bool = True,
    nbgs_override: list | None = None,
    input_paths: dict | None = None,
    return_case_timing: bool = False,
) -> tuple[pd.DataFrame, dict] | tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Method A(기존 RP-BEAR) 파이프라인을 실행하고 repaired log를 반환한다.

    use_multiprocessing=True  : 기존 RP-BEAR baseline 재현용
                                 (Manager + Pool.imap_unordered, 원본과 동일)
    use_multiprocessing=False : debugging / smoke test용
                                 (단일 프로세스 순차 처리, 처리 순서가 달라짐)

    nbgs_override : None(기본값)이면 기존과 완전히 같게 observed clean log로
                    discover_NBG(clean)을 만든다 (Method A).
                    NBG 목록을 주면 discover_NBG 대신 그 목록을 그대로 쓴다
                    (EXP-063 Method C: Petri Net 생성 trace로 만든 New NBG).
                    목록은 반드시 이 함수 안의 actset 순서
                    (= observed clean log의 activity 첫 등장 순서)로 정렬되어
                    있어야 한다. ADM이 NBGs[actset 위치]로 NBG를 찾기 때문이다.
                    actset, act_freq 등 NBG 외의 입력은 바뀌지 않는다.
    input_paths   : None(기본값)이면 기존과 같이 data/raw/{data_name}_1.00.csv,
                    data/raw/{data_name}.csv를 읽는다. dict를 주면
                    {"anomaly_log": ..., "ground_truth_log": ...} 파일을 읽는다
                    (예: 외부 P-BEAR-RL/Full_input). 저장 파일 이름 규칙은 바뀌지 않는다.
    return_case_timing : False(기본값)이면 기존과 같이 (repaired_log, stats)를 반환한다.
                    True이면 (repaired_log, stats, case_timing)을 반환한다.
                    case_timing = case별 처리 시간 DataFrame
                    (case_id, phase = val/test, path = RL/Mem, repair_time_ms).

    Timing 필드 (stats, 2026-10-12 instrumentation, 수리 결과에는 영향 없음):
        training_time_sec      = Validation(학습) phase wall-clock (= val_total_ms / 1000)
        total_repair_time_sec  = Test(추론) phase wall-clock (= test_total_ms / 1000)
        total_run_time_sec     = run_rpbear 전체 시간 (= total_sec)
        {val,test}_case_ms_{all,rl,mem}_{n,mean,median,std,min,max}
                               = worker 안에서 잰 case 하나의 처리 시간 (ms).
                                 RL = 새로 RL 학습/추론, Mem = repair memory 재사용.
                                 multiprocessing에서는 병렬 실행 중의 시간이다.

    Windows에서 use_multiprocessing=True로 호출하려면, 이 함수를
    호출하는 최상위 스크립트가 `if __name__ == "__main__":` 로
    감싸져 있어야 한다 (multiprocessing spawn 제약).
    """

    overall_start = time.time()

    random.seed(seed)
    np.random.seed(seed)

    if num_cores is None:
        num_cores = max(1, cpu_count() - 1)

    clean, anomaly, gt_traces, actset, filtered_actset, act_freq = _load_and_prepare_data(data_name, input_paths)

    nbg_start = time.time()
    if nbgs_override is None:
        NBGs = discover_NBG(clean, vis=False)
    else:
        if len(nbgs_override) != len(actset):
            raise ValueError(
                f"nbgs_override has {len(nbgs_override)} graphs, but actset has {len(actset)} activities"
            )
        NBGs = list(nbgs_override)
    nbg_ms = (time.time() - nbg_start) * 1000

    anomaly_nolabel, case_to_trace, val_caseids, test_caseids = _prepare_case_split(anomaly, val_ratio)

    if verbose:
        print("\n" + "=" * 70)
        print("RP-BEAR-RL 시작 (sim.run_rpbear, Method A)")
        print(
            f"data={data_name} | num_epi={num_epi} | alpha={alpha} | "
            f"val_ratio={val_ratio} | seed={seed} | max_step={max_step} | "
            f"use_multiprocessing={use_multiprocessing} | num_cores={num_cores}"
        )
        print("=" * 70)
        print(f"[INFO] Total anomalous cases = {len(val_caseids) + len(test_caseids)}")
        print(f"[INFO] Validation cases = {len(val_caseids)}")
        print(f"[INFO] Test cases       = {len(test_caseids)}")

    # Manager는 명시적으로 shutdown하지 않으면 백그라운드 서버 프로세스가
    # 남는다. run_rpbear()가 (비교 실험 등에서) 여러 번 호출될 수 있으므로
    # try/finally로 항상 정리되도록 한다.
    manager = Manager() if use_multiprocessing else None

    if use_multiprocessing:
        shared_q_table = manager.dict()
        repair_memory = manager.dict()
    else:
        shared_q_table = {}
        repair_memory = {}

    agent = QLearningAgent(
        filtered_actset=filtered_actset,
        act_freq=act_freq,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.5,
        exploration_decay_rate=0.0001,
    )
    agent.q_table = shared_q_table

    try:
        # ------------------------------------------------------
        # Validation phase (train_mode=True)
        # ------------------------------------------------------
        if verbose:
            print("\n[1/2] Validation phase 시작...")

        val_results, val_rl_count, val_success_count, val_total_ms, val_per_case_ms, val_timing = _run_phase(
            caseids=val_caseids,
            case_type="val",
            train_mode=True,
            case_to_trace=case_to_trace,
            repair_memory=repair_memory,
            anomaly_nolabel=anomaly_nolabel,
            filtered_actset=filtered_actset,
            act_freq=act_freq,
            actset=actset,
            NBGs=NBGs,
            num_epi=num_epi,
            alpha=alpha,
            agent=agent,
            max_step=max_step,
            gt_traces=gt_traces,
            use_multiprocessing=use_multiprocessing,
            num_cores=num_cores,
            seed=seed,
            verbose=verbose,
        )

        if verbose:
            print(f"\n[INFO] Validation 완료: total={val_total_ms:.2f} ms | per_case={val_per_case_ms:.2f} ms")

        model_path = None
        if save_outputs:
            model_dir = PROJECT_ROOT / "outputs" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file_name = f"{data_name}_{num_epi}_ppm{alpha}_valratio{val_ratio}_seed{seed}_agent.pkl"
            model_path = model_dir / model_file_name

            q_table_snapshot = dict(shared_q_table)
            agent.q_table = q_table_snapshot
            with open(model_path, "wb") as f:
                pickle.dump(agent, f)
            agent.q_table = shared_q_table

            if verbose:
                print(f"[INFO] Agent 저장 완료 -> {model_path}")

        # ------------------------------------------------------
        # Test phase (train_mode=False)
        # ------------------------------------------------------
        if verbose:
            print("\n[2/2] Test phase 시작...")

        test_results, test_rl_count, test_success_count, test_total_ms, test_per_case_ms, test_timing = _run_phase(
            caseids=test_caseids,
            case_type="test",
            train_mode=False,
            case_to_trace=case_to_trace,
            repair_memory=repair_memory,
            anomaly_nolabel=anomaly_nolabel,
            filtered_actset=filtered_actset,
            act_freq=act_freq,
            actset=actset,
            NBGs=NBGs,
            num_epi=num_epi,
            alpha=alpha,
            agent=agent,
            max_step=max_step,
            gt_traces=gt_traces,
            use_multiprocessing=use_multiprocessing,
            num_cores=num_cores,
            seed=seed,
            verbose=verbose,
        )

        if verbose:
            print(f"\n[INFO] Test 완료: total={test_total_ms:.2f} ms | per_case={test_per_case_ms:.2f} ms")

        # manager가 살아있는 동안(shutdown 전에) Q-table 크기를 미리 읽어둔다.
        q_table_size = len(agent.q_table)
    finally:
        if manager is not None:
            manager.shutdown()

    # ------------------------------------------------------
    # 결과 병합
    # ------------------------------------------------------
    val_repaired_df = pd.concat(val_results, ignore_index=True) if len(val_results) > 0 else pd.DataFrame()
    test_repaired_df = pd.concat(test_results, ignore_index=True) if len(test_results) > 0 else pd.DataFrame()
    repaired_log_a = pd.concat([val_repaired_df, test_repaired_df], ignore_index=True)

    total_execution_sec = time.time() - overall_start

    stats = {
        "data": data_name,
        "num_epi": num_epi,
        "alpha": alpha,
        "val_ratio": val_ratio,
        "seed": seed,
        "max_step": max_step,
        "use_multiprocessing": use_multiprocessing,
        "num_cores": num_cores,
        "n_val_cases": len(val_caseids),
        "val_rl_runs": val_rl_count,
        "val_p_anomaly": (val_success_count / len(val_caseids)) if len(val_caseids) > 0 else 0.0,
        "n_test_cases": len(test_caseids),
        "test_inference_runs": test_rl_count,
        "test_p_anomaly": (test_success_count / len(test_caseids)) if len(test_caseids) > 0 else 0.0,
        "nbg_ms": round(nbg_ms, 2),
        "val_total_ms": round(val_total_ms, 2),
        "val_per_case_ms": round(val_per_case_ms, 2),
        "test_total_ms": round(test_total_ms, 2),
        "test_per_case_ms": round(test_per_case_ms, 2),
        "total_sec": round(total_execution_sec, 4),
        "training_time_sec": round(val_total_ms / 1000, 4),
        "total_repair_time_sec": round(test_total_ms / 1000, 4),
        "total_run_time_sec": round(total_execution_sec, 4),
        **_timing_summary(val_timing, "val"),
        **_timing_summary(test_timing, "test"),
        "q_table_size": q_table_size,
        "model_path": str(model_path) if model_path is not None else None,
    }
    if nbgs_override is not None:
        stats["nbg_source"] = "override"
    if input_paths is not None:
        stats["input_paths"] = {k: str(v) for k, v in input_paths.items()}

    if save_outputs:
        output_dir = PROJECT_ROOT / "outputs" / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{data_name}_{num_epi}_ppm{alpha}_valratio{val_ratio}"

        repaired_log_a.to_csv(output_dir / f"{base_name}_repaired_log_a.csv", index=False)
        pd.DataFrame([stats]).to_csv(output_dir / f"{base_name}_repaired_log_a_stats.csv", index=False)

        if verbose:
            print(f"\n결과 저장 완료 -> {output_dir / (base_name + '_repaired_log_a.csv')}")
            print(f"통계 저장 완료 -> {output_dir / (base_name + '_repaired_log_a_stats.csv')}")

    if verbose:
        print("\n" + "=" * 70)
        print("run_rpbear 완료 (Method A)")
        print(f"Validation p.anomaly: {stats['val_p_anomaly'] * 100:.2f}%")
        print(f"Test p.anomaly      : {stats['test_p_anomaly'] * 100:.2f}%")
        print(f"전체 실행 시간       : {total_execution_sec:.4f} sec")
        print("=" * 70)

    if return_case_timing:
        return repaired_log_a, stats, pd.DataFrame(val_timing + test_timing,
                                                    columns=["case_id", "phase", "path", "repair_time_ms"])
    return repaired_log_a, stats
