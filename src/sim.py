"""RP-BEAR execution engine.

sim.py runs RP-BEAR once: data loading, NBG, train/test split, Q-learning training and test-phase
repair; it returns the repaired log. RA-Alignment training (utils/pn_gate_utils.py, tutorial 2) and
inference (tutorial 4) reuse its data loading, split and per-case functions.

The RP-BEAR core (NBG discovery, ADM, QLearningAgent, P_BEAR_RL) is imported unchanged from
src/rpbear/func_rpbear_final.py. This module does not implement a new RP-BEAR; it reproduces the
behaviour of the original RP-BEAR script as closely as possible inside this project's structure.
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
# 1. Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 2. Per-case processing (with repair memory)
# ============================================================
# Must be a top-level function so that multiprocessing.Pool can pickle it.
# The logic is the same as process_case_with_memory of the original RP-BEAR script.

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
    # Per-case processing time, measured inside the worker. perf_counter() consumes no random
    # numbers, so it does not affect the repair results.
    t0 = time.perf_counter()
    trace_str = case_to_trace[case_id]

    if trace_str in repair_memory:
        # trace already repaired: reuse the result from the repair memory (no RL run)
        result = repair_memory[trace_str].copy()
        result["type"] = case_type
        result["case:concept:name"] = case_id
        return case_id, result, 0, (time.perf_counter() - t0) * 1000  # 0: repair memory used, no RL run

    # new trace: run the RL repair
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
    return case_id, result, 1, (time.perf_counter() - t0) * 1000  # 1: new RL training / inference run


# ============================================================
# 3. Worker initialisation (seed reproducibility with multiprocessing)
# ============================================================
# The original RP-BEAR script seeds random / np.random at module level. With Windows spawn,
# every worker re-imports that script and therefore starts from the same seed. Here the seed is set
# inside run_rpbear() (parent process only); without an initializer the workers would be seeded from
# OS entropy. The Pool initializer therefore re-seeds every worker with the same seed.

def _worker_init(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 4. Data loading / clean-anomaly separation
# ============================================================

def _load_and_prepare_data(data_name: str, input_paths: dict | None = None):
    # input_paths=None: data/raw/{data_name}_1.00.csv and data/raw/{data_name}.csv
    # input_paths given: {"anomaly_log": Path, "ground_truth_log": Path}
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

    # ground truth (normal log) for the running accuracy printout
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
        print(f"[WARNING] could not load the normal data ({normal_data_path}); the running accuracy is skipped. Error: {e}")

    # add 'Start' and 'End' activities
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
# 5. Split anomalous cases into train ('val') / test (same logic as the original RP-BEAR script)
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
# 6. Progress printout (same format as the original RP-BEAR script)
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

        status = 'success(O)' if is_correct else 'failure(X)'
        acc_rate = (success_count / idx) * 100

        print(
            f"[{case_type.capitalize()} {idx:4d}/{total}] Case: {str(case_id):12s} | "
            f"path: {method_label} | repair: {status} | running p.anomaly: {acc_rate:6.2f}% "
            f"({success_count}/{idx})"
        )

    return success_count


# ============================================================
# 7. Run one phase (training or test)
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
    case_timing = []  # per-case processing time (phase, path = RL / Mem)
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
        # Debug / smoke path: single process, cases in order. The processing order differs from
        # Pool.imap_unordered, so the order of Q-table updates (and results that depend on alpha /
        # exploration) may differ from use_multiprocessing=True.
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
    """One row of the case-timing CSV. path: RL = new RL training/inference, Mem = repair-memory reuse."""
    return {"case_id": str(case_id), "phase": case_type, "path": "RL" if is_run else "Mem",
            "repair_time_ms": round(float(elapsed_ms), 4)}


def _timing_summary(case_timing, phase):
    """Per-phase case-time summary (all / RL / Mem: n, mean, median, std, min, max; ms).

    std is the sample standard deviation (ddof=1); None if there is at most one case.
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
# 8. Public entry point
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
    """Run the RP-BEAR pipeline and return the repaired log.

    use_multiprocessing=True  : reproduces the RP-BEAR baseline (Manager + Pool.imap_unordered).
    use_multiprocessing=False : debugging / smoke runs (single process, different processing order).

    nbgs_override : None (default) builds discover_NBG(clean) from the observed clean log (Method A).
                    A list of NBGs is used instead of discover_NBG (Method C: New NBGs built from
                    Petri-net-generated traces). The list must follow this function's actset order
                    (= first occurrence order of activities in the observed clean log), because the ADM
                    looks NBGs up by actset position. actset, act_freq and other inputs are unchanged.
    input_paths   : None reads data/raw/{data_name}_1.00.csv and data/raw/{data_name}.csv. A dict
                    {"anomaly_log": ..., "ground_truth_log": ...} reads those files instead.
                    Output file naming is unchanged.
    return_case_timing : False returns (repaired_log, stats); True returns (repaired_log, stats, case_timing),
                    where case_timing has one row per case (case_id, phase = val/test, path = RL/Mem, repair_time_ms).

    Timing fields in stats (they do not affect the repair results):
        training_time_sec      = wall-clock of the training ("validation") phase (= val_total_ms / 1000)
        total_repair_time_sec  = wall-clock of the test phase (= test_total_ms / 1000)
        total_run_time_sec     = total run_rpbear time (= total_sec)
        {val,test}_case_ms_{all,rl,mem}_{n,mean,median,std,min,max}
                               = per-case processing time measured inside the worker (ms);
                                 under multiprocessing these are times during parallel execution.

    On Windows, with use_multiprocessing=True the calling top-level script must be guarded by
    `if __name__ == "__main__":` (multiprocessing spawn).
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
        print('RP-BEAR-RL start (sim.run_rpbear, Method A)')
        print(
            f"data={data_name} | num_epi={num_epi} | alpha={alpha} | "
            f"val_ratio={val_ratio} | seed={seed} | max_step={max_step} | "
            f"use_multiprocessing={use_multiprocessing} | num_cores={num_cores}"
        )
        print("=" * 70)
        print(f"[INFO] Total anomalous cases = {len(val_caseids) + len(test_caseids)}")
        print(f"[INFO] Validation cases = {len(val_caseids)}")
        print(f"[INFO] Test cases       = {len(test_caseids)}")

    # Without an explicit shutdown the Manager leaves a background server process behind.
    # run_rpbear() may be called several times, so it is always cleaned up in try/finally.
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
        # Training phase ('validation' in the code; train_mode=True, Q-table updated)
        # ------------------------------------------------------
        if verbose:
            print('\n[1/2] Validation phase start...')

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
            print(f"\n[INFO] Validation finished: total={val_total_ms:.2f} ms | per_case={val_per_case_ms:.2f} ms")

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
                print(f"[INFO] Agent saved -> {model_path}")

        # ------------------------------------------------------
        # Test phase (train_mode=False, no Q-table update)
        # ------------------------------------------------------
        if verbose:
            print('\n[2/2] Test phase start...')

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
            print(f"\n[INFO] Test finished: total={test_total_ms:.2f} ms | per_case={test_per_case_ms:.2f} ms")

        # read the Q-table size while the manager is still alive (before shutdown)
        q_table_size = len(agent.q_table)
    finally:
        if manager is not None:
            manager.shutdown()

    # ------------------------------------------------------
    # Merge results
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
            print(f"\nResults saved -> {output_dir / (base_name + '_repaired_log_a.csv')}")
            print(f"Statistics saved -> {output_dir / (base_name + '_repaired_log_a_stats.csv')}")

    if verbose:
        print("\n" + "=" * 70)
        print('run_rpbear finished (Method A)')
        print(f"Validation p.anomaly: {stats['val_p_anomaly'] * 100:.2f}%")
        print(f"Test p.anomaly      : {stats['test_p_anomaly'] * 100:.2f}%")
        print(f"Total run time       : {total_execution_sec:.4f} sec")
        print("=" * 70)

    if return_case_timing:
        return repaired_log_a, stats, pd.DataFrame(val_timing + test_timing,
                                                    columns=["case_id", "phase", "path", "repair_time_ms"])
    return repaired_log_a, stats
