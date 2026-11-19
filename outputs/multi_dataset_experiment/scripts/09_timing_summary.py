"""Collect stored timing values of Credit / Pub into a common schema (no re-runs, no estimates).

Every dataset / method reports:
  A. training_ms                  : time to train the agent on the training data (ms)
  B. test_repair_mean_ms_per_case : time to produce a repaired test trace, mean per case (ms)

Timed sections (from src/sim.py and src/simulation.py):
  A / C / C+PN  training_ms     = wall-clock of the training ("validation") phase = `val_total_ms` in stats.json
                                  (excludes data loading / NBG construction / agent saving / test / evaluation)
                test (official) = test phase wall-clock `test_total_ms` (15-worker multiprocessing) / test cases,
                                  plus per-case timers (case_timing.csv, perf_counter inside the worker, includes CPU sharing)
                test (sequential) = per-case time of a single-process test-only replay with the trained agent
                C+PN test time includes the NBG decision and the Petri-net gate.
  B             training_ms     = N/A (no RL training; discovery is reported as preprocessing)
                test            = per-case align_case + repair_from_alignment time (perf_counter, sequential)
Preprocessing (not mixed into training_ms): discovery_ms, generation_ms, nbg_build_ms (if stored).

Missing values stay empty (None) with the reason in the note. Values keep the precision stored in the source files.
Output: outputs/multi_dataset_experiment/timing_summary.{csv,json}
"""
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIM = PROJECT_ROOT / "outputs/simulation"
OUT = PROJECT_ROOT / "outputs/multi_dataset_experiment"

SCOPE = {
    "rl_train": "training ('validation') phase wall-clock, sim.py val_total_ms (excludes data load / NBG / save / test)",
    "rl_test_official": 'mean / median = per-case timer (official run, perf_counter inside the workers of 15-worker multiprocessing, includes CPU sharing); test_wall_ms_per_case = test phase wall-clock / test cases (parallel throughput, original paper formula)',
    "rl_test_replay": 'sequential test-only replay (trained agent, 1 process); mean/median = per-case timer, all cases (RL + memory reuse)',
    "b_test": 'per-case align_case + repair_from_alignment (perf_counter, 1 process, sequential), test cases',
}


def rel(p):
    return str(Path(p).relative_to(PROJECT_ROOT)).replace("\\", "/")


def one(glob_dir, pattern):
    hits = sorted(Path(glob_dir).glob(pattern))
    return hits[0] if hits else None


def pre(ds):
    g = SIM / ds / "generate/generate_summary.json"
    gen = json.loads(g.read_text(encoding="utf-8")) if g.exists() else {}
    return {"generation_ms": round(gen["generation_time_sec"] * 1000, 4) if "generation_time_sec" in gen else None}


def rl_official(ds, method, d, solver_note):
    st = json.loads(one(d, "*_stats.json").read_text(encoding="utf-8"))
    ct = pd.read_csv(one(d, "*_case_timing.csv"))
    te = ct[ct.phase == "test"]
    n_test = int(st["n_test_cases"])
    return {
        "dataset": ds, "method": method, "run_id": 1, "timing_source": "official_run", "solver": solver_note,
        "train_cases": int(st["n_val_cases"]), "test_cases": n_test,
        "training_ms": st["val_total_ms"], "training_scope": SCOPE["rl_train"],
        "test_repair_total_ms": st["test_total_ms"],
        "test_wall_ms_per_case": round(st["test_total_ms"] / n_test, 4),
        "test_repair_mean_ms_per_case": round(float(te.repair_time_ms.mean()), 4),
        "test_repair_median_ms_per_case": round(float(te.repair_time_ms.median()), 4),
        "test_rl_path_cases": int((te.path == "RL").sum()),
        "test_rl_path_timer_mean_ms": round(float(te[te.path == "RL"].repair_time_ms.mean()), 4),
        "test_scope": SCOPE["rl_test_official"], "num_cores": st.get("num_cores"), "num_epi": st.get("num_epi"),
        "q_table_size": st.get("q_table_size"), "discovery_ms": None, **pre(ds), "nbg_build_ms": None,
        "nbg_ms_in_run": st.get("nbg_ms"), "source": rel(one(d, "*_stats.json")) + " ; " + rel(one(d, "*_case_timing.csv")),
        "note": 'nbg_ms_in_run=0: Method C/C+PN receive the NBGs built by the nbg stage as override and build no NBG inside the run; the nbg-stage build time is not stored' if method != "A" else
                'Method A: time to build the Original NBG inside the run = nbg_ms_in_run',
    }


def b_official(ds, d, solver_note):
    st = json.loads(one(d, "*_stats.json").read_text(encoding="utf-8"))
    tm = pd.read_csv(one(d, "*_alignment_timing.csv"))
    te = tm[tm.type == "test"]
    return {
        "dataset": ds, "method": "B", "run_id": 1, "timing_source": "official_run", "solver": solver_note,
        "train_cases": None, "test_cases": int(st["n_test"]), "training_ms": None, "training_scope": 'N/A (no RL training)',
        "test_repair_total_ms": round(float(te.align_ms.sum()), 4),
        "test_wall_ms_per_case": round(float(te.align_ms.sum()) / len(te), 4),
        "test_repair_mean_ms_per_case": round(float(te.align_ms.mean()), 4),
        "test_repair_median_ms_per_case": round(float(te.align_ms.median()), 4), "test_rl_path_cases": None, "test_rl_path_timer_mean_ms": None,
        "test_scope": SCOPE["b_test"], "num_cores": 1, "num_epi": None, "q_table_size": None,
        "discovery_ms": round(st["discovery_sec"] * 1000, 4), "generation_ms": None, "nbg_build_ms": None, "nbg_ms_in_run": None,
        "source": rel(one(d, "*_stats.json")) + " ; " + rel(one(d, "*_alignment_timing.csv")),
        "note": 'discovery_ms = IM discovery + Petri-net conversion (preprocessing, not training)',
    }


def replay(ds, method, f):
    s = json.loads(f.read_text(encoding="utf-8"))
    if method == "B":
        c = s["test_case_ms"]
        mean, med, n, total = c["mean"], c["median"], c["n"], s.get("test_phase_wall_ms")
        rl_n = rl_mean = None
    else:
        t = s["timing_summary"]
        mean, med, n, total = t["test_case_ms_all_mean"], t["test_case_ms_all_median"], t["test_case_ms_all_n"], t["test_phase_wall_ms"]
        rl_n, rl_mean = t["test_case_ms_rl_n"], t["test_case_ms_rl_mean"]
    return {"dataset": ds, "method": method, "run_id": 1, "timing_source": f"sequential_replay_{f.parent.name.split('_')[0]}",
            "solver": "cvxopt" if method == "B" else None, "train_cases": None, "test_cases": n, "training_ms": None,
            "training_scope": 'the replay does not train (see the official training_ms row)', "test_repair_total_ms": total,
            "test_wall_ms_per_case": round(total / n, 4) if total else None,
            "test_repair_mean_ms_per_case": mean, "test_repair_median_ms_per_case": med,
            "test_rl_path_cases": rl_n, "test_rl_path_timer_mean_ms": rl_mean,
            "test_scope": SCOPE["b_test"] if method == "B" else SCOPE["rl_test_replay"], "num_cores": 1,
            "num_epi": None, "q_table_size": None, "discovery_ms": None, "generation_ms": None, "nbg_build_ms": None,
            "nbg_ms_in_run": None, "source": rel(f), "note": 'timing-only replay (not an accuracy result)'}


def main():
    rows = []
    C, P = SIM / "Credit", SIM / "Pub"
    rows.append(rl_official("Credit", "A", C / "runs/methodA_run1", None))
    rows.append(b_official("Credit", C / "runs/methodB_run1", "cvxopt (official)"))
    rows.append(rl_official("Credit", "C", C / "runs/methodC_run1", None))
    rows.append(rl_official("Credit", "C+PN", C / "C_PN_Gate/run1", None))
    rows.append(b_official("Pub", P / "runs/methodB_run1", "scipy (shim; Credit 210/210 equivalent)"))
    rows.append(rl_official("Pub", "C", P / "runs/methodC_run1", None))
    rows.append(rl_official("Pub", "C+PN", P / "C_PN_Gate/run1", None))
    for m, sub in (("A", "test_timing/rep{}_A_alpha0"), ("B", "test_timing/rep{}_B"), ("C", "test_timing/rep{}_C_alpha0"),
                   ("C+PN", "C_PN_Gate/test_timing/rep{}_CPN")):
        for r in (1, 2, 3):
            f = C / sub.format(r) / "timing_summary.json"
            if f.exists():
                rows.append(replay("Credit", m, f))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "timing_summary.csv", index=False)
    missing = ['Pub: no sequential test-only replay (A/B/C/C+PN) -> not measured (official run values only)',
               'Pub: no Method A official run',
               'Credit / Pub: the nbg-stage NBG build time (nbg_build_ms) is not stored',
               'Credit / Pub: the discovery time of the generate stage is not in generate_summary (only discovery_sec of the Method B stats)']
    (OUT / "timing_summary.json").write_text(json.dumps({
        "definition": {"training_ms": SCOPE["rl_train"] + "; Method B = N/A",
                       "test_repair_mean_ms_per_case": {"official_run (A/C/C+PN)": SCOPE["rl_test_official"],
                                                        "sequential_replay (D037)": SCOPE["rl_test_replay"], "B": SCOPE["b_test"]},
                       "preprocessing (training과 분리)": "discovery_ms / generation_ms / nbg_build_ms"},
        "missing_values": missing, "rows": rows}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    show = ["dataset", "method", "timing_source", "train_cases", "test_cases", "training_ms", "test_wall_ms_per_case",
            "test_repair_mean_ms_per_case", "test_repair_median_ms_per_case", "test_rl_path_timer_mean_ms"]
    print(df[show].to_string())


if __name__ == "__main__":
    main()
