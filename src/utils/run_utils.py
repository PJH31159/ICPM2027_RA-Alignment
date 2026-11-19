"""Execution management around sim.run_rpbear.

- Method A: nbgs_override=None (Original NBG, unchanged RP-BEAR behaviour).
- Method C: nbgs_override=New NBG.
- Official runs call sim.run_rpbear(save_outputs=True) (same code path as Method A) and move the
  three files that sim writes to fixed paths into the run directory. Nothing is run if the run
  directory already exists or fixed-path files are left over (no overwriting).
- Continual / cumulative Q-table learning is not implemented; each dataset is trained independently.
"""

import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from utils.validation_utils import PROJECT_ROOT, Tee, environment_info, write_json

METHOD_FILE_SUFFIX = {"A": "repaired_log_a.csv", "C": "repaired_log_c.csv"}


def expected_split(data_name, input_paths, val_ratio, seed):
    """Compute the val/test case ids in the same order as run_rpbear (seed -> data load -> split).

    Data loading and NBG construction do not consume Python's random state, so the split depends only on the seed.
    """
    import numpy as np
    from sim import _load_and_prepare_data, _prepare_case_split

    random.seed(seed)
    np.random.seed(seed)
    _c, anomaly, _g, _a, _f, _af = _load_and_prepare_data(data_name, input_paths)
    _nolabel, _c2t, val_ids, test_ids = _prepare_case_split(anomaly, val_ratio)
    return [str(x) for x in val_ids], [str(x) for x in test_ids]


def split_ids_from_repaired(df):
    val = sorted(map(str, df.loc[df["type"] == "val", "case:concept:name"].unique()))
    test = sorted(map(str, df.loc[df["type"] == "test", "case:concept:name"].unique()))
    return val, test


def quick_run(data_name, input_paths, params, num_epi, use_multiprocessing, nbgs_override,
              return_case_timing=False):
    """Smoke run: save_outputs=False (no fixed-path files are written).

    With return_case_timing=True returns (df, stats, case_timing); otherwise (df, stats).
    """
    from sim import run_rpbear

    t0 = time.time()
    out = run_rpbear(data_name=data_name, num_epi=num_epi, alpha=params["alpha"],
                     val_ratio=params["val_ratio"], seed=params["seed"], max_step=params["max_step"],
                     num_cores=params["num_cores"], use_multiprocessing=use_multiprocessing,
                     save_outputs=False, verbose=False, nbgs_override=nbgs_override,
                     input_paths=input_paths, return_case_timing=return_case_timing)
    out[1]["wall_sec"] = round(time.time() - t0, 2)
    return out


def fixed_output_paths(data_name, params):
    """Fixed output paths written by sim.run_rpbear(save_outputs=True)."""
    base = f"{data_name}_{params['num_epi']}_ppm{params['alpha']}_valratio{params['val_ratio']}"
    return {
        "repaired_log.csv": PROJECT_ROOT / "outputs" / "tables" / f"{base}_repaired_log_a.csv",
        "stats.csv": PROJECT_ROOT / "outputs" / "tables" / f"{base}_repaired_log_a_stats.csv",
        "agent.pkl": PROJECT_ROOT / "outputs" / "models" / f"{base}_seed{params['seed']}_agent.pkl",
    }


def execute_official_run(data_name, input_paths, params, method, run_dir, file_prefix, nbgs_override,
                         extra_stats=None):
    """One official run. Returns an exit code (0 = success)."""
    import pandas as pd
    from multiprocessing import cpu_count

    params = dict(params)
    if params.get("num_cores") is None:
        params["num_cores"] = max(1, cpu_count() - 1)
    fixed = fixed_output_paths(data_name, params)

    problems = []
    if ".venv-rpbear" not in sys.prefix:
        problems.append(f"environment is not .venv-rpbear: {sys.prefix}")
    if not pd.__version__.startswith("2."):
        problems.append(f"pandas 2.x required (current {pd.__version__})")
    if Path(run_dir).exists():
        problems.append(f"run directory already exists (not overwritten): {run_dir}")
    for path in fixed.values():
        if path.exists():
            problems.append(f"fixed sim output file already exists (could mix with another run): {path}")
    if method == "C" and nbgs_override is None:
        problems.append('Method C requires New NBGs (nbgs_override)')
    if method == "A" and nbgs_override is not None:
        problems.append('Method A requires nbgs_override=None')
    if problems:
        print('[ABORT] pre-run checks failed:')
        for p in problems:
            print("  -", p)
        return 1

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)

    def f(suffix):
        return run_dir / f"{file_prefix}_{suffix}"

    from sim import run_rpbear

    with Tee(f("execution_log.txt")):
        print("=" * 70)
        print(f"Method {method} official run | data={data_name} | {params}")
        print("=" * 70)
        write_json(f("environment.json"), environment_info())
        started, t0 = datetime.now(), time.time()
        try:
            df, stats, case_timing = run_rpbear(
                data_name=data_name, num_epi=params["num_epi"], alpha=params["alpha"],
                val_ratio=params["val_ratio"], seed=params["seed"],
                max_step=params["max_step"], num_cores=params["num_cores"],
                use_multiprocessing=params["use_multiprocessing"], save_outputs=True,
                verbose=True, nbgs_override=nbgs_override, input_paths=input_paths,
                return_case_timing=True)
        except Exception:  # noqa: BLE001
            print('[FAILED] error during run_rpbear:')
            traceback.print_exc(file=sys.stdout)
            return 1
        wall = time.time() - t0

        missing = [n for n, p in fixed.items() if not p.exists() or p.stat().st_size == 0]
        if missing:
            print(f"[FAILED] outputs not created: {missing}")
            return 1
        moved = {"repaired_log.csv": METHOD_FILE_SUFFIX[method], "stats.csv": "stats.csv", "agent.pkl": "agent.pkl"}
        for key, path in fixed.items():
            shutil.move(str(path), str(f(moved[key])))

        stats_out = dict(stats)
        stats_out.update({"method": method, "repaired_rows": len(df), "wall_sec": round(wall, 4),
                          "model_path": str(f("agent.pkl")),
                          "started_at": started.isoformat(timespec="seconds"),
                          "finished_at": datetime.now().isoformat(timespec="seconds")})
        stats_out.update(extra_stats or {})
        write_json(f("stats.json"), stats_out)
        # per-case processing time (case_id, phase, path = RL / Mem, repair_time_ms)
        case_timing.to_csv(f("case_timing.csv"), index=False)
        val_ids, test_ids = split_ids_from_repaired(df)
        f("val_case_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
        f("test_case_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
        print(f"\n[SUMMARY] val p.anomaly {stats_out['val_p_anomaly']:.4f} | test p.anomaly "
              f"{stats_out['test_p_anomaly']:.4f} | Q-table {stats_out['q_table_size']} | {wall / 60:.1f} min")
    return 0
