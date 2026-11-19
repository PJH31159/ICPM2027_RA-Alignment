"""
run_utils — sim.run_rpbear 호출을 감싸는 공통 실행 관리 (REF-001)

출처: src/60 execute_run, src/63 stage_run / stage_smoke (같은 방식)

- Method A: nbgs_override=None (Original NBG, 기존 RP-BEAR 동작 그대로)
- Method C: nbgs_override=New NBG (D034)
- 공식 run은 sim.run_rpbear(save_outputs=True)로 실행하고(Method A와 같은 코드 경로),
  sim이 고정 경로에 저장한 3개 파일을 run 디렉터리로 옮긴다. run 디렉터리가 이미 있거나
  고정 경로 파일이 남아 있으면 실행하지 않는다 (덮어쓰기 금지).
- continual / cumulative Q-table learning은 구현하지 않는다 (dataset_config.CONTINUAL_LEARNING).
  나중에 시작 Q-table을 넘기게 되면 execute_official_run에 인자를 추가하는 방식으로 확장한다.
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
    """run_rpbear와 같은 순서(seed -> 데이터 로드 -> split)로 val/test case id를 계산한다.
    (데이터 로드와 NBG 생성은 python random을 쓰지 않는다: VAL-A01 V0)"""
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
    """smoke 용: save_outputs=False (고정 경로 파일을 만들지 않음).

    return_case_timing=True이면 (df, stats, case_timing)을 반환한다 (기본은 기존과 같이 (df, stats)).
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
    """sim.run_rpbear(save_outputs=True)가 저장하는 고정 경로 (sim.py 규칙과 같음)."""
    base = f"{data_name}_{params['num_epi']}_ppm{params['alpha']}_valratio{params['val_ratio']}"
    return {
        "repaired_log.csv": PROJECT_ROOT / "outputs" / "tables" / f"{base}_repaired_log_a.csv",
        "stats.csv": PROJECT_ROOT / "outputs" / "tables" / f"{base}_repaired_log_a_stats.csv",
        "agent.pkl": PROJECT_ROOT / "outputs" / "models" / f"{base}_seed{params['seed']}_agent.pkl",
    }


def execute_official_run(data_name, input_paths, params, method, run_dir, file_prefix, nbgs_override,
                         extra_stats=None):
    """공식 run 1회. return: exit code (0 = 성공)."""
    import pandas as pd
    from multiprocessing import cpu_count

    params = dict(params)
    if params.get("num_cores") is None:
        params["num_cores"] = max(1, cpu_count() - 1)
    fixed = fixed_output_paths(data_name, params)

    problems = []
    if ".venv-rpbear" not in sys.prefix:
        problems.append(f"환경이 .venv-rpbear가 아님: {sys.prefix}")
    if not pd.__version__.startswith("2."):
        problems.append(f"pandas 2.x 필요 (현재 {pd.__version__})")
    if Path(run_dir).exists():
        problems.append(f"run 디렉터리가 이미 있음 (덮어쓰지 않음): {run_dir}")
    for path in fixed.values():
        if path.exists():
            problems.append(f"sim 고정 출력 파일이 이미 있음 (다른 실행과 섞일 수 있음): {path}")
    if method == "C" and nbgs_override is None:
        problems.append("Method C에는 New NBG(nbgs_override)가 필요")
    if method == "A" and nbgs_override is not None:
        problems.append("Method A는 nbgs_override=None이어야 함")
    if problems:
        print("[ABORT] 실행 전 검사 실패:")
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
            print("[FAILED] run_rpbear 실행 중 오류:")
            traceback.print_exc(file=sys.stdout)
            return 1
        wall = time.time() - t0

        missing = [n for n, p in fixed.items() if not p.exists() or p.stat().st_size == 0]
        if missing:
            print(f"[FAILED] 생성되지 않은 출력: {missing}")
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
        # timing instrumentation: case별 처리 시간 (case_id, phase, path = RL / Mem, repair_time_ms)
        case_timing.to_csv(f("case_timing.csv"), index=False)
        val_ids, test_ids = split_ids_from_repaired(df)
        f("val_case_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
        f("test_case_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
        print(f"\n[SUMMARY] val p.anomaly {stats_out['val_p_anomaly']:.4f} | test p.anomaly "
              f"{stats_out['test_p_anomaly']:.4f} | Q-table {stats_out['q_table_size']} | {wall / 60:.1f} min")
    return 0
