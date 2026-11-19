"""Credit Method B: scipy LP solver (shim) vs official run (cvxopt), case by case.

Re-runs Credit with the same code as the run-b stage (utils.alignment_utils.alignment_repair) under the
scipy shim and compares it with the official Method B run 1 (cvxopt) on the test cases (val as a reference).

Criterion: all test repaired traces identical AND all alignment costs identical
    -> "official-equivalent for this experiment".
Visited / queued states may depend on the LP values; they are recorded but not used in the decision.

Output: outputs/simulation/Credit/scipy_equivalence_check/ (official runs/ folders are only read)
Usage:  .venv/Scripts/python outputs/multi_dataset_experiment/scripts/06_credit_scipy_equivalence.py
"""
import sys
import types

sys.modules["cvxopt"] = None
import lxml  # noqa: E402
from lxml import etree as _etree  # noqa: E402

_obj = types.ModuleType("lxml.objectify")
_obj.parse, _obj.fromstring = _etree.parse, _etree.fromstring
sys.modules["lxml.objectify"] = _obj
lxml.objectify = _obj

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pm4py  # noqa: E402
from pm4py.util.lp import solver as lp_solver  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import dataset_config as dc  # noqa: E402
from utils.alignment_utils import alignment_repair  # noqa: E402
from utils.data_utils import load_case_traces  # noqa: E402
from utils.discovery_utils import discover_model, model_report  # noqa: E402
from utils.validation_utils import environment_info, read_ids, read_json  # noqa: E402

CREDIT = PROJECT_ROOT / "outputs/simulation/Credit"
OFF = CREDIT / "runs/methodB_run1"
PFX = "credit-card_IM_noise0_alignment_run1_"
OUT = CREDIT / "scipy_equivalence_check"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {r["role"]: r for r in read_json(CREDIT / "inputs.json")["files"]}
    anomaly_log = Path(files["anomaly_log"]["resolved_path"])
    traces, labels, clean_ids, anomalous_ids = load_case_traces(anomaly_log)
    traces.index, labels.index = traces.index.map(str), labels.index.map(str)
    clean_ids = [str(c) for c in clean_ids]
    test_ids = read_ids(OFF / f"{PFX}test_case_ids.txt")
    val_ids = read_ids(OFF / f"{PFX}val_case_ids.txt")
    tree, net, im, fm, disc_sec = discover_model(traces, clean_ids, dc.DISCOVERY["noise_threshold"])
    mc, lang = model_report(tree, net, im, fm, traces, clean_ids, disc_sec,
                            dc.get_dataset("Credit")["trace_generation"]["safety_max_traces"])
    ref_mc = read_json(OFF / f"{PFX}model_check.json")
    model_same = all(mc[k] == ref_mc[k] for k in ["n_places", "n_transitions", "n_invisible_transitions",
                                                  "n_arcs", "process_tree"])
    t0 = time.perf_counter()
    rows_log, rows_case = alignment_repair(traces, labels, [("test", test_ids), ("val", val_ids)], net, im, fm,
                                           set(map(tuple, lang)) if lang is not None else None)
    align_sec = time.perf_counter() - t0
    new = pd.DataFrame(rows_case)
    new.drop(columns=["align_ms"]).to_csv(OUT / "scipy_alignment_per_case.csv", index=False)
    new_log = pd.DataFrame(rows_log)
    new_log.to_csv(OUT / "scipy_repaired_log_b.csv", index=False)

    off = pd.read_csv(OFF / f"{PFX}alignment_per_case.csv", dtype={"case:concept:name": str})
    off_log = pd.read_csv(OFF / f"{PFX}repaired_log_b.csv", dtype={"case:concept:name": str})
    new["case:concept:name"] = new["case:concept:name"].astype(str)
    m = off.merge(new, on=["case:concept:name", "type"], suffixes=("_official", "_scipy"), how="outer", indicator=True)
    cols = ["alignment_cost", "repaired_trace", "fitness", "alignment_failed", "n_sync_moves", "n_log_moves",
            "n_model_moves_visible", "n_tau_moves", "visited_states", "queued_states"]
    for c in cols:
        a, b = m[f"{c}_official"], m[f"{c}_scipy"]
        if c == "fitness":
            m[f"same_{c}"] = (a.astype(float) - b.astype(float)).abs() < 1e-12
        elif c in ("repaired_trace",):
            m[f"same_{c}"] = a.astype(str) == b.astype(str)
        else:
            m[f"same_{c}"] = a.astype(float) == b.astype(float)
    m.to_csv(OUT / "case_by_case_comparison.csv", index=False)

    # B output: row-level comparison of repaired_log_b.csv (activity sequence per case, order, type, failed flag)
    key = ["case:concept:name", "concept:name", "order", "type", "alignment_failed"]
    nl = new_log.copy()
    nl["case:concept:name"] = nl["case:concept:name"].astype(str)
    log_equal = off_log[key].astype(str).reset_index(drop=True).equals(nl[key].astype(str).reset_index(drop=True))

    summary = {"date": "2026-11-10", "purpose": "Credit Method B scipy(shim) vs official cvxopt equivalence",
               "lp_solver_used": lp_solver.DEFAULT_LP_SOLVER_VARIANT, "cvxopt_hidden": sys.modules.get("cvxopt") is None,
               "pm4py": pm4py.__version__, "official_run": str(OFF.relative_to(PROJECT_ROOT)),
               "official_solver": "cvxopt (cvxopt_solver_custom_align, pm4py default when cvxopt importable)",
               "anomaly_log": str(anomaly_log), "model_same_as_official": model_same,
               "only_in_one_side": int((m["_merge"] != "both").sum()), "alignment_sec": round(align_sec, 4),
               "repaired_log_b_rows_identical": bool(log_equal), "split": {}}
    for t in ("test", "val"):
        s = m[m["type"] == t]
        summary["split"][t] = {"cases": int(len(s))} | {f"same_{c}": int(s[f"same_{c}"].sum()) for c in cols}
    tst = summary["split"]["test"]
    summary["verdict_rule"] = "test 210/210 repaired trace AND 210/210 alignment cost"
    summary["official_equivalent"] = bool(model_same and tst["cases"] == 210 and tst["same_repaired_trace"] == 210
                                          and tst["same_alignment_cost"] == 210 and summary["only_in_one_side"] == 0)
    summary["environment"] = environment_info()
    (OUT / "equivalence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "environment"}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
