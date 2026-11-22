"""Fast exact agreement B-C (or B-C+PN), without any alignment search.

The all-optimal analysis (07) is expensive in time and memory; exact agreement only compares the official B
repaired trace with the official X repaired trace, so it can be computed independently.
Definition (same as all_optimal_agreement.py):
    - B = repaired_trace of the type == test rows of *_alignment_per_case.csv in --b-dir (">>"-separated)
    - X = type == test rows of the repaired log, Start / End removed, activity tuple per case
    - exact agreement = number of test cases with X trace == B trace / number of test cases
Check: --check-road verifies the result against the B-C exact value of 07 for Road 0.20 (agreement_summary.json).

Usage: python scripts/exact_agreement.py --b-dir <dir> --x-log <csv> --label C --out <json>
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def exact(b_dir, x_log):
    b_dir = PROJECT_ROOT / b_dir
    alg = pd.read_csv(next(b_dir.glob("*_alignment_per_case.csv")), dtype={"case:concept:name": str})
    alg = alg[alg.type == "test"].set_index("case:concept:name")
    ids = list(alg.index)
    df = pd.read_csv(PROJECT_ROOT / x_log, dtype=str, keep_default_na=False)
    df = df[(df["type"] == "test") & (~df["concept:name"].isin(["Start", "End"]))]
    tr = df.groupby("case:concept:name", sort=False)["concept:name"].agg(tuple)
    agree, missing = 0, 0
    for c in ids:
        b = tuple(alg.loc[c, "repaired_trace"].split(">>"))
        if c not in tr.index:
            missing += 1
            continue
        agree += tr[c] == b
    return {"n": len(ids), "exact": int(agree), "exact_percent": round(100 * agree / len(ids), 2),
            "x_missing_cases": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--x-log", required=True)
    ap.add_argument("--label", default="C")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-road", action="store_true")
    a = ap.parse_args()
    res = {"label": f"B-{a.label}", "b_dir": a.b_dir, "x_log": a.x_log, **exact(a.b_dir, a.x_log),
           "definition": 'exact agreement = official X repaired trace == official B repaired trace (same definition as 07)'}
    if a.check_road:
        rd = "outputs/simulation/RoadTraffic/threshold_0p20"
        s = json.loads((PROJECT_ROOT / rd / "alignment_all_optimal_analysis/run1/agreement_summary.json").read_text(encoding="utf-8"))
        road = exact(rd + "/runs/methodB_run1", s["c_log"])
        ref = s["agreement"]["B-C"]["counts"]["exact"]
        res["road_check"] = {"this_script": road["exact"], "script_07": ref, "n": road["n"], "ok": road["exact"] == ref}
    out = PROJECT_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
