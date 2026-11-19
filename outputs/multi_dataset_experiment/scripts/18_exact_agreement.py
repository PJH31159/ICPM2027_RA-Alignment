"""18_exact_agreement.py — B-C (또는 B-C+PN) exact agreement만 빠르게 계산 (alignment 탐색 없음)

왜: BPIC 2017 all-optimal 분석(07)은 무겁고 메모리 부족으로 중단된 적이 있다 (2026-11-16).
    exact agreement는 공식 B repaired trace와 공식 C repaired trace를 비교하기만 하면 되므로,
    all-optimal 결과와 상관없이 먼저 확보한다.
정의: 07_pub_all_optimal_agreement.py와 동일
    - B = --b-dir 의 *_alignment_per_case.csv, type == test 행의 repaired_trace (">>" 구분)
    - X = repaired log 의 type == test 행, Start / End 제외, case별 activity tuple
    - exact agreement = X trace == B trace 인 test case 수 / test case 수
검증: --check-road 를 주면 Road 0.20 에서 07 결과(agreement_summary.json의 B-C exact)와 같은지 확인한다.

실행: python outputs/multi_dataset_experiment/scripts/18_exact_agreement.py --b-dir ... --x-log ... --label C --out <json>
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
           "definition": "exact agreement = official X repaired trace == official B repaired trace (07과 같은 정의)"}
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
