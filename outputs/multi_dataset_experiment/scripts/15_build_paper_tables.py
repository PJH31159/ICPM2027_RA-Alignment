"""15_build_paper_tables.py — 저장된 실험 output만 읽어 논문용 결과표(CSV + Markdown)와 검증 결과를 만든다.

새 실험을 하지 않는다. 값이 output에 없으면 추정하지 않고 N/A로 둔다.
timing 단위는 모두 ms (Training = ms, Test Repair = mean ms/case). 병렬 throughput(wall/n)은 main 표에 쓰지 않는다.
main repair 시간 = 순차(sequential) Test-only 측정:
  B     : alignment + repair, 1 process, scipy LP (shim). threshold sweep은 14 script, 공식 replay는 test-timing-b
  C/C+PN: 학습된 agent로 Test만 순차 재실행 (test-timing / test-timing-pngate, D037). test 전체 case 평균(RL + Mem path)

출력 (outputs/multi_dataset_experiment/):
  threshold_sweep_summary.csv / .md
  paper_threshold_model_table.csv / .md, paper_threshold_timing_table.csv / .md,
  paper_threshold_agreement_table.csv / .md, paper_threshold_tradeoff_table.csv / .md,
  threshold_alignment_speed.csv, threshold_c_speed.csv,
  paper_result_table.csv / .md, table_validation.json
실행: python outputs/multi_dataset_experiment/scripts/15_build_paper_tables.py
"""
import json
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MD = PROJECT_ROOT / "outputs/multi_dataset_experiment"
SIM = PROJECT_ROOT / "outputs/simulation"
SWEEP = MD / "threshold_sweep"
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
SELECTED = {"RoadTraffic": 0.20, "BPIC2017": 0.10}
# BPIC 2017 (2026-11-12 교수님 방향 수정): coverage >= 90% 기준 폐기 -> "best executable trade-off".
# loop bound는 95% 규칙이 아니라 17_bpic_feasibility_tables.py의 후보 탐색에서 고른 값 (50k language 안 최대 coverage).
# BPIC 2017 C+PN: In progress (pending final C+PN result). 최종값이 나오기 전까지 표에는 N/A로 둔다.
BPIC_BOUND = "(1,9)"
BPIC_CPN_NOTE = "C+PN: In progress (pending final C+PN result)"
SAME_MODEL = {("RoadTraffic", 0.30): 0.20, ("BPIC2017", 0.30): 0.20}  # Process Tree가 0.20과 동일 (md5 확인)
NA = "N/A"


def tag(th):
    return f"threshold_{th:.2f}".replace(".", "p")


def rj(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def first(glob_dir, pattern):
    d = Path(glob_dir)
    fs = sorted(d.glob(pattern)) if d.exists() else []
    return fs[0] if fs else None


def pct(a, b):
    return None if a is None or b in (None, 0) else round(100 * a / b, 2)


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return NA
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e7:
            return f"{int(v):,}"
        if abs(v) >= 1e7 and (v.is_integer() or abs(v) >= 1e12):  # 큰 language 추정치만 지수 표기 (ms 값은 그대로)
            return f"{v:.3g}"
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def cnt(c, n):
    return NA if c is None else f"{int(c)} / {int(n)} ({100 * c / n:.2f}%)"


# ------------------------------------------------------------------ 결과 읽기
def b_sweep(ds, th):
    src = SAME_MODEL.get((ds, th), th) if ds == "BPIC2017" else th
    s = rj(SIM / ds / tag(src) / "b_alignment_timing" / "summary.json")
    if s and src != th:
        s = dict(s, note=f"identical Process Tree to {src:.2f}: measured once at {src:.2f}")
    return s


REPLAY_LABELS = {  # 순차 Test-only replay 폴더. Credit은 기존 D037 replay 3회 (평균 사용)
    "Credit": {"C": ["rep1_C_alpha0", "rep2_C_alpha0", "rep3_C_alpha0"], "CPN": ["rep1_CPN", "rep2_CPN", "rep3_CPN"]},
}


def rl_run(base, kind):
    """kind C: runs/methodC_run1 + test_timing/<label> ; CPN: C_PN_Gate/run1 + C_PN_Gate/test_timing/<label>."""
    run = base / ("runs/methodC_run1" if kind == "C" else "C_PN_Gate/run1")
    tj = first(run, "*_timing.json")
    t = rj(tj) if tj else None
    labels = REPLAY_LABELS.get(base.name, {}).get(kind, ["seq_C" if kind == "C" else "seq_CPN"])
    tdir = base / ("test_timing" if kind == "C" else "C_PN_Gate/test_timing")
    reps = [rj(tdir / lb / "timing_summary.json") for lb in labels]
    reps = [r["timing_summary"] for r in reps if r]
    ts = None
    if reps:
        ts = {k: round(sum(r[k] for r in reps) / len(reps), 4) for k in ("test_case_ms_all_mean", "test_case_ms_all_median")}
        ts["test_case_ms_all_n"] = reps[0]["test_case_ms_all_n"]
    train = t["training_ms"] if t else None
    if train is None:  # timing.json 도입(2026-11-11) 전 run: stats.json의 val_total_ms (같은 정의, timing_summary.csv와 같음)
        sj = first(run, "*_stats.json")
        train = rj(sj).get("val_total_ms") if sj else None
    return {"training_ms": train, "replays": len(reps),
            "seq_mean": ts["test_case_ms_all_mean"] if ts else None,
            "seq_median": ts["test_case_ms_all_median"] if ts else None,
            "seq_n": ts["test_case_ms_all_n"] if ts else None,
            "official_parallel_mean": t["test_repair_mean_ms_per_case"] if t else None,
            "timing_json": str(tj.relative_to(PROJECT_ROOT)) if tj else None}


def b_official_replay(base, label="seq_B"):
    s = rj(base / "test_timing" / label / "timing_summary.json")
    if not s:
        return None
    c = pd.read_csv(base / "test_timing" / label / "case_timing.csv")["repair_time_ms"]
    return {"mean": s["test_case_ms"]["mean"], "median": s["test_case_ms"]["median"], "n": s["test_case_ms"]["n"],
            "p95": round(float(c.quantile(0.95)), 4), "total": round(float(c.sum()), 4)}


def state_limit_note(base):
    """all-optimal exact search가 state 상한에 걸린 case 수 (그 case는 불일치로 셈 -> agreement는 하한)."""
    s = rj(base / "alignment_all_optimal_analysis/run1/agreement_summary.json")
    k = s["checks"].get("state_limit_hit", 0) if s else 0
    rl = s["checks"].get("resource_limit_hit", 0) if s else 0  # BPIC: 메모리 위험으로 탐색을 끝내지 못한 case (07 --resource-skip)
    if not k and not rl:
        return None
    txt = f"all-optimal search hit max_states for {k}/{s['cases']} cases"
    if rl:
        txt += f" and stopped at the memory limit for {rl} cases (resource-limit)"
    return txt + f" (counted as disagreement; all-optimal agreement is a lower bound, upper bound +{k + rl})"


def agreement(base):
    s = rj(base / "alignment_all_optimal_analysis/run1/agreement_summary.json")
    if not s:
        return None
    out = {"n": s["cases"]}
    for m, key in (("C", "B-C"), ("CPN", "B-CPN")):
        c = s["agreement"].get(key, {}).get("counts")
        out[m] = c
    return out


# ------------------------------------------------------------------ threshold sweep
def sweep_rows():
    rows = json.loads((SWEEP / "sweep_structure_language.json").read_text(encoding="utf-8"))
    road_feas = rj(SIM / "RoadTraffic/feasibility_11/road_11_feasibility_summary.json")
    out = []
    for r in rows:
        ds, th = r["Dataset"], float(r["NoiseThreshold"])
        base = SIM / ds / tag(th)
        b = b_sweep(ds, th)
        sel = SELECTED.get(ds) is not None and abs(SELECTED[ds] - th) < 1e-9
        nbg = rj(base / "nbg/nbg_comparison_summary.json")
        nbg_ms = nbg["timing_ms"]["new_nbg_build_ms"] if nbg and "timing_ms" in nbg else None
        c, cpn = rl_run(base, "C"), rl_run(base, "CPN")
        ag = agreement(base)
        notes, reason, measured, estimated = [], None, None, None
        if ds == "RoadTraffic":
            notes.append("Road exploratory preprocessing: duplicate-event derived copy")
        if r.get("dropped_activities"):
            notes.append(f"IMf removed activities from the model: {r['dropped_activities']}")
        if (ds, th) in SAME_MODEL:
            notes.append(f"identical Process Tree to {SAME_MODEL[(ds, th)]:.2f}")
        gen = r.get("GeneratedTraces")
        tot = r["CleanCases"]
        cov_cnt = r.get("CleanCaseCoverageCount")
        bpic_sel = ds == "BPIC2017" and sel
        if bpic_sel:  # sweep의 95% 규칙 bound 대신 실제 pipeline generate 결과 (best executable trade-off)
            gs = rj(base / "generate/generate_summary.json")
            r = dict(r, LoopBound=BPIC_BOUND, GeneratedTraces=gs["generated_unique_trace_count"],
                     ModelOnlyTraces=gs["model_only_count"], CleanCaseCoverageCount=gs["clean_cases_covered"],
                     CleanVariantCoverageCount=gs["observed_in_generated"], GenerationMs=gs["generation_ms"],
                     EstimatedLanguageSize=gs["generated_unique_trace_count"])
            gen, cov_cnt = r["GeneratedTraces"], r["CleanCaseCoverageCount"]
            notes.append("BPIC bound rule: best executable trade-off (coverage >= 90% abandoned); see paper_bpic_threshold_tradeoff")
            notes.append(BPIC_CPN_NOTE)
        if cov_cnt is None:  # 생성하지 않은 경우: 분석값(enumeration-free)으로 coverage 기록
            cov_cnt = round(r["AnalyticCoverage"] * tot)
            notes.append("coverage = enumeration-free analytic value (traces not generated)")
        # 상태
        if bpic_sel:
            status = "SELECTED" if c["training_ms"] is not None else "SELECTED (C run pending)"
            reason = BPIC_CPN_NOTE
        elif r["Status"] == "INFEASIBLE":
            status = "INFEASIBLE"
            reason = (f"bounded language for coverage target at {r['LoopBound']} = {r['EstimatedLanguageSize']:.3g} "
                      f"> generation guard 2,000,000; max coverage within guard = "
                      f"{100 * (r.get('max_coverage_within_guard') or 0):.2f}% at {r.get('bound_for_max_coverage_within_guard')}")
            estimated = f"language {r['EstimatedLanguageSize']:.3g} traces"
        elif sel:
            status = "SELECTED" if c["training_ms"] is not None else "SELECTED (C run pending)"
        elif ds == "RoadTraffic" and th == 0.0:
            status = "INFEASIBLE"
            reason = "W2 NBG construction did not finish (model-only 320k of 1.34M stopped at 3,600 s)"
            if road_feas:
                ext = road_feas["step2_w2_nbg"]["extrapolated_full_build_hours"]
                measured = "NBG 80k model-only: 784.3 s; 320k: > 3,600 s (stopped)"
                estimated = (f"full NBG {ext['lower_bound_linear_from_320k_stop (>=3600s * 1344075/320000)']}–"
                             f"{ext['quadratic_fit_5k_20k_80k']} h; acceptor check ~15 h")
        elif (r.get("AnalyticCoverage") or 0) < 0.95:
            status = "COVERAGE_LOSS"
            reason = f"max clean coverage {100 * r['max_coverage (unbounded)']:.2f}% < 95% target"
        elif (ds, th) in SAME_MODEL:
            status = "SAME_AS_" + f"{SAME_MODEL[(ds, th)]:.2f}"
            reason = "identical model -> identical generation / NBG / C"
        else:
            status = "NOT_RUN_C"
            reason = (f"C not run (time budget: one full RP-BEAR run per dataset); generated {gen:,} "
                      f"(model-only {r.get('ModelOnlyTraces'):,}) >> selected threshold")
        row = {
            "Dataset": ds, "NoiseThreshold": th, "IMVariant": r["im_variant"], "LoopBound": r["LoopBound"],
            "CleanCases": tot, "CleanVariants": r["CleanVariants"],
            "Places": r["Places"], "Transitions": r["Transitions"], "VisibleTransitions": r["VisibleTransitions"],
            "TauTransitions": r["TauTransitions"], "Arcs": r["Arcs"], "LoopCount": r["LoopCount"],
            "NestedLoopCount": r["NestedLoopCount"], "ParallelCount": r["ParallelCount"], "XORCount": r["XORCount"],
            "GeneratedTraces": gen, "EstimatedLanguageSize": r["EstimatedLanguageSize"],
            "ModelOnlyTraces": r.get("ModelOnlyTraces"),
            "CleanCaseCoverageCount": cov_cnt, "CleanCaseCoveragePercent": pct(cov_cnt, tot),
            "CleanVariantCoverageCount": r.get("CleanVariantCoverageCount"),
            "CleanVariantCoveragePercent": pct(r.get("CleanVariantCoverageCount"), r["CleanVariants"]),
            "MaxCoverageWithinGuardPercent": None if r.get("max_coverage_within_guard") is None else round(100 * r["max_coverage_within_guard"], 2),
            "BoundForMaxCoverageWithinGuard": r.get("bound_for_max_coverage_within_guard"),
            "DiscoveryMs": r["DiscoveryMs"], "GenerationMs": r.get("GenerationMs"), "NbgBuildMs": nbg_ms,
            "BAlignmentCases": None if not b else b["n"],
            "BAlignmentMeanMsPerCase": None if not b else b["mean_ms"],
            "BAlignmentMedianMsPerCase": None if not b else b["median_ms"],
            "BAlignmentP95MsPerCase": None if not b else b["p95_ms"],
            "BAlignmentTotalMs": None if not b else b["total_ms"],
            "BAlignmentTimeouts": None if not b else b["n_timeout_or_failed"],
            "CTrainingMs": c["training_ms"], "CRepairMeanMsPerCase": c["seq_mean"], "CRepairMedianMsPerCase": c["seq_median"],
            "CPNTrainingMs": cpn["training_ms"], "CPNRepairMeanMsPerCase": cpn["seq_mean"],
            "CPNRepairMedianMsPerCase": cpn["seq_median"],
            "TestCases": ag["n"] if ag else None,
            "BCExactAgreement": ag["C"]["exact"] if ag and ag.get("C") else None,
            "BCAllOptimalAgreement": ag["C"]["tau_excluded"] if ag and ag.get("C") else None,
            "BCPNExactAgreement": ag["CPN"]["exact"] if ag and ag.get("CPN") else None,
            "BCPNNativeAllOptimalAgreement": ag["CPN"]["native"] if ag and ag.get("CPN") else None,
            "BCPNAllOptimalAgreement": ag["CPN"]["tau_excluded"] if ag and ag.get("CPN") else None,
            "Status": status, "Reason": reason, "MeasuredRuntime": measured, "EstimatedRuntime": estimated,
            "Notes": "; ".join(notes + ([b["cases"]] if b else []) + ([b["note"]] if b and b.get("note") else [])) or None,
        }
        n = row["TestCases"]
        row["BCExactAgreementPercent"] = pct(row["BCExactAgreement"], n)
        row["BCAllOptimalAgreementPercent"] = pct(row["BCAllOptimalAgreement"], n)
        row["BCPNExactAgreementPercent"] = pct(row["BCPNExactAgreement"], n)
        row["BCPNAllOptimalAgreementPercent"] = pct(row["BCPNAllOptimalAgreement"], n)
        out.append(row)
    return pd.DataFrame(out)


SWEEP_COLS = ["Dataset", "NoiseThreshold", "CleanCases", "CleanVariants", "Places", "Transitions", "VisibleTransitions",
              "TauTransitions", "Arcs", "LoopCount", "NestedLoopCount", "ParallelCount", "XORCount", "LoopBound",
              "GeneratedTraces", "EstimatedLanguageSize", "ModelOnlyTraces", "CleanCaseCoverageCount", "CleanCaseCoveragePercent",
              "CleanVariantCoverageCount", "CleanVariantCoveragePercent", "MaxCoverageWithinGuardPercent",
              "BoundForMaxCoverageWithinGuard", "DiscoveryMs", "GenerationMs", "NbgBuildMs",
              "BAlignmentCases", "BAlignmentMeanMsPerCase", "BAlignmentMedianMsPerCase", "BAlignmentP95MsPerCase",
              "BAlignmentTotalMs", "BAlignmentTimeouts",
              "CTrainingMs", "CRepairMeanMsPerCase", "CRepairMedianMsPerCase",
              "CPNTrainingMs", "CPNRepairMeanMsPerCase", "CPNRepairMedianMsPerCase", "TestCases",
              "BCExactAgreement", "BCExactAgreementPercent", "BCAllOptimalAgreement",
              "BCAllOptimalAgreementPercent", "BCPNExactAgreement", "BCPNExactAgreementPercent",
              "BCPNAllOptimalAgreement", "BCPNAllOptimalAgreementPercent", "Status", "Reason", "MeasuredRuntime",
              "EstimatedRuntime", "Notes"]


def md_table(df, cols=None, headers=None):
    cols = cols or list(df.columns)
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if "Threshold" in c and isinstance(v, float) and not math.isnan(v):
                cells.append(f"{v:.2f}")
            else:
                cells.append(v if isinstance(v, str) else fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_py(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def speed(b, x):
    b, x = to_py(b), to_py(x)
    return None if b is None or x in (None, 0) else round(b / x, 3)


# ------------------------------------------------------------------ main
def main():
    S = sweep_rows()
    S[SWEEP_COLS].to_csv(MD / "threshold_sweep_summary.csv", index=False)
    md = ["# Threshold sweep summary (Road Traffic / BPIC 2017)", "",
          "- noise_threshold = PM4Py Inductive Miner noise threshold (0 = IM, > 0 = IMf; PM4Py 2.7.23.6). ADM alpha와 다른 값.",
          "- 모든 threshold는 PYTHONHASHSEED=0으로 discovery (IMf 결정성).",
          "- LoopBound 규칙 (모든 threshold 동일): clean case coverage >= 95%를 만족하는 Pareto-minimal bound 중 language가 가장 작은 것;"
          " 95%가 불가능하면 도달 가능한 최대 coverage의 bound. language > 2,000,000이면 생성하지 않고 INFEASIBLE.",
          "- timing 단위 ms. B Alignment = 순차 1 process, scipy LP, align + repair (discovery 제외).",
          "- N/A = output에 값이 없음 (추정하지 않음).", ""]
    for ds in ("RoadTraffic", "BPIC2017"):
        md += [f"## {ds}", "", md_table(S[S.Dataset == ds], SWEEP_COLS[1:]), ""]
    (MD / "threshold_sweep_summary.md").write_text("\n".join(md), encoding="utf-8")

    # 1) model table
    M = S[["Dataset", "NoiseThreshold", "Places", "Transitions", "TauTransitions", "LoopCount", "NestedLoopCount",
           "ParallelCount", "LoopBound", "GeneratedTraces", "EstimatedLanguageSize", "CleanCaseCoveragePercent"]].copy()
    M.columns = ["Dataset", "Threshold", "Places", "Transitions", "Tau", "Loops", "Nested Loops", "Parallel",
                 "Loop Bound", "Generated Traces", "Language Size (analytic)", "Clean Coverage (%)"]
    M.to_csv(MD / "paper_threshold_model_table.csv", index=False)
    (MD / "paper_threshold_model_table.md").write_text(
        "# Threshold → Petri Net / language\n\nGenerated Traces = 실제 생성 수 (생성하지 않은 경우 N/A, language size는 enumeration-free 계산값).\n\n"
        + md_table(M) + "\n", encoding="utf-8")

    # 2) timing table
    T = S[["Dataset", "NoiseThreshold", "BAlignmentCases", "BAlignmentMeanMsPerCase", "CTrainingMs", "CRepairMeanMsPerCase",
           "CPNTrainingMs", "CPNRepairMeanMsPerCase", "Status"]].copy()
    T["CSpeedupVsB"] = [speed(b, c) for b, c in zip(S.BAlignmentMeanMsPerCase, S.CRepairMeanMsPerCase)]
    T["CPNSpeedupVsB"] = [speed(b, c) for b, c in zip(S.BAlignmentMeanMsPerCase, S.CPNRepairMeanMsPerCase)]
    T.columns = ["Dataset", "Threshold", "B Cases", "B Alignment Mean ms/case", "C Training ms", "C Repair Mean ms/case",
                 "C+PN Training ms", "C+PN Repair Mean ms/case", "Status", "C Speedup vs B", "C+PN Speedup vs B"]
    T.to_csv(MD / "paper_threshold_timing_table.csv", index=False)
    (MD / "paper_threshold_timing_table.md").write_text(
        "# Threshold → timing (ms)\n\nspeedup = B mean / C mean (1.0 = 동일, > 1.0 = C가 더 빠름, < 1.0 = B가 더 빠름).\n"
        "B, C, C+PN repair = 모두 순차 Test-only 측정. C 계열은 선택 threshold에서만 실행.\n"
        "B 열은 threshold 간 비교를 위해 모두 sweep Petri Net으로 같은 방식(14_threshold_b_alignment.py)으로 잰 값이다 "
        "(일부는 다른 작업과 동시에 측정되어 약 10% 오차; 같은 모델인 Road 0.20 / 0.30이 55.92 / 61.29). "
        "main result table의 Road B 값은 공식 run-b Petri Net으로 다른 작업 없이 다시 잰 순차 replay(seq_B, 48.83 ms/case)이다.\n\n"
        + md_table(T) + "\n", encoding="utf-8")

    # 3) agreement table
    A = pd.DataFrame([{"Dataset": r.Dataset, "Threshold": r.NoiseThreshold, "TestCases": to_py(r.TestCases),
                       "B-C Exact": cnt(to_py(r.BCExactAgreement), r.TestCases) if to_py(r.TestCases) else NA,
                       "B-C Tau-Excluded All-Optimal": cnt(to_py(r.BCAllOptimalAgreement), r.TestCases) if to_py(r.TestCases) else NA,
                       "B-C+PN Exact": cnt(to_py(r.BCPNExactAgreement), r.TestCases) if to_py(r.TestCases) else NA,
                       "B-C+PN Native All-Optimal": cnt(to_py(r.BCPNNativeAllOptimalAgreement), r.TestCases) if to_py(r.TestCases) else NA,
                       "B-C+PN Tau-Excluded All-Optimal": cnt(to_py(r.BCPNAllOptimalAgreement), r.TestCases) if to_py(r.TestCases) else NA,
                       "Status": r.Status} for r in S.itertuples()])
    A.to_csv(MD / "paper_threshold_agreement_table.csv", index=False)
    (MD / "paper_threshold_agreement_table.md").write_text(
        "# Threshold → agreement\n\nrepair accuracy는 main 표에서 제외 (교수님 지시).\n"
        + "".join(f"- {r.Dataset} {r.NoiseThreshold:.2f}: {n}\n" for r in S.itertuples()
                  if (n := state_limit_note(SIM / r.Dataset / tag(r.NoiseThreshold))))
        + "\n" + md_table(A) + "\n", encoding="utf-8")

    # 4) trade-off table
    bt = MD / "paper_bpic_threshold_tradeoff.csv"  # 17_bpic_feasibility_tables.py (BPIC: best executable trade-off)
    bpic_dec = ({round(float(x.Threshold), 2): x.Decision for x in pd.read_csv(bt).itertuples()} if bt.exists() else {})

    def decision(r):
        if r.Dataset == "BPIC2017" and not r.Status.startswith("SELECTED") and round(r.NoiseThreshold, 2) in bpic_dec:
            return f"{bpic_dec[round(r.NoiseThreshold, 2)]} (BPIC rule; 95%-rule status {r.Status})"
        if r.Status == "SELECTED":
            return "**SELECTED**"
        if r.Status.startswith("SELECTED"):
            return "**SELECTED** (C pending)"
        if r.NoiseThreshold == 0.0:
            return "Baseline (" + ("Too large" if r.Status == "INFEASIBLE" else r.Status) + ")"
        return {"INFEASIBLE": "Too large", "COVERAGE_LOSS": "Coverage loss", "NOT_RUN_C": "Candidate (larger language)"}.get(
            r.Status, "Same model as 0.20" if r.Status.startswith("SAME_AS") else r.Status)
    D = pd.DataFrame([{"Dataset": r.Dataset, "Threshold": r.NoiseThreshold, "Loop Bound": r.LoopBound,
                       "Coverage (%)": to_py(r.CleanCaseCoveragePercent),
                       "Generated Traces": to_py(r.GeneratedTraces) if to_py(r.GeneratedTraces) is not None else
                       f"(est. {r.EstimatedLanguageSize:.3g})",
                       "Petri Net Transitions": r.Transitions, "Alignment ms/case": to_py(r.BAlignmentMeanMsPerCase),
                       "C ms/case": to_py(r.CRepairMeanMsPerCase),
                       "Agreement (B-C+PN tau-excl. all-opt., %)": to_py(r.BCPNAllOptimalAgreementPercent),
                       "Decision": decision(r)} for r in S.itertuples()])
    D.to_csv(MD / "paper_threshold_tradeoff_table.csv", index=False)
    rationale = (
        "\n## Threshold selection rationale\n\n"
        "규칙 (모든 threshold 공통): 같은 loop-bound 규칙(clean coverage ≥ 95%인 Pareto-minimal bound 중 language 최소)을 "
        "적용한 뒤, Method C를 실행할 수 있는(생성 language ≤ 2M) threshold 중 coverage ≥ 90%(가능하면 ≥ 95%)를 유지하면서 "
        "language가 가장 작은 지점을 고른다. Credit / Pub는 0.00에서 이미 실행 가능하므로 0.00 그대로 쓴다.\n\n"
        "**Road Traffic → 0.20 (2,2).** 0.00은 1,344,244 traces (95.11%)로 W2 NBG가 1시간 안에 끝나지 않았다. "
        "0.05는 241,528 traces / 99.42%로 실행은 가능하지만 language가 0.20보다 6.7배 크다. "
        "0.10 / 0.15는 최대 coverage 94.13%로 95%에 못 미친다(language는 작음). "
        "0.20은 36,174 traces로 99.16%를 유지해 language가 급격히 줄어드는 elbow이며, 0.30은 0.20과 같은 모델이다. "
        "0.40은 activity h, i, j가 모델에서 빠지고 coverage가 81.03%로 떨어진다.\n\n"
        "**BPIC 2017 → 0.10 (1,9), best executable trade-off.** 위 95% 규칙으로는 어떤 threshold도 실행 가능한 bound가 없다 "
        "(0.00은 95%에 약 10^21 traces; threshold > 0은 unbounded 최대 coverage도 90% 미만: 0.05 87.80%, 0.15–0.30 58.57%, 0.40 2.05%). "
        "교수님 방향 수정(2026-11-12)에 따라 BPIC는 coverage 기준을 버리고, 현재 PC에서 기존 B / C / C+PN을 수정 없이 실행할 수 있는 "
        "조건 중 clean coverage가 가장 높은 조건을 고른다 (paper_bpic_threshold_tradeoff.md, bpic_candidate_feasibility.md). "
        "threshold 0.05는 language를 키우면 coverage가 더 높지만((1,10) 97,056 traces / 47.82%), 같은 50k budget에서는 "
        "(1,8) 20,032 traces / 40.09%로 0.10 (1,9) 31,440 traces / 43.49%보다 낮다. 0.15–0.30은 39.71%에서 포화된다. "
        "따라서 0.10 (1,9)가 trade-off에서 가장 적절하다. coverage 43.49%는 clean case의 절반 미만이므로 Road와 같은 수준의 "
        "모델 coverage로 해석하면 안 된다. BPIC 2017 C+PN: In progress (pending final C+PN result).\n")
    (MD / "paper_threshold_tradeoff_table.md").write_text("# Threshold trade-off\n\n" + md_table(D) + "\n" + rationale,
                                                           encoding="utf-8")

    # 5) relation tables (RQ4 / RQ5)
    S[["Dataset", "NoiseThreshold", "Places", "Transitions", "TauTransitions", "Arcs", "LoopCount", "BAlignmentCases",
       "BAlignmentMeanMsPerCase", "BAlignmentMedianMsPerCase", "BAlignmentP95MsPerCase", "BAlignmentTimeouts"]].to_csv(
        MD / "threshold_alignment_speed.csv", index=False)
    nbg_sizes = []
    for r in S.itertuples():
        p = rj(SIM / r.Dataset / tag(r.NoiseThreshold) / "preparation/expanded_w2_nbg_summary.json")
        nbg_sizes.append((p["nodes"], p["edges"]) if p else (None, None))
    Cr = S[["Dataset", "NoiseThreshold", "GeneratedTraces", "ModelOnlyTraces", "NbgBuildMs", "CTrainingMs",
            "CRepairMeanMsPerCase", "CPNTrainingMs", "CPNRepairMeanMsPerCase", "Status"]].copy()
    Cr.insert(5, "W2NbgNodes", [a for a, _ in nbg_sizes])
    Cr.insert(6, "W2NbgEdges", [b for _, b in nbg_sizes])
    Cr.to_csv(MD / "threshold_c_speed.csv", index=False)

    # 6) main paper table
    main_rows = []
    for ds, base, th, bound, b_label in (("Credit", SIM / "Credit", 0.0, "(5,7)", "seq_scipy_B"),
                                         ("Pub", SIM / "Pub", 0.0, "(4,4)", "seq_B")):
        gen = rj(base / "generate/generate_summary.json")
        b = b_official_replay(base, b_label)
        c, cpn = rl_run(base, "C"), rl_run(base, "CPN")
        ag = None
        if ds == "Credit":
            j = rj(base / "alignment_all_optimal_analysis/bcpn_all_optimal_agreement.json")
            if j:  # "dev" = tau-excluded (deviation-only cost) all-optimal, 예: "162/210 = 0.7714"
                c_, n_ = j["dev"]["all_optimal_agreement"].split(" = ")[0].split("/")
                ag = {"n": int(n_), "tau": int(c_)}
        else:
            a2 = agreement(base)
            ag = {"n": a2["n"], "tau": a2["CPN"]["tau_excluded"]} if a2 else None
        main_rows.append({"Dataset": ds, "Selected Noise Threshold": th, "Loop Bound": bound,
                          "Test Cases": ag["n"] if ag else None, "B-C+PN Agreement": ag["tau"] if ag else None,
                          "B Alignment Repair (ms/case)": b["mean"] if b else None,
                          "C Training (ms)": c["training_ms"], "C Repair (ms/case)": c["seq_mean"],
                          "C+PN Training (ms)": cpn["training_ms"], "C+PN Repair (ms/case)": cpn["seq_mean"],
                          "Generated Traces": gen["generated_unique_trace_count"] if gen else None,
                          "Clean Coverage (%)": round(100 * gen["clean_case_coverage"], 2) if gen else None,
                          "Status": "OK", "Notes": state_limit_note(base) if ds != "Credit" else None})
    for ds in ("RoadTraffic", "BPIC2017"):
        th = SELECTED[ds]
        if th is None:
            r0 = S[(S.Dataset == ds) & (S.NoiseThreshold == 0.0)].iloc[0]
            main_rows.append({"Dataset": ds, "Selected Noise Threshold": None, "Loop Bound": None, "Test Cases": None,
                              "B-C+PN Agreement": None, "B Alignment Repair (ms/case)": None, "C Training (ms)": None,
                              "C Repair (ms/case)": None, "C+PN Training (ms)": None, "C+PN Repair (ms/case)": None,
                              "Generated Traces": None, "Clean Coverage (%)": None, "Status": "INFEASIBLE",
                              "Notes": "no noise_threshold in {0.00,...,0.40} gives a generatable bounded language with >= 90% coverage; "
                                       f"baseline needs {r0.EstimatedLanguageSize:.3g} traces for 95%"})
            continue
        r = S[(S.Dataset == ds) & (S.NoiseThreshold == th)].iloc[0]
        base = SIM / ds / tag(th)
        b = b_official_replay(base, "seq_B")
        n_test = to_py(r.TestCases)
        if ds == "BPIC2017" and n_test is None:  # C+PN 없음 -> all-optimal 분석 대신 run-b test case 수
            sj = first(base / "runs/methodB_run1", "*_stats.json")
            n_test = rj(sj)["n_test"] if sj else None
        main_rows.append({"Dataset": ds, "Selected Noise Threshold": th, "Loop Bound": r.LoopBound,
                          "Test Cases": n_test, "B-C+PN Agreement": to_py(r.BCPNAllOptimalAgreement),
                          # BPIC: sweep 30-case sample로 대체하지 않음 (seq_B 없으면 N/A)
                          "B Alignment Repair (ms/case)": b["mean"] if b else (
                              None if ds == "BPIC2017" else to_py(r.BAlignmentMeanMsPerCase)),
                          "C Training (ms)": to_py(r.CTrainingMs), "C Repair (ms/case)": to_py(r.CRepairMeanMsPerCase),
                          "C+PN Training (ms)": to_py(r.CPNTrainingMs), "C+PN Repair (ms/case)": to_py(r.CPNRepairMeanMsPerCase),
                          "Generated Traces": to_py(r.GeneratedTraces), "Clean Coverage (%)": to_py(r.CleanCaseCoveragePercent),
                          "Status": r.Status, "Notes": "; ".join(x for x in (
                              "Road exploratory preprocessing: duplicate-event derived copy" if ds == "RoadTraffic" else None,
                              "best executable trade-off (coverage >= 90% abandoned); clean coverage < 50%" if ds == "BPIC2017" else None,
                              BPIC_CPN_NOTE if ds == "BPIC2017" else None,
                              (f"B-C exact {cnt(to_py(r.BCExactAgreement), n_test)}, B-C tau-excluded all-optimal "
                               f"{cnt(to_py(r.BCAllOptimalAgreement), n_test)} (C+PN in progress, so B-C is reported instead)")
                              if ds == "BPIC2017" and to_py(r.BCExactAgreement) is not None else None,
                              state_limit_note(base)) if x) or None})
    P = pd.DataFrame(main_rows)
    P.insert(5, "B-C+PN Agreement (%)", [pct(to_py(a), to_py(n)) for a, n in zip(P["B-C+PN Agreement"], P["Test Cases"])])
    P["C Speedup vs B"] = [speed(b, c) for b, c in zip(P["B Alignment Repair (ms/case)"], P["C Repair (ms/case)"])]
    P["C+PN Speedup vs B"] = [speed(b, c) for b, c in zip(P["B Alignment Repair (ms/case)"], P["C+PN Repair (ms/case)"])]
    P = P[["Dataset", "Selected Noise Threshold", "Loop Bound", "Clean Coverage (%)", "Generated Traces", "Test Cases",
           "B-C+PN Agreement", "B-C+PN Agreement (%)", "B Alignment Repair (ms/case)", "C Training (ms)",
           "C Repair (ms/case)", "C+PN Training (ms)", "C+PN Repair (ms/case)", "C Speedup vs B", "C+PN Speedup vs B",
           "Status", "Notes"]]  # 사용자 지정 순서 (2026-11-12) + Status / Notes
    P.to_csv(MD / "paper_result_table.csv", index=False)
    (MD / "paper_result_table.md").write_text(
        "# Main result table (selected condition per dataset)\n\n"
        "- Agreement = B–C+PN tau-excluded all-optimal agreement (count / test cases). "
        "BPIC 2017: C+PN in progress -> N/A; B–C agreement is in Notes and paper_threshold_agreement_table.md.\n"
        "- Training = RP-BEAR validation(학습) phase ms (official run, 15 workers). Repair = 순차 Test-only mean ms/case.\n"
        "- B = Alignment (PM4Py state-equation A*), 순차, scipy LP (모든 dataset 동일 solver). Training N/A.\n"
        "- speedup = B mean / C mean (> 1.0 = C가 더 빠름).\n\n" + md_table(P) + "\n", encoding="utf-8")

    # 7) validation
    checks = []

    def chk(name, ok, detail=None):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    for r in S.itertuples():
        if to_py(r.CleanCaseCoverageCount) is not None:
            chk(f"{r.Dataset} {r.NoiseThreshold}: coverage % == count / clean cases",
                abs(r.CleanCaseCoveragePercent - round(100 * r.CleanCaseCoverageCount / r.CleanCases, 2)) < 1e-9)
        if to_py(r.TestCases):
            for c, p in (("BCExactAgreement", "BCExactAgreementPercent"), ("BCAllOptimalAgreement", "BCAllOptimalAgreementPercent"),
                         ("BCPNAllOptimalAgreement", "BCPNAllOptimalAgreementPercent")):
                if to_py(getattr(r, c)) is not None:
                    chk(f"{r.Dataset} {r.NoiseThreshold}: {p} == count / test", abs(getattr(r, p) - round(100 * getattr(r, c) / r.TestCases, 2)) < 1e-9)
        sw = rj(SIM / r.Dataset / tag(r.NoiseThreshold) / "sweep/sweep_result.json")["row"]
        gs = rj(SIM / r.Dataset / tag(r.NoiseThreshold) / "generate/generate_summary.json")
        if r.Dataset == "BPIC2017" and r.Status.startswith("SELECTED"):  # 95% 규칙 sweep 대신 16 frontier와 비교
            fr = pd.read_csv(MD / "bpic_candidates" / f"frontier_{tag(r.NoiseThreshold)}.csv")
            n_, t_ = map(int, BPIC_BOUND.strip("()").split(","))
            ana = fr[(fr.loop_num == n_) & (fr.loop_total == t_)].iloc[0]
            chk(f"BPIC2017 {r.NoiseThreshold}: pipeline generate == analytic language {BPIC_BOUND}",
                gs["generated_unique_trace_count"] == int(ana.language), (gs["generated_unique_trace_count"], int(ana.language)))
            chk(f"BPIC2017 {r.NoiseThreshold}: pipeline coverage == analytic coverage {BPIC_BOUND}",
                abs(gs["clean_case_coverage"] - ana.case_cov) < 1e-6, (gs["clean_case_coverage"], ana.case_cov))
            continue
        chk(f"{r.Dataset} {r.NoiseThreshold}: generated traces == sweep output", to_py(r.GeneratedTraces) == sw.get("GeneratedTraces"))
        if gs:
            chk(f"{r.Dataset} {r.NoiseThreshold}: pipeline generate == sweep generated", gs["generated_unique_trace_count"] == sw.get("GeneratedTraces"),
                (gs["generated_unique_trace_count"], sw.get("GeneratedTraces")))
    for r in T.itertuples():
        for sp, x in (("C Speedup vs B", "C Repair Mean ms/case"), ("C+PN Speedup vs B", "C+PN Repair Mean ms/case")):
            pass
    for _, r in P.iterrows():
        for sp, x in (("C Speedup vs B", "C Repair (ms/case)"), ("C+PN Speedup vs B", "C+PN Repair (ms/case)")):
            if to_py(r[sp]) is not None:
                chk(f"main {r.Dataset}: {sp} == B / {x}", abs(r[sp] - round(r["B Alignment Repair (ms/case)"] / r[x], 3)) < 1e-9)
        if to_py(r["B-C+PN Agreement (%)"]) is not None:
            chk(f"main {r.Dataset}: agreement % == count / test", abs(r["B-C+PN Agreement (%)"] - round(100 * r["B-C+PN Agreement"] / r["Test Cases"], 2)) < 1e-9)
        for col in ("B Alignment Repair (ms/case)", "C Repair (ms/case)", "C+PN Repair (ms/case)"):
            v = to_py(r[col])
            if v is not None:
                chk(f"main {r.Dataset}: {col} is a per-case ms value (0 < v < 600,000)", 0 < v < 600_000, v)
        for col in ("C Training (ms)", "C+PN Training (ms)"):
            v = to_py(r[col])
            if v is not None:
                chk(f"main {r.Dataset}: {col} > 1,000 ms (not seconds)", v > 1000, v)
    for ds, th in SELECTED.items():
        if th is not None:
            chk(f"selected threshold {ds} {th} exists in sweep", ((S.Dataset == ds) & (S.NoiseThreshold == th)).any())
            row = P[P.Dataset == ds].iloc[0]
            srow = S[(S.Dataset == ds) & (S.NoiseThreshold == th)].iloc[0]
            chk(f"main {ds} row == sweep row (generated / coverage / C training)",
                (to_py(row["Generated Traces"]), to_py(row["Clean Coverage (%)"]), to_py(row["C Training (ms)"])) ==
                (to_py(srow.GeneratedTraces), to_py(srow.CleanCaseCoveragePercent), to_py(srow.CTrainingMs)))
    chk("timing columns are named in ms only (no seconds columns in paper tables)",
        not any(c.lower().endswith(("sec", "(s)", " s")) for df in (S[SWEEP_COLS], T, P) for c in df.columns))
    val = {"all_ok": all(c["ok"] for c in checks), "n_checks": len(checks), "n_failed": sum(not c["ok"] for c in checks),
           "checks": checks}
    (MD / "table_validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"validation: {val['n_checks']} checks, failed {val['n_failed']}")
    for c in checks:
        if not c["ok"]:
            print("  FAIL", c)


if __name__ == "__main__":
    main()
