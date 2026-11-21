"""Build the paper result tables (CSV + Markdown) and their validation from stored outputs only.

No new experiments are run. Values missing in the outputs are reported as N/A, never estimated.
All timings are in ms (training = ms, test repair = mean ms/case). Parallel throughput (wall/n) is not used in the main table.
Main repair time = sequential test-only measurement:
  B     : alignment + repair, one process, scipy LP (shim). Threshold sweep: script 14; official replay: test-timing-b
  C/C+PN: test-only sequential replay with the trained agent (test-timing / test-timing-pngate), mean over all test
          cases (RL + Mem paths)

Output (outputs/multi_dataset_experiment/):
  threshold_sweep_summary.csv / .md
  paper_threshold_model_table.csv / .md, paper_threshold_timing_table.csv / .md,
  paper_threshold_agreement_table.csv / .md, paper_threshold_tradeoff_table.csv / .md,
  threshold_alignment_speed.csv, threshold_c_speed.csv,
  paper_result_table.csv / .md, table_validation.json
Usage: python outputs/multi_dataset_experiment/scripts/15_build_paper_tables.py
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
# BPIC 2017: the coverage >= 90 % criterion is not reachable -> "best executable trade-off".
# The loop bound comes from the candidate search of 17_bpic_feasibility_tables.py (maximal coverage within a
# 50k-trace language), not from the 95 % rule.
# BPIC 2017 C+PN: the official run finished (2026-11-18 to 11-20). BPIC_CPN_NOTE is added only when the C+PN output
# does not exist; B-C+PN all-optimal agreement comes from 22_agreement_from_checkpoint.py (same B optimal sets).
BPIC_BOUND = "(1,9)"
BPIC_CPN_NOTE = "C+PN: In progress (pending final C+PN result)"
SAME_MODEL = {("RoadTraffic", 0.30): 0.20, ("BPIC2017", 0.30): 0.20}  # same process tree as 0.20 (md5 checked)
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
        if abs(v) >= 1e7 and (v.is_integer() or abs(v) >= 1e12):  # exponent notation only for large language estimates (ms values unchanged)
            return f"{v:.3g}"
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def cnt(c, n):
    return NA if c is None else f"{int(c)} / {int(n)} ({100 * c / n:.2f}%)"


# ------------------------------------------------------------------ read results
def b_sweep(ds, th):
    src = SAME_MODEL.get((ds, th), th) if ds == "BPIC2017" else th
    s = rj(SIM / ds / tag(src) / "b_alignment_timing" / "summary.json")
    if s and src != th:
        s = dict(s, note=f"identical Process Tree to {src:.2f}: measured once at {src:.2f}")
    return s


REPLAY_LABELS = {  # sequential test-only replay folders. Credit: mean of three earlier sequential replays
    "Credit": {"C": ["rep1_C_alpha0", "rep2_C_alpha0", "rep3_C_alpha0"], "CPN": ["rep1_CPN", "rep2_CPN", "rep3_CPN"]},
}


def rl_run(base, kind):
    """kind C: runs/methodC_run1 + test_timing/<label>; CPN: C_PN_Gate/run1 + C_PN_Gate/test_timing/<label>."""
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
    if train is None:  # runs without timing.json: val_total_ms from stats.json (same definition, same as timing_summary.csv)
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
    """Number of cases where the exact all-optimal search hit the state limit (counted as disagreement -> lower bound)."""
    s = rj(base / "alignment_all_optimal_analysis/run1/agreement_summary.json")
    k = s["checks"].get("state_limit_hit", 0) if s else 0
    rl = s["checks"].get("resource_limit_hit", 0) if s else 0  # cases whose search could not finish because of the memory guard (07 --resource-skip)
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
    if out["CPN"] is None:  # BPIC 2017: script 07 computed B-C only -> B-C+PN from 22_agreement_from_checkpoint.py
        s2 = rj(base / "alignment_all_optimal_analysis/bcpn_from_checkpoint/agreement_summary.json")
        if s2 and s2["cases"] == s["cases"]:
            out["CPN"] = s2["agreement"]["B-CPN"]["counts"]
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
        if bpic_sel:  # the actual pipeline generate result instead of the sweep's 95 % bound (best executable trade-off)
            gs = rj(base / "generate/generate_summary.json")
            r = dict(r, LoopBound=BPIC_BOUND, GeneratedTraces=gs["generated_unique_trace_count"],
                     ModelOnlyTraces=gs["model_only_count"], CleanCaseCoverageCount=gs["clean_cases_covered"],
                     CleanVariantCoverageCount=gs["observed_in_generated"], GenerationMs=gs["generation_ms"],
                     EstimatedLanguageSize=gs["generated_unique_trace_count"])
            gen, cov_cnt = r["GeneratedTraces"], r["CleanCaseCoverageCount"]
            notes.append("BPIC bound rule: best executable trade-off (coverage >= 90% abandoned); see paper_bpic_threshold_tradeoff")
            if cpn["training_ms"] is None:
                notes.append(BPIC_CPN_NOTE)
        if cov_cnt is None:  # not generated: coverage from the analytic (enumeration-free) values
            cov_cnt = round(r["AnalyticCoverage"] * tot)
            notes.append("coverage = enumeration-free analytic value (traces not generated)")
        # status
        if bpic_sel:
            status = "SELECTED" if c["training_ms"] is not None else "SELECTED (C run pending)"
            reason = BPIC_CPN_NOTE if cpn["training_ms"] is None else None
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
          '- noise_threshold = PM4Py Inductive Miner noise threshold (0 = IM, > 0 = IMf; PM4Py 2.7.23.6). Not the ADM alpha.',
          '- Discovery for every threshold uses PYTHONHASHSEED=0 (IMf determinism).',
          '- LoopBound rule (same for every threshold): the Pareto-minimal bound with the smallest language among those with clean-case coverage >= 95%; if 95% is unreachable, the bound with the maximal reachable coverage. If language > 2,000,000, no traces are generated (INFEASIBLE).',
          '- Timings in ms. B Alignment = sequential, 1 process, scipy LP, align + repair (discovery excluded).',
          '- N/A = value not present in the outputs (not estimated).', ""]
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
        '# Threshold → Petri Net / language\n\nGenerated Traces = number actually generated (N/A if not generated; language size = enumeration-free computation).\n\n'
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
        '# Threshold → timing (ms)\n\nspeedup = B mean / C mean (1.0 = equal, > 1.0 = C faster, < 1.0 = B faster).\nB, C, C+PN repair = all sequential test-only measurements. The C variants run only at the selected threshold.\nThe B column is measured for every threshold in the same way with the sweep Petri net (14_threshold_b_alignment.py) for comparison across thresholds (some values were measured concurrently with other jobs, about 10% error; Road 0.20 / 0.30, the same model, give 55.92 / 61.29). The Road B value of the main result table is a sequential replay with the official run-b Petri net without concurrent jobs (seq_B, 48.83 ms/case).\n'
        '**The BPIC 2017 B column is a random sample of 30 test cases (seed 42, 60 s timeout per case; B Cases = 30).** '
        'The C / C+PN columns are sequential measurements over all 2,360 test cases, so the BPIC speedups in this table compare '
        'different measurement scopes and are not used for the main comparison. The main BPIC timing is in paper_result_table.md '
        '(seq_B, all 2,360 test cases, sequential, 4,887.86 ms/case).\n'
        'Training = RP-BEAR training (validation) phase of the official run (multiprocessing, 15 workers). Repair = sequential '
        'test inference. The multiprocessing test time (throughput) of the official run is not used in this table.\n\n'
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
        '# Threshold → agreement\n\nRepair accuracy is not part of the main table.\n'
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
        '\n## Threshold selection rationale\n\nRule (same for every threshold): apply the same loop-bound rule (Pareto-minimal bound with the smallest language among those with clean coverage ≥ 95%), then choose, among the thresholds at which Method C can run (generated language ≤ 2M), the point with the smallest language that keeps coverage ≥ 90% (≥ 95% if possible). Credit / Pub are already executable at 0.00, so 0.00 is kept.\n\n**Road Traffic → 0.20 (2,2).** 0.00 gives 1,344,244 traces (95.11%) and the W2 NBG did not finish within one hour. 0.05 gives 241,528 traces / 99.42%, executable but with a language 6.7 times larger than 0.20. 0.10 / 0.15 reach at most 94.13%, below 95% (small language). 0.20 keeps 99.16% with 36,174 traces, the elbow where the language drops sharply; 0.30 is the same model as 0.20. At 0.40 the activities h, i, j disappear from the model and coverage drops to 81.03%.\n\n**BPIC 2017 → 0.10 (1,9), best executable trade-off.** With the 95% rule above no threshold has an executable bound (0.00 needs about 10^21 traces for 95%; for thresholds > 0 even the unbounded maximal coverage is below 90%: 0.05 87.80%, 0.15–0.30 58.57%, 0.40 2.05%). BPIC 2017 therefore drops the coverage criterion and uses the condition with the highest clean coverage among those under which B / C / C+PN run unchanged (paper_bpic_threshold_tradeoff.md, bpic_candidate_feasibility.md). Threshold 0.05 reaches higher coverage with a larger language ((1,10) 97,056 traces / 47.82%), but within the same 50k budget (1,8) gives 20,032 traces / 40.09%, below 0.10 (1,9) with 31,440 traces / 43.49%. 0.15–0.30 saturate at 39.71%. Hence 0.10 (1,9) is the best trade-off. A coverage of 43.49% is below half of the clean cases and must not be read as model coverage comparable to Road. '
        + ("BPIC 2017 C+PN: In progress (pending final C+PN result).\n"
           if S.loc[(S.Dataset == "BPIC2017") & (S.Status == "SELECTED"), "CPNTrainingMs"].isna().all() else
           "BPIC 2017 C+PN was run afterwards (measured training time: see the timing table).\n"))
    tnote = ("- Alignment ms/case = threshold-sweep measurement (14_threshold_b_alignment.py, sequential). "
             "BPIC 2017 uses a 30-case test sample, Road all test cases. Not the main timing (main = paper_result_table.md).\n"
             "- C ms/case = sequential measurement over all test cases (run only at the selected threshold).\n\n")
    (MD / "paper_threshold_tradeoff_table.md").write_text("# Threshold trade-off\n\n" + tnote + md_table(D) + "\n" + rationale,
                                                           encoding="utf-8")

    # 5) relation tables
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
            if j:  # "dev" = tau-excluded (deviation-only cost) all-optimal, e.g. "162/210 = 0.7714"
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
        if ds == "BPIC2017" and n_test is None:  # no C+PN -> number of run-b test cases instead of the all-optimal analysis
            sj = first(base / "runs/methodB_run1", "*_stats.json")
            n_test = rj(sj)["n_test"] if sj else None
        main_rows.append({"Dataset": ds, "Selected Noise Threshold": th, "Loop Bound": r.LoopBound,
                          "Test Cases": n_test, "B-C+PN Agreement": to_py(r.BCPNAllOptimalAgreement),
                          # BPIC: not replaced by the 30-case sweep sample (N/A if seq_B is missing)
                          "B Alignment Repair (ms/case)": b["mean"] if b else (
                              None if ds == "BPIC2017" else to_py(r.BAlignmentMeanMsPerCase)),
                          "C Training (ms)": to_py(r.CTrainingMs), "C Repair (ms/case)": to_py(r.CRepairMeanMsPerCase),
                          "C+PN Training (ms)": to_py(r.CPNTrainingMs), "C+PN Repair (ms/case)": to_py(r.CPNRepairMeanMsPerCase),
                          "Generated Traces": to_py(r.GeneratedTraces), "Clean Coverage (%)": to_py(r.CleanCaseCoveragePercent),
                          "Status": r.Status, "Notes": "; ".join(x for x in (
                              "Road exploratory preprocessing: duplicate-event derived copy" if ds == "RoadTraffic" else None,
                              "best executable trade-off (coverage >= 90% abandoned); clean coverage < 50%" if ds == "BPIC2017" else None,
                              BPIC_CPN_NOTE if ds == "BPIC2017" and to_py(r.CPNTrainingMs) is None else None,
                              (f"B-C exact {cnt(to_py(r.BCExactAgreement), n_test)}, B-C tau-excluded all-optimal "
                               f"{cnt(to_py(r.BCAllOptimalAgreement), n_test)} "
                               + ("(C+PN in progress, so B-C is reported instead)" if to_py(r.BCPNAllOptimalAgreement) is None else
                                  f"(for comparison; B-C+PN exact {cnt(to_py(r.BCPNExactAgreement), n_test)})"))
                              if ds == "BPIC2017" and to_py(r.BCExactAgreement) is not None else None,
                              state_limit_note(base)) if x) or None})
    P = pd.DataFrame(main_rows)
    P.insert(5, "B-C+PN Agreement (%)", [pct(to_py(a), to_py(n)) for a, n in zip(P["B-C+PN Agreement"], P["Test Cases"])])
    P["C Speedup vs B"] = [speed(b, c) for b, c in zip(P["B Alignment Repair (ms/case)"], P["C Repair (ms/case)"])]
    P["C+PN Speedup vs B"] = [speed(b, c) for b, c in zip(P["B Alignment Repair (ms/case)"], P["C+PN Repair (ms/case)"])]
    P = P[["Dataset", "Selected Noise Threshold", "Loop Bound", "Clean Coverage (%)", "Generated Traces", "Test Cases",
           "B-C+PN Agreement", "B-C+PN Agreement (%)", "B Alignment Repair (ms/case)", "C Training (ms)",
           "C Repair (ms/case)", "C+PN Training (ms)", "C+PN Repair (ms/case)", "C Speedup vs B", "C+PN Speedup vs B",
           "Status", "Notes"]]  # column order + Status / Notes
    P.to_csv(MD / "paper_result_table.csv", index=False)
    (MD / "paper_result_table.md").write_text(
        "# Main result table (selected condition per dataset)\n\n- Agreement = B–C+PN tau-excluded all-optimal agreement (count / test cases). "
        + ("BPIC 2017: C+PN in progress -> N/A; B–C agreement is in Notes and paper_threshold_agreement_table.md.\n"
           if P.loc[P.Dataset == "BPIC2017", "B-C+PN Agreement"].isna().all() else
           "BPIC 2017: B–C+PN computed from the B optimal sets of the B–C all-optimal run (22_agreement_from_checkpoint.py); "
           "B–C agreement is in Notes and paper_threshold_agreement_table.md.\n")
        + "- Training = RP-BEAR training ('validation') phase ms (official run, 15 workers). Repair = sequential test-only mean ms/case.\n"
        "- Repair ms/case is a sequential (one process) measurement over all test cases for every dataset (B seq_B, C seq_C, "
        "C+PN seq_CPN; Credit: mean of 3 replays). The multiprocessing test time (throughput) of the official runs and the "
        "30-case BPIC sweep sample are not used.\n- B = Alignment (PM4Py state-equation A*), sequential, scipy LP (same solver for every dataset). Training N/A.\n- speedup = B mean / C mean (> 1.0 = C faster).\n\n" + md_table(P) + "\n", encoding="utf-8")

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
        if r.Dataset == "BPIC2017" and r.Status.startswith("SELECTED"):  # compare with the frontier of script 16 instead of the 95 % rule
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
