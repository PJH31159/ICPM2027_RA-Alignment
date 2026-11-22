"""Rebuild the paper Tables 1-3 from the stored evidence in results/supporting/ and compare them with the paper.

Why: every number of the paper tables should be traceable to an official result file. This script reads ONLY the
files listed in results/supporting/MANIFEST.csv (after checking their SHA-256), recomputes each table value, and
writes it next to the value printed in the paper. Nothing is re-run (no training, inference or alignment).

Outputs (overwritten on every run):
  results/table1_dataset_summary.csv
  results/table2_alignment_results.csv
  results/table3_bpic2017_case_study.csv

Each paper value has a `*_paper` column and a `*_recomputed` column. `status` is
  MATCH     : equal after rounding to the paper's decimals
  ROUNDING  : differs by at most one unit of the last printed decimal
  MISMATCH  : larger difference
  UNKNOWN   : no source file for the recomputation was found (the paper value is kept, nothing is estimated)
Differences are reported, never corrected. Notes on mixed methods / criteria / LP solvers are in the `notes` column
and in results/README.md.

Usage: python scripts/make_paper_tables.py          (any Python 3 with pandas)
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES = PROJECT_ROOT / "results"
SUP = RES / "supporting"

# ------------------------------------------------------------------ paper values (copied from the PDF, not edited)
PAPER_T1 = {  # dataset: (|L|, |L_dev|, #Act, #Var, NT, (num, tot), |L_N|, Cov %)
    "Small": (12500, 1047, 20, 672, 0.00, "no loop", 6, 100.00),
    "Medium": (12500, 665, 32, 570, 0.00, "no loop", 25, 100.00),
    "Large": (12500, 1285, 42, 1128, 0.00, "no loop", 28, 100.00),
    "Wide": (12500, 674, 34, 546, 0.00, "no loop", 19, 100.00),
    "Credit": (5000, 420, 15, 392, 0.00, "(5, 7)", 5000, 98.03),
    "Pub": (5000, 537, 15, 626, 0.00, "(4, 4)", 6339, 99.98),
    "RoadTraffic": (150370, 5503, 14, 1549, 0.20, "(2, 2)", 36174, 99.16),
    "BPIC2017": (31509, 4719, 26, 7459, 0.10, "(1, 9)", 31440, 43.49),
}
PAPER_T2 = {  # dataset: (test cases, ACC %, training, training unit, Optimal-A., RA-Alignment, time unit)
    "Small": (524, 95.61, 60.75, "min", 1.43, 106.49, "ms/case"),
    "Medium": (333, 84.98, 56.01, "min", 1.62, 205.28, "ms/case"),
    "Large": (643, 85.23, 142.65, "min", 2.48, 299.86, "ms/case"),
    "Wide": (337, 86.94, 60.78, "min", 1.63, 140.63, "ms/case"),
    "Credit": (210, 77.14, 35.64, "min", 122.91, 305.81, "ms/case"),
    "Pub": (269, 89.22, 38.89, "min", 50.23, 665.75, "ms/case"),
    "RoadTraffic": (2752, 85.54, 94.86, "min", 48.83, 100.76, "ms/case"),
    "BPIC2017": (10, 72.00, 42.87, "h", 19.02, 3.31, "s (median per case, mean over cases)"),
}
PAPER_T3 = {  # case: (ACC correct of 5, Optimal-A. s, RA-Alignment s), in paper order
    "Application_1210575946": (0, 29.88, 4.01), "Application_1953341305": (5, 18.82, 3.58),
    "Application_683856223": (1, 26.63, 1.89), "Application_1740309505": (3, 25.98, 3.90),
    "Application_1866718658": (5, 18.28, 3.61), "Application_709809813": (5, 19.35, 5.79),
    "Application_835296163": (3, 24.41, 4.66), "Application_26668050": (4, 2.56, 1.20),
    "Application_283291072": (5, 20.96, 3.55), "Application_488035660": (5, 3.35, 0.88),
}
DIR = {"Small": "simulation/Small", "Medium": "simulation/Medium", "Large": "simulation/Large", "Wide": "simulation/Wide",
       "Credit": "simulation/Credit", "Pub": "simulation/Pub", "RoadTraffic": "simulation/RoadTraffic/threshold_0p20",
       "BPIC2017": "simulation/BPIC2017/threshold_0p10"}
NT = {"RoadTraffic": 0.20, "BPIC2017": 0.10}  # from the output folder name threshold_0pXX; others: default 0.00
CT = "simulation/BPIC2017/threshold_0p10/complex_trace_timing"


# ------------------------------------------------------------------ helpers
def verify_manifest():
    """Every supporting file must match the public SHA-256 recorded in MANIFEST.csv."""
    bad = []
    with open(SUP / "MANIFEST.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        p = PROJECT_ROOT / r["public_path"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != r["public_sha256"]:
            bad.append(r["public_path"])
    return len(rows), bad


def rj(rel):
    return json.loads((SUP / rel).read_text(encoding="utf-8"))


def one(pattern):
    fs = sorted(SUP.glob(pattern))
    if len(fs) != 1:
        raise FileNotFoundError(f"{pattern}: {len(fs)} files")
    return fs[0].relative_to(SUP).as_posix()


def decimals(x):
    s = f"{x}"
    return len(s.split(".")[1]) if "." in s else 0


def status(paper, recomputed, nd=None):
    if recomputed is None:
        return "UNKNOWN"
    if isinstance(paper, str) or isinstance(recomputed, str):
        return "MATCH" if str(paper) == str(recomputed) else "MISMATCH"
    if isinstance(paper, int) and isinstance(recomputed, int):  # counts must be equal
        return "MATCH" if paper == recomputed else "MISMATCH"
    nd = decimals(paper) if nd is None else nd
    d = abs(round(recomputed, nd) - paper)
    return "MATCH" if d < 10 ** -(nd + 3) else ("ROUNDING" if d <= 10 ** -nd + 1e-9 else "MISMATCH")


def frac(s):
    """'162/210 = 0.7714' -> (162, 210)"""
    a, b = s.split("=")[0].strip().split("/")
    return int(a), int(b)


def worst(*st):
    order = ["MATCH", "ROUNDING", "UNKNOWN", "MISMATCH"]
    return max(st, key=order.index)


# ------------------------------------------------------------------ Table 1
def table1():
    rows = []
    for ds, (L, Ld, act, var, nt, bound, ln, cov) in PAPER_T1.items():
        mc_rel, gs_rel = f"{DIR[ds]}/discover/model_check.json", f"{DIR[ds]}/generate/generate_summary.json"
        mc, gs = rj(mc_rel), rj(gs_rel)
        L_r = mc["n_clean_cases"] + mc["n_anomalous_cases"]
        act_r = len(mc["visible_activities"])
        if gs.get("strategy") == "bounded_loop":
            bound_r, ln_r = f"({gs['loop_num']}, {gs['loop_total']})", gs["generated_unique_trace_count"]
            cov_r = round(100 * gs["clean_case_coverage"], 2)
        else:  # acyclic: the full model language is enumerated; coverage 100 % if every observed clean variant is in it
            bound_r, ln_r = "no loop", gs["generated_unique_trace_count"]
            cov_r = 100.0 if gs.get("observed_only_count") == 0 else None
        st = {"L": status(L, L_r), "L_dev": status(Ld, mc["n_anomalous_cases"]), "Act": status(act, act_r),
              "Var": "UNKNOWN", "NT": status(nt, NT.get(ds, 0.0), 2), "bound": status(bound, bound_r),
              "L_N": status(ln, ln_r), "Cov": status(cov, cov_r, 2)}
        rows.append({
            "dataset": ds, "log_size_paper": L, "log_size_recomputed": L_r,
            "deviating_traces_paper": Ld, "deviating_traces_recomputed": mc["n_anomalous_cases"],
            "activities_paper": act, "activities_recomputed": act_r, "activities_basis": "visible transitions of the discovered Petri net",
            "variants_paper": var, "variants_recomputed": "Unknown",
            "noise_threshold_paper": f"{nt:.2f}", "noise_threshold_recomputed": f"{NT.get(ds, 0.0):.2f}",
            "loop_bound_paper": bound, "loop_bound_recomputed": bound_r,
            "generated_traces_paper": ln, "generated_traces_recomputed": ln_r,
            "clean_coverage_pct_paper": f"{cov:.2f}", "clean_coverage_pct_recomputed": "" if cov_r is None else f"{cov_r:.2f}",
            "status": worst(*st.values()),
            "status_detail": "; ".join(f"{k}={v}" for k, v in st.items() if v != "MATCH"),
            "notes": ("#Var: no source found in the stored results (clean-variant count "
                      f"{mc['n_clean_variants']} is a different definition)."
                      + ("" if st["Act"] == "MATCH" else f" #Act: paper {act} vs {act_r} visible activities; basis of the paper value not found.")),
            "sources": f"{mc_rel}; {gs_rel}; src/dataset_config.py",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Table 3 (needed by Table 2's BPIC row)
def table3():
    main = pd.read_csv(SUP / CT / "summary_by_case.csv")
    add = pd.read_csv(SUP / CT / "additional_cases/summary_by_case.csv")
    plan = rj(f"{CT}/additional_cases/candidate_plan.json")
    runs = {src: pd.read_csv(SUP / CT / sub / "runs_CPN.csv") for src, sub in (("main", ""), ("additional", "additional_cases"))}
    main["source"], add["source"] = "main", "additional"
    allc = pd.concat([main[main.case_id == "Application_1953341305"], add[add.case_id != "Application_1953341305"]])
    rows = []
    for r in allc.itertuples():
        rr = runs[r.source]
        rr = rr[(rr.case_id == r.case_id) & (~rr.warmup.astype(str).str.lower().eq("true"))]
        ok_runs = int(rr.in_tau_excl_set.astype(str).str.lower().eq("true").sum())
        ok = int(str(r.CPN_tau).split("/")[0])
        p = PAPER_T3.get(r.case_id)
        st = [status(p[0], ok), status(p[1], round(r.B_median_ms / 1000, 2)), status(p[2], round(r.CPN_median_ms / 1000, 2))] if p else []
        rank = plan["verify_cases"].index(r.case_id) + 1 if r.case_id in plan.get("verify_cases", []) else None
        rows.append({
            "case_id": r.case_id, "in_paper_table3": "yes" if p else "no",
            "study": "first study (4 cases)" if r.source == "main" else f"additional candidates (rank {rank} of 10)",
            "runs": int(r.CPN_n), "acc_correct_paper": p[0] if p else "",
            "acc_correct_recomputed": ok, "acc_correct_from_runs_csv": ok_runs,
            "optimal_alignment_s_paper": p[1] if p else "", "optimal_alignment_median_s": round(r.B_median_ms / 1000, 2),
            "ra_alignment_s_paper": p[2] if p else "", "ra_alignment_cpn_median_s": round(r.CPN_median_ms / 1000, 2),
            "method_c_median_s": round(r.C_median_ms / 1000, 2), "method_c_correct": int(str(r.C_tau).split("/")[0]),
            "b_failed_runs": int(r.B_failed), "cpn_failed_runs": int(r.CPN_failed),
            "status": worst(*st, "MATCH" if ok == ok_runs else "MISMATCH") if p else "NOT_IN_PAPER",
            "acc_criterion": "C+PN repaired trace in the tau-excluded optimal set of B (5 runs, seeds 42-46)",
            "sources": f"{CT}/{'additional_cases/' if r.source == 'additional' else ''}summary_by_case.csv; runs_CPN.csv",
        })
    T = pd.DataFrame(rows)
    T = pd.concat([T[T.in_paper_table3 == "yes"].set_index("case_id").loc[list(PAPER_T3)].reset_index(),
                   T[T.in_paper_table3 == "no"]], ignore_index=True)
    agg = []
    for label, sub in (("PAPER_10_CASES", T[T.in_paper_table3 == "yes"]), ("ALL_11_MEASURED_CASES", T)):
        ok, n = int(sub.acc_correct_recomputed.sum()), int(sub.runs.sum())
        b, c = sub.optimal_alignment_median_s.mean(), sub.ra_alignment_cpn_median_s.mean()
        # the averages are computed from the unrounded medians
        idx = sub.case_id
        src = allc.set_index("case_id").loc[idx]
        b, c = src.B_median_ms.mean() / 1000, src.CPN_median_ms.mean() / 1000
        paper = label == "PAPER_10_CASES"
        agg.append({"case_id": label, "in_paper_table3": "yes" if paper else "no", "study": f"{len(sub)} cases",
                    "runs": n, "acc_correct_paper": "36" if paper else "", "acc_correct_recomputed": ok,
                    "acc_correct_from_runs_csv": int(sub.acc_correct_from_runs_csv.sum()),
                    "optimal_alignment_s_paper": 19.02 if paper else "", "optimal_alignment_median_s": round(b, 2),
                    "ra_alignment_s_paper": 3.31 if paper else "", "ra_alignment_cpn_median_s": round(c, 2),
                    "method_c_median_s": round(src.C_median_ms.mean() / 1000, 2),
                    "method_c_correct": int(sub.method_c_correct.sum()), "b_failed_runs": int(sub.b_failed_runs.sum()),
                    "cpn_failed_runs": int(sub.cpn_failed_runs.sum()),
                    "status": worst(status(36, ok), status(19.02, b), status(3.31, c)) if paper else "NOT_IN_PAPER",
                    "acc_criterion": f"{ok}/{n} = {100 * ok / n:.2f} %; speedup B/C+PN = {b / c:.2f}x",
                    "sources": "aggregate of the rows above"})
    paper_agg = dict(agg[0], n_cases=int((T.in_paper_table3 == "yes").sum()))
    return pd.concat([T, pd.DataFrame(agg)], ignore_index=True), paper_agg


# ------------------------------------------------------------------ Table 2
def table2(t3_paper):
    rows = []
    for ds, (n_test, acc, tr, tr_unit, b_p, ra_p, unit) in PAPER_T2.items():
        d, notes, src = DIR[ds], [], []
        c_stats = one(f"{d}/runs/methodC_run1/*_stats.json") if ds != "Small" else one("exp063_generated_nbg_rpbear/run1/*_stats.json")
        src.append(c_stats)
        tr_ms = rj(c_stats)["val_total_ms"]
        tr_r = tr_ms / (3.6e6 if tr_unit == "h" else 6e4)
        cpn_tr = cpn_ra = None
        if ds in ("Small", "Medium", "Large", "Wide"):
            s = pd.read_csv(SUP / f"repaired_log_agreement/{ds}_abc_agreement_summary.csv").set_index("metric").loc["B vs C"]
            num, den = int(s["same"]), int(s["total"])
            acc_m, crit = "C", "exact: C repaired trace == the single B repaired trace (no all-optimal set)"
            bs = pd.read_csv(SUP / f"{d}/test_timing/repeat_summary_3reps_B.csv").iloc[0]
            b_r, solver, b_agg = float(bs["mean_avg"]), "cvxopt", "mean of 3 sequential replays (mean ms/case)"
            cs = pd.read_csv(SUP / f"{d}/test_timing/repeat_summary_3reps.csv").set_index("condition").loc["C_alpha0"]
            ra_r, ra_agg = float(cs["seq_wall_per_case_avg"]), "3 sequential test replays: test-phase wall / case, averaged"
            src += [f"repaired_log_agreement/{ds}_abc_agreement_summary.csv", f"{d}/test_timing/repeat_summary_3reps_B.csv",
                    f"{d}/test_timing/repeat_summary_3reps.csv"]
            notes.append("No C+PN for this dataset (the PN gate is defined only for the W2 datasets).")
        else:
            cpn_stats = one(f"{d}/C_PN_Gate/run1/*_stats.json")
            cpn_tr = rj(cpn_stats)["val_total_ms"]
            src.append(cpn_stats)
            acc_m = "C+PN"
            crit = "tau-excluded all-optimal: C+PN repaired trace in the set of optimal B repairs (cost log/model move 1, tau 0)"
            if ds == "Credit":
                num, den = frac(rj(f"{d}/alignment_all_optimal_analysis/bcpn_all_optimal_agreement.json")["dev"]["all_optimal_agreement"])
                b_r = rj(f"{d}/test_timing/seq_scipy_B/timing_summary.json")["test_case_ms"]["mean"]
                b_agg = "one sequential replay (mean ms/case)"
                cs = pd.read_csv(SUP / f"{d}/test_timing/repeat_summary_3reps.csv").set_index("condition").loc["C_alpha0"]
                ra_r, ra_agg = float(cs["seq_wall_per_case_avg"]), "3 sequential test replays: test-phase wall / case, averaged"
                cpn_ra = float(pd.read_csv(SUP / f"{d}/C_PN_Gate/test_timing/repeat_summary_3reps.csv").iloc[0]["seq_wall_per_case_avg"])
                cvx = float(pd.read_csv(SUP / f"{d}/test_timing/repeat_summary_3reps_B.csv").iloc[0]["mean_avg"])
                notes.append(f"Same Credit B measured with cvxopt: {cvx} ms/case (solver dependence).")
                src += [f"{d}/alignment_all_optimal_analysis/bcpn_all_optimal_agreement.json", f"{d}/test_timing/seq_scipy_B/timing_summary.json",
                        f"{d}/test_timing/repeat_summary_3reps.csv", f"{d}/C_PN_Gate/test_timing/repeat_summary_3reps.csv"]
            elif ds in ("Pub", "RoadTraffic"):
                ag = rj(f"{d}/alignment_all_optimal_analysis/run1/agreement_summary.json")["agreement"]["B-CPN"]["counts"]
                num, den = ag["tau_excluded"], ag["n"]
                b_r = rj(f"{d}/test_timing/seq_B/timing_summary.json")["test_case_ms"]["mean"]
                b_agg = "one sequential replay (mean ms/case)"
                ra_r = rj(f"{d}/test_timing/seq_C/timing_summary.json")["timing_summary"]["test_case_ms_all_mean"]
                ra_agg = "one sequential test replay (mean ms/case)"
                cpn_ra = rj(f"{d}/C_PN_Gate/test_timing/seq_CPN/timing_summary.json")["timing_summary"]["test_case_ms_all_mean"]
                src += [f"{d}/alignment_all_optimal_analysis/run1/agreement_summary.json", f"{d}/test_timing/seq_B/timing_summary.json",
                        f"{d}/test_timing/seq_C/timing_summary.json", f"{d}/C_PN_Gate/test_timing/seq_CPN/timing_summary.json"]
                if ds == "RoadTraffic":
                    notes.append("Road: duplicate-event derived input copy (scripts/prepare_road_traffic.py); "
                                 "all-optimal search hit max_states for 11 cases (counted as disagreement, lower bound).")
            else:  # BPIC2017: the paper row is the 10-case study of Table 3
                num, den = int(t3_paper["acc_correct_recomputed"]), int(t3_paper["runs"])
                crit = "Table 3 case study: 10 selected test traces x 5 runs, C+PN in the tau-excluded optimal set of B"
                b_r, ra_r = t3_paper["optimal_alignment_median_s"], t3_paper["ra_alignment_cpn_median_s"]
                b_agg = ra_agg = "median of 5 runs per case, mean over the 10 cases (seconds)"
                full = rj(f"{d}/alignment_all_optimal_analysis/bcpn_from_checkpoint/agreement_summary.json")["agreement"]["B-CPN"]["counts"]
                notes.append(f"Full BPIC test set (2,360 cases): B-C+PN tau-excluded all-optimal {full['tau_excluded']}/{full['n']} "
                             f"= {100 * full['tau_excluded'] / full['n']:.2f} % (lower bound; 81 state/resource-limited cases).")
                notes.append("RA-Alignment time here is C+PN; training time is Method C.")
                cpn_ra = None
                src += [f"{CT}/summary_by_case.csv", f"{CT}/additional_cases/summary_by_case.csv",
                        f"{d}/alignment_all_optimal_analysis/bcpn_from_checkpoint/agreement_summary.json"]
            solver = "scipy"
            if ds != "BPIC2017":
                notes.append("ACC is C+PN, while the paper's training and RA-Alignment times are Method C "
                             f"(C+PN: training {cpn_tr / 6e4:.2f} min, repair {cpn_ra:.2f} ms/case).")
            else:
                notes.append(f"C+PN training {cpn_tr / 3.6e6:.2f} h.")
        acc_r = 100 * num / den
        n_test_r = t3_paper["n_cases"] if ds == "BPIC2017" else den  # BPIC: 10 traces (x 5 runs = ACC denominator 50)
        st ={"test": status(n_test, n_test_r), "ACC": status(acc, acc_r, 2), "training": status(tr, tr_r, 2),
              "optimal": status(b_p, b_r, 2), "RA": status(ra_p, ra_r, 2)}
        rows.append({
            "dataset": ds, "test_cases_paper": n_test, "test_cases_recomputed": n_test_r,
            "acc_denominator": den, "acc_numerator": num,
            "acc_pct_paper": f"{acc:.2f}", "acc_pct_recomputed": f"{acc_r:.2f}", "acc_method": acc_m, "acc_criterion": crit,
            "training_paper": tr, "training_unit": tr_unit, "training_recomputed": round(tr_r, 2), "training_method": "C",
            "training_ms_c": tr_ms, "training_ms_cpn_reference": cpn_tr if cpn_tr is not None else "",
            "optimal_alignment_paper": b_p, "optimal_alignment_recomputed": round(b_r, 3), "time_unit": unit,
            "optimal_alignment_method": "B (PM4Py state-equation A*, align + repair)", "lp_solver": solver,
            "optimal_alignment_aggregation": b_agg,
            "ra_alignment_paper": ra_p, "ra_alignment_recomputed": round(ra_r, 3),
            "ra_alignment_method": "C+PN" if ds == "BPIC2017" else "C", "ra_alignment_aggregation": ra_agg,
            "ra_alignment_cpn_reference": round(cpn_ra, 2) if cpn_ra is not None else "",
            "status": worst(*st.values()), "status_detail": "; ".join(f"{k}={v}" for k, v in st.items() if v != "MATCH"),
            "notes": " ".join(notes), "sources": "; ".join(src),
        })
    return pd.DataFrame(rows)


def main():
    n, bad = verify_manifest()
    print(f"MANIFEST: {n} files, SHA-256 mismatches: {len(bad)}")
    if bad:
        print("  ", bad)
        sys.exit(1)
    t1 = table1()
    t3, t3_paper = table3()
    t2 = table2(t3_paper)
    for name, df in (("table1_dataset_summary.csv", t1), ("table2_alignment_results.csv", t2),
                     ("table3_bpic2017_case_study.csv", t3)):
        df.to_csv(RES / name, index=False)
        print(f"\n== {name}: status {df['status'].value_counts().to_dict()}")
    print("\nTable 1"); print(t1[["dataset", "status", "status_detail"]].to_string(index=False))
    print("\nTable 2"); print(t2[["dataset", "acc_numerator", "acc_denominator", "acc_pct_recomputed", "acc_method",
                                   "training_recomputed", "optimal_alignment_recomputed", "lp_solver",
                                   "ra_alignment_recomputed", "status", "status_detail"]].to_string(index=False))
    print("\nTable 3"); print(t3[["case_id", "in_paper_table3", "acc_correct_recomputed", "acc_correct_from_runs_csv",
                                   "optimal_alignment_median_s", "ra_alignment_cpn_median_s", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
