"""Tutorial 5 - Performance comparison: Optimal Alignment vs RA-Alignment (Credit test cases).

    python tutorials/05_compare_performance.py                    # PM4Py environment (requirements-pm4py.txt)
    python tutorials/05_compare_performance.py --use-reference    # use the shipped all-optimal sets (no search)
    python tutorials/05_compare_performance.py --model-dir artifacts/models/<run>   # training row of your own model

Alignment accuracy (ACC, paper Sec. 4.1): a repaired trace is correct if it is one of the optimal repairs of the
test trace. Because a trace can have several optimal alignments, ALL optimal repaired traces are computed with an
exhaustive search on the Petri net that counts only visible deviations (log move 1, visible model move 1, silent
transition 0; "tau-excluded" all-optimal set, src/alignment/all_optimal.py). Agreement with one single reference
alignment is NOT used as ACC (it is reported separately as "exact agreement with Optimal Alignment").

1. tau-excluded all-optimal sets of the test traces (recomputed by default and compared with the official sets)
2. ACC of Optimal Alignment (tutorial 3) and of RA-Alignment (tutorial 4, this run)
3. ACC of the training run of the model (test rows of the model's repaired log; for the pre-trained model this is the
   paper run, 163 / 210)
4. time per case of both methods (this machine) and training time of the model (from its training statistics)
Inputs: results/optimal_alignment/, results/ra_alignment/ (run tutorials 3 and 4 first)
Outputs: results/comparison/
"""
import argparse
import time
from pathlib import Path

import _common as C


def tau_sets_reference():
    import pandas as pd

    tr = pd.read_csv(C.REF / "credit_all_optimal_repaired_traces.csv", dtype=str, keep_default_na=False)
    return tr[tr["set"] == "deviation_only"].groupby("case_id")["trace"].agg(set).to_dict()


def model_test_traces(repaired_log):
    import pandas as pd

    df = pd.read_csv(repaired_log, dtype=str, keep_default_na=False)
    df = df[(df["type"] == "test") & (~df["concept:name"].isin(["Start", "End"]))]
    return df.groupby("case:concept:name", sort=False)["concept:name"].agg(" ".join).to_dict()


def acc(traces, sets, cases):
    ok = sum(traces.get(c) in sets.get(c, set()) for c in cases)
    return {"correct": ok, "n": len(cases), "acc_pct": round(100 * ok / len(cases), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--use-reference", action="store_true", help="skip the search, use the official all-optimal sets")
    ap.add_argument("--model-dir", default=None, help="model whose training-run ACC is reported (default: pre-trained)")
    a = ap.parse_args()
    import pandas as pd

    from utils.timing_utils import summarize_values

    oa = pd.read_csv(C.need(C.OA_RESULTS / "optimal_alignment_repaired_traces.csv", "run tutorial 3 first"),
                     dtype={"case_id": str}, keep_default_na=False)
    ra = pd.read_csv(C.need(C.RA_RESULTS / "ra_alignment_repaired_traces.csv", "run tutorial 4 first"),
                     dtype={"case_id": str}, keep_default_na=False)
    ra_meta = C.jload(C.RA_RESULTS / "ra_alignment_summary.json")
    mdir = (C.ROOT / a.model_dir).resolve() if a.model_dir else C.PRETRAINED
    mfiles = C.model_files(mdir)
    cases = list(oa["case_id"])
    C.check(sorted(cases) == sorted(ra["case_id"]), "same test cases in tutorial 3 and tutorial 4 outputs")

    ref_sets = tau_sets_reference()
    if a.use_reference:
        print("1) tau-excluded all-optimal sets: official files (artifacts/reference/), no search")
        sets, search = ref_sets, None
    else:
        print("1) tau-excluded all-optimal sets: exhaustive search on the Petri net (tutorial 1)")
        C.scipy_lp_shim()
        import pm4py

        from alignment.all_optimal import dev_optimal, net_struct

        net, im, fm = pm4py.read_pnml(str(C.need(C.PNML, "run tutorial 1 first")))
        ns = net_struct(net, im, fm)
        orig = dict(zip(oa["case_id"], oa["original_trace"]))
        sets, limited, t0 = {}, [], time.perf_counter()
        for c in cases:
            _cost, outs, _pops = dev_optimal(tuple(orig[c].split()), ns, move_cost=1, tau_cost=0)
            if outs is None:
                limited.append(c)
                continue
            sets[c] = {" ".join(t) for t in outs}
        search = {"sec": round(time.perf_counter() - t0, 2), "state_limit_hit": limited}
        same = sum(sets.get(c) == ref_sets.get(c) for c in cases)
        C.check(same == len(cases), f"recomputed sets == official sets ({same}/{len(cases)})", strict=False)
        search["same_as_official"] = f"{same}/{len(cases)}"
    print(f"   optimal repaired traces per case: mean {sum(len(sets.get(c, ())) for c in cases) / len(cases):.2f}, "
          f"cases with more than one: {sum(len(sets.get(c, ())) > 1 for c in cases)}")

    print("2) ACC (tau-excluded all-optimal)")
    oa_tr, ra_tr = dict(zip(oa["case_id"], oa["repaired_trace"])), dict(zip(ra["case_id"], ra["repaired_trace"]))
    acc_oa, acc_ra = acc(oa_tr, sets, cases), acc(ra_tr, sets, cases)
    train_tr = {str(k): v for k, v in model_test_traces(mfiles["repaired_log"]).items()}
    acc_train = acc(train_tr, sets, cases)
    exact = sum(oa_tr[c] == ra_tr[c] for c in cases)
    stats = C.jload(mfiles["stats"])
    t_oa = summarize_values([float(x) for x in oa["time_ms"]])
    t_ra = ra_meta["timing"]
    rows = [
        {"method": "Optimal Alignment", "source": "this run (tutorial 3)", "acc_correct": acc_oa["correct"],
         "n_test": acc_oa["n"], "acc_pct": acc_oa["acc_pct"], "time_ms_per_case_mean": t_oa["mean"],
         "time_ms_per_case_median": t_oa["median"], "training_time_sec": None},
        {"method": "RA-Alignment", "source": "this run (tutorial 4, frozen Q-table, sequential)",
         "acc_correct": acc_ra["correct"], "n_test": acc_ra["n"], "acc_pct": acc_ra["acc_pct"],
         "time_ms_per_case_mean": t_ra["test_case_ms_all_mean"], "time_ms_per_case_median": t_ra["test_case_ms_all_median"],
         "training_time_sec": None},
        {"method": "RA-Alignment", "source": ("paper run (pre-trained model, training-run test rows)" if mdir == C.PRETRAINED
                                              else f"training run of {Path(mdir).name} (test rows)"),
         "acc_correct": acc_train["correct"], "n_test": acc_train["n"], "acc_pct": acc_train["acc_pct"],
         "time_ms_per_case_mean": None, "time_ms_per_case_median": None,
         "training_time_sec": stats.get("training_time_sec")},
    ]
    for r in rows:
        print(f"   {r['method']:<18} {r['source']:<58} ACC {r['acc_correct']}/{r['n_test']} = {r['acc_pct']:.2f}%")
    print(f"   exact agreement RA-Alignment (this run) == Optimal Alignment: {exact}/{len(cases)} (not ACC)")
    if mdir == C.PRETRAINED:
        C.check((acc_train["correct"], acc_train["n"]) == (163, 210), "paper-run ACC recomputed == 163/210", strict=False)

    C.CMP_RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.CMP_RESULTS / "performance_comparison.csv", index=False)
    pd.DataFrame([{"case_id": c, "n_optimal": len(sets.get(c, ())), "optimal_alignment": oa_tr[c],
                   "oa_correct": oa_tr[c] in sets.get(c, set()), "ra_alignment": ra_tr[c],
                   "ra_correct": ra_tr[c] in sets.get(c, set()), "ra_equals_oa": oa_tr[c] == ra_tr[c]}
                  for c in cases]).to_csv(C.CMP_RESULTS / "per_case_comparison.csv", index=False)
    md = ["| Method | Source | ACC | Time per case (mean / median, ms) | Training time (s) |", "|---|---|---|---|---|"]
    for r in rows:
        tm = "-" if r["time_ms_per_case_mean"] is None else f"{r['time_ms_per_case_mean']:.1f} / {r['time_ms_per_case_median']:.1f}"
        tr = "-" if r["training_time_sec"] is None else f"{r['training_time_sec']:.1f}"
        md.append(f"| {r['method']} | {r['source']} | {r['acc_correct']}/{r['n_test']} ({r['acc_pct']:.2f}%) | {tm} | {tr} |")
    md += ["", f"Exact agreement RA-Alignment (this run) vs Optimal Alignment: {exact}/{len(cases)} (not used as ACC).",
           "ACC: repaired trace in the tau-excluded all-optimal set. Times are machine dependent (single process)."]
    (C.CMP_RESULTS / "performance_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    C.jdump(C.CMP_RESULTS / "comparison_metadata.json", {
        "acc_definition": "repaired trace in the tau-excluded all-optimal set (log move 1, visible model move 1, tau 0)",
        "all_optimal_sets": "official files" if a.use_reference else "recomputed (src/alignment/all_optimal.py)",
        "search": search, "exact_agreement_ra_vs_oa": f"{exact}/{len(cases)}",
        "sources": {"optimal_alignment": "results/optimal_alignment/optimal_alignment_repaired_traces.csv",
                    "ra_alignment": "results/ra_alignment/ra_alignment_repaired_traces.csv",
                    "training_run": str(Path(mfiles["repaired_log"]).relative_to(C.ROOT)).replace("\\", "/"),
                    "reference_sets": "artifacts/reference/credit_all_optimal_repaired_traces.csv (set deviation_only)"},
        "ra_model": ra_meta.get("model_dir"), "timing_oa": t_oa, "rows": rows})
    print(f"outputs: {C.CMP_RESULTS.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
