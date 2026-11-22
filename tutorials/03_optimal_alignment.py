"""Tutorial 3 - Optimal Alignment baseline (Credit test cases).

    python tutorials/03_optimal_alignment.py            # PM4Py environment (requirements-pm4py.txt)

1. reads the Petri net of tutorial 1 (PNML with initial / final marking) and the test cases L_test of the paper run
2. aligns every test trace to the Petri net with PM4Py's optimal alignment (A* with state equation, standard costs)
3. repairs the trace from the alignment: synchronous move -> keep, log move -> delete, visible model move -> insert,
   silent (tau) model move -> no event
4. measures the time per case (perf_counter around alignment + repair; loading, PNML reading and file writing excluded)
5. compares the repaired traces with the official Optimal Alignment output of the paper run
Outputs: results/optimal_alignment/
"""
import argparse
import time

import _common as C


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    C.scipy_lp_shim()
    import pandas as pd
    import pm4py

    from alignment.exact_alignment import ALIGNMENT_VARIANT, align_case, repair_from_alignment
    from utils.data_utils import build_event_log, load_case_traces
    from utils.timing_utils import summarize_values

    pnml = C.need(C.PNML, "run tutorial 1 first")
    net, im, fm = pm4py.read_pnml(str(pnml))
    traces, labels, _clean, _dev = load_case_traces(C.need(C.ANOMALY_LOG, "Credit log"))
    by_str = {str(k): k for k in traces.index}
    test_ids = C.read_ids(C.REF_TEST_IDS)
    print(f"1) Petri net {pnml.relative_to(C.ROOT)} (places {len(net.places)}, transitions {len(net.transitions)}) | "
          f"L_test {len(test_ids)} cases")

    print(f"2) optimal alignment ({ALIGNMENT_VARIANT.name}) + repair, case by case")
    rows = []
    t_phase = time.perf_counter()
    for cid in test_ids:
        key = by_str[cid]
        trace_obj = build_event_log([key], traces)[0]
        t0 = time.perf_counter()
        try:
            res = align_case(trace_obj, net, im, fm)
            repaired, moves = repair_from_alignment(res["alignment"])
            failed, cost = False, res["cost"]
        except Exception:  # noqa: BLE001  (as in the paper run: keep the original trace and flag the case)
            repaired, moves, failed, cost = list(traces[key]), {}, True, None
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"case_id": cid, "anomaly_label": labels[key], "original_trace": C.space(traces[key]),
                     "repaired_trace": C.space(repaired), "alignment_cost": cost, "alignment_failed": failed,
                     "time_ms": round(ms, 4), **moves})
    phase_ms = (time.perf_counter() - t_phase) * 1000
    df = pd.DataFrame(rows)
    C.OA_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OA_RESULTS / "optimal_alignment_repaired_traces.csv", index=False)

    print("3) comparison with the official Optimal Alignment output (artifacts/reference/)")
    ref = pd.read_csv(C.REF / "credit_optimal_alignment_official.csv", dtype={"case:concept:name": str})
    ref = ref[ref["type"] == "test"].set_index("case:concept:name")
    same = sum(ref.loc[r["case_id"], "repaired_trace"].replace(">>", " ") == r["repaired_trace"] for r in rows)
    checks = {
        "all test cases aligned": C.check(not df["alignment_failed"].any(), "every test case aligned", strict=False),
        "test cases == official": C.check(set(test_ids) == set(ref.index), "test cases == official", strict=False),
        "repaired traces == official": C.check(same == len(rows), f"repaired traces == official ({same}/{len(rows)})",
                                               strict=False),
    }
    summary = {"method": "Optimal Alignment (PM4Py)", "alignment_variant": ALIGNMENT_VARIANT.name,
               "pm4py": pm4py.__version__, "petri_net": str(pnml.relative_to(C.ROOT)).replace("\\", "/"),
               "test_cases": len(rows), "failed": int(df["alignment_failed"].sum()),
               "time_ms_per_case": summarize_values(df["time_ms"].tolist()), "phase_wall_ms": round(phase_ms, 2),
               "same_repaired_trace_as_official": f"{same}/{len(rows)}",
               "timed_section": "align_case + repair_from_alignment (perf_counter), single process",
               "checks": checks}
    C.jdump(C.OA_RESULTS / "optimal_alignment_summary.json", summary)
    print({"test_cases": len(rows), "mean_ms": summary["time_ms_per_case"]["mean"],
           "median_ms": summary["time_ms_per_case"]["median"], "same_as_official": summary["same_repaired_trace_as_official"]})
    print(f"outputs: {C.OA_RESULTS.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
