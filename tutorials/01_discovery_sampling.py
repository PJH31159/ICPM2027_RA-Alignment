"""Tutorial 1 - Process discovery and bounded model trace sampling (Credit).

    python tutorials/01_discovery_sampling.py          # PM4Py environment (requirements-pm4py.txt)

1. loads the Credit log and keeps the conforming cases L_clean (empty deviation label, column type_res_trace)
2. discovers a process tree with the Inductive Miner (noise threshold 0.00) and converts it into an accepting Petri net
3. writes the Petri net (PNML with initial / final marking) and the process tree
4. samples the bounded model trace set L_N with (b_num, b_tot) = (5, 7) (Definition 2): exhaustive enumeration of the
   tree language under the loop bounds; silent transitions produce no event; every trace is replayed on the Petri net
5. writes L_N (one row per unique trace, and as a Case / Activity log for tutorial 2) and sampling metadata
6. compares the process tree, the Petri net and L_N with the official files of the paper run (artifacts/reference/)
Outputs: artifacts/petri_net/, artifacts/sampling/
"""
import argparse
import time

import _common as C


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    C.scipy_lp_shim()
    import pandas as pd
    import pm4py

    from utils.data_utils import load_case_traces, write_generated_log
    from utils.discovery_utils import discover_model
    from utils.loop_bound_utils import generate_bounded, make_petri_acceptor

    print("1) Credit log -> conforming cases (L_clean)")
    traces, labels, clean_ids, dev_ids = load_case_traces(C.need(C.ANOMALY_LOG, "Credit log"))
    print(f"   cases {len(traces)}, clean {len(clean_ids)}, deviating {len(dev_ids)}")

    print("2) Inductive Miner (noise threshold 0.00) -> process tree -> Petri net")
    tree, net, im, fm, sec = discover_model(traces, clean_ids, C.NOISE_THRESHOLD)
    C.TREE.parent.mkdir(parents=True, exist_ok=True)
    C.TREE.write_text(str(tree) + "\n", encoding="utf-8")
    pm4py.write_pnml(net, im, fm, str(C.PNML))
    print(f"   discovery {sec:.2f} s | places {len(net.places)}, transitions {len(net.transitions)}, arcs {len(net.arcs)}")

    print(f"3) bounded sampling L_N with (b_num, b_tot) = ({C.B_NUM}, {C.B_TOT})")
    t0 = time.time()
    lang, gst = generate_bounded(tree, C.B_NUM, C.B_TOT, C.SAFETY["max_trace_length"], C.SAFETY["max_intermediate"],
                                 C.SAFETY["max_seconds"], C.SAFETY["max_rss_gb"])
    ln = sorted(lang, key=lambda z: (len(z), z))
    accepts = make_petri_acceptor(net, im, fm)
    n_accepted = sum(accepts(t) for t in ln)
    gen_sec = time.time() - t0
    C.LN_TRACES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trace": [">>".join(t) for t in ln], "length": [len(t) for t in ln]}).to_csv(C.LN_TRACES, index=False)
    write_generated_log(ln, C.LN_LOG)
    print(f"   L_N = {len(ln)} unique traces ({gen_sec:.2f} s), replayed on the Petri net: {n_accepted}/{len(ln)}")

    clean = [tuple(traces[c]) for c in clean_ids]
    v_clean, s_ln = set(clean), set(ln)
    fp = pm4py.precision_footprints(pm4py.discover_footprints(pm4py.objects.log.obj.EventLog(
        [pm4py.objects.log.obj.Trace([pm4py.objects.log.obj.Event({"concept:name": a}) for a in t]) for t in ln])),
        pm4py.discover_footprints(tree))

    print("4) comparison with the official files of the paper run (artifacts/reference/)")
    ref_tree = (C.REF / "credit_process_tree.txt").read_text(encoding="utf-8").splitlines()[0].strip()
    ref_net = pm4py.read_pnml(str(C.REF / "credit_petri_net.pnml"))
    ref_ln = {tuple(t.split(">>")) for t in pd.read_csv(C.REF / "credit_LN.csv", dtype={"trace": str})["trace"]}
    checks = {
        "process tree == official": C.check(str(tree) == ref_tree, "process tree == official", strict=False),
        "Petri net structure == official PNML": C.check(C.net_signature(net, im, fm) == C.net_signature(*ref_net),
                                                        "Petri net structure == official PNML", strict=False),
        "L_N == official L_N (5,000 traces)": C.check(s_ln == ref_ln and len(ln) == 5000, "L_N == official L_N (5,000 unique traces)", strict=False),
        "L_N has no duplicates": C.check(len(ln) == len(s_ln), "no duplicate trace in L_N", strict=False),
        "every L_N trace replays on the Petri net": C.check(n_accepted == len(ln), "every L_N trace replays on the Petri net", strict=False),
        "no silent / Start / End label in L_N": C.check(all(a and a not in ("Start", "End") for t in ln for a in t),
                                                        "no silent / Start / End label in L_N", strict=False),
    }
    meta = {"dataset": "Credit", "noise_threshold": C.NOISE_THRESHOLD, "b_num": C.B_NUM, "b_tot": C.B_TOT,
            "safety_bounds": C.SAFETY, "clean_cases": len(clean_ids), "deviating_cases": len(dev_ids),
            "L_N_unique_traces": len(ln), "max_trace_length": max(map(len, ln)), "generation_sec": round(gen_sec, 3),
            "peak_intermediate": gst.get("peak_intermediate"),
            "clean_variant_coverage_pct": round(100 * len(v_clean & s_ln) / len(v_clean), 2),
            "clean_variant_coverage": f"{len(v_clean & s_ln)}/{len(v_clean)} (|V_clean ∩ L_N| / |V_clean|)",
            "clean_case_coverage_pct": round(100 * sum(t in s_ln for t in clean) / len(clean), 2),
            "footprints_precision_pct": round(100 * fp, 2), "pm4py": pm4py.__version__,
            "checks": checks, "reproduces_official": all(checks.values())}
    C.jdump(C.LN_TRACES.parent / "sampling_metadata.json", meta)
    print({k: meta[k] for k in ["L_N_unique_traces", "clean_variant_coverage_pct", "clean_case_coverage_pct",
                                "footprints_precision_pct", "reproduces_official"]})
    print(f"outputs: {C.PNML.relative_to(C.ROOT)}, {C.TREE.relative_to(C.ROOT)}, {C.LN_TRACES.relative_to(C.ROOT)}, "
          f"{C.LN_LOG.relative_to(C.ROOT)}")
    if not meta["reproduces_official"]:
        raise SystemExit("tutorial 1 finished, but the result differs from the official files (see the FAILED checks)")


if __name__ == "__main__":
    main()
