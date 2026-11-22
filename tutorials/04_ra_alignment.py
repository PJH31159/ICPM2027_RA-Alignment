"""Tutorial 4 - RA-Alignment inference on the Credit test cases (no training).

    python tutorials/04_ra_alignment.py                                   # pre-trained model of the paper run
    python tutorials/04_ra_alignment.py --model-dir artifacts/models/<run> # your own model from tutorial 2
RP-BEAR environment (requirements-rpbear.txt). For the reference comparison run with PYTHONHASHSEED=0.

1. loads the agent (Q-table) with an allow-list unpickler (only the RA-Alignment agent class, numpy and pandas objects;
   the pre-trained file is also checked against its SHA-256 in artifacts/MANIFEST.csv)
2. rebuilds the L_N-only Normal Behavior Graphs from tutorial 1 and checks that they equal the graphs used in training
3. rebuilds the train / test split (seed 42) and the repair memory from the training rows of the model's repaired log
4. repairs every test case one by one in a single process with the FROZEN Q-table (no Q-value update):
   a trace already repaired during training is taken from the repair memory, otherwise the agent runs up to 100
   episodes (exploration 0.5, at most 8 steps) and an episode can only end as "aligned" if the Petri net accepts it
5. measures the time per case and, for the pre-trained model, compares the output with the reference run
Outputs: results/ra_alignment/
"""
import argparse
import os
from pathlib import Path

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=None, help="model folder (default: artifacts/pretrained_models/credit_ra_alignment)")
    a = ap.parse_args()
    mdir = (C.ROOT / a.model_dir).resolve() if a.model_dir else C.PRETRAINED
    pretrained = mdir == C.PRETRAINED.resolve() or mdir == C.PRETRAINED
    files = C.model_files(mdir)
    for k, p in files.items():
        if p is None:
            raise SystemExit(f"[missing] {k} file in {mdir}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("  [note] PYTHONHASHSEED is not 0: the run is valid, but may differ from the reference run (PYTHONHASHSEED=0)")

    import numpy as np
    import pandas as pd

    from sim import _load_and_prepare_data
    from utils.nbg_utils import build_new_nbgs, edge_set_digest
    from utils.pn_gate_utils import PNMLAcceptor, replay_test_phase_gated
    from utils.timing_utils import prepare_rpbear_inputs, rebuild_repair_memory, summarize, test_result_frame

    stats = C.jload(files["stats"])
    print(f"1) model {Path(files['agent']).relative_to(C.ROOT)}")
    if pretrained:
        rel = str(Path(files["agent"]).relative_to(C.ROOT)).replace("\\", "/")
        C.check(C.sha256(files["agent"]) == C.manifest_sha(rel), "agent file SHA-256 == artifacts/MANIFEST.csv")
    agent = C.load_agent(files["agent"])
    print(f"   Q-table entries {len(agent.q_table)} | exploration rate {agent.exploration_rate}")

    print("2) L_N-only NBG (tutorial 1 output) and Petri-net gate")
    _c, _a, _g, actset, _f, _af = _load_and_prepare_data(C.DATA_NAME, C.INPUT_PATHS)
    actset = np.asarray(actset)
    nbgs, _info = build_new_nbgs(C.need(C.LN_LOG, "run tutorial 1 first"), actset)
    digest = edge_set_digest(nbgs, actset)
    C.check(digest == stats["nbg_edge_set_digest"], f"NBG edge-set digest {digest} == training run")
    gate = PNMLAcceptor(C.need(C.PNML, "run tutorial 1 first"))
    gate.accepts(("a",))

    print("3) split and repair memory")
    rl = {k: stats[k] for k in ["num_epi", "alpha", "val_ratio", "seed", "max_step"]}
    prep = prepare_rpbear_inputs(C.DATA_NAME, C.INPUT_PATHS, float(rl["val_ratio"]), int(rl["seed"]), nbgs)
    C.check(sorted(map(str, prep["test_ids"])) == sorted(C.read_ids(files["test_ids"])), "test split == model's test split")
    C.check(sorted(map(str, prep["val_ids"])) == sorted(C.read_ids(files["train_ids"])), "train split == model's train split")
    memory, minfo = rebuild_repair_memory(files["repaired_log"], prep["case_to_trace"], prep["val_ids"])
    print(f"   repair memory: {minfo['memory_traces']} traces from {minfo['validation_cases']} training cases")

    print(f"4) inference on {len(prep['test_ids'])} test cases (frozen Q-table, single process)")
    q_before = dict(agent.q_table)
    results, timing, wall = replay_test_phase_gated(prep, agent, memory, float(rl["alpha"]), int(rl["num_epi"]),
                                                    int(rl["max_step"]), gate, int(rl["seed"]))
    C.check(dict(agent.q_table) == q_before, "Q-table unchanged by inference")
    rep = test_result_frame(results)
    rep["case:concept:name"] = rep["case:concept:name"].astype(str)
    t_by = {str(r["case_id"]): r for r in timing}
    rows = []
    for cid, g in rep.groupby("case:concept:name", sort=False):
        acts = [x for x in g["concept:name"] if x not in ("Start", "End")]
        rows.append({"case_id": cid, "original_trace": C.space(prep["case_to_trace"][cid]), "repaired_trace": C.space(acts),
                     "path": t_by[cid]["path"], "time_ms": t_by[cid]["repair_time_ms"],
                     "pn_gate_rejected_episodes": int(g["pn_gate_rejected_episodes"].iloc[0])
                     if "pn_gate_rejected_episodes" in g else None})
    C.RA_RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.RA_RESULTS / "ra_alignment_repaired_traces.csv", index=False)
    rep.to_csv(C.RA_RESULTS / "ra_alignment_repaired_log.csv", index=False)
    summ = summarize(timing, wall)
    accepted = sum(gate.accepts(tuple(r["repaired_trace"].split())) for r in rows)

    checks = {"every test case repaired": C.check(len(rows) == len(prep["test_ids"]), "every test case has an output",
                                                  strict=False)}
    if pretrained:
        print("5) comparison with the reference run of the pre-trained model (artifacts/reference/)")
        ref = pd.read_csv(C.REF / "credit_ra_alignment_exp068_sequential_test.csv")
        ref["case:concept:name"] = ref["case:concept:name"].astype(str)
        ref_tr = {c: [x for x in g["concept:name"] if x not in ("Start", "End")] for c, g in ref.groupby("case:concept:name")}
        same = sum(ref_tr.get(r["case_id"]) == r["repaired_trace"].split() for r in rows)
        checks["repaired traces == reference run"] = C.check(same == len(rows),
                                                             f"repaired traces == reference run ({same}/{len(rows)})", strict=False)
        ref_t = C.jload(C.REF / "credit_ra_alignment_exp068_sequential_timing.json")["timing_summary"]
        print(f"   mean time per case: this run {summ['test_case_ms_all_mean']:.1f} ms | reference run "
              f"{ref_t['test_case_ms_all_mean']:.1f} ms (machine dependent)")
    out = {"method": "RA-Alignment (L_N-only NBG + Petri-net gate), frozen Q-table",
           "model_dir": str(Path(mdir).relative_to(C.ROOT)).replace("\\", "/"), "pretrained": pretrained,
           "agent_sha256": C.sha256(files["agent"]), "nbg_edge_set_digest": digest, "settings": rl,
           "test_cases": len(rows), "accepted_by_petri_net": f"{accepted}/{len(rows)}",
           "rl_path_cases": sum(r["path"] == "RL" for r in rows), "memory_path_cases": sum(r["path"] != "RL" for r in rows),
           "timing": summ, "repair_memory": minfo, "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "checks": checks}
    C.jdump(C.RA_RESULTS / "ra_alignment_summary.json", out)
    print({"test_cases": len(rows), "accepted_by_petri_net": out["accepted_by_petri_net"],
           "mean_ms": summ["test_case_ms_all_mean"], "median_ms": summ["test_case_ms_all_median"]})
    print(f"outputs: {C.RA_RESULTS.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
