"""Tutorial 2 - RA-Alignment training (Credit, L_N-only NBG, Petri-net gate).

    python tutorials/02_train_ra_alignment.py                # RP-BEAR environment (requirements-rpbear.txt)
    python tutorials/02_train_ra_alignment.py --workers 7    # worker processes (default: CPU count - 1, at most 15)
    python tutorials/02_train_ra_alignment.py --smoke        # code-path check (2 cases, 2 episodes, 2 workers); NOT a model

Optional: you can skip this tutorial; tutorial 4 uses the pre-trained model of the paper run by default.

1. builds the Normal Behavior Graphs from L_N ONLY (tutorial 1 output; each unique trace once, no clean-log traces)
2. loads the Credit log, splits the deviating cases 50 / 50 into L_train / L_test (seed 42, same split as the paper run)
3. trains a NEW RA-Alignment agent (empty shared Q-table) on L_train with the official settings:
   1000 episodes per case, alpha 0, max 8 steps, learning rate 0.1, discount 0.9, exploration 0.5 (decay 0.0001),
   Petri-net gate (an episode can only end as "aligned" if the Petri net accepts the repaired trace)
4. saves the agent, its training statistics and the training-run repaired log (train rows = repair memory used by
   tutorial 4) in artifacts/models/<run name>/
The training function is utils.pn_gate_utils.run_rpbear_gated (unchanged). Like the paper run it also repairs L_test
at the end of training (rows "test" of the repaired log); tutorial 4 measures the test alignment separately.
Results of a new training are not identical to the paper run (parallel workers update the shared Q-table in a
non-deterministic order).
"""
import argparse
import os
import pickle
import time
from datetime import datetime

import _common as C  # noqa: F401  (also puts src/ on sys.path for the worker processes)


def main():
    from multiprocessing import cpu_count

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default=None, help="model folder name under artifacts/models/")
    a = ap.parse_args()
    workers = a.workers or max(1, min(15, cpu_count() - 1))
    rl, limit = dict(C.RL), None
    if a.smoke:
        rl["num_epi"], limit, workers = 2, 2, 2
    name = a.name or f"RA-Alignment_credit_{'SMOKE_' if a.smoke else ''}{datetime.now():%Y%m%d_%H%M%S}"
    out = C.MODELS / name
    if out.exists():
        raise SystemExit(f"[ABORT] {out} exists (not overwritten)")
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("  [note] PYTHONHASHSEED is not 0 (the paper runs used PYTHONHASHSEED=0)")

    import numpy as np

    from sim import _load_and_prepare_data
    from utils.nbg_utils import build_new_nbgs, edge_set_digest
    from utils.pn_gate_utils import PNMLAcceptor, run_rpbear_gated

    ln_log = C.need(C.LN_LOG, "run tutorial 1 first")
    pnml = C.need(C.PNML, "run tutorial 1 first")
    print(f"1) L_N-only NBG from {ln_log.relative_to(C.ROOT)}")
    _c, _a, _g, actset, _f, _af = _load_and_prepare_data(C.DATA_NAME, C.INPUT_PATHS)
    actset = np.asarray(actset)
    nbgs, info = build_new_nbgs(ln_log, actset)
    digest = edge_set_digest(nbgs, actset)
    print(f"   NBG input {info['generated_cases']} cases (each unique L_N trace once) | edge-set digest {digest}")
    gate = PNMLAcceptor(pnml)
    gate.accepts(("a",))
    print(f"2) Petri-net gate {pnml.relative_to(C.ROOT)} {gate.info}")

    print(f"3) training a new RA-Alignment agent | workers {workers} | {rl}{' | SMOKE' if a.smoke else ''}")
    t0 = time.time()
    df, stats, timing, agent = run_rpbear_gated(C.DATA_NAME, rl["num_epi"], rl["alpha"], rl["val_ratio"], rl["seed"],
                                                rl["max_step"], workers, True, nbgs, C.INPUT_PATHS, pn_gate=gate,
                                                limit=limit, verbose=False)
    out.mkdir(parents=True)
    with open(out / f"{name}_agent.pkl", "wb") as fh:
        pickle.dump(agent, fh)
    df.to_csv(out / f"{name}_repaired_log.csv", index=False)
    timing.to_csv(out / f"{name}_case_timing.csv", index=False)
    types = df.groupby("case:concept:name", sort=False, observed=True)["type"].first()
    train_ids = sorted(map(str, types[types == "val"].index))
    test_ids = sorted(map(str, types[types == "test"].index))
    (out / f"{name}_train_case_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (out / f"{name}_test_case_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    stats.update({"method": "RA-Alignment (L_N-only NBG + Petri-net gate)", "nbg_edge_set_digest": digest,
                  "new_nbg_input": "generated_only (L_N-only)", "new_nbg_info": info, "pn_gate": pnml.name,
                  "smoke": a.smoke, "wall_sec": round(time.time() - t0, 2), "agent_sha256": C.sha256(out / f"{name}_agent.pkl"),
                  "pythonhashseed": os.environ.get("PYTHONHASHSEED")})
    C.jdump(out / f"{name}_training_stats.json", stats)
    if not a.smoke:
        C.check(test_ids == sorted(C.read_ids(C.REF_TEST_IDS)), "test split == paper run", strict=False)
    print({k: stats[k] for k in ["n_val_cases", "n_test_cases", "val_total_ms", "q_table_size", "wall_sec"]})
    print(f"model saved: {out.relative_to(C.ROOT)}  (use it in tutorial 4 with --model-dir {out.relative_to(C.ROOT)})")


if __name__ == "__main__":
    main()
