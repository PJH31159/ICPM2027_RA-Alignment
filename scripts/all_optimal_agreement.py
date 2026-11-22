"""All-optimal agreement of Method B with C / C+PN (.venv).

Definitions (search code shared with scripts/credit_all_optimal_analysis.py, whose repair_from_alignment / net_struct /
dev_optimal definitions are executed from that file):
  exact        : X repaired == B repaired
  native-cost  : X in the set of optimal B repairs under the PM4Py default cost
                 (log / visible model move 10000, tau 1, sync 0), own exact search;
                 optionally cross-checked against PM4Py generator_dijkstra_less_memory (the decision uses the own search)
  tau-excluded : X in the set of optimal B repairs under the deviation-only cost
                 (log / visible model move 1, tau 0, sync 0) <- main metric.
                 tau moves are invisible routing steps of the model and produce no event in the repaired
                 trace, so they should not decide which repair is optimal.
A multiple-optimal case has >= 2 optimal repaired traces. Accuracy (EMR) is stored only as an auxiliary value.
The search is bounded by --max-states; a case that hits it is counted as disagreement (lower bound).

Input:  B = *_alignment_per_case.csv (test) in --b-dir; C / C+PN repaired logs (optional)
Output: --out folder (case_summary.csv, optimal_repaired_traces.csv, agreement_summary.json)
Example: .venv/Scripts/python scripts/all_optimal_agreement.py --b-dir <methodB_run1> --c-log <repaired_log_c.csv> --cpn-log <repaired_log_cpn.csv> --out <dir>
"""
import sys
import types

sys.modules["cvxopt"] = None
import lxml  # noqa: E402
from lxml import etree as _etree  # noqa: E402

_obj = types.ModuleType("lxml.objectify")
_obj.parse, _obj.fromstring = _etree.parse, _etree.fromstring
sys.modules["lxml.objectify"] = _obj
lxml.objectify = _obj

import argparse  # noqa: E402
import heapq  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pm4py  # noqa: E402
from pm4py.algo.conformance.alignments.petri_net.variants import generator_dijkstra_less_memory as gen_lm  # noqa: E402
from pm4py.objects.log.obj import Event, Trace  # noqa: E402
from pm4py.util.lp import solver as lp_solver  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _data_root(cli_root=None):
    """Data root: --data-root argument > environment variable ICPM2027_DATA_ROOT > <repo>/data/raw.

    Same priority as src/utils/data_utils.resolve_data_root; it must point to the P-BEAR-RL Full_input folder.
    """
    import os
    if cli_root is None and "--data-root" in sys.argv:
        cli_root = sys.argv[sys.argv.index("--data-root") + 1]
    if cli_root:
        return Path(cli_root).resolve()
    env = os.environ.get("ICPM2027_DATA_ROOT")
    return Path(env).resolve() if env else (PROJECT_ROOT / "data" / "raw").resolve()


def _strip_data_root(argv):
    """Remove '--data-root X' from argv before positional arguments are read (existing argument handling unchanged)."""
    if "--data-root" in argv:
        i = argv.index("--data-root")
        del argv[i:i + 2]
CREDIT_SCRIPT = PROJECT_ROOT / "scripts/credit_all_optimal_analysis.py"


def credit_defs():
    """Load the repair_from_alignment / net_struct / fire / dev_optimal definitions from the Credit script unchanged."""
    src = CREDIT_SCRIPT.read_text(encoding="utf-8")
    ns = {"heapq": heapq, "Counter": Counter, "Event": Event, "Trace": Trace,
          "alignments_algo": None, "VARIANT": None, "gen_lm": gen_lm, "gen_nh": gen_lm}
    exec(compile(src[src.index("def trace_obj"):src.index("# ---------------------------------------------------------------- toy")],
                 "credit_all_optimal_defs", "exec"), ns)
    return ns


def load_repaired(path, ids):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["type"] == "test"]
    df = df[~df["concept:name"].isin(["Start", "End"])]
    tr = df.groupby("case:concept:name", sort=False)["concept:name"].agg(tuple)
    return {c: tr[c] for c in ids if c in tr.index}


def gt_traces(path):
    df = pd.read_csv(path, usecols=["Case", "Activity"], dtype=str)
    return df.groupby("Case", sort=False)["Activity"].agg(tuple).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--c-log")
    ap.add_argument("--cpn-log")
    ap.add_argument("--gt", default=None, help="ground-truth normal log (default: <data root>/encoded_normal/mccloud.csv)")
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-states", type=int, default=5_000_000)
    # Execution options (the decision definition is unchanged): each case is saved to a checkpoint (jsonl);
    # a re-run skips finished cases.
    #   --shard K/N : compute only ids[K::N] of the test cases (parallel runs; results go to the checkpoint only)
    #   --merge     : merge all checkpoints and write the final files (all test cases must be present)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--merge", action="store_true")
    # --skip-gen-check: skip the PM4Py generator_dijkstra_less_memory cross-check (gen_less_memory_eq_native).
    #   That auxiliary check enumerates all alignments without a state limit and can exhaust memory on large models.
    #   The agreement decision (dev_optimal, --max-states) is unchanged; gen_* columns are None for skipped cases.
    ap.add_argument("--skip-gen-check", action="store_true")
    # --resource-skip FILE: file whose first (tab-separated) column is a case id. These cases reached the
    #   memory guard even when run alone in a single process. They are recorded with resource_limit_hit=True and an
    #   empty all-optimal set (no search) -> counted as disagreement in all-optimal agreement (lower bound).
    #   Exact agreement is a plain trace comparison and is computed normally. This is a separate category from the
    #   state limit (--max-states) and is not merged into state_limit_hit.
    ap.add_argument("--resource-skip", default=None)
    a = ap.parse_args()
    if a.gt is None:
        a.gt = str(_data_root(a.data_root) / "encoded_normal" / "mccloud.csv")
    ns = credit_defs()
    import psutil  # noqa: E402  (per-case memory record)
    _proc = psutil.Process()
    bdir, out = PROJECT_ROOT / a.b_dir, PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    net, im, fm = pm4py.read_pnml(str(next(bdir.glob("*_petri_net.pnml"))))
    st = ns["net_struct"](net, im, fm)
    alg = pd.read_csv(next(bdir.glob("*_alignment_per_case.csv")), dtype={"case:concept:name": str})
    alg = alg[alg.type == "test"].set_index("case:concept:name")
    ids = list(alg.index)
    X = {}
    if a.c_log:
        X["C"] = load_repaired(PROJECT_ROOT / a.c_log, ids)
    if a.cpn_log:
        X["CPN"] = load_repaired(PROJECT_ROOT / a.cpn_log, ids)
    gt = gt_traces(a.gt)
    ck_dir = out / "checkpoint"
    ck_dir.mkdir(exist_ok=True)
    done = {}
    for f in sorted(ck_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["row"]["case_id"]] = rec
    todo = ids
    if a.shard:
        kk, nn = map(int, a.shard.split("/"))
        todo = ids[kk::nn]
    ck_file = ck_dir / (f"shard{a.shard.replace('/', 'of')}.jsonl" if a.shard else "main.jsonl")
    if a.merge:
        missing = [c for c in ids if c not in done]
        if missing:
            print(f"[ERROR] merge: {len(missing)}/{len(ids)} cases unfinished (e.g. {missing[:5]})")
            sys.exit(2)
    print(f"checkpoint: {len(done)} cases done, to do in this run {sum(c not in done for c in todo)}/{len(todo)}", flush=True)
    res_skip = set()
    if a.resource_skip and Path(a.resource_skip).exists():
        res_skip = {ln.split("	")[0].strip() for ln in Path(a.resource_skip).read_text(encoding="utf-8").splitlines() if ln.strip()}
    rows, trows, t_all = [], [], time.time()
    elapsed_prev = sum(r["row"].get("_case_sec", 0) for r in done.values())
    for k, c in enumerate(ids if a.merge else todo, 1):
        if c in done:
            if a.merge or not a.shard:
                rows.append({kx: v for kx, v in done[c]["row"].items() if kx != "_case_sec"})
                trows.extend(done[c]["trows"])
            continue
        t_case = time.time()
        an = tuple(alg.loc[c, "original_trace"].split(">>"))
        b = tuple(alg.loc[c, "repaired_trace"].split(">>"))
        cost_b = int(alg.loc[c, "alignment_cost"])
        rskip = c in res_skip
        if rskip:
            xcost, S_ex, xpops, ms_ex, dcost, S_dev, dpops, ms_dev = None, set(), None, None, None, set(), None, None
        else:
            t0 = time.perf_counter()
            xcost, S_ex, xpops = ns["dev_optimal"](an, st, 10000, 1, max_states=a.max_states)
            ms_ex = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dcost, S_dev, dpops = ns["dev_optimal"](an, st, 1, 0, max_states=a.max_states)
            ms_dev = (time.perf_counter() - t0) * 1000
        if a.skip_gen_check or rskip:
            als, ms_gen, peak_kb, S_gen = None, None, None, None
        else:
            tracemalloc.start()
            t0 = time.perf_counter()
            als = list(gen_lm.apply(ns["trace_obj"](an), net, im, fm))
            ms_gen = (time.perf_counter() - t0) * 1000
            peak_kb = tracemalloc.get_traced_memory()[1] / 1024
            tracemalloc.stop()
            S_gen = {ns["repair_from_alignment"](x["alignment"])[0] for x in als}
        S_ex, S_dev = set(S_ex or ()), set(S_dev or ())
        g = gt.get(c)
        r = {"case_id": c, "anomaly_trace": " ".join(an), "ground_truth": " ".join(g) if g else None, "B": " ".join(b),
             "B_cost": cost_b, "exact_cost": xcost, "exact_cost_eq_B": xcost == cost_b, "state_limit_hit": (xcost is None or dcost is None) and not rskip,
             "resource_limit_hit": rskip,
             "native_n_optimal": len(S_ex), "tau_excl_n_optimal": len(S_dev), "tau_excl_deviations": dcost,
             "B_in_native_set": b in S_ex, "B_in_tau_excl_set": b in S_dev, "native_subset_of_tau_excl": S_ex <= S_dev,
             "gen_less_memory_n_alignments": None if als is None else len(als),
             "gen_less_memory_n_optimal": None if S_gen is None else len(S_gen),
             "gen_less_memory_eq_native": None if S_gen is None else S_gen == S_ex, "B_correct": b == g,
             "trace_len": len(an), "rss_mb_after_case": round(_proc.memory_info().rss / 2**20, 1),
             "peak_wset_mb_process": round(getattr(_proc.memory_info(), "peak_wset", 0) / 2**20, 1),
             "ms_native_search": None if ms_ex is None else round(ms_ex, 2), "native_pops": xpops, "ms_tau_excl_search": None if ms_dev is None else round(ms_dev, 2),
             "tau_excl_pops": dpops, "ms_gen_less_memory": None if ms_gen is None else round(ms_gen, 2),
             "peak_kb_gen": None if peak_kb is None else round(peak_kb, 1)}
        for m, tr in X.items():
            x = tr.get(c)
            r[f"{m}"] = " ".join(x) if x is not None else None
            r[f"{m}_exact_eq_B"] = x == b
            r[f"{m}_in_native_set"] = x in S_ex
            r[f"{m}_in_tau_excl_set"] = x in S_dev
            r[f"{m}_correct"] = x == g
        rows.append(r)
        tr_c = []
        for name, S in (("native", S_ex), ("tau_excluded", S_dev), ("gen_less_memory", S_gen)):
            for t in sorted(S or ()):  # S_gen = None (--skip-gen-check) -> no rows
                tr_c.append({"case_id": c, "set": name, "trace": " ".join(t), "is_B": t == b, "is_GT": t == g,
                             **{f"is_{m}": t == tr.get(c) for m, tr in X.items()}})
        trows.extend(tr_c)
        with ck_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"row": {**r, "_case_sec": round(time.time() - t_case, 3)}, "trows": tr_c},
                                ensure_ascii=False, default=str) + "\n")
        if k % 50 == 0 or k == len(todo):
            print(f"{k}/{len(todo)} {time.time() - t_all:.1f}s max_native_set {max(x['native_n_optimal'] for x in rows)} "
                  f"max_tau_excl_set {max(x['tau_excl_n_optimal'] for x in rows)}", flush=True)
    if a.shard:
        print(f"[shard {a.shard}] done -> {ck_file}")
        return
    if len(rows) != len(ids):
        print(f"[ERROR] {len(rows)}/{len(ids)} cases present")
        sys.exit(2)
    T = pd.DataFrame(rows)
    T.to_csv(out / "case_summary.csv", index=False)
    pd.DataFrame(trows).to_csv(out / "optimal_repaired_traces.csv", index=False)
    n = len(T)
    summ = {"cases": n, "lp_solver_in_process": lp_solver.DEFAULT_LP_SOLVER_VARIANT, "b_dir": a.b_dir,
            "c_log": a.c_log, "cpn_log": a.cpn_log, "total_sec": round(elapsed_prev + time.time() - t_all, 2),
            "total_sec_note": 'sum of checkpointed case times + this run (with parallel shards this is closer to the sequential CPU sum than to wall time)',
            "checks": {"exact_cost_eq_B": int(T.exact_cost_eq_B.sum()), "B_in_native_set": int(T.B_in_native_set.sum()),
                       "B_in_tau_excl_set": int(T.B_in_tau_excl_set.sum()), "native_subset_of_tau_excl": int(T.native_subset_of_tau_excl.sum()),
                       "state_limit_hit": int(T.state_limit_hit.sum()), "gen_less_memory_eq_native": int((T.gen_less_memory_eq_native == True).sum()),  # noqa: E712
                       "gen_less_memory_checked": int(T.gen_less_memory_eq_native.notna().sum()),
                       "resource_limit_hit": int((T.get("resource_limit_hit", pd.Series(dtype=object)) == True).sum())},  # noqa: E712
            "multiple_optimal_cases": {"native": int((T.native_n_optimal >= 2).sum()), "tau_excluded": int((T.tau_excl_n_optimal >= 2).sum()),
                                       "native_set_size_dist": T.native_n_optimal.value_counts().sort_index().to_dict(),
                                       "tau_excl_set_size_dist": T.tau_excl_n_optimal.value_counts().sort_index().to_dict()},
            "agreement": {}, "accuracy_aux": {"B": f"{int(T.B_correct.sum())}/{n}"},
            "runtime_ms": {k: {"mean": round(float(pd.to_numeric(T[k]).mean()), 2), "median": round(float(pd.to_numeric(T[k]).median()), 2),
                               "max": round(float(pd.to_numeric(T[k]).max()), 2)}
                           for k in ("ms_native_search", "ms_tau_excl_search", "ms_gen_less_memory", "native_pops", "tau_excl_pops", "peak_kb_gen")},
            "definition": "all-optimal agreement = X repaired trace ∈ B optimal repaired trace set (official exact metric unchanged)"}
    for m in X:
        e, nat, dev = int(T[f"{m}_exact_eq_B"].sum()), int(T[f"{m}_in_native_set"].sum()), int(T[f"{m}_in_tau_excl_set"].sum())
        summ["agreement"][f"B-{m}"] = {"exact": f"{e}/{n} = {e / n:.4f}", "native_all_optimal": f"{nat}/{n} = {nat / n:.4f}",
                                       "tau_excluded_all_optimal": f"{dev}/{n} = {dev / n:.4f}",
                                       "counts": {"exact": e, "native": nat, "tau_excluded": dev, "n": n}}
        summ["accuracy_aux"][m] = f"{int(T[f'{m}_correct'].sum())}/{n}"
    (out / "agreement_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
