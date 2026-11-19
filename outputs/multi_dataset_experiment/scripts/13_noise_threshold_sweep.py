"""Road / BPIC 2017: model structure, bounded language and clean coverage per IM noise_threshold (.venv).

Why: for large models, a higher discovery noise_threshold simplifies the Petri net so that the methods can run
unchanged. The threshold is the Inductive Miner noise_threshold (PM4Py 2.7.23.6: IMf if > 0); it is not the
ADM alpha. Method C itself is not changed.

Per threshold:
  1. discover_model on the clean cases (same function as the pipeline) -> store process tree / Petri net, count structures
  2. Pareto front of the required (loop_num, loop_total) per clean variant (requirement of script 12, no enumeration)
  3. per coverage target (90 / 95 / 99 %): the Pareto-minimal bound with the smallest language and its size (lang_count)
  4. selection rule (same for every threshold):
       chosen bound = the bound with the smallest language among those with clean-case coverage >= 95 %
                      (if 95 % is unreachable: smallest language among the bounds with the maximal reachable coverage)
       only if language <= GUARD (2,000,000, the generation guard) the traces are generated with generate_bounded
       to confirm count and coverage; otherwise nothing is generated and INFEASIBLE (estimated size) is recorded
  A tree without loops has a finite language regardless of the bound -> chosen bound = (0,0), generated as is.

Output: outputs/simulation/<Dataset>/threshold_0pXX/sweep/ (tree, pnml, coverage grid, json)
        outputs/multi_dataset_experiment/threshold_sweep/sweep_structure_language.csv / .json
Usage:  PYTHONHASHSEED=0 .venv/Scripts/python outputs/multi_dataset_experiment/scripts/13_noise_threshold_sweep.py [Road|BPIC ...]
"""
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lra", HERE / "12_loop_requirement_analysis.py")  # includes the LP-solver shim
lra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lra)

import pandas as pd  # noqa: E402
import pm4py  # noqa: E402
from pm4py.objects.process_tree.obj import Operator  # noqa: E402

PROJECT_ROOT = HERE.parents[2]


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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.data_utils import load_case_traces  # noqa: E402
from utils.discovery_utils import discover_model, operator_kind_counts, petri_net_has_cycle  # noqa: E402
from utils.loop_bound_utils import ExplosionAborted, generate_bounded  # noqa: E402

SIM = PROJECT_ROOT / "outputs/simulation"
OUT = PROJECT_ROOT / "outputs/multi_dataset_experiment/threshold_sweep"
FULL = _data_root() / "encoded_anomaly"
INPUTS = {
    "RoadTraffic": SIM / "RoadTraffic/exploratory_dedup/derived_data/Road_Traffic_1.00_dedup.csv",
    "BPIC2017": FULL / "b17_1.00.csv",
}
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]  # 0.15 / 0.40: Road 0.10 is non-monotone and 0.20 == 0.30 -> auxiliary candidates
GUARD = 2_000_000
TARGET = 0.95
NMAX, TMAX = 12, 60


def tag(th):
    return f"threshold_{th:.2f}".replace(".", "p")


def nested_loops(t, inside=False):
    n = 1 if (t.operator == Operator.LOOP and inside) else 0
    inside = inside or t.operator == Operator.LOOP
    return n + sum(nested_loops(c, inside) for c in t.children)


def run(ds, th):
    d = SIM / ds / tag(th) / "sweep"
    d.mkdir(parents=True, exist_ok=True)
    traces, _lab, clean_ids, anom_ids = load_case_traces(INPUTS[ds])
    tree, net, im, fm, sec = discover_model(traces, clean_ids, th)
    (d / "process_tree.txt").write_text(str(tree) + "\n", encoding="utf-8")
    pm4py.write_pnml(net, im, fm, str(d / "petri_net.pnml"))
    k = operator_kind_counts(tree)
    vis = [t for t in net.transitions if t.label is not None]
    row = {"Dataset": ds, "NoiseThreshold": th, "im_variant": "IMf" if th > 0 else "IM",
           "DiscoveryMs": round(sec * 1000, 2),
           "Places": len(net.places), "Transitions": len(net.transitions), "VisibleTransitions": len(vis),
           "TauTransitions": len(net.transitions) - len(vis), "Arcs": len(net.arcs),
           "LoopCount": k.get("LOOP", 0), "NestedLoopCount": nested_loops(tree), "ParallelCount": k.get("PARALLEL", 0),
           "XORCount": k.get("XOR", 0), "SequenceCount": k.get("SEQUENCE", 0), "TauLeaves": k.get("tau", 0),
           "petri_net_has_cycle": petri_net_has_cycle(net), "labels_unique": lra.labels_unique(tree),
           "visible_activities": len(set(t.label for t in vis))}
    obs = Counter(tuple(traces[c]) for c in clean_ids)
    tot = sum(obs.values())
    clean_acts = Counter(a for v, c in obs.items() for a in v for _ in range(c))
    row["dropped_activities"] = ";".join(f"{a}({clean_acts[a]} events)" for a in sorted(set(clean_acts) - lra.alphabet(tree))) or None
    row.update({"CleanCases": tot, "CleanVariants": len(obs), "AnomalousCases": len(anom_ids)})
    if not row["labels_unique"]:
        row["Status"] = "ANALYSIS_NOT_APPLICABLE"
        row["Notes"] = "duplicate labels: enumeration-free method not validated for this tree"
        return row
    R = lra.Requirement(tree)
    t0 = time.time()
    fronts = {v: R.trace_front(v) for v in obs}
    row["requirement_sec"] = round(time.time() - t0, 2)
    row["not_coverable_cases"] = sum(c for v, c in obs.items() if not fronts[v])
    row["max_coverage (unbounded)"] = round(lra.coverage(obs, fronts, lra.INF, lra.INF), 6)
    has_loop = row["LoopCount"] > 0
    if has_loop:
        grid = pd.DataFrame([{"loop_num": n, "loop_total": t, "coverage": lra.coverage(obs, fronts, n, t)}
                             for n in range(1, NMAX + 1) for t in range(n, TMAX + 1)])
    else:
        grid = pd.DataFrame([{"loop_num": 0, "loop_total": 0, "coverage": lra.coverage(obs, fronts, 0, 0)}])
    grid.to_csv(d / "coverage_grid.csv", index=False)
    row["coverage_at_(1,1)"] = round(lra.coverage(obs, fronts, 1, 1), 6)
    row["language_at_(1,1)"] = lra.lang_count(tree, 1, 1)

    def best_for(p):
        ok = grid[grid.coverage >= p - 1e-12]
        if ok.empty:
            return None
        cand = sorted({(int(r.loop_num), int(r.loop_total)) for r in ok.itertuples()})
        cand = [(n, t) for n, t in cand if not any(n2 <= n and t2 <= t and (n2, t2) != (n, t) for n2, t2 in cand)]
        rows = [{"loop_num": n, "loop_total": t, "coverage": round(lra.coverage(obs, fronts, n, t), 6),
                 "language_size": lra.lang_count(tree, n, t)} for n, t in cand]
        return min(rows, key=lambda r: r["language_size"])

    targets = {}
    for p in (0.90, 0.95, 0.99):
        b = best_for(p)
        targets[str(p)] = b
        row[f"bound_for_{int(p*100)}"] = None if b is None else f"({b['loop_num']},{b['loop_total']})"
        row[f"language_for_{int(p*100)}"] = None if b is None else b["language_size"]
    # maximal coverage within the guard (2M): for each n the largest t with language <= GUARD (language grows monotonically in t)
    best_g = None
    for n in (range(1, NMAX + 1) if has_loop else [0]):
        lo, hi = (n, TMAX) if has_loop else (0, 0)
        if lra.lang_count(tree, n, lo) > GUARD:
            continue
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if lra.lang_count(tree, n, mid) <= GUARD else (lo, mid - 1)
        cov = lra.coverage(obs, fronts, n, lo)
        size = lra.lang_count(tree, n, lo)
        if best_g is None or (cov, -size) > (best_g[2], -best_g[3]):
            best_g = (n, lo, cov, size)
    row["max_coverage_within_guard"] = None if best_g is None else round(best_g[2], 6)
    row["bound_for_max_coverage_within_guard"] = None if best_g is None else f"({best_g[0]},{best_g[1]})"
    row["language_for_max_coverage_within_guard"] = None if best_g is None else best_g[3]
    chosen = targets[str(TARGET)] or best_for(float(grid.coverage.max()))
    row["target_coverage"] = TARGET
    row["chosen_rule"] = ("min language among Pareto-minimal bounds with coverage >= 95%" if targets[str(TARGET)]
                          else "95% unreachable -> min language among bounds with max reachable coverage")
    row["LoopBound"] = f"({chosen['loop_num']},{chosen['loop_total']})"
    row["EstimatedLanguageSize"] = chosen["language_size"]
    row["AnalyticCoverage"] = chosen["coverage"]
    if chosen["language_size"] > GUARD:
        row.update({"Status": "INFEASIBLE", "GeneratedTraces": None, "ModelOnlyTraces": None,
                    "Notes": f"estimated bounded language {chosen['language_size']:.3g} > generation guard {GUARD:,}; not generated"})
    else:
        n, t = (chosen["loop_num"], chosen["loop_total"]) if has_loop else (1, 1)
        try:
            lang, gst = generate_bounded(tree, n, t, 100, GUARD, 900, 16)
        except ExplosionAborted as e:
            row.update({"Status": "INFEASIBLE", "GeneratedTraces": None, "Notes": f"generation aborted: {e}"})
            lang = None
        if lang is not None:
            gen = set(lang)
            cov_cases = sum(c for v, c in obs.items() if v in gen)
            cov_var = sum(1 for v in obs if v in gen)
            row.update({"Status": "GENERATED", "GeneratedTraces": len(gen), "ModelOnlyTraces": len(gen - set(obs)),
                        "GenerationMs": round(gst["generation_time_sec"] * 1000, 2),
                        "CleanCaseCoverageCount": cov_cases, "CleanVariantCoverageCount": cov_var,
                        "generated_equals_analytic": len(gen) == round(chosen["language_size"]),
                        "coverage_equals_analytic": abs(cov_cases / tot - chosen["coverage"]) < 1e-6})
    (d / "sweep_result.json").write_text(json.dumps({"row": row, "targets": targets}, ensure_ascii=False, indent=1,
                                                    default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in row.items()}, default=str), flush=True)
    return row


def main():
    import os
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("PYTHONHASHSEED=0 으로 실행하세요 (PM4Py IMf는 hash 순서에 따라 process마다 다른 tree를 낼 수 있음)")
    OUT.mkdir(parents=True, exist_ok=True)
    which = [x for x in sys.argv[1:]] or ["Road", "BPIC"]
    rows = []
    f = OUT / "sweep_structure_language.json"
    if f.exists():
        rows = [r for r in json.loads(f.read_text(encoding="utf-8"))
                if not any(r["Dataset"].startswith(w[:4]) for w in which)]
    for w in which:
        ds = "RoadTraffic" if w.startswith("Road") else "BPIC2017"
        for th in THRESHOLDS:
            print(f"\n== {ds} noise_threshold={th}", flush=True)
            rows.append(run(ds, th))
            f.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "sweep_structure_language.csv", index=False)


if __name__ == "__main__":
    _strip_data_root(sys.argv)
    main()
