"""All-optimal alignment analysis for Credit Method B (.venv). Official outputs are only read.

(Formerly outputs/simulation/Credit/alignment_all_optimal_analysis/scripts/credit_all_optimal.py.)
Run as a script: `.venv/Scripts/python scripts/credit_all_optimal_analysis.py [--toy | --limit N]`.
This file runs the Credit analysis at module level, so do not import it; scripts/all_optimal_agreement.py reads
only its definition block (see below) as source text.

Environment shim: cvxopt is hidden inside this process so that the PM4Py LP solver uses scipy, and
lxml.objectify (used only by the PTML importer, not by PNML reading or alignments) is replaced by a stub.
This is needed where the cvxopt / lxml.objectify binaries cannot be loaded; the installation is not changed.

Modes:
  --toy                  toy tie tests only
  --toy-once <name>      (subprocess) one state-equation A* run on one toy -> stdout
  --astar-only <out>     (subprocess) state-equation A* on the Credit test cases only -> json
  --limit N              first N Credit cases only (smoke)

The definitions from the trace_obj function up to the toy section (repair_from_alignment, net_struct, fire,
dev_optimal) are also loaded by scripts/all_optimal_agreement.py. It locates that block by searching for the first
function header of trace_obj and for the toy-section comment, so do not write either text literally in this docstring.
"""
import sys
import types

sys.modules["cvxopt"] = None
import lxml  # noqa: E402

from lxml import etree as _etree  # noqa: E402

_obj = types.ModuleType("lxml.objectify")
_obj.parse, _obj.fromstring = _etree.parse, _etree.fromstring  # the PNML importer passes etree.XMLParser explicitly
sys.modules["lxml.objectify"] = _obj
lxml.objectify = _obj

import heapq  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # repository root (this file is in scripts/)
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
import pm4py  # noqa: E402
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo  # noqa: E402
from pm4py.algo.conformance.alignments.petri_net.variants import generator_dijkstra_less_memory as gen_lm  # noqa: E402
from pm4py.algo.conformance.alignments.petri_net.variants import generator_dijkstra_no_heuristics as gen_nh  # noqa: E402
from pm4py.objects.log.obj import Event, Trace  # noqa: E402
from pm4py.objects.process_tree.utils.generic import parse  # noqa: E402
from pm4py.util.lp import solver as lp_solver  # noqa: E402

D = ROOT / "outputs/simulation/Credit"
OUT = D / "alignment_all_optimal_analysis"
B_DIR = D / "runs/methodB_run1"
VARIANT = alignments_algo.Variants.VERSION_STATE_EQUATION_A_STAR  # official Method B alignment variant
SCRIPT = Path(__file__).resolve()
PY = sys.executable


def trace_obj(acts):
    t = Trace()
    for a in acts:
        t.append(Event({"concept:name": a}))
    return t


def repair_from_alignment(alignment):
    """Same rule as the Method B repair: keep sync moves, delete log moves, insert visible model moves, skip tau."""
    rep, n = [], Counter()
    for log_label, model_label in alignment:
        if log_label != ">>" and model_label != ">>":
            rep.append(model_label)
            n["sync"] += 1
        elif model_label == ">>":
            n["log"] += 1
        elif model_label is None:
            n["tau"] += 1
        else:
            rep.append(model_label)
            n["model_visible"] += 1
    return tuple(rep), n


def astar(acts, net, im, fm):
    r = alignments_algo.apply(trace_obj(acts), net, im, fm, variant=VARIANT)
    return repair_from_alignment(r["alignment"])[0], r["cost"]


def all_opt_pm4py(acts, net, im, fm, variant=gen_nh):
    """PM4Py generator variant: all alignments with the same minimal PM4Py cost (log/model 10000, tau 1, sync 0)."""
    als = list(variant.apply(trace_obj(acts), net, im, fm))
    costs = {a["cost"] for a in als}
    return als, costs


# ---------------------------------------------------------------- deviation-only all-optimal (own search; auxiliary definition)
def net_struct(net, im, fm):
    trans = []
    for t in net.transitions:
        pre = tuple(sorted((a.source.name, a.weight) for a in t.in_arcs))
        post = tuple(sorted((a.target.name, a.weight) for a in t.out_arcs))
        trans.append((t.label, pre, post))
    mk = lambda m: tuple(sorted((p.name, c) for p, c in m.items() if c > 0))  # noqa: E731
    return trans, mk(im), mk(fm)


def fire(m, pre, post):
    d = dict(m)
    for p, w in pre:
        if d.get(p, 0) < w:
            return None
        d[p] -= w
    for p, w in post:
        d[p] = d.get(p, 0) + w
    return tuple(sorted((p, c) for p, c in d.items() if c > 0))


def dev_optimal(acts, ns, move_cost=1, tau_cost=0, max_states=5_000_000):
    """All repaired traces of minimal cost (sync 0, log / visible model move = move_cost, tau = tau_cost).

    move_cost=1, tau_cost=0     : tau-excluded (deviation-only): minimise visible deviations only;
                                  invisible transitions are routing steps of the model and produce no event
    move_cost=10000, tau_cost=1 : same cost as the PM4Py standard (native / exact all-optimal)
    state = (marking, trace index, output). The output is part of the state, so all distinct repaired traces are found.
    """
    trans, m0, mf = ns
    n = len(acts)
    start = (m0, 0, ())
    best = {start: 0}
    pq = [(0, 0, start)]
    goal_cost, outs, pops, k = None, set(), 0, 1
    while pq:
        c, _, s = heapq.heappop(pq)
        if best.get(s, 1 << 60) < c:
            continue
        if goal_cost is not None and c > goal_cost:
            break
        pops += 1
        if pops > max_states:
            return None, None, pops
        m, i, out = s
        if m == mf and i == n:
            goal_cost = c
            outs.add(out)
            continue
        nxt = []
        if i < n:
            nxt.append((c + move_cost, (m, i + 1, out)))  # log move
        for lab, pre, post in trans:
            m2 = fire(m, pre, post)
            if m2 is None:
                continue
            if lab is None:
                nxt.append((c + tau_cost, (m2, i, out)))  # tau
            else:
                if i < n and lab == acts[i]:
                    nxt.append((c, (m2, i + 1, out + (lab,))))  # sync
                nxt.append((c + move_cost, (m2, i, out + (lab,))))  # visible model move
        for c2, s2 in nxt:
            if c2 < best.get(s2, 1 << 60):
                best[s2] = c2
                heapq.heappush(pq, (c2, k, s2))
                k += 1
    return goal_cost, outs, pops


# ---------------------------------------------------------------- toy
TOYS = {
    "xor_choice": ("->( 'a', X( 'b', 'c' ), 'd' )", ["a", "d"]),
    "parallel": ("->( 'a', +( 'b', 'c' ), 'd' )", ["a", "d"]),
    "xor_vs_parallel_mixed": ("->( 'a', X( 'b', +( 'c', 'e' ) ), 'd' )", ["a", "c", "b", "d"]),
}


def toy_net(name):
    tree = parse(TOYS[name][0])
    return pm4py.convert_to_petri_net(tree)


if "--toy-once" in sys.argv:
    name = sys.argv[sys.argv.index("--toy-once") + 1]
    net, im, fm = toy_net(name)
    rep, cost = astar(TOYS[name][1], net, im, fm)
    print(json.dumps({"repaired": " ".join(rep), "cost": cost}))
    sys.exit(0)


def run_toy():
    res = {"solver_in_this_process": lp_solver.DEFAULT_LP_SOLVER_VARIANT, "toys": {}}
    for name, (tree_s, tr) in TOYS.items():
        net, im, fm = toy_net(name)
        inproc = Counter(" ".join(astar(tr, net, im, fm)[0]) for _ in range(20))
        sub = Counter()
        for _ in range(8):
            o = subprocess.run([PY, str(SCRIPT), "--toy-once", name], capture_output=True, text=True)
            line = [x for x in o.stdout.splitlines() if x.startswith("{")][-1]
            sub[json.loads(line)["repaired"]] += 1
        als, costs = all_opt_pm4py(tr, net, im, fm)
        reps = Counter(" ".join(repair_from_alignment(a["alignment"])[0]) for a in als)
        als_lm, costs_lm = all_opt_pm4py(tr, net, im, fm, gen_lm)
        reps_lm = Counter(" ".join(repair_from_alignment(a["alignment"])[0]) for a in als_lm)
        dc, douts, _ = dev_optimal(tuple(tr), net_struct(net, im, fm))
        res["toys"][name] = {
            "process_tree": tree_s, "trace": " ".join(tr),
            "state_equation_a_star_same_process_x20": dict(inproc),
            "state_equation_a_star_separate_processes_x8": dict(sub),
            "generator_dijkstra_no_heuristics": {"n_alignments": len(als), "costs": sorted(costs), "repaired_traces": dict(reps)},
            "generator_dijkstra_less_memory": {"n_alignments": len(als_lm), "costs": sorted(costs_lm), "repaired_traces": dict(reps_lm)},
            "deviation_only_search": {"deviations": dc, "repaired_traces": sorted(" ".join(o) for o in douts)},
        }
        print(name, json.dumps(res["toys"][name], ensure_ascii=False), flush=True)
    return res


# ---------------------------------------------------------------- Credit
def credit_inputs():
    net, im, fm = pm4py.read_pnml(str(next(B_DIR.glob("*_petri_net.pnml"))))
    mc = json.loads(next(B_DIR.glob("*_model_check.json")).read_text(encoding="utf-8"))
    info = (len(net.places), len(net.transitions), sum(t.label is None for t in net.transitions), len(net.arcs))
    assert info == (mc["n_places"], mc["n_transitions"], mc["n_invisible_transitions"], mc["n_arcs"]), (info, mc)
    alg = pd.read_csv(next(B_DIR.glob("*_alignment_per_case.csv")), dtype={"case:concept:name": str})
    alg = alg[alg.type == "test"].set_index("case:concept:name")
    assert len(alg) == 210
    return net, im, fm, alg


if "--astar-only" in sys.argv:
    outp = sys.argv[sys.argv.index("--astar-only") + 1]
    net, im, fm, alg = credit_inputs()
    r = {c: " ".join(astar(tuple(alg.loc[c, "original_trace"].split(">>")), net, im, fm)[0]) for c in alg.index}
    Path(outp).write_text(json.dumps(r), encoding="utf-8")
    sys.exit(0)

OUT.mkdir(exist_ok=True)
if "--toy" in sys.argv:
    toy = run_toy()
    (OUT / "toy_tie_test.json").write_text(json.dumps(toy, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0)

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
net, im, fm, alg = credit_inputs()
ns = net_struct(net, im, fm)
cl = pd.read_csv(D / "pn_gate_analysis/cpn_case_level.csv", dtype={"case:concept:name": str}).set_index("case:concept:name")
ids = sorted(alg.index, key=lambda x: int(x))[: limit or None]
rows, trows = [], []
t_all = time.time()
for idx, c in enumerate(ids, 1):
    an = tuple(alg.loc[c, "original_trace"].split(">>"))
    b_off = tuple(alg.loc[c, "repaired_trace"].split(">>"))
    cost_off = int(alg.loc[c, "alignment_cost"])
    assert tuple(cl.loc[c, "anomalous_trace"].split(">>")) == an and tuple(cl.loc[c, "B_repaired"].split(">>")) == b_off
    g = tuple(cl.loc[c, "ground_truth"].split(">>"))
    cc = tuple(cl.loc[c, "C_repaired"].split(">>"))
    cp = tuple(cl.loc[c, "CPN_repaired"].split(">>"))
    t0 = time.perf_counter()
    rep_a, cost_a = astar(an, net, im, fm)
    ms_astar = (time.perf_counter() - t0) * 1000
    tracemalloc.start()
    t0 = time.perf_counter()
    als, costs = all_opt_pm4py(an, net, im, fm)
    ms_gen = (time.perf_counter() - t0) * 1000
    peak_kb = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    reps = Counter(repair_from_alignment(a["alignment"])[0] for a in als)
    t0 = time.perf_counter()
    dcost, douts, dpops = dev_optimal(an, ns)
    ms_dev = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    xcost, xouts, xpops = dev_optimal(an, ns, 10000, 1)
    ms_exact = (time.perf_counter() - t0) * 1000
    S_gen, S_dev, S_ex = set(reps), set(douts), set(xouts)
    rows.append({
        "case_id": c, "anomaly_trace": " ".join(an), "ground_truth": " ".join(g), "B_official": " ".join(b_off),
        "C_repaired": " ".join(cc), "CPN_repaired": " ".join(cp),
        "B_official_cost": cost_off, "B_official_deviations": cost_off // 10000, "B_official_tau": cost_off % 10000,
        "astar_rerun_repaired": " ".join(rep_a), "astar_rerun_cost": int(cost_a), "astar_rerun_eq_official": rep_a == b_off,
        # native all-optimal (PM4Py cost, own search) = main definition
        "exact_cost": xcost, "exact_cost_eq_official": xcost == cost_off, "exact_n_unique_repaired": len(S_ex),
        "B_official_in_exact_set": b_off in S_ex, "GT_in_exact_set": g in S_ex, "C_in_exact_set": cc in S_ex, "CPN_in_exact_set": cp in S_ex,
        # tau-excluded / deviation-only (tau cost 0) = auxiliary definition
        "dev_deviations": dcost, "dev_n_unique_repaired": len(S_dev), "exact_subset_of_dev": S_ex <= S_dev,
        "B_official_in_dev_set": b_off in S_dev, "GT_in_dev_set": g in S_dev, "C_in_dev_set": cc in S_dev, "CPN_in_dev_set": cp in S_dev,
        # PM4Py generator_dijkstra_no_heuristics (diagnostic: completeness)
        "gen_cost": sorted(costs)[0] if costs else None, "gen_n_alignments": len(als), "gen_n_unique_repaired": len(S_gen),
        "gen_duplicate_alignments": len(als) - len(S_gen), "gen_subset_of_exact": S_gen <= S_ex, "gen_complete": S_gen == S_ex,
        "B_official_in_gen_set": b_off in S_gen,
        "B_correct": b_off == g, "C_correct": cc == g, "CPN_correct": cp == g, "B_eq_C": b_off == cc, "B_eq_CPN": b_off == cp,
        "ms_astar": round(ms_astar, 2), "ms_gen": round(ms_gen, 2), "peak_kb_gen": round(peak_kb, 1),
        "ms_exact_search": round(ms_exact, 2), "exact_search_pops": xpops, "ms_dev_search": round(ms_dev, 2), "dev_search_pops": dpops,
    })
    for setname, S in (("exact_pm4py_cost", S_ex), ("deviation_only", S_dev), ("pm4py_generator", S_gen)):
        for t in sorted(S):
            trows.append({"case_id": c, "set": setname, "trace": " ".join(t), "gen_n_alignments": reps.get(t, 0) if setname == "pm4py_generator" else None,
                          "is_B_official": t == b_off, "is_GT": t == g, "is_C": t == cc, "is_CPN": t == cp})
    if idx % 20 == 0 or idx == len(ids):
        print(f"{idx}/{len(ids)} elapsed {time.time() - t_all:.1f}s max_exact_ms {max(r['ms_exact_search'] for r in rows):.1f} "
              f"max_exact_set {max(r['exact_n_unique_repaired'] for r in rows)}", flush=True)

T = pd.DataFrame(rows)
TR = pd.DataFrame(trows)
suffix = f"_smoke{limit}" if limit else ""
T.to_csv(OUT / f"all_optimal_case_summary{suffix}.csv", index=False)
TR.to_csv(OUT / f"all_optimal_repaired_traces{suffix}.csv", index=False)
rt = {"cases": len(T), "total_sec": round(time.time() - t_all, 2), "solver_in_this_process": lp_solver.DEFAULT_LP_SOLVER_VARIANT,
      **{f"{k}_{s}": round(float(getattr(T[k], s)()), 2) for k in ("ms_astar", "ms_exact_search", "exact_search_pops", "ms_dev_search",
                                                                   "ms_gen", "peak_kb_gen", "gen_n_alignments", "exact_n_unique_repaired",
                                                                   "dev_n_unique_repaired") for s in ("mean", "median", "max")}}
(OUT / f"runtime_summary{suffix}.json").write_text(json.dumps(rt, indent=2), encoding="utf-8")
print(json.dumps(rt, indent=1))
print("astar rerun == official:", int(T.astar_rerun_eq_official.sum()), "/", len(T),
      "| exact cost == official:", int(T.exact_cost_eq_official.sum()), "| official in exact set:", int(T.B_official_in_exact_set.sum()),
      "| exact ⊆ dev:", int(T.exact_subset_of_dev.sum()), "| gen ⊆ exact:", int(T.gen_subset_of_exact.sum()),
      "| gen complete:", int(T.gen_complete.sum()), "| official in gen:", int(T.B_official_in_gen_set.sum()))
