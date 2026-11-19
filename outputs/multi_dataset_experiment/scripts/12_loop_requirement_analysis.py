"""Bounded clean coverage and bounded-language size without enumerating traces (.venv).

Why: for BPIC 2017 the bound (1,4) already yields 1,360,844 traces but only 1.37 % clean coverage, and larger
bounds exceed the generation guard. Instead of generating millions of traces, this script computes from the
process-tree structure (1) the minimal loop bound each observed clean trace requires and (2) the number of
traces in the bounded language for a given bound.

Loop semantics (same as loop_bound_utils):
    loop_num   : upper bound on do-part executions per loop instance (instance-local; nested loops restart
                 for every outer iteration)
    loop_total : upper bound on the total do-part executions of all loop instances in one trace
                 (the first execution counts as 1)
    max trace length 100 (same safety guard as the generator)

Method 1, requirement (coverage): if every visible label occurs once in the tree (no duplicate labels), each event
maps to exactly one leaf, so each subtree only needs the projection of the trace onto its alphabet.
    SEQ: children's events must follow in order / XOR: exactly one child / PAR: projection per child, free interleaving
    LOOP: all ways to split into do (redo do)^(k-1) are explored by dynamic programming (k = do executions)
  Result: per-trace Pareto front of (required loop_num, required loop_total). A trace is covered by (n, t) iff the
  front has a point with n' <= n and t' <= t.
Method 2, language size: per node, (length, loop_total usage) -> number of traces.
    SEQ: convolution / XOR: sum / PAR: C(l1+l2, l1) interleavings of two traces of lengths l1, l2
    (disjoint alphabets, so all interleavings are distinct) / LOOP: k = 1..n do executions, +1 loop_total each.
  The same computation with PAR replaced by SEQ separates the contribution of parallel interleaving.
Validation: both methods are compared with existing enumeration results (Pub, Credit, Road raw/dedup, BPIC (1,4))
and are used for BPIC only if they match.

Output: outputs/multi_dataset_experiment/loop_requirement_analysis/
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

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pm4py.objects.process_tree.obj import Operator  # noqa: E402
from pm4py.objects.process_tree.utils.generic import parse  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
FULL = _data_root() / "encoded_anomaly"
SIM = PROJECT_ROOT / "outputs/simulation"
OUT = PROJECT_ROOT / "outputs/multi_dataset_experiment/loop_requirement_analysis"
MAX_LEN = 100
INF = 10 ** 9

DATASETS = {  # tree = discovered tree stored by the earlier diagnostics (IM noise 0, clean cases)
    "Credit": (SIM / "Credit/dataset_audit/process_tree.txt", FULL / "credit-card_1.00.csv"),
    "Pub": (SIM / "Pub/dataset_audit/process_tree.txt", FULL / "mccloud_1.00.csv"),
    "RoadTraffic_raw": (SIM / "RoadTraffic/exploratory_dedup/A_raw_process_tree.txt", FULL / "Road_Traffic_1.00.csv"),
    "RoadTraffic_dedup": (SIM / "RoadTraffic/exploratory_dedup/B_dedup_process_tree.txt",
                          SIM / "RoadTraffic/exploratory_dedup/derived_data/Road_Traffic_1.00_dedup.csv"),
    "BPIC2017": (SIM / "BPIC2017/dataset_audit/process_tree.txt", FULL / "b17_1.00.csv"),
}
# existing enumeration results (values from the source files). (n, t) -> (generated_unique, covered clean cases)
REFERENCE = {
    "Pub": ("Pub/dataset_audit/generation_feasibility.csv", None),
    "Credit": ("Credit/dataset_audit/generation_feasibility.csv", None),
    "RoadTraffic_raw": ("RoadTraffic/exploratory_dedup/A_raw_bounded_growth.csv", None),
    "RoadTraffic_dedup": ("RoadTraffic/exploratory_dedup/B_dedup_bounded_growth.csv", None),
    "BPIC2017": ("BPIC2017/dataset_audit/generation_feasibility.csv", None),
}


# ------------------------------------------------------------------ tree helpers
def read_tree(path):
    return parse(path.read_text(encoding="utf-8").strip())


def nodes(t):
    yield t
    for c in t.children:
        yield from nodes(c)


def alphabet(t):
    return frozenset(n.label for n in nodes(t) if not n.children and n.label is not None)


def labels_unique(t):
    labs = [n.label for n in nodes(t) if not n.children and n.label is not None]
    return len(labs) == len(set(labs))


def loop_name(t):
    a = sorted(alphabet(t.children[0]))
    return "*(" + ",".join(a[:8]) + (",…" if len(a) > 8 else "") + ")"


def depth_of_loops(t, d=0, out=None):
    out = {} if out is None else out
    if t.operator == Operator.LOOP:
        out[id(t)] = d
        d += 1
    for c in t.children:
        depth_of_loops(c, d, out)
    return out


# ------------------------------------------------------------------ method 1: requirement
def pareto2(points):
    """points: (a, b, w). 2-D Pareto front of (a, b); for equal (a, b) the first witness is kept."""
    best = {}
    for a, b, w in points:
        if (a, b) not in best:
            best[(a, b)] = w
    out, minb = [], INF + 1
    for (a, b), w in sorted(best.items()):
        if b < minb:
            out.append((a, b, w))
            minb = b
    return tuple(out)


def pareto3(points):
    """points: (k, imax, itot, w). Remove a point if another point is <= in all three values."""
    best = {}
    for k, a, b, w in points:
        best.setdefault((k, a, b), w)
    keys = sorted(best)
    keep = []
    for x in keys:
        if not any(y[0] <= x[0] and y[1] <= x[1] and y[2] <= x[2] for y in keep):
            keep.append(x)
    return [(k, a, b, best[(k, a, b)]) for k, a, b in keep]


class Requirement:
    def __init__(self, tree):
        self.tree = tree
        self.alpha = {id(n): alphabet(n) for n in nodes(tree)}
        self.loops = [n for n in nodes(tree) if n.operator == Operator.LOOP]
        self.loop_idx = {id(n): i for i, n in enumerate(self.loops)}
        self.zero = tuple((0, 0) for _ in self.loops)
        self.memo = {}

    def wadd(self, w1, w2):
        return tuple((max(a[0], b[0]), a[1] + b[1]) for a, b in zip(w1, w2))

    def comb(self, P1, P2):
        return pareto2((max(a1, a2), b1 + b2, self.wadd(w1, w2)) for a1, b1, w1 in P1 for a2, b2, w2 in P2)

    def req(self, node, seq):
        key = (id(node), seq)
        if key in self.memo:
            return self.memo[key]
        r = self._req(node, seq)
        self.memo[key] = r
        return r

    def _req(self, node, seq):
        op = node.operator
        if op is None:
            if node.label is None:
                return ((0, 0, self.zero),) if not seq else ()
            return ((0, 0, self.zero),) if seq == (node.label,) else ()
        kids = node.children
        if op == Operator.XOR:
            if not seq:
                return pareto2(p for c in kids for p in self.req(c, ()))
            for c in kids:
                if seq[0] in self.alpha[id(c)]:
                    return self.req(c, seq) if all(x in self.alpha[id(c)] for x in seq) else ()
            return ()
        if op == Operator.SEQUENCE:
            idx = []
            for x in seq:
                i = next(j for j, c in enumerate(kids) if x in self.alpha[id(c)])
                if idx and i < idx[-1]:
                    return ()
                idx.append(i)
            P = ((0, 0, self.zero),)
            for j, c in enumerate(kids):
                P = self.comb(P, self.req(c, tuple(x for x, i in zip(seq, idx) if i == j)))
                if not P:
                    return ()
            return P
        if op == Operator.PARALLEL:
            P = ((0, 0, self.zero),)
            for c in kids:
                P = self.comb(P, self.req(c, tuple(x for x in seq if x in self.alpha[id(c)])))
                if not P:
                    return ()
            return P
        if op == Operator.LOOP:
            return self._loop(node, seq)
        raise ValueError(op)

    def _loop(self, node, seq):
        do, redo = node.children
        ad, ar = self.alpha[id(do)], self.alpha[id(redo)]
        m = len(seq)
        li = self.loop_idx[id(node)]

        def blocks(i, A, child):
            """The end indices j for which seq[i:j] consists only of labels in A, with their requirements."""
            out = []
            j = i
            while True:
                P = self.req(child, seq[i:j])
                if P:
                    out.append((j, P))
                if j >= m or seq[j] not in A:
                    break
                j += 1
            return out

        S = [[] for _ in range(m + 1)]
        for j, P in blocks(0, ad, do):
            S[j].extend((1, a, b, w) for a, b, w in P)
        for i in range(m + 1):
            if not S[i]:
                continue
            S[i] = pareto3(S[i])
            if i == m:
                break
            for j1, PR in blocks(i, ar, redo):
                for j2, PD in blocks(j1, ad, do):
                    if j2 <= i:
                        continue  # each iteration consumes at least one event (prevents infinite empty iterations)
                    for k, a, b, w in S[i]:
                        for ar_, br, wr in PR:
                            for ad_, bd, wd in PD:
                                S[j2].append((k + 1, max(a, ar_, ad_), b + br + bd, self.wadd(self.wadd(w, wr), wd)))
        res = []
        for k, a, b, w in S[m]:
            w = list(w)
            w[li] = (max(w[li][0], k), w[li][1] + k)
            res.append((max(k, a), k + b, tuple(w)))
        return pareto2(res)

    def trace_front(self, trace):
        if not set(trace) <= self.alpha[id(self.tree)]:
            return ()
        self.memo.clear() if len(self.memo) > 2_000_000 else None
        return self.req(self.tree, tuple(trace))


def covered(front, n, t):
    return any(a <= n and b <= t for a, b, _ in front)


# ------------------------------------------------------------------ method 2: language size
def lang_count(node, n, t, par_as_seq=False):
    """(length 0..MAX_LEN, loop_total usage 0..t) -> number of traces (float64; interleavings via EGF)."""
    L = MAX_LEN

    def norm(A):
        # there is only one empty trace: merge it into the smallest loop_total usage (same as the generator's minimal-count dedup)
        if A[0].any():
            j = int(np.argmax(A[0] > 0))
            A[0] = 0
            A[0, j] = 1
        return A

    def conv(A, B):
        C = np.zeros((L + 1, t + 1))
        nzA = np.argwhere(A > 0)
        for la, ta in nzA:
            v = A[la, ta]
            C[la:, ta:] += v * B[: L + 1 - la, : t + 1 - ta]
        return C

    fact = np.array([math.factorial(i) for i in range(L + 1)], dtype=float)

    def rec(nd):
        op = nd.operator
        A = np.zeros((L + 1, t + 1))
        if op is None:
            A[0 if nd.label is None else 1, 0] = 1
            return A
        ks = [rec(c) for c in nd.children]
        if op == Operator.XOR:
            return norm(sum(ks))
        if op == Operator.SEQUENCE or (op == Operator.PARALLEL and par_as_seq):
            A = ks[0]
            for B in ks[1:]:
                A = norm(conv(A, B))
            return A
        if op == Operator.PARALLEL:
            A = ks[0] / fact[:, None]
            for B in ks[1:]:
                A = conv(A, B / fact[:, None])
            return norm(A * fact[:, None])
        if op == Operator.LOOP:
            D, R = ks
            out = np.zeros((L + 1, t + 1))
            cur = D.copy()
            for k in range(1, n + 1):
                if k > t:
                    break
                out[:, k:] += cur[:, : t + 1 - k]   # k do executions of this loop instance -> usage +k
                if k == n:
                    break
                lim = t - (k + 1)  # with k+1 do executions the inner usage must be <= t-(k+1)
                if lim < 0:
                    break
                cur = conv(conv(cur, R), D)
                cur[:, lim + 1:] = 0
            return norm(out)
        raise ValueError(op)

    A = rec(node)
    return float(A.sum())


# ------------------------------------------------------------------ data
def clean_traces(csv):
    df = pd.read_csv(csv, usecols=["Case", "Activity", "type_res_trace"], dtype={"Case": str, "Activity": str})
    tr = df.groupby("Case", sort=False)["Activity"].agg(tuple)
    lab = df.groupby("Case", sort=False)["type_res_trace"].first()
    return Counter(tr[c] for c in lab.index if pd.isna(lab[c]))


def analyze(name):
    tree_path, csv = DATASETS[name]
    tree = read_tree(tree_path)
    obs = clean_traces(csv)
    t0 = time.time()
    R = Requirement(tree)
    fronts = {v: R.trace_front(v) for v in obs}
    sec = time.time() - t0
    return tree, R, obs, fronts, sec


def coverage(obs, fronts, n, t):
    tot = sum(obs.values())
    return sum(c for v, c in obs.items() if covered(fronts[v], n, t)) / tot


def validate(name, tree, obs, fronts):
    ref = pd.read_csv(SIM / REFERENCE[name][0])
    rows = []
    tot = sum(obs.values())
    for _, r in ref.iterrows():
        if r["status"] != "OK":
            continue
        n, t = int(r.loop_num), int(r.loop_total)
        cov_ref = float(r["clean_case_coverage"])
        cov = coverage(obs, fronts, n, t)
        cnt = lang_count(tree, n, t)
        rows.append({"dataset": name, "loop_num": n, "loop_total": t, "enum_generated": int(r.generated_unique),
                     "analytic_generated": round(cnt), "generated_match": round(cnt) == int(r.generated_unique),
                     "enum_coverage": cov_ref, "analytic_coverage": round(cov, 6),
                     "coverage_match": abs(round(cov, 6) - cov_ref) < 1e-6, "clean_cases": tot})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or list(DATASETS)
    val_rows, summary = [], {}
    for name in which:
        print(f"\n== {name}", flush=True)
        tree, R, obs, fronts, sec = analyze(name)
        uniq = labels_unique(tree)
        print(f"   labels unique {uniq}, variants {len(obs)}, cases {sum(obs.values())}, requirement {sec:.1f}s", flush=True)
        rows = validate(name, tree, obs, fronts)
        for r in rows:
            print("   ", {k: r[k] for k in ("loop_num", "loop_total", "enum_generated", "analytic_generated", "enum_coverage", "analytic_coverage")}, flush=True)
        val_rows += rows
        summary[name] = {"labels_unique": uniq, "variants": len(obs), "cases": sum(obs.values()), "requirement_sec": round(sec, 2),
                         "validation_points": len(rows), "generated_match": sum(r["generated_match"] for r in rows),
                         "coverage_match": sum(r["coverage_match"] for r in rows),
                         "not_coverable_variants": sum(1 for v in obs if not fronts[v]),
                         "not_coverable_cases": sum(c for v, c in obs.items() if not fronts[v])}
    pd.DataFrame(val_rows).to_csv(OUT / "validation_vs_enumeration.csv", index=False)
    (OUT / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------ BPIC 2017 analysis (no enumeration)
def bpic(nmax=12, tmax=60):
    name = "BPIC2017"
    tree, R, obs, fronts, sec = analyze(name)
    tot = sum(obs.values())
    depth = depth_of_loops(tree)
    lnames = [f"L{i}{loop_name(lp)}@depth{depth[id(lp)]}" for i, lp in enumerate(R.loops)]
    out = {"dataset": name, "clean_cases": tot, "clean_variants": len(obs), "requirement_sec": round(sec, 2),
           "loops": lnames}
    lens = Counter()
    for v, c in obs.items():
        lens[len(v)] += c
    arr = np.repeat(list(lens), list(lens.values()))
    out["trace_length"] = {"min": int(arr.min()), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
                           "max": int(arr.max()), "cases_longer_than_100": int((arr > 100).sum())}

    # 1) per-case requirement
    rows = []
    for v, c in obs.items():
        F = fronts[v]
        if not F:
            rows.append({"variant": " ".join(v), "cases": c, "length": len(v), "coverable": False})
            continue
        a1 = [b for a, b, _ in F if a == 1]
        wit = min(F, key=lambda x: (x[1], x[0]))[2]  # per loop of the parse with the smallest loop_total: (instance max k, sum of k)
        r = {"variant": " ".join(v), "cases": c, "length": len(v), "coverable": True,
             "front": ";".join(f"({a},{b})" for a, b, _ in F), "min_loop_num": min(a for a, _, _ in F),
             "min_loop_total": min(b for _, b, _ in F), "min_total_if_no_repetition": a1[0] if a1 else None,
             "loops_entered": sum(1 for mk, _ in wit if mk > 0), "loops_repeated": sum(1 for mk, _ in wit if mk > 1)}
        for nm, (mk, _sk) in zip(lnames, wit):
            r[f"max_k {nm}"] = mk
        rows.append(r)
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "bpic_case_requirements.csv", index=False)
    G = pd.DataFrame([{"loop_num": n, "loop_total": t, "coverage": coverage(obs, fronts, n, t)}
                      for n in range(1, nmax + 1) for t in range(n, tmax + 1)])
    G.to_csv(OUT / "bpic_coverage_grid.csv", index=False)

    w = C[C.coverable].copy()

    def dist(col):
        return {str(k): int(v) for k, v in w.groupby(col)["cases"].sum().sort_index().items()}

    out["not_coverable_cases"] = int(C.loc[~C.coverable, "cases"].sum())
    out["min_loop_num_distribution (cases)"] = dist("min_loop_num")
    out["min_loop_total_distribution (cases)"] = dist("min_loop_total")
    out["loops_entered_distribution (cases, min-total parse)"] = dist("loops_entered")
    out["loops_repeated_distribution (cases, min-total parse)"] = dist("loops_repeated")
    out["coverage (1, unlimited total)"] = round(coverage(obs, fronts, 1, INF), 6)
    out["coverage (unlimited, unlimited)"] = round(coverage(obs, fronts, INF, INF), 6)

    # 2) why cases are not covered at (1,4)
    unc = w[[not covered(fronts[tuple(x.split())], 1, 4) for x in w.variant]]
    reason = Counter()
    for _, r in unc.iterrows():
        key = ("no repetition needed, but more than 4 do-executions in total (= more than 4 loops entered)"
               if r.min_loop_num == 1 else "needs repetition (loop_num >= 2)")
        reason[key] += int(r.cases)
    out["uncovered_at_(1,4)"] = {
        "cases": int(unc.cases.sum()), "reason": dict(reason),
        "per_loop (min-total parse)": {nm: {"cases_entering": int(unc.loc[unc[f"max_k {nm}"] >= 1, "cases"].sum()),
                                            "cases_repeating (k>=2)": int(unc.loc[unc[f"max_k {nm}"] >= 2, "cases"].sum()),
                                            "max_k": int(unc[f"max_k {nm}"].max())} for nm in lnames}}
    per = {}
    for nm in lnames:
        ent = w[w[f"max_k {nm}"] >= 1]
        ks = np.repeat(ent[f"max_k {nm}"].to_numpy(), ent["cases"].to_numpy())
        per[nm] = {"cases_entering": int(ent.cases.sum()), "cases_repeating (k>=2)": int(w.loc[w[f"max_k {nm}"] >= 2, "cases"].sum()),
                   "k_p50_p95_max_among_entering": [float(np.percentile(ks, 50)), float(np.percentile(ks, 95)), int(ks.max())] if len(ks) else None}
    out["all_clean_per_loop (min-total parse)"] = per

    # 3) Pareto-minimal bounds per coverage target and their language sizes
    targets, best_rows = [], []
    for p in (0.5, 0.8, 0.9, 0.95, 0.99):
        ok = G[G.coverage >= p]
        if ok.empty:
            targets.append({"target": p, "status": f"not reached within n<={nmax}, t<={tmax}"})
            continue
        cand = sorted({(int(r.loop_num), int(r.loop_total)) for r in ok.itertuples()})
        cand = [(n, t) for n, t in cand if not any(n2 <= n and t2 <= t and (n2, t2) != (n, t) for n2, t2 in cand)]
        best = None
        for n, t in cand:
            t0 = time.time()
            size = lang_count(tree, n, t)
            row = {"target": p, "loop_num": n, "loop_total": t, "coverage": round(coverage(obs, fronts, n, t), 6),
                   "language_size": size, "language_size_PAR_as_SEQ": lang_count(tree, n, t, par_as_seq=True),
                   "sec": round(time.time() - t0, 2), "pareto_candidates": len(cand)}
            row["parallel_factor"] = row["language_size"] / row["language_size_PAR_as_SEQ"]
            targets.append(row)
            print("   target", p, row, flush=True)
            if best is None or size < best["language_size"]:
                best = row
        best_rows.append(best)
    pd.DataFrame(targets).to_csv(OUT / "bpic_bound_targets_all_pareto.csv", index=False)
    pd.DataFrame(best_rows).to_csv(OUT / "bpic_bound_targets_best.csv", index=False)
    out["targets_best (smallest language among Pareto-minimal bounds)"] = best_rows

    # 4) explosion decomposition at the same bound: PAR->SEQ / no loop repetition (n=1, same t)
    dec = []
    for n, t in [(1, 4), (1, 10)] + [(r["loop_num"], r["loop_total"]) for r in best_rows]:
        full = lang_count(tree, n, t)
        seq = lang_count(tree, n, t, par_as_seq=True)
        no_rep = lang_count(tree, 1, t)
        no_rep_seq = lang_count(tree, 1, t, par_as_seq=True)
        dec.append({"loop_num": n, "loop_total": t, "coverage": round(coverage(obs, fronts, n, t), 6), "language_size": full,
                    "PAR_as_SEQ": seq, "no_repetition (n=1, same t)": no_rep, "no_repetition_and_PAR_as_SEQ": no_rep_seq,
                    "log10_total": round(math.log10(full), 2),
                    "log10_parallel_factor (full / PAR_as_SEQ)": round(math.log10(full / seq), 2),
                    "log10_repetition_factor (full / no_repetition)": round(math.log10(full / no_rep), 2),
                    "log10_base (no_repetition & PAR_as_SEQ)": round(math.log10(no_rep_seq), 2)})
        print("   decomposition", dec[-1], flush=True)
    pd.DataFrame(dec).to_csv(OUT / "bpic_explosion_decomposition.csv", index=False)
    out["explosion_decomposition"] = dec
    (OUT / "bpic_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    _strip_data_root(sys.argv)
    bpic() if sys.argv[1:2] == ["bpic"] else main()
