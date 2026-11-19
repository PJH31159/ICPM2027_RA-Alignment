"""12_loop_requirement_analysis.py — trace를 enumerate하지 않고 bounded coverage와 trace language 크기를 계산 (탐색, .venv)

왜: BPIC 2017은 (1,4)에서 1,360,844 trace를 만들어도 clean coverage가 1.37%다.
    더 큰 bound는 생성 guard를 넘으므로, 수백만~수억 trace를 실제로 만들지 않고
    (1) observed clean trace 하나하나가 "최소 어떤 loop bound를 요구하는지"와
    (2) 그 bound에서 bounded trace language가 몇 개인지
    를 Process Tree 구조로 직접 계산한다.

loop semantics (D004, loop_bound_utils와 같음):
    loop_num   : loop instance 하나에서 do-part 실행 횟수의 상한 (instance-local, nested loop는 바깥 반복마다 새로 셈)
    loop_total : 한 trace 안 모든 loop instance의 do-part 실행 횟수 총합의 상한 (첫 실행도 1로 셈)
    max trace length 100 (generator 안전장치와 같음)

방법 1 — requirement (coverage):
    모든 visible label이 tree에서 한 번씩만 나오면(duplicate label 없음) trace의 각 event는 leaf 하나에만 대응한다.
    그래서 subtree마다 "자기 alphabet으로 projection한 부분 trace"만 보면 된다.
      SEQ  : child 순서대로 event가 이어져야 함 / XOR: 한 child만 / PAR: child마다 projection, 자유 interleaving
      LOOP : do (redo do)^(k-1) 로 쪼개는 모든 방법을 DP로 탐색 (k = do 실행 횟수)
    결과 = trace별 (필요 loop_num, 필요 loop_total) Pareto front. bound (n, t)에서 covered ⇔ front에 n'<=n, t'<=t 인 점이 있음.
방법 2 — language size:
    node마다 (길이, loop_total 사용량) -> trace 수 를 계산한다.
      SEQ: 합성곱 / XOR: 합 / PAR: 길이 l1, l2 두 trace의 interleaving 수 C(l1+l2, l1) (alphabet이 서로 달라 모두 다른 trace)
      LOOP: k = 1..n 번 do 실행, 각 반복마다 loop_total 사용량 +1
    PAR을 SEQ로 바꾼 같은 계산과 비교해 parallel interleaving의 기여를 분리한다.
검증: 두 방법을 기존 enumeration 결과(Pub, Credit, Road raw/dedup, BPIC (1,4))와 비교해 같을 때만 BPIC 해석에 쓴다.

출력: outputs/multi_dataset_experiment/loop_requirement_analysis/
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
    """data root: --data-root 인자 > 환경변수 ICPM2027_DATA_ROOT > <repo>/data/raw
    (src/utils/data_utils.resolve_data_root와 같은 우선순위). P-BEAR-RL Full_input 폴더를 가리켜야 한다."""
    import os
    if cli_root is None and "--data-root" in sys.argv:
        cli_root = sys.argv[sys.argv.index("--data-root") + 1]
    if cli_root:
        return Path(cli_root).resolve()
    env = os.environ.get("ICPM2027_DATA_ROOT")
    return Path(env).resolve() if env else (PROJECT_ROOT / "data" / "raw").resolve()


def _strip_data_root(argv):
    """positional 인자를 읽기 전에 '--data-root X'를 argv에서 뺀다 (기존 인자 처리는 그대로)."""
    if "--data-root" in argv:
        i = argv.index("--data-root")
        del argv[i:i + 2]
FULL = _data_root() / "encoded_anomaly"
SIM = PROJECT_ROOT / "outputs/simulation"
OUT = PROJECT_ROOT / "outputs/multi_dataset_experiment/loop_requirement_analysis"
MAX_LEN = 100
INF = 10 ** 9

DATASETS = {  # tree = 기존 진단에서 저장한 discovered tree (IM noise 0, clean case)
    "Credit": (SIM / "Credit/dataset_audit/process_tree.txt", FULL / "credit-card_1.00.csv"),
    "Pub": (SIM / "Pub/dataset_audit/process_tree.txt", FULL / "mccloud_1.00.csv"),
    "RoadTraffic_raw": (SIM / "RoadTraffic/exploratory_dedup/A_raw_process_tree.txt", FULL / "Road_Traffic_1.00.csv"),
    "RoadTraffic_dedup": (SIM / "RoadTraffic/exploratory_dedup/B_dedup_process_tree.txt",
                          SIM / "RoadTraffic/exploratory_dedup/derived_data/Road_Traffic_1.00_dedup.csv"),
    "BPIC2017": (SIM / "BPIC2017/dataset_audit/process_tree.txt", FULL / "b17_1.00.csv"),
}
# 기존 enumeration 결과 (근거 파일의 값). (n, t) -> (generated_unique, covered clean cases)
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


# ------------------------------------------------------------------ 방법 1: requirement
def pareto2(points):
    """points: (a, b, w). (a, b) 2차원 Pareto front, 같은 (a, b)는 첫 witness 유지."""
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
    """points: (k, imax, itot, w). 세 값 모두 작거나 같은 점이 있으면 제거."""
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
            """seq[i:j]가 A 안의 label로만 이루어진 j들과 그 requirement."""
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
                        continue  # 반복마다 최소 1 event를 소비 (빈 반복 무한 루프 방지)
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


# ------------------------------------------------------------------ 방법 2: language size
def lang_count(node, n, t, par_as_seq=False):
    """(길이 0..MAX_LEN, loop_total 사용량 0..t) -> trace 수 (float64, EGF로 interleaving 계산)."""
    L = MAX_LEN

    def norm(A):
        # 빈 trace는 1개뿐: 가장 작은 loop_total 사용량 하나로 합친다 (generator의 최소 count dedup과 같음)
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
                out[:, k:] += cur[:, : t + 1 - k]   # 이 loop instance의 do 실행 k회 -> 사용량 +k
                if k == n:
                    break
                lim = t - (k + 1)  # do를 k+1번 실행하면 안쪽 사용량은 t-(k+1) 이하여야 한다
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


# ------------------------------------------------------------------ BPIC 2017 분석 (enumeration 없음)
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

    # 1) case별 requirement
    rows = []
    for v, c in obs.items():
        F = fronts[v]
        if not F:
            rows.append({"variant": " ".join(v), "cases": c, "length": len(v), "coverable": False})
            continue
        a1 = [b for a, b, _ in F if a == 1]
        wit = min(F, key=lambda x: (x[1], x[0]))[2]  # loop_total이 가장 작은 parse의 loop별 (instance max k, k 합)
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

    # 2) (1,4)에서 빠진 case의 이유
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

    # 3) coverage 목표별 Pareto-minimal bound와 그 language 크기
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

    # 4) 폭발 원인 분해: 같은 bound에서 PAR->SEQ / loop 반복 제거(n=1, 같은 t)
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
