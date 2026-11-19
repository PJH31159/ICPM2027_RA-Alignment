"""
loop_bound_utils — loop가 있는 Process Tree의 bounded trace language 생성과 coverage 계산 (EXP-066)

공식 loop semantics (D004, 2026-10-14 확정):
    loop_num   : loop instance 하나에서 do-part가 실행될 수 있는 최대 횟수 (instance-local)
    loop_total : 한 trace 안 모든 loop instance의 do-part 실행 횟수 총합의 상한 (redo가 아니라 do 실행 기준)
    nested loop: instance-local. 바깥 loop가 다시 돌면서 안쪽 loop가 새로 시작되면 안쪽 counter도 새로 센다.
    trace length: 연구 파라미터가 아님. max length 100은 안전장치(safety guard)로만 쓴다.

PM4Py LOOP *(do, redo) 의 실행 = do (redo do)^k  ->  do 실행 횟수 = k + 1.

출처: src/experiments/p1_toy_trace_generation/15_full_bounded_process_tree_generator.py 의
generate_bounded_traces와 같은 의미를 dict(trace -> 최소 loop count) 형태로 다시 구현했다.
    - 같은 trace가 여러 방법으로 만들어지면 가장 작은 do 실행 총합을 그 trace의 count로 쓴다 (원본 dedup과 같음).
    - 원본과 달리 중간 결과를 개수로 자르지 않는다. 대신 안전장치를 넘으면 ExplosionAborted를 올려
      그 조합 전체를 ABORTED로 기록한다 (잘린 결과로 coverage를 계산하지 않기 위해).
"""

import time
from collections import deque

from pm4py.objects.process_tree.obj import Operator


class ExplosionAborted(Exception):
    """안전장치(중간 집합 크기 / 시간 / 메모리)를 넘어서 생성을 중단했다."""


class _Guard:
    def __init__(self, max_intermediate, max_seconds, max_rss_gb):
        self.max_intermediate = max_intermediate
        self.max_seconds = max_seconds
        self.max_rss_gb = max_rss_gb
        self.t0 = time.time()
        self.peak_intermediate = 0
        try:
            import psutil
            self._proc = psutil.Process()
        except ModuleNotFoundError:
            self._proc = None

    def check(self, size, where):
        self.peak_intermediate = max(self.peak_intermediate, size)
        if size > self.max_intermediate:
            raise ExplosionAborted(f"intermediate set {size:,} > {self.max_intermediate:,} at {where}")
        if time.time() - self.t0 > self.max_seconds:
            raise ExplosionAborted(f"time > {self.max_seconds}s at {where}")
        if self._proc is not None and self._proc.memory_info().rss / 2 ** 30 > self.max_rss_gb:
            raise ExplosionAborted(f"process memory > {self.max_rss_gb} GiB at {where}")


def _add(out, trace, count):
    """trace -> 최소 count."""
    old = out.get(trace)
    if old is None or count < old:
        out[trace] = count


def _interleavings(x, y):
    """두 sequence의 모든 interleaving (각 sequence 안의 순서는 유지)."""
    if not x:
        return {y}
    if not y:
        return {x}
    res = set()
    for rest in _interleavings(x[1:], y):
        res.add((x[0],) + rest)
    for rest in _interleavings(x, y[1:]):
        res.add((y[0],) + rest)
    return res


def bounded_language(node, loop_num, loop_total, max_len, guard, path="root"):
    """node가 만들 수 있는 trace -> 최소 do 실행 총합 (loop_num / loop_total / max_len 안에서)."""
    op = node.operator
    if op is None:
        return {(): 0} if node.label is None else {(node.label,): 0}
    kids = [bounded_language(c, loop_num, loop_total, max_len, guard, f"{path}/{i}")
            for i, c in enumerate(node.children)]

    if op == Operator.SEQUENCE:
        cur = kids[0]
        for nxt in kids[1:]:
            new = {}
            for t1, c1 in cur.items():
                for t2, c2 in nxt.items():
                    if c1 + c2 <= loop_total and len(t1) + len(t2) <= max_len:
                        _add(new, t1 + t2, c1 + c2)
            guard.check(len(new), path)
            cur = new
        return cur

    if op == Operator.XOR:
        out = {}
        for k in kids:
            for t, c in k.items():
                _add(out, t, c)
        return out

    if op == Operator.PARALLEL:
        cur = kids[0]
        for nxt in kids[1:]:
            new = {}
            for t1, c1 in cur.items():
                for t2, c2 in nxt.items():
                    if c1 + c2 <= loop_total and len(t1) + len(t2) <= max_len:
                        for t in _interleavings(t1, t2):
                            _add(new, t, c1 + c2)
                guard.check(len(new), path)
            cur = new
        return cur

    if op == Operator.LOOP:
        if len(kids) != 2:
            raise ValueError(f"LOOP with {len(kids)} children is not supported ({path})")
        do, redo = kids
        out = {}
        cur = dict(do)  # do가 k번 실행된 partial trace (count = 안쪽 do 실행 합, 이 loop의 k는 아직 안 더함)
        for k in range(1, loop_num + 1):
            cur = {t: c for t, c in cur.items() if c + k <= loop_total}
            for t, c in cur.items():
                _add(out, t, c + k)
            if k == loop_num or not cur:
                break
            new = {}
            for t1, c1 in cur.items():
                for t2, c2 in redo.items():
                    for t3, c3 in do.items():
                        c = c1 + c2 + c3
                        if c + k + 1 <= loop_total and len(t1) + len(t2) + len(t3) <= max_len:
                            _add(new, t1 + t2 + t3, c)
            guard.check(len(new), path)
            cur = new
        return out

    raise ValueError(f"unsupported operator {op} ({path})")


def generate_bounded(tree, loop_num, loop_total, max_len, max_intermediate, max_seconds, max_rss_gb):
    """return (language dict, stats). 안전장치를 넘으면 ExplosionAborted."""
    guard = _Guard(max_intermediate, max_seconds, max_rss_gb)
    t0 = time.time()
    lang = bounded_language(tree, loop_num, loop_total, max_len, guard)
    return lang, {"generation_time_sec": time.time() - t0, "peak_intermediate": guard.peak_intermediate}


# ------------------------------------------------------------
# Petri Net 정확 accepting execution (tau closure 포함) — 생성 trace의 conformance 검증용
# ------------------------------------------------------------

def _marking_key(m):
    return frozenset((p, n) for p, n in m.items() if n > 0)


def _fire(m, t):
    if all(m.get(a.source, 0) >= a.weight for a in t.in_arcs):
        new = dict(m)
        for a in t.in_arcs:
            new[a.source] -= a.weight
        for a in t.out_arcs:
            new[a.target] = new.get(a.target, 0) + a.weight
        return new
    return None


def _tau_closure(markings, taus, max_states=100_000):
    seen = {_marking_key(m): m for m in markings}
    q = deque(seen.values())
    while q:
        m = q.popleft()
        for t in taus:
            n = _fire(m, t)
            if n is not None:
                k = _marking_key(n)
                if k not in seen:
                    seen[k] = n
                    q.append(n)
                    if len(seen) > max_states:
                        raise RuntimeError("tau closure too large")
    return list(seen.values())


def make_petri_acceptor(net, im, fm):
    """trace를 받아 Petri Net이 정확히 받아들이는지(초기 marking -> 최종 marking) 판정하는 함수."""
    taus = [t for t in net.transitions if t.label is None]
    by_label = {}
    for t in net.transitions:
        if t.label is not None:
            by_label.setdefault(t.label, []).append(t)
    m0 = {p: n for p, n in im.items()}
    final_key = _marking_key({p: n for p, n in fm.items()})
    start = _tau_closure([m0], taus)

    def accepts(trace):
        states = start
        for a in trace:
            nxt = {}
            for m in states:
                for t in by_label.get(a, []):
                    n = _fire(m, t)
                    if n is not None:
                        nxt[_marking_key(n)] = n
            if not nxt:
                return False
            states = _tau_closure(list(nxt.values()), taus)
        return any(_marking_key(m) == final_key for m in states)

    return accepts


# ------------------------------------------------------------
# 관측 trace의 loop 실행 횟수 (Credit discovered tree 전용) — sensitivity 분석 / coverage cross-check
# ------------------------------------------------------------

CREDIT_TREE = ("->( 'a', +( X( tau, *( 'e', tau ) ), *( ->( *( 'b', ->( 'j', 'k' ) ), 'c', 'd' ), 'l' ) ), "
               "X( tau, ->( 'h', 'i' ) ), 'f', X( tau, 'g' ) )")


def credit_loop_requirements(trace):
    """Credit tree의 loop별 do 실행 횟수와, 이 trace를 포함하는 데 필요한 최소 (loop_num, loop_total).

    loop 3개의 label이 서로 겹치지 않아서 실행 횟수가 하나로 정해진다:
        e-loop *(e, tau): do 실행 = e의 수 (0이면 XOR의 tau 가지, loop 실행 없음)
        바깥 loop *(…, l): do 실행 = l의 수 + 1
        안쪽 b-loop *(b, ->(j,k)): instance(바깥 loop의 do 한 번)마다 b의 수
    D004 semantics: need_loop_num = 가장 큰 instance의 do 실행, need_loop_total = 모든 do 실행의 합.
    """
    e = trace.count("e")
    outer = trace.count("l") + 1
    seg, inner = 0, []
    for a in trace:
        if a == "b":
            seg += 1
        elif a == "l":
            inner.append(seg)
            seg = 0
    inner.append(seg)
    return {"e_do": e, "outer_do": outer, "inner_do_list": inner, "inner_do_max": max(inner),
            "need_loop_num": max(e, outer, max(inner)), "need_loop_total": e + outer + sum(inner)}
