"""Bounded trace language of a process tree with loops, and clean-case coverage.

Loop-bound semantics:
    loop_num   : maximum number of do-part executions within one loop instance (instance-local).
    loop_total : upper bound on the total number of do-part executions over all loop instances
                 in one trace (counted on do executions, not redo executions).
    nested loop: instance-local; when an outer loop iterates and restarts an inner loop, the
                 inner counter restarts as well.
    trace length is not a research parameter; max length 100 is only a safety guard.

A PM4Py LOOP *(do, redo) executes as do (redo do)^k, i.e. k + 1 do executions.

The language is stored as dict(trace -> minimal loop count): if a trace can be produced in
several ways, its count is the smallest total number of do executions. Intermediate results are
never truncated; if a safety guard is exceeded, ExplosionAborted is raised and the whole bound
is recorded as ABORTED, so coverage is never computed from a truncated language.
"""

import time
from collections import deque

from pm4py.objects.process_tree.obj import Operator


class ExplosionAborted(Exception):
    """Generation stopped because a safety guard (intermediate set size / time / memory) was exceeded."""


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
    """trace -> minimal count."""
    old = out.get(trace)
    if old is None or count < old:
        out[trace] = count


def _interleavings(x, y):
    """All interleavings of two sequences (the order inside each sequence is kept)."""
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
    """Traces a node can produce -> minimal total do executions (within loop_num / loop_total / max_len)."""
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
        cur = dict(do)  # partial trace with k do executions (count = inner do executions; k of this loop not added yet)
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
    """Returns (language dict, stats). Raises ExplosionAborted when a guard is exceeded."""
    guard = _Guard(max_intermediate, max_seconds, max_rss_gb)
    t0 = time.time()
    lang = bounded_language(tree, loop_num, loop_total, max_len, guard)
    return lang, {"generation_time_sec": time.time() - t0, "peak_intermediate": guard.peak_intermediate}


# ------------------------------------------------------------
# Exact accepting execution of the Petri net (with tau closure), used to check generated traces
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
    """Return a function that decides whether the Petri net accepts a trace (initial -> final marking)."""
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
# Loop execution counts of observed traces (Credit discovered tree only): sensitivity / coverage cross-check
# ------------------------------------------------------------

CREDIT_TREE = ("->( 'a', +( X( tau, *( 'e', tau ) ), *( ->( *( 'b', ->( 'j', 'k' ) ), 'c', 'd' ), 'l' ) ), "
               "X( tau, ->( 'h', 'i' ) ), 'f', X( tau, 'g' ) )")


def credit_loop_requirements(trace):
    """Do-execution counts per loop of the Credit tree, and the minimal (loop_num, loop_total) for the trace.

    The three loops have disjoint labels, so the execution counts are unique:
        e-loop *(e, tau): do executions = number of e (0 means the XOR tau branch, no loop execution)
        outer loop *(..., l): do executions = number of l + 1
        inner b-loop *(b, ->(j,k)): number of b per instance (one do execution of the outer loop)
    need_loop_num = largest per-instance do count; need_loop_total = sum of all do executions.
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
