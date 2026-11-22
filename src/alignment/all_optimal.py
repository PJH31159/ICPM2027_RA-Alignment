"""Tau-excluded all-optimal reference for the Alignment Accuracy (ACC).

The functions below are copied verbatim from the search used for the official ACC of the paper runs
(research script credit_all_optimal_analysis.py / all_optimal_agreement.py, function dev_optimal).

dev_optimal(acts, net_struct(net, im, fm), move_cost=1, tau_cost=0) returns (minimal cost, set of ALL repaired traces of
minimal cost, popped states). Cost model: synchronous move 0, log move 1, visible model move 1, silent (tau) move 0,
i.e. only visible deviations are counted ("tau-excluded"). move_cost=10000, tau_cost=1 gives the PM4Py standard cost
("native"). A repaired trace is the model-side projection without tau.
"""
import heapq


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
