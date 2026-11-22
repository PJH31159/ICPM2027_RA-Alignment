"""Normal Behavior Graphs (NBGs) from the bounded model trace set L_N only (paper Definition 3).

build_new_nbgs(L_N event log, actset):
    - every unique trace of L_N is used exactly once (weight 1); the clean log is NOT added and its case
      frequencies are NOT restored;
    - discover_NBG of the RP-BEAR core (rpbear/func_rpbear_final.py) is used unchanged;
    - the NBGs are returned in the order of the observed activity set `actset`, because RP-BEAR's ADM looks an NBG up
      by its actset position.
Two former hard errors were removed at the supervisor's request (they are normal situations, not errors):
    - "generated activity set != observed actset": an observed activity that never occurs in L_N (for example
      activity 'w' of BPIC 2017) gets an NBG with its anchor node and no edges, so every directly-follows relation
      involving it is voted deviating by the ADM; activities of L_N that are not in actset are not used by the ADM;
    - "NBG ordering mismatch": the NBGs are mapped to actset by activity name, so NBG i always belongs to actset[i];
      the anchor check is kept as information only.
"""

import hashlib


class NBGValidationError(RuntimeError):
    pass


def nbg_anchor(graph):
    """Anchor activity of an NBG = activity of its level-0 nodes."""
    return {d["activity"] for _, d in graph.nodes(data=True) if d.get("level") == 0}


def order_mismatches(nbgs, actset):
    return [(i, act, sorted(nbg_anchor(g))) for i, (act, g) in enumerate(zip(actset, nbgs))
            if nbg_anchor(g) != {act}]


def _anchor_only_nbg(activity):
    """NBG of an activity that does not occur in L_N: only the anchor node (same naming as discover_NBG), no edges."""
    import networkx as nx

    g = nx.DiGraph()
    g.add_node(f"{activity}_0", level=0, activity=activity)
    return g


def build_new_nbgs(generated_log_path, actset):
    """L_N event log (Case / Activity, one case per unique trace) -> (Start/End as in sim) -> discover_NBG -> actset order.

    Returns (ordered_nbgs, info dict). Raises NBGValidationError only if L_N contains a duplicate trace.
    """
    from rpbear.func_rpbear_final import discover_NBG

    from utils.data_utils import add_start_end_like_sim, read_log_like_sim

    gen_clean = add_start_end_like_sim(read_log_like_sim(generated_log_path, has_label=False))

    # weight 1: every case of the generated log must be a distinct trace
    body = gen_clean[~gen_clean["concept:name"].isin(["Start", "End"])]
    per_case = body.groupby("case:concept:name", observed=True)["concept:name"].agg(lambda s: ">>".join(map(str, s)))
    if per_case.nunique() != len(per_case):
        raise NBGValidationError("generated log has duplicate traces (weight must be 1 per unique trace)")

    gen_actset = list(gen_clean["concept:name"].unique())
    raw = discover_NBG(gen_clean, vis=False)
    by_activity = dict(zip(gen_actset, raw))
    missing = [str(a) for a in actset if a not in by_activity]
    extra = [str(a) for a in gen_actset if a not in set(map(str, actset))]
    ordered = [by_activity[act] if act in by_activity else _anchor_only_nbg(act) for act in actset]
    mismatches = order_mismatches(ordered, actset)

    info = {
        "generated_cases": int(per_case.size),
        "generated_rows_with_start_end": int(len(gen_clean)),
        "generated_activity_first_seen_order_equals_actset": list(gen_actset) == list(actset),
        "anchor_match": f"{len(actset) - len(mismatches)}/{len(actset)}",
        "activities_without_LN_occurrence": missing,
        "LN_activities_not_in_actset": extra,
        "weighting": "each unique generated trace once (model language as a set; observed frequencies NOT restored)",
    }
    return ordered, info


def edge_table(graph):
    import networkx as nx
    return nx.get_edge_attributes(graph, "likelihood")


def edge_set_digest(nbgs, actset):
    h = hashlib.sha256()
    for act, g in zip(actset, nbgs):
        h.update(str(act).encode())
        for e in sorted(g.edges()):
            h.update(repr(e).encode())
    return h.hexdigest()[:16]


def trace_edge_set(traces):
    """Independently compute the NBG edge set of traces with the discover_NBG rule (validation).

    Traces are activity tuples without Start/End. edge = (graph activity, f"{a}_{i-p}", f"{b}_{i+1-p}"),
    where p ranges over the positions of the graph activity in the trace (with Start/End).
    """
    out = set()
    for t in traces:
        full = ("Start",) + tuple(t) + ("End",)
        for p, anchor in enumerate(full):
            for i in range(len(full) - 1):
                out.add((anchor, f"{full[i]}_{i - p}", f"{full[i + 1]}_{i + 1 - p}"))
    return out


def nbg_edge_set(nbgs, actset):
    return {(str(a), f, t) for a, g in zip(actset, nbgs) for f, t in g.edges()}
