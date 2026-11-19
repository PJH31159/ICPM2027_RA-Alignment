"""Inductive Miner -> process tree -> Petri net, and model structure checks.

tree_length_distribution / estimate_tree_language_size compute the size of the process-tree
language without generating traces (multinomial counting). They return None if the tree has a
LOOP (infinite language). With duplicate labels in several branches the count is an upper bound
(raw candidates). petri_net_has_cycle reports whether the Petri net graph contains a cycle.
The process-tree language is enumerated only when its expected size is below the safety bound.
"""

import time
from collections import Counter
from math import comb

ALIGNMENT_VARIANT_NAME = "VERSION_STATE_EQUATION_A_STAR"  # PM4Py state-equation A* alignments


# ------------------------------------------------------------
# 1. Discovery
# ------------------------------------------------------------

def discover_model(traces, clean_ids, noise_threshold):
    """Clean cases -> Inductive Miner -> process tree -> Petri net."""
    import pm4py

    from utils.data_utils import build_event_log

    clean_log = build_event_log(clean_ids, traces)
    t0 = time.time()
    tree = pm4py.discover_process_tree_inductive(clean_log, noise_threshold=noise_threshold)
    net, im, fm = pm4py.convert_to_petri_net(tree)
    return tree, net, im, fm, time.time() - t0


# ------------------------------------------------------------
# 2. Process tree structure / language
# ------------------------------------------------------------

def count_operators(node, counter=None):
    counter = Counter() if counter is None else counter
    counter[str(node.operator) if node.operator is not None else ("tau" if node.label is None else "activity")] += 1
    for c in node.children:
        count_operators(c, counter)
    return counter


def operator_kind_counts(node, counter=None):
    """Count operators via the Operator enum.

    str(operator) is a symbol such as '*', '+', 'X', '->', so searching for names like 'LOOP' would always give 0.
    """
    counter = Counter() if counter is None else counter
    if node.operator is None:
        counter["tau" if node.label is None else "activity"] += 1
    else:
        counter[node.operator.name] += 1  # SEQUENCE / XOR / PARALLEL / LOOP / OR / INTERLEAVING
    for c in node.children:
        operator_kind_counts(c, counter)
    return counter


def interleave(a, b):
    if not a:
        return [b]
    if not b:
        return [a]
    return [[a[0]] + rest for rest in interleave(a[1:], b)] + [[b[0]] + rest for rest in interleave(a, b[1:])]


def enumerate_tree_language(node):
    """Enumerate the full language of a tree with only SEQUENCE / XOR / PARALLEL / tau.

    Returns None for other operators (e.g. LOOP). Check the size first (use tree_language_safe).
    """
    from pm4py.objects.process_tree.obj import Operator

    if node.operator is None:
        return [[]] if node.label is None else [[node.label]]
    child_langs = [enumerate_tree_language(c) for c in node.children]
    if any(lang is None for lang in child_langs):
        return None
    if node.operator == Operator.XOR:
        return [t for lang in child_langs for t in lang]
    if node.operator == Operator.SEQUENCE:
        result = [[]]
        for lang in child_langs:
            result = [prefix + suffix for prefix in result for suffix in lang]
        return result
    if node.operator == Operator.PARALLEL:
        result = [[]]
        for lang in child_langs:
            result = [mix for prefix in result for suffix in lang for mix in interleave(prefix, suffix)]
        return result
    return None


def tree_length_distribution(node):
    """{trace length: number of raw traces}. None if the tree contains LOOP / OR."""
    from pm4py.objects.process_tree.obj import Operator

    if node.operator is None:
        return {0: 1} if node.label is None else {1: 1}
    dists = [tree_length_distribution(c) for c in node.children]
    if any(d is None for d in dists):
        return None
    if node.operator == Operator.XOR:
        out = Counter()
        for d in dists:
            out.update(d)
        return dict(out)
    if node.operator in (Operator.SEQUENCE, Operator.PARALLEL):
        out = {0: 1}
        for d in dists:
            nxt = Counter()
            for l1, c1 in out.items():
                for l2, c2 in d.items():
                    mult = comb(l1 + l2, l1) if node.operator == Operator.PARALLEL else 1
                    nxt[l1 + l2] += c1 * c2 * mult
            out = dict(nxt)
        return out
    return None


def estimate_tree_language_size(tree):
    dist = tree_length_distribution(tree)
    return None if dist is None else sum(dist.values())


def petri_net_has_cycle(net):
    """True if the place -> transition -> place graph has a cycle (repeatable behaviour)."""
    succ = {}
    for arc in net.arcs:
        succ.setdefault(arc.source, []).append(arc.target)
    color = {}

    def visit(start):
        stack = [(start, iter(succ.get(start, [])))]
        color[start] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = 2
                stack.pop()
            elif color.get(nxt, 0) == 1:
                return True
            elif color.get(nxt, 0) == 0:
                color[nxt] = 1
                stack.append((nxt, iter(succ.get(nxt, []))))
        return False

    nodes = list(net.places) + list(net.transitions)
    return any(color.get(n, 0) == 0 and visit(n) for n in nodes)


# ------------------------------------------------------------
# 3. Model check report
# ------------------------------------------------------------

def model_report(tree, net, im, fm, traces, clean_ids, discovery_sec, safety_max_traces):
    """Model check (fitness, soundness, language) plus a structure block.

    The process-tree language is enumerated only if its expected size is <= safety_max_traces.
    Returns (model_check dict, tree language set or None).
    """
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo
    from pm4py.objects.log.obj import Event, Trace

    variant = getattr(alignments_algo.Variants, ALIGNMENT_VARIANT_NAME)
    clean_variants = Counter(tuple(traces[cid]) for cid in clean_ids)

    ops = count_operators(tree)
    estimated = estimate_tree_language_size(tree)
    if estimated is not None and estimated <= safety_max_traces:
        language = enumerate_tree_language(tree)
        language_set = None if language is None else set(tuple(t) for t in language)
        language_note = "enumerated"
    else:
        language_set = None
        language_note = ("not enumerated: tree has LOOP/other operator (infinite or unbounded language)"
                         if estimated is None else
                         f"not enumerated: estimated raw size {estimated} > safety_max_traces {safety_max_traces}")

    variant_rows = []
    for v, count in clean_variants.most_common():
        trace_obj = Trace()
        for act in v:
            trace_obj.append(Event({"concept:name": act}))
        res = alignments_algo.apply(trace_obj, net, im, fm, variant=variant)
        if res is None or res.get("alignment") is None:
            raise RuntimeError("alignment returned None")
        variant_rows.append({
            "variant": ">>".join(v), "n_cases": count, "length": len(v),
            "alignment_cost": res["cost"], "fitness": res["fitness"],
            "in_model_language": None if language_set is None else v in language_set,
        })
    n_clean = sum(clean_variants.values())
    weighted_fitness = sum(r["fitness"] * r["n_cases"] for r in variant_rows) / n_clean

    visible = [t for t in net.transitions if t.label is not None]
    model_check = {
        "process_tree": str(tree),
        "tree_node_counts": dict(ops),
        "n_places": len(net.places),
        "n_transitions": len(net.transitions),
        "n_visible_transitions": len(visible),
        "n_invisible_transitions": len(net.transitions) - len(visible),
        "n_arcs": len(net.arcs),
        "visible_activities": sorted(t.label for t in visible),
        "n_clean_cases": n_clean,
        "n_clean_variants": len(clean_variants),
        "model_language_size": None if language_set is None else len(language_set),
        "model_language_equals_clean_variants": None if language_set is None else language_set == set(clean_variants),
        "clean_variants_not_in_model": None if language_set is None else [">>".join(v) for v in clean_variants if v not in language_set],
        "model_traces_not_in_clean": None if language_set is None else sorted(">>".join(t) for t in language_set - set(clean_variants)),
        "clean_log_fitness_alignment_weighted": weighted_fitness,
        "clean_variants": variant_rows,
        "discovery_sec": round(discovery_sec, 4),
    }
    kinds = operator_kind_counts(tree)
    model_check["structure"] = {
        "operator_kinds": dict(kinds),
        "n_loop_operators": kinds.get("LOOP", 0),
        "n_parallel_operators": kinds.get("PARALLEL", 0),
        "n_xor_operators": kinds.get("XOR", 0),
        "n_other_operators": sum(v for k, v in kinds.items()
                                 if k not in ("SEQUENCE", "XOR", "PARALLEL", "LOOP", "tau", "activity")),
        "n_tau_leaves": kinds.get("tau", 0),
        "petri_net_has_cycle": petri_net_has_cycle(net),
        "estimated_raw_tree_language_size": estimated,
        "tree_language_note": language_note,
    }
    return model_check, language_set
