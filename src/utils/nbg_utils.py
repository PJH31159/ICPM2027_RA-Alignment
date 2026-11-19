"""
nbg_utils — New NBG 생성, actset 순서 검증, Original/New 비교, ADM 동일성 (REF-001)

출처: src/63_generated_nbg_rpbear.py (build_new_nbgs, nbg_anchor, edge_set_digest,
      stage_nbg의 비교 / ADM 부분)

D034 규칙을 이 모듈 안에서 강제한다 (호출하는 쪽에서 끌 수 없음):
    - generated unique trace weight = 1 (생성 log의 case마다 서로 다른 trace)
      (D039 이후: model-only trace가 있는 Credit은 build_expanded_nbgs = clean log + model-only, build_method_c_nbgs로 선택)
    - New NBG는 observed actset 순서로 재정렬
    - 모든 index i에서 actset[i] == anchor(NBG[i])가 아니면 예외
    - structure / edge set / likelihood는 따로 판단
RP-BEAR core(rpbear/func_rpbear_final.py)는 수정하지 않고 discover_NBG를 그대로 쓴다.
"""

import hashlib


class NBGValidationError(RuntimeError):
    pass


def nbg_anchor(graph):
    """NBG의 기준(anchor) activity = level 0 node들의 activity."""
    return {d["activity"] for _, d in graph.nodes(data=True) if d.get("level") == 0}


def order_mismatches(nbgs, actset):
    return [(i, act, sorted(nbg_anchor(g))) for i, (act, g) in enumerate(zip(actset, nbgs))
            if nbg_anchor(g) != {act}]


def build_new_nbgs(generated_log_path, actset):
    """생성 trace log -> (sim과 같은 Start/End) -> discover_NBG -> actset 순서 재정렬 + 검증.

    return: (ordered_nbgs, info dict). 규칙 위반 시 NBGValidationError.
    """
    from rpbear.func_rpbear_final import discover_NBG

    from utils.data_utils import add_start_end_like_sim, read_log_like_sim

    gen_clean = add_start_end_like_sim(read_log_like_sim(generated_log_path, has_label=False))

    # weight 1: 생성 log의 case마다 서로 다른 trace여야 한다
    body = gen_clean[~gen_clean["concept:name"].isin(["Start", "End"])]
    per_case = body.groupby("case:concept:name", observed=True)["concept:name"].agg(lambda s: ">>".join(map(str, s)))
    if per_case.nunique() != len(per_case):
        raise NBGValidationError("generated log has duplicate traces (weight must be 1 per unique trace)")

    gen_actset = list(gen_clean["concept:name"].unique())
    if set(gen_actset) != set(actset):
        raise NBGValidationError(f"generated activity set != observed actset "
                                 f"(only generated: {sorted(set(gen_actset) - set(actset))}, "
                                 f"only observed: {sorted(set(actset) - set(gen_actset))})")

    raw = discover_NBG(gen_clean, vis=False)
    by_activity = dict(zip(gen_actset, raw))
    ordered = [by_activity[act] for act in actset]
    mismatches = order_mismatches(ordered, actset)
    if mismatches:
        raise NBGValidationError(f"New NBG ordering mismatch: {mismatches}")

    info = {
        "generated_cases": int(per_case.size),
        "generated_rows_with_start_end": int(len(gen_clean)),
        "generated_activity_first_seen_order_equals_actset": list(gen_actset) == list(actset),
        "anchor_match": f"{len(actset)}/{len(actset)}",
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


def compare_nbgs(original, new, actset):
    """structure / edge set / likelihood를 따로 비교. return: (summary, per_graph df, edges df)"""
    import pandas as pd

    per_graph, edge_rows = [], []
    for act, g_o, g_n in zip(actset, original, new):
        nodes_o = {(n, d["activity"], d["level"]) for n, d in g_o.nodes(data=True)}
        nodes_n = {(n, d["activity"], d["level"]) for n, d in g_n.nodes(data=True)}
        lik_o, lik_n = edge_table(g_o), edge_table(g_n)
        common = set(lik_o) & set(lik_n)
        diffs = [abs(lik_o[e] - lik_n[e]) for e in common]
        per_graph.append({
            "activity": act,
            "orig_nodes": g_o.number_of_nodes(), "new_nodes": g_n.number_of_nodes(),
            "orig_edges": g_o.number_of_edges(), "new_edges": g_n.number_of_edges(),
            "node_set_equal": nodes_o == nodes_n,
            "edge_set_equal": set(lik_o) == set(lik_n),
            "likelihood_equal": set(lik_o) == set(lik_n) and all(d == 0 for d in diffs),
            "n_edges_likelihood_differs": sum(d > 0 for d in diffs),
            "max_abs_likelihood_diff": max(diffs) if diffs else 0.0,
        })
        for e in sorted(set(lik_o) | set(lik_n)):
            edge_rows.append({"activity": act, "from": e[0], "to": e[1],
                              "orig_likelihood": lik_o.get(e), "new_likelihood": lik_n.get(e),
                              "abs_diff": (abs(lik_o[e] - lik_n[e]) if e in common else None)})
    graphs, edges = pd.DataFrame(per_graph), pd.DataFrame(edge_rows)

    # edge validity(likelihood > alpha)가 두 NBG에서 달라지기 시작하는 가장 작은 alpha
    cand = sorted(set(edges["orig_likelihood"].dropna()) | set(edges["new_likelihood"].dropna()))
    o = edges["orig_likelihood"].fillna(-1.0)
    n = edges["new_likelihood"].fillna(-1.0)
    first = next((a for a in [0.0] + cand if ((o > a) != (n > a)).any()), None)

    summary = {
        "n_graphs": {"original": len(original), "new": len(new)},
        "total_nodes": {"original": int(graphs.orig_nodes.sum()), "new": int(graphs.new_nodes.sum())},
        "total_edges": {"original": int(graphs.orig_edges.sum()), "new": int(graphs.new_edges.sum())},
        "structure_equal (all node sets incl. activity/level)": bool(graphs.node_set_equal.all()),
        "edge_set_equal (all graphs)": bool(graphs.edge_set_equal.all()),
        "likelihood_equal (all edges)": bool(graphs.likelihood_equal.all()),
        "graphs_with_likelihood_difference": int((~graphs.likelihood_equal).sum()),
        "edges_with_likelihood_difference": int(graphs.n_edges_likelihood_differs.sum()),
        "max_abs_likelihood_diff": float(graphs.max_abs_likelihood_diff.max()),
        "min_likelihood": {"original": float(edges["orig_likelihood"].min()),
                           "new": float(edges["new_likelihood"].min())},
        "smallest_alpha_where_edge_validity_differs": first,
        "edge_set_digest": {"original": edge_set_digest(original, actset), "new": edge_set_digest(new, actset)},
    }
    return summary, graphs, edges


def adm_equality(anomaly, ids_by_split, actset, filtered_actset, original, new, alphas):
    """같은 trace를 넣었을 때 ADM matrix / anomaly location이 같은지 (P_BEAR_RL.ADM 그대로 사용)."""
    import numpy as np
    import pandas as pd
    from rpbear.func_rpbear_final import P_BEAR_RL

    case_rows = {cid: df[["case:concept:name", "concept:name", "order"]].reset_index(drop=True)
                 for cid, df in anomaly.groupby("case:concept:name", observed=True)}
    results, per_case = {}, []
    for alpha in alphas:
        stats = {}
        for split_name, ids in ids_by_split.items():
            n_matrix = n_loc = 0
            for cid in ids:
                case = case_rows[cid]
                env = P_BEAR_RL(case, actset, filtered_actset, original, alpha=alpha)
                m_o, m_n = env.ADM(case, alpha, original), env.ADM(case, alpha, new)
                loc_o = np.where(np.sum(m_o, axis=0) > 1)[0].tolist() if len(m_o) else []
                loc_n = np.where(np.sum(m_n, axis=0) > 1)[0].tolist() if len(m_n) else []
                n_matrix += m_o == m_n
                n_loc += loc_o == loc_n
                if alpha == alphas[0]:
                    per_case.append({"split": split_name, "case": cid, "adm_matrix_equal": m_o == m_n,
                                     "anomaly_locations_equal": loc_o == loc_n,
                                     "orig_locations": loc_o, "new_locations": loc_n})
            stats[split_name] = {"n_cases": len(ids), "adm_matrix_equal": n_matrix,
                                 "anomaly_locations_equal": n_loc}
        results[f"alpha={alpha}"] = stats
    return results, pd.DataFrame(per_case)


# ------------------------------------------------------------
# D039: Expanded New NBG (W2) = Original clean log + model-only generated traces
# ------------------------------------------------------------

def build_expanded_nbgs(clean, model_only_log_path, actset):
    """W2 Expanded New NBG (D039, Credit official Method C).

    입력 = 관측 clean log (sim의 `clean`, 관측 빈도 그대로, Start/End 포함)
         + model-only generated trace log (각 1회, 관측 clean variant와 겹치면 안 됨).
    discover_NBG는 그대로 호출하고, 결과를 observed actset 순서로 재정렬 + anchor 검증한다.
    return: (ordered_nbgs, info). 규칙 위반 시 NBGValidationError.
    """
    import pandas as pd

    from rpbear.func_rpbear_final import discover_NBG

    from utils.data_utils import add_start_end_like_sim, read_log_like_sim

    cols = ["case:concept:name", "concept:name", "order"]
    obs = clean[cols].astype({"case:concept:name": str, "concept:name": str})
    gen = add_start_end_like_sim(read_log_like_sim(model_only_log_path, has_label=False))[cols]
    gen = gen.astype({"case:concept:name": str, "concept:name": str})

    def per_case(df):
        body = df[~df["concept:name"].isin(["Start", "End"])]
        return body.groupby("case:concept:name", sort=False)["concept:name"].agg(">>".join)

    obs_tr, gen_tr = per_case(obs), per_case(gen)
    if gen_tr.nunique() != len(gen_tr):
        raise NBGValidationError("model-only log has duplicate traces (each model-only trace must be used once)")
    overlap = set(gen_tr) & set(obs_tr)
    if overlap:
        raise NBGValidationError(f"model-only log contains {len(overlap)} observed clean variants")
    if set(obs["case:concept:name"]) & set(gen["case:concept:name"]):
        raise NBGValidationError("case id collision between clean log and model-only log")
    if not set(gen["concept:name"]) <= set(map(str, actset)):
        raise NBGValidationError(f"model-only activities outside observed actset: {sorted(set(gen['concept:name']) - set(map(str, actset)))}")

    log = pd.concat([obs, gen], ignore_index=True)
    raw = discover_NBG(log, vis=False)
    by_activity = dict(zip(log["concept:name"].unique(), raw))
    ordered = [by_activity[str(act)] for act in actset]
    mismatches = order_mismatches(ordered, actset)
    if mismatches:
        raise NBGValidationError(f"Expanded New NBG ordering mismatch: {mismatches}")
    info = {
        "nbg_input": "original_plus_model_only (W2, D039)",
        "observed_clean_cases": int(obs_tr.size), "observed_clean_variants": int(obs_tr.nunique()),
        "model_only_traces": int(gen_tr.size),
        "anchor_match": f"{len(actset)}/{len(actset)}",
        "weighting": "observed clean log with observed frequencies + each model-only generated trace once",
    }
    return ordered, info


def trace_edge_set(traces):
    """discover_NBG와 같은 규칙으로 trace들의 NBG edge 집합을 독립 계산한다 (검증용).

    trace는 Start/End 없는 activity tuple. edge = (graph activity, f"{a}_{i-p}", f"{b}_{i+1-p}"),
    p = graph activity가 trace(Start/End 포함)에서 나타나는 각 위치.
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


def build_method_c_nbgs(nbg_input, generate_dir, clean, actset):
    """dataset 설정 `new_nbg_input`에 따라 Method C New NBG를 만든다.

    "generated_only"           (D034, Small~Wide): generate/generated_normal_log.csv, 각 1회 (기존 build_new_nbgs 그대로)
    "original_plus_model_only" (D039, Credit)    : clean log + generate/model_only_generated_log.csv
    """
    from pathlib import Path

    generate_dir = Path(generate_dir)
    if nbg_input == "generated_only":
        return build_new_nbgs(generate_dir / "generated_normal_log.csv", actset)
    if nbg_input == "original_plus_model_only":
        return build_expanded_nbgs(clean, generate_dir / "model_only_generated_log.csv", actset)
    raise NBGValidationError(f"unknown new_nbg_input '{nbg_input}'")
