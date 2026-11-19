"""
trace_generation_utils — 생성 전략 판단, Petri Net 직접 나열, 관측 vs 생성 비교 (REF-001)

출처: enumerate_petri_net_language <- src/63_generated_nbg_rpbear.py (같은 알고리즘)

보수적인 원칙 (D034, 사용자 지시 2026-10-11):
    - 전략(strategy)이 config에 명시되지 않으면 trace를 만들지 않는다 (BLOCKED).
    - "exhaustive_acyclic"은 다음을 모두 만족할 때만 허용한다:
        Process Tree에 LOOP / 기타(OR 등) operator 없음, Petri Net에 cycle 없음,
        예상 언어 크기(raw) <= safety_max_traces.
    - loop가 있는 모델은 자동으로 exhaustive enumeration을 하지 않는다. 모델 구조를
      보고하고 멈추며, bounded generation 설계는 따로 결정한다.
    - safety_max_firings_per_transition / safety_max_trace_length / safety_max_traces는
      무한 탐색 방지용 안전장치이며, 연구 개념인 loop_num / loop_total과 다르다.
    - 잘림 / 가지치기 / dead end 수는 항상 기록한다.
"""

from collections import Counter

SUPPORTED_STRATEGIES = ("exhaustive_acyclic",)


def decide_strategy(tg_cfg, model_check):
    """return: {"allowed": bool, "strategy": ..., "reasons": [...]}"""
    strategy = tg_cfg.get("strategy")
    s = model_check["structure"]
    reasons = []
    if strategy is None:
        reasons.append("trace_generation.strategy가 설정되지 않음 (TODO: 모델 구조를 보고 결정)")
    elif strategy not in SUPPORTED_STRATEGIES:
        reasons.append(f"지원하지 않는 strategy '{strategy}' (구현된 것: {SUPPORTED_STRATEGIES})")
    else:
        if s["n_loop_operators"] > 0:
            reasons.append(f"Process Tree에 LOOP {s['n_loop_operators']}개 -> exhaustive 금지, bounded 설계 필요")
        if s["n_other_operators"] > 0:
            reasons.append(f"Process Tree에 지원하지 않는 operator {s['n_other_operators']}개")
        if s["petri_net_has_cycle"]:
            reasons.append("Petri Net에 cycle 있음 -> exhaustive 금지")
        est = s["estimated_raw_tree_language_size"]
        if est is None:
            reasons.append("예상 언어 크기를 계산할 수 없음")
        elif est > tg_cfg["safety_max_traces"]:
            reasons.append(f"예상 raw 언어 크기 {est} > safety_max_traces {tg_cfg['safety_max_traces']}")
    return {"allowed": not reasons, "strategy": strategy, "reasons": reasons}


def enumerate_petri_net_language(net, im, fm, safety_max_firings_per_transition,
                                 safety_max_trace_length, safety_max_traces):
    """initial marking -> final marking까지 가능한 firing sequence를 DFS로 모두 따라간다.

    - visible transition의 label만 trace에 넣고 tau(label None)는 넣지 않는다.
    - 단순 visited-marking pruning은 하지 않는다 (D025).
    """
    transitions = sorted(net.transitions, key=lambda t: (t.label is None, t.label or "", t.name))
    pre = {t: [(a.source, a.weight) for a in t.in_arcs] for t in transitions}
    post = {t: [(a.target, a.weight) for a in t.out_arcs] for t in transitions}
    final = {p: c for p, c in fm.items() if c > 0}

    stats = Counter()
    accepted = []
    stack = [({p: c for p, c in im.items() if c > 0}, (), {})]
    while stack:
        marking, trace, fired = stack.pop()
        stats["states_visited"] += 1
        if marking == final:
            accepted.append(trace)
            if len(accepted) >= safety_max_traces:
                stats["truncated_max_traces"] = 1
                break
            continue

        enabled = [t for t in transitions if all(marking.get(p, 0) >= w for p, w in pre[t])]
        if not enabled:
            stats["dead_end_markings"] += 1
            continue

        for t in reversed(enabled):
            if fired.get(t, 0) >= safety_max_firings_per_transition:
                stats["pruned_by_firing_bound"] += 1
                continue
            new_trace = trace + ((t.label,) if t.label is not None else ())
            if len(new_trace) > safety_max_trace_length:
                stats["pruned_by_length_bound"] += 1
                continue
            new_marking = dict(marking)
            for p, w in pre[t]:
                new_marking[p] -= w
                if new_marking[p] == 0:
                    del new_marking[p]
            for p, w in post[t]:
                new_marking[p] = new_marking.get(p, 0) + w
            new_fired = dict(fired)
            new_fired[t] = new_fired.get(t, 0) + 1
            stack.append((new_marking, new_trace, new_fired))

    for key in ["truncated_max_traces", "dead_end_markings", "pruned_by_firing_bound", "pruned_by_length_bound"]:
        stats.setdefault(key, 0)
    return accepted, dict(stats)


def bound_was_hit(enum_stats):
    return (enum_stats["truncated_max_traces"] > 0 or enum_stats["pruned_by_firing_bound"] > 0
            or enum_stats["pruned_by_length_bound"] > 0)


def compare_generated_observed(accepted, observed_counter, tree_language_set):
    generated_set = set(accepted)
    observed_set = set(observed_counter)
    return {
        "generated_trace_count (accepted firing sequences)": len(accepted),
        "generated_unique_trace_count": len(generated_set),
        "observed_clean_variant_count": len(observed_set),
        "process_tree_language_size": None if tree_language_set is None else len(tree_language_set),
        "generated_equals_observed": generated_set == observed_set,
        "generated_equals_process_tree_language": (None if tree_language_set is None
                                                   else generated_set == tree_language_set),
        "generated_only_count": len(generated_set - observed_set),
        "observed_only_count": len(observed_set - generated_set),
        "generated_only": [">>".join(t) for t in sorted(generated_set - observed_set)],
        "observed_only": [">>".join(t) for t in sorted(observed_set - generated_set)],
    }
