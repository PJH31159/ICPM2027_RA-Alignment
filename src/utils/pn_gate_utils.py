"""
pn_gate_utils — Method C + Petri Net Conformance Gate (C+PN), 탐색 variant (2026-11)

목적:
    W2 Expanded New NBG의 ADM이 "anomaly 없음"이라고 판단해도, 최종 trace가 공식 discovered Petri Net에서
    exact replay(initial marking -> visible / invisible transitions -> final marking)로 실행 가능할 때만
    repair를 "완료"로 인정한다.

현재 RP-BEAR 종료 구조 (rpbear/func_rpbear_final.py, 수정하지 않음):
    - P_BEAR_RL.step(): 행동 전 loc_anomalies(prev_state)가 비면 즉시 done=True. 행동 후 done = _is_done()
      (= loc_anomalies(state)가 비었는지). current_step > max_step 또는 len(state) < 4이면 game_over.
    - process_case_id_wrapper(): episode 반복 (train = num_epi, test = 최대 100). `done and not game_over`인 episode만
      성공으로 기록하고, test는 첫 성공에서 멈춘다. 성공 episode가 없으면 원래 trace(init_state)를 반환한다.

Gate (이 모듈):
    process_case_id_wrapper의 성공 조건을 `done and not game_over and gate(final trace)`로 바꾼 사본.
    - gate가 거부하면 그 episode는 완료로 인정하지 않고 다음 episode로 repair를 계속한다
      (ADM이 정상이면 anomaly 위치가 없어 같은 episode 안에서는 행동할 위치가 없다 — 위치를 새로 정하는 규칙은 만들지 않음).
    - max_step, train num_epi, test 최대 100 episode는 기존 값 그대로 (새 제한 없음). RL reward는 바꾸지 않는다.
    - pn_gate=None이면 core / sim과 똑같이 동작한다 (smoke에서 byte-identical 확인).

core 파일(func_rpbear_final.py, main_rpbear_final.py)과 sim.py는 수정하지 않는다. 아래 함수들은 core / sim 로직의
사본이며, 바뀐 곳은 `# [PN gate]` 표시가 있는 줄뿐이다.
.venv-rpbear에는 pm4py가 없으므로 Petri Net 판정은 PNML을 직접 읽는 PNMLAcceptor로 한다
(loop_bound_utils.make_petri_acceptor와 같은 의미; smoke에서 결과 동일성 확인).
"""

import random
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np


# ------------------------------------------------------------
# 1. Petri Net exact acceptor (pm4py 없이, PNML)
# ------------------------------------------------------------

class PNMLAcceptor:
    """공식 PNML의 Petri Net으로 trace가 initial marking -> final marking까지 정확히 실행 가능한지 판정.

    invisible transition(tau)은 closure로 자유롭게 발화, visible transition은 label이 trace activity와 같을 때 발화.
    loop_bound_utils.make_petri_acceptor와 같은 알고리즘. pickle 시 경로만 넘기고 worker에서 다시 읽는다.
    """

    def __init__(self, pnml_path):
        self.pnml_path = str(pnml_path)
        self._built = False
        self.n_calls = 0

    def __getstate__(self):
        return {"pnml_path": self.pnml_path}

    def __setstate__(self, state):
        self.__init__(state["pnml_path"])

    def _build(self):
        root = ET.parse(self.pnml_path).getroot()
        net = root.find("net")
        page = net.find("page")
        im, label, pre, post = {}, {}, {}, {}
        for p in page.findall("place"):
            m = p.find("initialMarking/text")
            if m is not None and int(m.text) > 0:
                im[p.get("id")] = int(m.text)
        for t in page.findall("transition"):
            tid = t.get("id")
            ts = t.find("toolspecific")
            label[tid] = None if (ts is not None and ts.get("activity") == "$invisible$") else t.find("name/text").text
            pre[tid], post[tid] = {}, {}
        for a in page.findall("arc"):
            s, d = a.get("source"), a.get("target")
            w = a.find("inscription/text")
            w = int(w.text) if w is not None else 1
            if s in label:
                post[s][d] = post[s].get(d, 0) + w
            else:
                pre[d][s] = pre[d].get(s, 0) + w
        fm = {}
        for p in net.findall("finalmarkings/marking/place"):
            fm[p.get("idref")] = int(p.find("text").text)
        self.taus = [t for t, l in label.items() if l is None]
        self.by_label = {}
        for t, l in label.items():
            if l is not None:
                self.by_label.setdefault(l, []).append(t)
        self.pre, self.post = pre, post
        self.m0 = im
        self.final_key = frozenset(fm.items())
        self.info = {"places": len({p.get("id") for p in page.findall("place")}), "transitions": len(label),
                     "invisible": len(self.taus), "arcs": len(page.findall("arc"))}
        self._built = True

    @staticmethod
    def _key(m):
        return frozenset((p, n) for p, n in m.items() if n > 0)

    def _fire(self, m, t):
        if all(m.get(p, 0) >= w for p, w in self.pre[t].items()):
            n = dict(m)
            for p, w in self.pre[t].items():
                n[p] -= w
            for p, w in self.post[t].items():
                n[p] = n.get(p, 0) + w
            return n
        return None

    def _closure(self, markings, max_states=100_000):
        seen = {self._key(m): m for m in markings}
        q = deque(seen.values())
        while q:
            m = q.popleft()
            for t in self.taus:
                n = self._fire(m, t)
                if n is not None:
                    k = self._key(n)
                    if k not in seen:
                        seen[k] = n
                        q.append(n)
                        if len(seen) > max_states:
                            raise RuntimeError("tau closure too large")
        return list(seen.values())

    def accepts(self, trace):
        if not self._built:
            self._build()
        self.n_calls += 1
        states = self._closure([self.m0])
        for a in trace:
            nxt = {}
            for m in states:
                for t in self.by_label.get(a, []):
                    n = self._fire(m, t)
                    if n is not None:
                        nxt[self._key(n)] = n
            if not nxt:
                return False
            states = self._closure(list(nxt.values()))
        return any(self._key(m) == self.final_key for m in states)

    def accepts_case(self, case_df):
        """RP-BEAR state DataFrame (Start / End 포함) -> trace 판정."""
        acts = [a for a in case_df["concept:name"].tolist() if a not in ("Start", "End")]
        return self.accepts(tuple(acts))


# ------------------------------------------------------------
# 2. core process_case_id_wrapper 사본 + gate
# ------------------------------------------------------------

def gated_process_case_id_wrapper(case_id_for_processing, p_anomaly_nolabel, p_filtered_actset, p_act_freq, p_actset,
                                  p_NBGs, p_num_epi, alpha, agent, train_mode=True, case_type='val', max_step=8,
                                  pn_gate=None):
    from rpbear.func_rpbear_final import P_BEAR_RL, discover_pattern

    obs_case = p_anomaly_nolabel.loc[p_anomaly_nolabel['case:concept:name'] == case_id_for_processing].reset_index(drop=True)
    env = P_BEAR_RL(obs_case, p_actset, p_filtered_actset, p_NBGs, alpha=alpha)

    episodes = p_num_epi if train_mode else 100
    all_episode_histories = []
    gate_rejected = 0  # [PN gate] ADM normal인데 Petri Net이 거부한 episode 수

    for _ in range(episodes):
        state = env.reset()
        steps = 0
        done = False
        game_over = False
        episode_history = []

        while not done and not game_over:
            action = agent.select_action(state, greedy_only=0)
            next_state, reward, done, game_over, info, loc, label_act, rework = env.step(action,
                max_step=max_step,
                deterministic=0
            )
            if train_mode:
                agent.update_q_value(state, action, reward, next_state)
                agent.decay_exploration_rate()
            steps += 1
            if label_act is not None:
                episode_history.append((state.copy(), env.action_labels[action].split('_')[0], loc, label_act, rework, reward, next_state.copy()))
            state = next_state

        if done and not game_over:
            # [PN gate] ADM이 anomaly 없음이라고 해도 Petri Net exact replay를 통과해야 완료로 인정
            if pn_gate is not None and not pn_gate.accepts_case(next_state):
                gate_rejected += 1
                continue
            adm_last = env.ADM(next_state, 0, p_NBGs)
            if len(adm_last) > 0 and len(adm_last[0]) > 0:
                score = np.sum(adm_last) / (len(adm_last) * len(adm_last[0]))
            else:
                score = 0
            all_episode_histories.append((steps - 1, score, episode_history))
            if not train_mode:
                break

    if len(all_episode_histories) > 0:
        positive_histories = [hist for hist in all_episode_histories if hist[0] >= 0]
        if len(positive_histories) > 0:
            min_step = min(h[0] for h in positive_histories)
            shortest_episodes = [hist for hist in positive_histories if hist[0] == min_step]
            best_episode = min(shortest_episodes, key=lambda x: x[1])
            if len(best_episode[2]) > 0:
                env.state = best_episode[2][-1][-1]
                env.state['predict_patterns'] = discover_pattern(best_episode[2])
            else:
                env.state = env.init_state.copy()
                env.state['predict_patterns'] = ''
            case_result_df = env.state.copy()
            rl_steps = best_episode[0]
            rl_success = 1
        else:
            state = env.init_state.copy()
            state['predict_patterns'] = ''
            case_result_df = state
            rl_steps = 0
            rl_success = 0
    else:
        state = env.init_state.copy()
        state['predict_patterns'] = ''
        case_result_df = state
        rl_steps = 0
        rl_success = 0

    case_result_df['type'] = case_type
    case_result_df['rl_steps'] = rl_steps
    case_result_df['rl_success'] = rl_success
    if pn_gate is not None:  # [PN gate] 진단용 열 (gate off일 때는 core와 같은 열만)
        case_result_df['pn_gate_rejected_episodes'] = gate_rejected
    return case_result_df


# ------------------------------------------------------------
# 3. sim._process_case_with_memory / _run_phase / run_rpbear 사본 + gate
# ------------------------------------------------------------

def gated_process_case_with_memory(case_id, case_to_trace, repair_memory, p_anomaly_nolabel, p_filtered_actset, p_act_freq,
                                   p_actset, p_NBGs, p_num_epi, alpha, agent, train_mode, case_type, max_step, pn_gate=None):
    t0 = time.perf_counter()
    trace_str = case_to_trace[case_id]
    if trace_str in repair_memory:
        result = repair_memory[trace_str].copy()
        result["type"] = case_type
        result["case:concept:name"] = case_id
        return case_id, result, 0, (time.perf_counter() - t0) * 1000
    result = gated_process_case_id_wrapper(
        case_id_for_processing=case_id, p_anomaly_nolabel=p_anomaly_nolabel, p_filtered_actset=p_filtered_actset,
        p_act_freq=p_act_freq, p_actset=p_actset, p_NBGs=p_NBGs, p_num_epi=p_num_epi, alpha=alpha, agent=agent,
        train_mode=train_mode, case_type=case_type, max_step=max_step, pn_gate=pn_gate)
    repair_memory[trace_str] = result.copy()
    return case_id, result, 1, (time.perf_counter() - t0) * 1000


def _run_phase_gated(caseids, case_type, train_mode, case_to_trace, repair_memory, anomaly_nolabel, filtered_actset,
                     act_freq, actset, NBGs, num_epi, alpha, agent, max_step, gt_traces, use_multiprocessing, num_cores,
                     seed, verbose, pn_gate):
    from functools import partial
    from multiprocessing import Pool

    from sim import _report_progress, _timing_row, _worker_init

    func = partial(gated_process_case_with_memory, case_to_trace=case_to_trace, repair_memory=repair_memory,
                   p_anomaly_nolabel=anomaly_nolabel, p_filtered_actset=filtered_actset, p_act_freq=act_freq,
                   p_actset=actset, p_NBGs=NBGs, p_num_epi=num_epi, alpha=alpha, agent=agent, train_mode=train_mode,
                   case_type=case_type, max_step=max_step, pn_gate=pn_gate)
    results, case_timing = [], []
    rl_count = success_count = 0
    start_time = time.time()
    if use_multiprocessing:
        with Pool(processes=num_cores, initializer=_worker_init, initargs=(seed,)) as pool:
            for idx, (case_id, result, is_run, elapsed_ms) in enumerate(pool.imap_unordered(func, caseids), start=1):
                results.append(result)
                case_timing.append(_timing_row(case_id, case_type, is_run, elapsed_ms))
                rl_count += is_run
                success_count = _report_progress(result, case_id, gt_traces, idx, len(caseids), success_count,
                                                 case_type, is_run, verbose)
    else:
        for idx, case_id in enumerate(caseids, start=1):
            case_id_out, result, is_run, elapsed_ms = func(case_id)
            results.append(result)
            case_timing.append(_timing_row(case_id_out, case_type, is_run, elapsed_ms))
            rl_count += is_run
            success_count = _report_progress(result, case_id_out, gt_traces, idx, len(caseids), success_count,
                                             case_type, is_run, verbose)
    total_ms = (time.time() - start_time) * 1000
    per_case_ms = total_ms / len(caseids) if len(caseids) > 0 else 0.0
    return results, rl_count, success_count, total_ms, per_case_ms, case_timing


def run_rpbear_gated(data_name, num_epi, alpha, val_ratio, seed, max_step, num_cores, use_multiprocessing, nbgs_override,
                     input_paths, pn_gate=None, limit=None, verbose=False):
    """sim.run_rpbear와 같은 순서(seed -> load -> NBG -> split -> agent -> val -> test). 저장은 호출하는 쪽에서 한다.

    limit: smoke 전용. split 후 val / test 앞의 N개 case만 처리 (official run에서는 None).
    return: (repaired_log df, stats dict, case_timing df, agent with q_table snapshot)
    """
    import pandas as pd
    from multiprocessing import Manager, cpu_count

    from rpbear.func_rpbear_final import QLearningAgent, discover_NBG
    from sim import _load_and_prepare_data, _prepare_case_split, _timing_summary

    overall_start = time.time()
    random.seed(seed)
    np.random.seed(seed)
    if num_cores is None:
        num_cores = max(1, cpu_count() - 1)
    clean, anomaly, gt_traces, actset, filtered_actset, act_freq = _load_and_prepare_data(data_name, input_paths)
    nbg_start = time.time()
    if nbgs_override is None:
        NBGs = discover_NBG(clean, vis=False)
    else:
        if len(nbgs_override) != len(actset):
            raise ValueError(f"nbgs_override has {len(nbgs_override)} graphs, but actset has {len(actset)} activities")
        NBGs = list(nbgs_override)
    nbg_ms = (time.time() - nbg_start) * 1000
    anomaly_nolabel, case_to_trace, val_caseids, test_caseids = _prepare_case_split(anomaly, val_ratio)
    if limit is not None:
        val_caseids, test_caseids = val_caseids[:limit], test_caseids[:limit]

    manager = Manager() if use_multiprocessing else None
    if use_multiprocessing:
        shared_q_table = manager.dict()
        repair_memory = manager.dict()
    else:
        shared_q_table = {}
        repair_memory = {}
    agent = QLearningAgent(filtered_actset=filtered_actset, act_freq=act_freq, learning_rate=0.1, discount_factor=0.9,
                           exploration_rate=0.5, exploration_decay_rate=0.0001)
    agent.q_table = shared_q_table
    common = dict(case_to_trace=case_to_trace, repair_memory=repair_memory, anomaly_nolabel=anomaly_nolabel,
                  filtered_actset=filtered_actset, act_freq=act_freq, actset=actset, NBGs=NBGs, num_epi=num_epi,
                  alpha=alpha, agent=agent, max_step=max_step, gt_traces=gt_traces,
                  use_multiprocessing=use_multiprocessing, num_cores=num_cores, seed=seed, verbose=verbose,
                  pn_gate=pn_gate)
    try:
        val_results, val_rl, val_succ, val_ms, val_pc, val_timing = _run_phase_gated(
            caseids=val_caseids, case_type="val", train_mode=True, **common)
        q_snapshot = dict(shared_q_table)
        test_results, test_rl, test_succ, test_ms, test_pc, test_timing = _run_phase_gated(
            caseids=test_caseids, case_type="test", train_mode=False, **common)
        q_table_size = len(agent.q_table)
    finally:
        if manager is not None:
            manager.shutdown()
    val_df = pd.concat(val_results, ignore_index=True) if val_results else pd.DataFrame()
    test_df = pd.concat(test_results, ignore_index=True) if test_results else pd.DataFrame()
    repaired = pd.concat([val_df, test_df], ignore_index=True)
    total = time.time() - overall_start
    stats = {"data": data_name, "num_epi": num_epi, "alpha": alpha, "val_ratio": val_ratio, "seed": seed,
             "max_step": max_step, "use_multiprocessing": use_multiprocessing, "num_cores": num_cores,
             "n_val_cases": len(val_caseids), "val_rl_runs": val_rl, "n_test_cases": len(test_caseids),
             "test_inference_runs": test_rl, "nbg_ms": round(nbg_ms, 2), "val_total_ms": round(val_ms, 2),
             "val_per_case_ms": round(val_pc, 2), "test_total_ms": round(test_ms, 2), "test_per_case_ms": round(test_pc, 2),
             "total_sec": round(total, 4), "training_time_sec": round(val_ms / 1000, 4),
             "total_repair_time_sec": round(test_ms / 1000, 4), "total_run_time_sec": round(total, 4),
             **_timing_summary(val_timing, "val"), **_timing_summary(test_timing, "test"),
             "q_table_size": q_table_size, "nbg_source": "override" if nbgs_override is not None else None,
             "pn_gate": pn_gate.pnml_path if pn_gate is not None else None, "limit": limit,
             "note": "val/test p_anomaly omitted (invalid for Credit numeric case ids; use evaluate)"}
    timing = pd.DataFrame(val_timing + test_timing, columns=["case_id", "phase", "path", "repair_time_ms"])
    agent.q_table = q_snapshot
    return repaired, stats, timing, agent


def replay_test_phase_gated(prep, agent, repair_memory, alpha, num_epi, max_step, pn_gate, seed):
    """timing_utils.replay_test_phase와 같은 순차 Test-only 재실행 (gate 포함)."""
    from functools import partial

    from sim import _timing_row

    random.seed(seed)
    np.random.seed(seed)
    func = partial(gated_process_case_with_memory, case_to_trace=prep["case_to_trace"], repair_memory=repair_memory,
                   p_anomaly_nolabel=prep["anomaly_nolabel"], p_filtered_actset=prep["filtered_actset"],
                   p_act_freq=prep["act_freq"], p_actset=prep["actset"], p_NBGs=prep["nbgs"], p_num_epi=num_epi,
                   alpha=alpha, agent=agent, train_mode=False, case_type="test", max_step=max_step, pn_gate=pn_gate)
    results, timing = [], []
    t0 = time.perf_counter()
    for case_id in prep["test_ids"]:
        cid, result, is_run, elapsed_ms = func(case_id)
        results.append(result)
        timing.append(_timing_row(cid, "test", is_run, elapsed_ms))
    return results, timing, (time.perf_counter() - t0) * 1000


class AcceptAllGate:
    """smoke 전용: 항상 통과하는 gate. gate 경로 자체가 결과를 바꾸지 않는지(conformant면 기존처럼 종료) 확인."""

    pnml_path = "accept-all (smoke only)"

    def accepts_case(self, case_df):
        return True


def case_df_from_trace(trace):
    """activity tuple (Start / End 없이) -> RP-BEAR case DataFrame (Start / End / order 포함)."""
    import pandas as pd

    acts = ["Start"] + list(trace) + ["End"]
    return pd.DataFrame({"case:concept:name": "x", "concept:name": acts, "order": range(len(acts))})


def adm_normal(trace, actset, filtered_actset, nbgs, alpha=0.0):
    """official P_BEAR_RL.ADM으로 trace에 anomaly 위치(2표 이상)가 없는지."""
    from rpbear.func_rpbear_final import P_BEAR_RL

    case = case_df_from_trace(trace)
    env = P_BEAR_RL(case, actset, filtered_actset, nbgs, alpha=alpha)
    m = env.ADM(case, alpha, nbgs)
    return (len(np.where(np.sum(m, axis=0) > 1)[0]) == 0) if len(m) else True
