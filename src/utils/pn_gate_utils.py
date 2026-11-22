"""Petri-net gate of RA-Alignment (training and inference).

Purpose: even when the ADM of the (L_N-only) NBG reports "no anomaly", a repair is accepted
as complete only if the final trace replays exactly on the discovered Petri net
(initial marking -> visible / invisible transitions -> final marking).

RP-BEAR termination (rpbear/func_rpbear_final.py, unchanged):
    - P_BEAR_RL.step(): done=True immediately if loc_anomalies(prev_state) is empty before the action;
      after the action done = _is_done() (loc_anomalies(state) is empty).
      game_over if current_step > max_step or len(state) < 4.
    - process_case_id_wrapper(): repeats episodes (train = num_epi, test = at most 100). Only episodes with
      `done and not game_over` count as successes; test stops at the first success. Without a successful
      episode the original trace (init_state) is returned.

Gate (this module): a copy of process_case_id_wrapper whose success condition is
`done and not game_over and gate(final trace)`.
    - If the gate rejects, the episode is not accepted and repair continues with the next episode
      (when the ADM is normal there is no anomaly location left to act on within the same episode;
      no new location rule is introduced).
    - max_step, train num_epi and the test limit of 100 episodes are unchanged; the RL reward is unchanged.
    - With pn_gate=None the behaviour is identical to the core / sim (checked byte-identical in the smoke stage).

The functions below are copies of the core / sim logic; the only changed lines are marked `# [PN gate]`.
.venv-rpbear has no PM4Py, so Petri-net acceptance uses PNMLAcceptor, which reads the PNML directly
(same semantics as loop_bound_utils.make_petri_acceptor; equality checked in the smoke stage).
"""

import random
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np


# ------------------------------------------------------------
# 1. Exact Petri-net acceptor (PNML, without PM4Py)
# ------------------------------------------------------------

class PNMLAcceptor:
    """Decide whether a trace runs exactly from the initial to the final marking of the official PNML Petri net.

    Invisible transitions (tau) fire freely via closure; visible transitions fire when their label equals the
    trace activity. Same algorithm as loop_bound_utils.make_petri_acceptor. When pickled, only the path is
    passed and the net is re-read in the worker.
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
        """RP-BEAR state DataFrame (with Start / End) -> acceptance decision."""
        acts = [a for a in case_df["concept:name"].tolist() if a not in ("Start", "End")]
        return self.accepts(tuple(acts))


# ------------------------------------------------------------
# 2. Copy of core process_case_id_wrapper + gate
# ------------------------------------------------------------

def gated_process_case_id_wrapper(case_id_for_processing, p_anomaly_nolabel, p_filtered_actset, p_act_freq, p_actset,
                                  p_NBGs, p_num_epi, alpha, agent, train_mode=True, case_type='val', max_step=8,
                                  pn_gate=None):
    from rpbear.func_rpbear_final import P_BEAR_RL, discover_pattern

    obs_case = p_anomaly_nolabel.loc[p_anomaly_nolabel['case:concept:name'] == case_id_for_processing].reset_index(drop=True)
    env = P_BEAR_RL(obs_case, p_actset, p_filtered_actset, p_NBGs, alpha=alpha)

    episodes = p_num_epi if train_mode else 100
    all_episode_histories = []
    gate_rejected = 0  # [PN gate] number of episodes that the ADM accepts but the Petri net rejects

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
            # [PN gate] even if the ADM reports no anomaly, the trace must pass exact Petri-net replay
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
    if pn_gate is not None:  # [PN gate] diagnostic columns (gate off: same columns as the core)
        case_result_df['pn_gate_rejected_episodes'] = gate_rejected
    return case_result_df


# ------------------------------------------------------------
# 3. Copies of sim._process_case_with_memory / _run_phase / run_rpbear + gate
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
    """Same order as sim.run_rpbear (seed -> load -> NBG -> split -> agent -> val -> test). The caller saves outputs.

    limit: smoke only; process the first N val / test cases after the split (None in official runs).
    Returns (repaired_log df, stats dict, case_timing df, agent with q_table snapshot).
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
    """Sequential test-only replay as in timing_utils.replay_test_phase, with the gate."""
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
    """Smoke only: a gate that always accepts. Checks that the gate path itself does not change results."""

    pnml_path = "accept-all (smoke only)"

    def accepts_case(self, case_df):
        return True


def case_df_from_trace(trace):
    """Activity tuple (without Start / End) -> RP-BEAR case DataFrame (with Start / End / order)."""
    import pandas as pd

    acts = ["Start"] + list(trace) + ["End"]
    return pd.DataFrame({"case:concept:name": "x", "concept:name": acts, "order": range(len(acts))})


def adm_normal(trace, actset, filtered_actset, nbgs, alpha=0.0):
    """Whether the official P_BEAR_RL.ADM finds no anomaly location (2 or more votes) in the trace."""
    from rpbear.func_rpbear_final import P_BEAR_RL

    case = case_df_from_trace(trace)
    env = P_BEAR_RL(case, actset, filtered_actset, nbgs, alpha=alpha)
    m = env.ADM(case, alpha, nbgs)
    return (len(np.where(np.sum(m, axis=0) > 1)[0]) == 0) if len(m) else True
