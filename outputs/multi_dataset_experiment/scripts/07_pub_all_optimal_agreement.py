"""07_pub_all_optimal_agreement.py — Pub: Method B all-optimal 집합 + B–C / B–C+PN agreement (탐색, .venv)

왜: Credit에서 정한 main metric(B–C+PN, tau-excluded all-optimal agreement)을 Pub에 같은 정의로 계산한다.
    정의와 search 코드는 Credit alignment_all_optimal_analysis/scripts/credit_all_optimal.py 와 같다
    (repair_from_alignment / net_struct / dev_optimal 을 그 파일에서 그대로 실행해 가져온다).
      exact          : X repaired == B repaired (official metric과 같은 정의)
      native-cost    : X ∈ B optimal 집합, PM4Py 기본 cost (log / visible model 10000, tau 1, sync 0), 직접 exact search
                       + PM4Py generator_dijkstra_less_memory 집합과 비교(교차 확인, 판정에는 직접 search 사용)
      tau-excluded   : X ∈ B optimal 집합, deviation-only cost (log / visible model 1, tau 0, sync 0)  <- main
    multiple-optimal case = optimal repaired trace 집합 크기 >= 2.
    accuracy(EMR)는 보조로만 같이 저장한다.

입력: B = --b-dir 의 *_alignment_per_case.csv (test), C / C+PN repaired log (선택)
출력: --out 폴더 (case_summary.csv, optimal_repaired_traces.csv, agreement_summary.json)
실행 예:
  .venv\\Scripts\\python outputs\\multi_dataset_experiment\\scripts\\07_pub_all_optimal_agreement.py \\
      --b-dir outputs/simulation/Pub/smoke_b --c-log ... --cpn-log ... --out outputs/simulation/Pub/alignment_all_optimal_analysis/smoke
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

import argparse  # noqa: E402
import heapq  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pm4py  # noqa: E402
from pm4py.algo.conformance.alignments.petri_net.variants import generator_dijkstra_less_memory as gen_lm  # noqa: E402
from pm4py.objects.log.obj import Event, Trace  # noqa: E402
from pm4py.util.lp import solver as lp_solver  # noqa: E402

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
CREDIT_SCRIPT = PROJECT_ROOT / "outputs/simulation/Credit/alignment_all_optimal_analysis/scripts/credit_all_optimal.py"


def credit_defs():
    """Credit script의 repair_from_alignment / net_struct / fire / dev_optimal 정의를 그대로 가져온다."""
    src = CREDIT_SCRIPT.read_text(encoding="utf-8")
    ns = {"heapq": heapq, "Counter": Counter, "Event": Event, "Trace": Trace,
          "alignments_algo": None, "VARIANT": None, "gen_lm": gen_lm, "gen_nh": gen_lm}
    exec(compile(src[src.index("def trace_obj"):src.index("# ---------------------------------------------------------------- toy")],
                 "credit_all_optimal_defs", "exec"), ns)
    return ns


def load_repaired(path, ids):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["type"] == "test"]
    df = df[~df["concept:name"].isin(["Start", "End"])]
    tr = df.groupby("case:concept:name", sort=False)["concept:name"].agg(tuple)
    return {c: tr[c] for c in ids if c in tr.index}


def gt_traces(path):
    df = pd.read_csv(path, usecols=["Case", "Activity"], dtype=str)
    return df.groupby("Case", sort=False)["Activity"].agg(tuple).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--c-log")
    ap.add_argument("--cpn-log")
    ap.add_argument("--gt", default=None, help="ground-truth normal log (default: <data root>/encoded_normal/mccloud.csv)")
    ap.add_argument("--data-root", dest="data_root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-states", type=int, default=5_000_000)
    # 실행 방식 옵션 (판정 정의는 그대로): case마다 checkpoint(jsonl)에 저장하고, 다시 실행하면 끝난 case는 건너뛴다.
    #   --shard K/N : test case 중 ids[K::N]만 계산 (병렬 실행용, 결과는 checkpoint에만 저장)
    #   --merge     : 모든 checkpoint를 합쳐 최종 파일을 쓴다 (모든 test case가 있어야 함)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--merge", action="store_true")
    # --skip-gen-check (2026-11-16, BPIC): PM4Py generator_dijkstra_less_memory 교차검증(gen_less_memory_eq_native)을 건너뛴다.
    #   이 보조 검증은 state 상한 없이 모든 alignment를 나열해 BPIC에서 process당 ~59 GB까지 올라가 PC가 멈췄다
    #   (Windows Event 2004). agreement 판정(dev_optimal, --max-states)은 그대로이며, 건너뛴 case는 gen_* 열이 None.
    ap.add_argument("--skip-gen-check", action="store_true")
    # --resource-skip FILE (2026-11-16, BPIC): 첫 열(tab 구분)이 case id인 파일. 이 case들은 1 process 단독 실행에서도
    #   메모리 위험 수준(watchdog)에 도달해 탐색을 끝내지 못한 case다. 탐색 없이 resource_limit_hit=True로 기록하고
    #   all-optimal 집합은 비어 있는 것으로 둔다 -> all-optimal agreement에서 불일치 (하한). exact agreement는 trace 비교라 정상 계산.
    #   state-limit(--max-states) 정의와는 별개 범주이며 state_limit_hit에 섞지 않는다.
    ap.add_argument("--resource-skip", default=None)
    a = ap.parse_args()
    if a.gt is None:
        a.gt = str(_data_root(a.data_root) / "encoded_normal" / "mccloud.csv")
    ns = credit_defs()
    import psutil  # noqa: E402  (case별 메모리 기록용)
    _proc = psutil.Process()
    bdir, out = PROJECT_ROOT / a.b_dir, PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    net, im, fm = pm4py.read_pnml(str(next(bdir.glob("*_petri_net.pnml"))))
    st = ns["net_struct"](net, im, fm)
    alg = pd.read_csv(next(bdir.glob("*_alignment_per_case.csv")), dtype={"case:concept:name": str})
    alg = alg[alg.type == "test"].set_index("case:concept:name")
    ids = list(alg.index)
    X = {}
    if a.c_log:
        X["C"] = load_repaired(PROJECT_ROOT / a.c_log, ids)
    if a.cpn_log:
        X["CPN"] = load_repaired(PROJECT_ROOT / a.cpn_log, ids)
    gt = gt_traces(a.gt)
    ck_dir = out / "checkpoint"
    ck_dir.mkdir(exist_ok=True)
    done = {}
    for f in sorted(ck_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["row"]["case_id"]] = rec
    todo = ids
    if a.shard:
        kk, nn = map(int, a.shard.split("/"))
        todo = ids[kk::nn]
    ck_file = ck_dir / (f"shard{a.shard.replace('/', 'of')}.jsonl" if a.shard else "main.jsonl")
    if a.merge:
        missing = [c for c in ids if c not in done]
        if missing:
            print(f"[ERROR] merge: {len(missing)}/{len(ids)} case 미완료 (예: {missing[:5]})")
            sys.exit(2)
    print(f"checkpoint: {len(done)} case 완료됨, 이번 대상 {sum(c not in done for c in todo)}/{len(todo)}", flush=True)
    res_skip = set()
    if a.resource_skip and Path(a.resource_skip).exists():
        res_skip = {ln.split("	")[0].strip() for ln in Path(a.resource_skip).read_text(encoding="utf-8").splitlines() if ln.strip()}
    rows, trows, t_all = [], [], time.time()
    elapsed_prev = sum(r["row"].get("_case_sec", 0) for r in done.values())
    for k, c in enumerate(ids if a.merge else todo, 1):
        if c in done:
            if a.merge or not a.shard:
                rows.append({kx: v for kx, v in done[c]["row"].items() if kx != "_case_sec"})
                trows.extend(done[c]["trows"])
            continue
        t_case = time.time()
        an = tuple(alg.loc[c, "original_trace"].split(">>"))
        b = tuple(alg.loc[c, "repaired_trace"].split(">>"))
        cost_b = int(alg.loc[c, "alignment_cost"])
        rskip = c in res_skip
        if rskip:
            xcost, S_ex, xpops, ms_ex, dcost, S_dev, dpops, ms_dev = None, set(), None, None, None, set(), None, None
        else:
            t0 = time.perf_counter()
            xcost, S_ex, xpops = ns["dev_optimal"](an, st, 10000, 1, max_states=a.max_states)
            ms_ex = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dcost, S_dev, dpops = ns["dev_optimal"](an, st, 1, 0, max_states=a.max_states)
            ms_dev = (time.perf_counter() - t0) * 1000
        if a.skip_gen_check or rskip:
            als, ms_gen, peak_kb, S_gen = None, None, None, None
        else:
            tracemalloc.start()
            t0 = time.perf_counter()
            als = list(gen_lm.apply(ns["trace_obj"](an), net, im, fm))
            ms_gen = (time.perf_counter() - t0) * 1000
            peak_kb = tracemalloc.get_traced_memory()[1] / 1024
            tracemalloc.stop()
            S_gen = {ns["repair_from_alignment"](x["alignment"])[0] for x in als}
        S_ex, S_dev = set(S_ex or ()), set(S_dev or ())
        g = gt.get(c)
        r = {"case_id": c, "anomaly_trace": " ".join(an), "ground_truth": " ".join(g) if g else None, "B": " ".join(b),
             "B_cost": cost_b, "exact_cost": xcost, "exact_cost_eq_B": xcost == cost_b, "state_limit_hit": (xcost is None or dcost is None) and not rskip,
             "resource_limit_hit": rskip,
             "native_n_optimal": len(S_ex), "tau_excl_n_optimal": len(S_dev), "tau_excl_deviations": dcost,
             "B_in_native_set": b in S_ex, "B_in_tau_excl_set": b in S_dev, "native_subset_of_tau_excl": S_ex <= S_dev,
             "gen_less_memory_n_alignments": None if als is None else len(als),
             "gen_less_memory_n_optimal": None if S_gen is None else len(S_gen),
             "gen_less_memory_eq_native": None if S_gen is None else S_gen == S_ex, "B_correct": b == g,
             "trace_len": len(an), "rss_mb_after_case": round(_proc.memory_info().rss / 2**20, 1),
             "peak_wset_mb_process": round(getattr(_proc.memory_info(), "peak_wset", 0) / 2**20, 1),
             "ms_native_search": None if ms_ex is None else round(ms_ex, 2), "native_pops": xpops, "ms_tau_excl_search": None if ms_dev is None else round(ms_dev, 2),
             "tau_excl_pops": dpops, "ms_gen_less_memory": None if ms_gen is None else round(ms_gen, 2),
             "peak_kb_gen": None if peak_kb is None else round(peak_kb, 1)}
        for m, tr in X.items():
            x = tr.get(c)
            r[f"{m}"] = " ".join(x) if x is not None else None
            r[f"{m}_exact_eq_B"] = x == b
            r[f"{m}_in_native_set"] = x in S_ex
            r[f"{m}_in_tau_excl_set"] = x in S_dev
            r[f"{m}_correct"] = x == g
        rows.append(r)
        tr_c = []
        for name, S in (("native", S_ex), ("tau_excluded", S_dev), ("gen_less_memory", S_gen)):
            for t in sorted(S or ()):  # S_gen = None (--skip-gen-check) -> 행 없음
                tr_c.append({"case_id": c, "set": name, "trace": " ".join(t), "is_B": t == b, "is_GT": t == g,
                             **{f"is_{m}": t == tr.get(c) for m, tr in X.items()}})
        trows.extend(tr_c)
        with ck_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"row": {**r, "_case_sec": round(time.time() - t_case, 3)}, "trows": tr_c},
                                ensure_ascii=False, default=str) + "\n")
        if k % 50 == 0 or k == len(todo):
            print(f"{k}/{len(todo)} {time.time() - t_all:.1f}s max_native_set {max(x['native_n_optimal'] for x in rows)} "
                  f"max_tau_excl_set {max(x['tau_excl_n_optimal'] for x in rows)}", flush=True)
    if a.shard:
        print(f"[shard {a.shard}] done -> {ck_file}")
        return
    if len(rows) != len(ids):
        print(f"[ERROR] {len(rows)}/{len(ids)} case만 있음")
        sys.exit(2)
    T = pd.DataFrame(rows)
    T.to_csv(out / "case_summary.csv", index=False)
    pd.DataFrame(trows).to_csv(out / "optimal_repaired_traces.csv", index=False)
    n = len(T)
    summ = {"cases": n, "lp_solver_in_process": lp_solver.DEFAULT_LP_SOLVER_VARIANT, "b_dir": a.b_dir,
            "c_log": a.c_log, "cpn_log": a.cpn_log, "total_sec": round(elapsed_prev + time.time() - t_all, 2),
            "total_sec_note": "checkpoint case 시간의 합 + 이번 실행 시간 (shard 병렬 실행이면 wall time이 아니라 CPU 순차 합에 가까움)",
            "checks": {"exact_cost_eq_B": int(T.exact_cost_eq_B.sum()), "B_in_native_set": int(T.B_in_native_set.sum()),
                       "B_in_tau_excl_set": int(T.B_in_tau_excl_set.sum()), "native_subset_of_tau_excl": int(T.native_subset_of_tau_excl.sum()),
                       "state_limit_hit": int(T.state_limit_hit.sum()), "gen_less_memory_eq_native": int((T.gen_less_memory_eq_native == True).sum()),  # noqa: E712
                       "gen_less_memory_checked": int(T.gen_less_memory_eq_native.notna().sum()),
                       "resource_limit_hit": int((T.get("resource_limit_hit", pd.Series(dtype=object)) == True).sum())},  # noqa: E712
            "multiple_optimal_cases": {"native": int((T.native_n_optimal >= 2).sum()), "tau_excluded": int((T.tau_excl_n_optimal >= 2).sum()),
                                       "native_set_size_dist": T.native_n_optimal.value_counts().sort_index().to_dict(),
                                       "tau_excl_set_size_dist": T.tau_excl_n_optimal.value_counts().sort_index().to_dict()},
            "agreement": {}, "accuracy_aux": {"B": f"{int(T.B_correct.sum())}/{n}"},
            "runtime_ms": {k: {"mean": round(float(pd.to_numeric(T[k]).mean()), 2), "median": round(float(pd.to_numeric(T[k]).median()), 2),
                               "max": round(float(pd.to_numeric(T[k]).max()), 2)}
                           for k in ("ms_native_search", "ms_tau_excl_search", "ms_gen_less_memory", "native_pops", "tau_excl_pops", "peak_kb_gen")},
            "definition": "all-optimal agreement = X repaired trace ∈ B optimal repaired trace set (official exact metric unchanged)"}
    for m in X:
        e, nat, dev = int(T[f"{m}_exact_eq_B"].sum()), int(T[f"{m}_in_native_set"].sum()), int(T[f"{m}_in_tau_excl_set"].sum())
        summ["agreement"][f"B-{m}"] = {"exact": f"{e}/{n} = {e / n:.4f}", "native_all_optimal": f"{nat}/{n} = {nat / n:.4f}",
                                       "tau_excluded_all_optimal": f"{dev}/{n} = {dev / n:.4f}",
                                       "counts": {"exact": e, "native": nat, "tau_excluded": dev, "n": n}}
        summ["accuracy_aux"][m] = f"{int(T[f'{m}_correct'].sum())}/{n}"
    (out / "agreement_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
