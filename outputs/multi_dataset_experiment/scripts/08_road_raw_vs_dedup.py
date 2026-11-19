"""08_road_raw_vs_dedup.py — Road Traffic: raw clean log vs 중복 event 복원 derived copy의 discovery 비교 (탐색, official 아님)

왜: Road_Traffic_1.00.csv에서 k(Appeal to Judge)를 포함한 clean case 475건은 모든 event가 두 번씩(x x y y ...) 들어 있다.
    1차 진단에서 raw model은 LOOP 11개(거의 모든 activity가 *(x, tau))였고, k case를 discovery에서 빼면 LOOP가 1개만 남았다.
    여기서는 k case를 빼지 않고 "한 번씩으로 복원"한 derived copy로 discovery를 다시 하고 raw와 비교한다.
      A. raw clean log (현재 pipeline 입력 그대로)
      B. derived copy: clean case 중 exact doubling(t[2i] == t[2i+1] 전부)인 case만 t[0::2]로 복원.
         anomaly case는 바꾸지 않는다 (doubling 위에 주입된 anomaly를 복원하는 규칙은 정의되지 않음).
      R. 참고: 예전 EXP-039~058의 Road XES model (원본 XES 전체, pm4py.discover_process_tree_inductive 기본값)
    원본 파일은 읽기만 한다 (실행 전후 SHA-256 확인). official run / 연구 문서 반영은 하지 않는다.

비교: Process Tree, LOOP / PARALLEL / XOR, places / transitions / tau / arcs,
      clean fitness (exact Petri Net acceptance 비율 + token-based replay fitness),
      anomaly acceptance (anomaly 200건 sample, seed 42, 05 script와 같은 sample), bounded trace growth.

출력: outputs/simulation/RoadTraffic/exploratory_dedup/
      derived_data/Road_Traffic_1.00_dedup.csv (90 MB급, gitignore) + manifest / 비교 결과
실행: .venv\\Scripts\\python outputs\\multi_dataset_experiment\\scripts\\08_road_raw_vs_dedup.py
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

import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pm4py  # noqa: E402

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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.data_utils import build_event_log  # noqa: E402
from utils.discovery_utils import discover_model, operator_kind_counts, petri_net_has_cycle  # noqa: E402
from utils.loop_bound_utils import ExplosionAborted, generate_bounded, make_petri_acceptor  # noqa: E402

RAW = _data_root() / "encoded_anomaly" / "Road_Traffic_1.00.csv"
XES = PROJECT_ROOT / "data/raw/Road_Traffic_Fine_Management_Process.xes"
OUT = PROJECT_ROOT / "outputs/simulation/RoadTraffic/exploratory_dedup"
DERIVED = OUT / "derived_data" / "Road_Traffic_1.00_dedup.csv"
GUARD = dict(max_intermediate=2_000_000, max_seconds=120, max_rss_gb=6)  # 1차 진단(03)과 같은 guard
ACCEPT_BUDGET_SEC = 600


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def exact_double(t):
    return len(t) > 0 and len(t) % 2 == 0 and all(t[2 * i] == t[2 * i + 1] for i in range(len(t) // 2))


def make_derived(df):
    """clean exact-doubling case만 짝수 위치 event(0, 2, 4, ...)를 남긴다. Event 열은 case 안에서 다시 번호를 매긴다."""
    lab = df.groupby("Case", sort=False)["type_res_trace"].first()
    tr = df.groupby("Case", sort=False)["Activity"].agg(tuple)
    target = [c for c in tr.index if pd.isna(lab[c]) and exact_double(tr[c])]
    pos = df.groupby("Case", sort=False).cumcount()
    tset = set(target)
    keep = ~df["Case"].isin(tset) | (pos % 2 == 0)
    out = df[keep].copy()
    m = out["Case"].isin(tset)
    out.loc[m, "Event"] = out[m].groupby("Case", sort=False).cumcount()
    if "order" in out:
        out.loc[m, "order"] = out.loc[m, "Event"] + 1
    return out, target


def traces_of(df):
    tr = df.groupby("Case", sort=False)["Activity"].agg(lambda s: list(s.astype(str)))
    lab = df.groupby("Case", sort=False)["type_res_trace"].first()
    tr.index, lab.index = tr.index.map(str), lab.index.map(str)
    clean = [c for c in lab.index if pd.isna(lab[c])]
    anom = [c for c in lab.index if not pd.isna(lab[c])]
    return tr, lab, clean, anom


def labels_of(node):
    if node.operator is None:
        return [] if node.label is None else [node.label]
    return [x for c in node.children for x in labels_of(c)]


def loops(tree):
    from pm4py.objects.process_tree.obj import Operator
    res, stack = [], [tree]
    while stack:
        n = stack.pop()
        if n.operator == Operator.LOOP:
            do, redo = n.children[0], n.children[1]
            res.append({"do": sorted(set(labels_of(do))), "redo_is_tau": redo.operator is None and redo.label is None,
                        "self_loop_single_activity": do.operator is None and do.label is not None
                        and redo.operator is None and redo.label is None})
        stack.extend(n.children)
    return res


def structure(tree, net):
    k = operator_kind_counts(tree)
    lp = loops(tree)
    return {"process_tree": str(tree), "n_loop": k.get("LOOP", 0), "n_parallel": k.get("PARALLEL", 0), "n_xor": k.get("XOR", 0),
            "n_sequence": k.get("SEQUENCE", 0), "n_tau_leaves": k.get("tau", 0),
            "n_self_loops_single_activity": sum(x["self_loop_single_activity"] for x in lp), "loops": lp,
            "n_places": len(net.places), "n_transitions": len(net.transitions),
            "n_tau_transitions": sum(t.label is None for t in net.transitions), "n_arcs": len(net.arcs),
            "petri_net_has_cycle": petri_net_has_cycle(net)}


def clean_fitness(tr, clean_ids, net, im, fm):
    variants = Counter(tuple(tr[c]) for c in clean_ids)
    acc, t0, checked = make_petri_acceptor(net, im, fm), time.time(), {}
    for v, _n in variants.most_common():
        if time.time() - t0 > ACCEPT_BUDGET_SEC:
            break
        try:
            checked[v] = acc(v)
        except RuntimeError:
            checked[v] = None
    n_chk = sum(variants[v] for v, x in checked.items() if x is not None)
    n_acc = sum(variants[v] for v, x in checked.items() if x is True)
    # token-based replay fitness (variant 대표 1건씩 x 빈도로 가중) — 빠른 보조 지표
    vlist = list(variants)
    vid = {v: f"v{i}" for i, v in enumerate(vlist)}
    log = build_event_log([vid[v] for v in vlist], {vid[v]: list(v) for v in vlist})
    tbr = pm4py.conformance.fitness_token_based_replay(log, net, im, fm)
    from pm4py.algo.conformance.tokenreplay import algorithm as tr_alg
    per = tr_alg.apply(log, net, im, fm)
    fit_w = sum(r["trace_fitness"] * variants[v] for r, v in zip(per, vlist)) / sum(variants.values())
    return {"clean_cases": sum(variants.values()), "clean_variants": len(variants), "variants_checked": len(checked),
            "clean_cases_checked": n_chk, "clean_cases_accepted_exact": n_acc,
            "exact_acceptance_rate": round(n_acc / n_chk, 6) if n_chk else None,
            "token_replay_fitness_case_weighted": round(fit_w, 6),
            "token_replay_fitness_variant_log": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in tbr.items()},
            "acceptance_sec": round(time.time() - t0, 1), "budget_sec": ACCEPT_BUDGET_SEC}


def anomaly_acceptance(tr, lab, anom_ids, net, im, fm, pick):
    acc, out, by, tot, t0 = make_petri_acceptor(net, im, fm), {}, Counter(), Counter(), time.time()
    for c in pick:
        v = tuple(tr[c])
        if v not in out:
            try:
                out[v] = acc(v)
            except RuntimeError:
                out[v] = None
        t = lab[c].split("'")[1] if "'" in str(lab[c]) else str(lab[c])
        tot[t] += 1
        by[t] += bool(out[v])
    n = sum(tot.values())
    return {"sampled": len(pick), "accepted": sum(by.values()), "rate": round(sum(by.values()) / n, 4),
            "by_first_anomaly_type": {k: f"{by[k]}/{tot[k]}" for k in sorted(tot)}, "sec": round(time.time() - t0, 1)}


def growth(tree, tr, clean_ids, grid):
    observed = Counter(tuple(tr[c]) for c in clean_ids)
    total, rows = sum(observed.values()), []
    for n, t in grid:
        t0 = time.time()
        try:
            lang, st = generate_bounded(tree, n, t, 100, **GUARD)
            gen = set(lang)
            obs_in = [v for v in observed if v in gen]
            cov = sum(observed[v] for v in obs_in)
            rows.append({"loop_num": n, "loop_total": t, "status": "OK", "generated_unique": len(gen),
                         "model_only": len(gen) - len(obs_in), "observed_variants_in": len(obs_in),
                         "observed_variants": len(observed), "clean_case_coverage": round(cov / total, 6),
                         "peak_intermediate": st["peak_intermediate"], "sec": round(time.time() - t0, 2)})
            del lang, gen
        except ExplosionAborted as e:
            rows.append({"loop_num": n, "loop_total": t, "status": "ABORTED", "reason": str(e)[:200], "sec": round(time.time() - t0, 2)})
        print("   growth", rows[-1], flush=True)
    return rows


def grid_for(n_loop):
    """loop가 L개일 때: 한 trace가 여러 loop를 1회씩 지나면 loop_total >= L 이 필요하다 (D004)."""
    L = max(1, n_loop)
    g = []
    for n in (1, 2, 3, 4):
        for t in sorted({n, L, L + n, L * n}):
            if t >= n:
                g.append((n, t))
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    sha_before = sha(RAW)
    raw = pd.read_csv(RAW, low_memory=False)
    der, target = make_derived(raw)
    der.to_csv(DERIVED, index=False)
    sha_after = sha(RAW)
    tr_r, lab_r, clean_r, anom_r = traces_of(raw)
    tr_d, lab_d, clean_d, anom_d = traces_of(der)
    changed = [str(c) for c in target]
    assert set(anom_r) == set(anom_d) and all(tr_r[c] == tr_d[c] for c in anom_r), "anomaly cases must be unchanged"
    assert all(tr_d[c] == tr_r[c][0::2] for c in changed)
    k_clean = [c for c in clean_r if "k" in tr_r[c]]
    manifest = {"source": str(RAW), "source_sha256_before": sha_before, "source_sha256_after": sha_after,
                "source_unchanged": sha_before == sha_after, "derived_file": str(DERIVED.relative_to(PROJECT_ROOT)),
                "derived_sha256": sha(DERIVED), "rule": "clean case (type_res_trace empty) whose trace is an exact doubling "
                "(t[2i] == t[2i+1] for all i) -> keep events at positions 0, 2, 4, ... (Event / order renumbered); "
                "anomaly cases unchanged", "restored_clean_cases": len(changed),
                "clean_cases_with_k": len(k_clean), "restored_cases_all_contain_k": all("k" in tr_r[c] for c in changed),
                "k_clean_cases_restored": len(set(k_clean) & set(changed)),
                "anomaly_cases_with_k_unchanged": sum("k" in tr_r[c] for c in anom_r),
                "anomaly_cases_exact_doubling_unchanged": sum(exact_double(tuple(tr_r[c])) for c in anom_r),
                "rows_raw": len(raw), "rows_derived": len(der), "restored_case_ids": changed}
    (OUT / "derived_copy_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: v for k, v in manifest.items() if k != "restored_case_ids"}, flush=True)

    pick = random.Random(42).sample(sorted(anom_r), 200)  # 05 script와 같은 sample (ids 정렬 후 seed 42)
    res = {}
    for name, tr, lab, clean, anom in (("A_raw", tr_r, lab_r, clean_r, anom_r), ("B_dedup", tr_d, lab_d, clean_d, anom_d)):
        print(f"\n[{name}] discovery", flush=True)
        tree, net, im, fm, sec = discover_model(tr, clean, 0.0)
        s = structure(tree, net)
        s["discovery_sec"] = round(sec, 2)
        print("  ", {k: v for k, v in s.items() if k not in ("loops",)}, flush=True)
        s["clean_fitness"] = clean_fitness(tr, clean, net, im, fm)
        print("   clean fitness", s["clean_fitness"], flush=True)
        s["anomaly_acceptance_200"] = anomaly_acceptance(tr, lab, anom, net, im, fm, pick)
        print("   anomaly acceptance", s["anomaly_acceptance_200"], flush=True)
        s["bounded_growth"] = growth(tree, tr, clean, grid_for(s["n_loop"]))
        pd.DataFrame(s["bounded_growth"]).to_csv(OUT / f"{name}_bounded_growth.csv", index=False)
        (OUT / f"{name}_process_tree.txt").write_text(str(tree) + "\n", encoding="utf-8")
        pm4py.write_pnml(net, im, fm, str(OUT / f"{name}_petri_net.pnml"))
        res[name] = s

    print("\n[R_old_xes] reference: EXP-039~058 XES model (pm4py.discover_process_tree_inductive(read_xes) default)", flush=True)
    t0 = time.time()
    xdf = pm4py.read_xes(str(XES))
    xtree = pm4py.discover_process_tree_inductive(xdf)
    xnet, xim, xfm = pm4py.convert_to_petri_net(xtree)
    r = structure(xtree, xnet)
    r["discovery_sec"] = round(time.time() - t0, 2)
    r["note"] = "activity names (XES); letter mapping in audit/raw_dataset_audit.json road_traffic_csv_vs_xes.letter_to_name"
    mp = json.loads((PROJECT_ROOT / "outputs/multi_dataset_experiment/audit/raw_dataset_audit.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in mp["road_traffic_csv_vs_xes"]["letter_to_name"].items()}
    tree_letters = str(xtree)
    for name_, letter in sorted(inv.items(), key=lambda x: -len(x[0])):
        tree_letters = tree_letters.replace(f"'{name_}'", f"'{letter}'")
    r["process_tree_letters"] = tree_letters
    (OUT / "R_old_xes_process_tree.txt").write_text(str(xtree) + "\n" + tree_letters + "\n", encoding="utf-8")
    print("  ", {k: v for k, v in r.items() if k not in ("loops",)}, flush=True)
    res["R_old_xes"] = r

    summary = {"date": "2026-11-10", "status": "exploratory (official Road data NOT decided)", "manifest": {k: v for k, v in manifest.items() if k != "restored_case_ids"},
               "guard": GUARD, "results": res}
    (OUT / "raw_vs_dedup_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
