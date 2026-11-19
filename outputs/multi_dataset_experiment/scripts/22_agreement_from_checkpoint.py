"""22_agreement_from_checkpoint.py — 기존 all-optimal checkpoint의 B optimal set을 재사용해 B-X agreement 계산 (탐색 없음)

왜 (BPIC 2017 B-C+PN, 2026-11-17):
  07_pub_all_optimal_agreement.py는 case마다 B의 optimal repaired-trace 집합(native / tau-excluded)을 exact search로 구한 뒤
  X(C 또는 C+PN)의 repaired trace가 그 집합에 속하는지만 본다. B의 집합은 X와 무관하므로, BPIC B-C 계산(약 21.5 h, 메모리 사고 1회)
  때 checkpoint에 저장된 집합을 그대로 재사용하면 같은 탐색을 다시 할 필요가 없다. 판정 정의는 07과 동일하다.
  - exact                 : X trace == B trace
  - native all-optimal    : X trace ∈ B native optimal set
  - tau-excluded all-opt. : X trace ∈ B tau-excluded optimal set
  - state-limit / resource-limit case: checkpoint의 집합이 비어 있으므로 불일치 (공식 하한). 상한 = 하한 + limit case 수.
검증: --check 를 주면 같은 checkpoint의 C log로 B-C를 다시 계산해 checkpoint 자체의 C_* 열 및 agreement_summary.json과
      일치하는지 확인한다 (Road: B-C+PN 2293 / 2297 / 2354 재현, BPIC: B-C 926 / 983 / 998 재현).

실행: python 22_agreement_from_checkpoint.py --ckpt-run <run1 dir> --x-log <repaired log> --label CPN --gt <gt csv> --out <dir> [--check]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_ckpt(run_dir):
    """checkpoint/*.jsonl -> {case_id: (row, native_set, tau_set)} ; 중복 case는 내용이 같은지 확인."""
    out = {}
    for f in sorted((run_dir / "checkpoint").glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            r = rec["row"]
            nat = {t["trace"] for t in rec["trows"] if t["set"] == "native"}
            tau = {t["trace"] for t in rec["trows"] if t["set"] == "tau_excluded"}
            cid = r["case_id"]
            if cid in out:
                assert out[cid][1] == nat and out[cid][2] == tau, f"checkpoint conflict for {cid}"
                continue
            out[cid] = (r, nat, tau)
    return out


def load_repaired(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[(df["type"] == "test") & (~df["concept:name"].isin(["Start", "End"]))]
    return df.groupby("case:concept:name", sort=False)["concept:name"].agg(lambda s: " ".join(s)).to_dict()


def agreement(ck, xtr, gt, label):
    rows = []
    for cid, (r, nat, tau) in ck.items():
        x = xtr.get(cid)
        limited = bool(r.get("state_limit_hit")) or bool(r.get("resource_limit_hit"))
        rows.append({"case_id": cid, "B": r["B"], label: x, "ground_truth": r.get("ground_truth"),
                     "state_limit_hit": bool(r.get("state_limit_hit")), "resource_limit_hit": bool(r.get("resource_limit_hit")),
                     "limited": limited, "native_n_optimal": len(nat), "tau_excl_n_optimal": len(tau),
                     f"{label}_exact_eq_B": x == r["B"], f"{label}_in_native_set": x in nat,
                     f"{label}_in_tau_excl_set": x in tau, f"{label}_correct": x == r.get("ground_truth") and x is not None})
    T = pd.DataFrame(rows)
    n = len(T)
    e, na, ta = int(T[f"{label}_exact_eq_B"].sum()), int(T[f"{label}_in_native_set"].sum()), int(T[f"{label}_in_tau_excl_set"].sum())
    lim = int(T.limited.sum())
    summ = {"cases": n, "x_missing_cases": int(T[label].isna().sum()),
            "checks": {"state_limit_hit": int(T.state_limit_hit.sum()), "resource_limit_hit": int(T.resource_limit_hit.sum()),
                       "limited_total": lim, "resolved": n - lim,
                       "limited_but_exact_eq_B": int((T.limited & T[f"{label}_exact_eq_B"]).sum())},
            "agreement": {f"B-{label}": {
                "exact": f"{e}/{n} = {e / n:.4f}", "native_all_optimal": f"{na}/{n} = {na / n:.4f}",
                "tau_excluded_all_optimal": f"{ta}/{n} = {ta / n:.4f}",
                "counts": {"exact": e, "native": na, "tau_excluded": ta, "n": n},
                "bounds": {"native_lower": na, "native_upper": na + lim, "tau_excluded_lower": ta, "tau_excluded_upper": ta + lim,
                           "native_lower_pct": round(100 * na / n, 2), "native_upper_pct": round(100 * (na + lim) / n, 2),
                           "tau_excluded_lower_pct": round(100 * ta / n, 2), "tau_excluded_upper_pct": round(100 * (ta + lim) / n, 2),
                           "note": "lower = limited cases counted as disagreement (official); upper = all limited cases assumed to agree"}}},
            "accuracy_aux": {label: f"{int(T[f'{label}_correct'].sum())}/{n}"},
            "definition": "all-optimal agreement = X repaired trace ∈ B optimal repaired trace set (07 definition; B sets reused from checkpoint)"}
    return T, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-run", required=True, help="07 output dir containing checkpoint/ (e.g. .../alignment_all_optimal_analysis/run1)")
    ap.add_argument("--x-log", required=True)
    ap.add_argument("--label", default="CPN")
    ap.add_argument("--gt", default=None, help="unused (ground truth is taken from the checkpoint rows); kept for symmetry with 07")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="re-derive B-C from the checkpoint's own C column and compare")
    a = ap.parse_args()
    run = PROJECT_ROOT / a.ckpt_run
    ck = load_ckpt(run)
    xtr = load_repaired(PROJECT_ROOT / a.x_log)
    T, summ = agreement(ck, xtr, None, a.label)
    summ.update({"ckpt_run": a.ckpt_run, "x_log": a.x_log})
    checks = []
    ref = run / "agreement_summary.json"
    if a.check and ref.exists():
        rs = json.loads(ref.read_text(encoding="utf-8"))
        # 1) checkpoint 자체의 C 열로 B-C 재계산 == checkpoint C_* 열 == agreement_summary B-C counts
        ctr = {cid: r["C"] for cid, (r, _, _) in ck.items()}
        Tc, sc = agreement(ck, ctr, None, "C")
        own = {k: int(sum(bool(r.get(k)) for r, _, _ in ck.values())) for k in ("C_exact_eq_B", "C_in_native_set", "C_in_tau_excl_set")}
        got = sc["agreement"]["B-C"]["counts"]
        checks.append({"check": "B-C from checkpoint sets == checkpoint C_* columns",
                       "ok": (got["exact"], got["native"], got["tau_excluded"]) == (own["C_exact_eq_B"], own["C_in_native_set"], own["C_in_tau_excl_set"]),
                       "detail": {"recomputed": got, "checkpoint_columns": own}})
        rc = rs["agreement"]["B-C"]["counts"]
        checks.append({"check": "B-C from checkpoint sets == agreement_summary.json B-C", "ok": got == rc, "detail": {"recomputed": got, "summary": rc}})
        # 2) summary에 B-<label>가 이미 있으면(Road) 그것도 재현해야 함
        key = f"B-{a.label}"
        if key in rs["agreement"]:
            rr = rs["agreement"][key]["counts"]
            g2 = summ["agreement"][key]["counts"]
            checks.append({"check": f"{key} from checkpoint sets == agreement_summary.json {key}", "ok": g2 == rr, "detail": {"recomputed": g2, "summary": rr}})
        checks.append({"check": "checkpoint covers all summary cases", "ok": len(ck) == rs["cases"], "detail": (len(ck), rs["cases"])})
    summ["checks_vs_reference"] = checks
    out = PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "case_summary.csv", index=False)
    (out / "agreement_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("cases", "x_missing_cases", "checks", "agreement")}, ensure_ascii=False, indent=1))
    for c in checks:
        print("[CHECK]", "PASS" if c["ok"] else "FAIL", c["check"], c["detail"])
    if checks and not all(c["ok"] for c in checks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
