"""B-X agreement from the B optimal sets stored in an existing all-optimal checkpoint (no search).

all_optimal_agreement.py computes, per case, the set of optimal B repaired traces (native / tau-excluded) by
exact search and then checks whether the repaired trace of X (C or C+PN) belongs to it. The B sets do not depend on X,
so the sets stored in the checkpoint can be reused without repeating the search. Same decision definition as 07:
  - exact                 : X trace == B trace
  - native all-optimal    : X trace in the B native optimal set
  - tau-excluded all-opt. : X trace in the B tau-excluded optimal set
  - state-limit / resource-limit cases: the stored set is empty -> disagreement (official lower bound);
    upper bound = lower bound + number of limited cases.
Check: --check recomputes B-C from the C column of the same checkpoint and verifies it against the checkpoint's own
C_* columns and agreement_summary.json.

Usage: python scripts/agreement_from_checkpoint.py --ckpt-run <run1 dir> --x-log <repaired log> --label CPN --gt <gt csv> --out <dir> [--check]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_ckpt(run_dir):
    """checkpoint/*.jsonl -> {case_id: (row, native_set, tau_set)}; duplicate cases must have identical content."""
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
        # 1) B-C recomputed from the checkpoint's own C column == checkpoint C_* columns == agreement_summary B-C counts
        ctr = {cid: r["C"] for cid, (r, _, _) in ck.items()}
        Tc, sc = agreement(ck, ctr, None, "C")
        own = {k: int(sum(bool(r.get(k)) for r, _, _ in ck.values())) for k in ("C_exact_eq_B", "C_in_native_set", "C_in_tau_excl_set")}
        got = sc["agreement"]["B-C"]["counts"]
        checks.append({"check": "B-C from checkpoint sets == checkpoint C_* columns",
                       "ok": (got["exact"], got["native"], got["tau_excluded"]) == (own["C_exact_eq_B"], own["C_in_native_set"], own["C_in_tau_excl_set"]),
                       "detail": {"recomputed": got, "checkpoint_columns": own}})
        rc = rs["agreement"]["B-C"]["counts"]
        checks.append({"check": "B-C from checkpoint sets == agreement_summary.json B-C", "ok": got == rc, "detail": {"recomputed": got, "summary": rc}})
        # 2) if the summary already contains B-<label>, it must be reproduced as well
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
