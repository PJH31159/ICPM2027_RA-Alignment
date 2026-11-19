"""Shared evaluation functions.

Rules:
    - Traces are rebuilt in CSV row order, Start/End removed, as activity tuples
      (the order column of Method A is not used).
    - Ground truth is used for evaluation only.
    - Exact Match Rate = share of test cases with repaired trace == ground truth.
"""

import math
import re

PATTERN_ORDER = ["insert", "skip", "replace", "rework", "moved", "multiple"]


def canonical_traces(path, case_col, act_col, type_filter=None):
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if type_filter is not None:
        df = df[df["type"] == type_filter]
    df = df[~df[act_col].isin(["Start", "End"])]
    return df.groupby(case_col, sort=False)[act_col].agg(tuple)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def pattern_key(label):
    # Pattern key: keep letters only, sort, and join with '+'.
    parts = sorted(re.findall(r"[a-zA-Z]+", label))
    return "+".join(parts) if parts else "normal"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def fmt(t):
    return ">>".join(t)


def load_evaluation_basis(anomaly_log, ground_truth_log):
    """Ground truth, anomalous traces, anomaly labels and the set of clean normal variants."""
    import pandas as pd

    gt = canonical_traces(ground_truth_log, "Case", "Activity")
    anomalous = canonical_traces(anomaly_log, "Case", "Activity")
    raw = pd.read_csv(anomaly_log, dtype=str, keep_default_na=False, usecols=["Case", "Activity", "type_res_trace"])
    labels = raw.groupby("Case", sort=False)["type_res_trace"].first()
    normal_variants = set(anomalous[labels[labels == ""].index])
    return gt, anomalous, labels, normal_variants


def error_type(rep, an, g, normal_variants):
    if rep == g:
        return "correct"
    if rep == an:
        return "unrepaired"
    return "wrong_normal_variant" if rep in normal_variants else "non_normal_trace"


def case_level_table(test_ids, gt, anomalous, labels, normal_variants, runs):
    """runs: {label: canonical Series (test)} -> per-case table."""
    import pandas as pd

    rows = []
    for c in test_ids:
        g, an = gt[c], anomalous[c]
        pk = pattern_key(labels[c])
        row = {"case:concept:name": c, "anomaly_label": labels[c],
               "pattern_group": pk if "+" not in pk else "multiple",
               "anomalous_trace": fmt(an), "ground_truth": fmt(g),
               "changed_by_anomaly": an != g, "lev_before": levenshtein(an, g)}
        for label, traces in runs.items():
            rep = traces[c]
            row[f"{label}_repaired"] = fmt(rep)
            row[f"{label}_correct"] = rep == g
            row[f"{label}_lev_after"] = levenshtein(rep, g)
            row[f"{label}_normal_variant"] = rep in normal_variants
            row[f"{label}_error_type"] = error_type(rep, an, g, normal_variants)
        rows.append(row)
    return pd.DataFrame(rows)


def official_metrics(cl, labels):
    import pandas as pd

    out = []
    for m in labels:
        out.append({"run": m, "n_cases": len(cl), "n_exact_match": int(cl[f"{m}_correct"].sum()),
                    "exact_match_rate": cl[f"{m}_correct"].mean(), "recon_err_before": cl["lev_before"].mean(),
                    "recon_err_after": cl[f"{m}_lev_after"].mean(),
                    "normal_variant_rate": cl[f"{m}_normal_variant"].mean(),
                    "changed_only_emr": cl.loc[cl.changed_by_anomaly, f"{m}_correct"].mean(),
                    "n_wrong_normal_variant": int((cl[f"{m}_error_type"] == "wrong_normal_variant").sum()),
                    "n_non_normal": int((cl[f"{m}_error_type"] == "non_normal_trace").sum()),
                    "n_unrepaired_wrong": int((cl[f"{m}_error_type"] == "unrepaired").sum())})
    return pd.DataFrame(out).set_index("run")


def pattern_table(cl, labels):
    import pandas as pd

    rows = []
    for g in PATTERN_ORDER + ["ALL"]:
        d = cl if g == "ALL" else cl[cl.pattern_group == g]
        row = {"pattern_group": g, "n": len(d)}
        for m in labels:
            row[f"{m}_correct"] = int(d[f"{m}_correct"].sum())
            row[f"{m}_emr"] = d[f"{m}_correct"].mean() if len(d) else None
        rows.append(row)
    return pd.DataFrame(rows)


def contingency(cl, x, y):
    ct = {"both_correct": int((cl[f"{x}_correct"] & cl[f"{y}_correct"]).sum()),
          f"{x}_only_correct": int((cl[f"{x}_correct"] & ~cl[f"{y}_correct"]).sum()),
          f"{y}_only_correct": int((~cl[f"{x}_correct"] & cl[f"{y}_correct"]).sum()),
          "both_wrong": int((~cl[f"{x}_correct"] & ~cl[f"{y}_correct"]).sum())}
    ct["mcnemar_exact_p (supplementary)"] = mcnemar_exact(ct[f"{x}_only_correct"], ct[f"{y}_only_correct"])
    ct["n_different_repairs"] = int((cl[f"{x}_repaired"] != cl[f"{y}_repaired"]).sum())
    return ct


def case_agreement(full_runs, ids):
    """full_runs: {label: {case: trace}} -> share of identical repairs for every pair."""
    labels = list(full_runs)
    out = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            out[f"{a} vs {b}"] = sum(full_runs[a][c] == full_runs[b][c] for c in ids) / len(ids)
    return out
