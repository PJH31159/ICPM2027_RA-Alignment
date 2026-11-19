"""Agreement between the repaired logs of two methods.

Unlike the Exact Match Rate (repaired == ground truth), this compares the repaired trace of
method X with the repaired trace of method Y for the same case id. Two cases agree only if
the full activity sequences are identical. Only existing repaired logs are read.
Trace reconstruction follows evaluation_utils.canonical_traces (file row order, Start/End removed).
"""


def category(a, b, c):
    if a == b == c:
        return "A=B=C"
    if a == b:
        return "A=B only"
    if a == c:
        return "A=C only"
    if b == c:
        return "B=C only"
    return "all different"


def abc_agreement(paths, test_ids, anomaly_log, ground_truth_log):
    """paths: {"A": csv, "B": csv, "C": csv}. Returns (per-case DataFrame, summary DataFrame)."""
    import pandas as pd

    from utils.evaluation_utils import canonical_traces, fmt, load_evaluation_basis

    gt, _anomalous, labels, _nv = load_evaluation_basis(anomaly_log, ground_truth_log)
    tr = {m: canonical_traces(p, "case:concept:name", "concept:name", "test") for m, p in paths.items()}
    for m, t in tr.items():
        if set(t.index) != set(test_ids):
            raise ValueError(f"{m}: test case set != split")
    rows = []
    for c in test_ids:
        a, b, cc = tr["A"][c], tr["B"][c], tr["C"][c]
        rows.append({"case_id": c, "anomaly_label": labels[c], "ground_truth": fmt(gt[c]),
                     "A_repaired": fmt(a), "B_repaired": fmt(b), "C_repaired": fmt(cc),
                     "A_eq_B": a == b, "A_eq_C": a == cc, "B_eq_C": b == cc, "category": category(a, b, cc),
                     "A_correct": a == gt[c], "B_correct": b == gt[c], "C_correct": cc == gt[c]})
    pc = pd.DataFrame(rows)
    n = len(pc)
    summ = [{"metric": f"{x} vs {y}", "same": int(pc[f"{x}_eq_{y}"].sum()), "total": n,
             "rate": round(pc[f"{x}_eq_{y}"].sum() / n, 4), "different": int(n - pc[f"{x}_eq_{y}"].sum())}
            for x, y in (("A", "B"), ("A", "C"), ("B", "C"))]
    ncorrect = pc[["A_correct", "B_correct", "C_correct"]].sum(axis=1)
    for cat in ["A=B=C", "A=B only", "A=C only", "B=C only", "all different"]:
        sel = pc["category"] == cat
        k = int(sel.sum())
        summ.append({"metric": f"category: {cat}", "same": k, "total": n, "rate": round(k / n, 4), "different": "",
                     "all_correct": int((sel & (ncorrect == 3)).sum()),
                     "some_correct": int((sel & ncorrect.between(1, 2)).sum()),
                     "all_wrong": int((sel & (ncorrect == 0)).sum())})
    return pc, pd.DataFrame(summ)
