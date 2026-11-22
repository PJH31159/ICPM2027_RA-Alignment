"""Method B: Petri-net alignment repair -> repaired log B.

Method B discovers a process model from the clean cases and aligns each anomalous trace to it, so
the trace is repaired to the closest trace allowed by the model:
    clean cases of the anomaly log -> Inductive Miner (noise threshold 0) -> process tree -> Petri net
    -> alignment -> repaired log B

Repair rule, with alignment move = (log label, model label):
    - synchronous move   (a, a)       -> keep a
    - log move           (a, '>>')    -> delete
    - visible model move ('>>', b)    -> insert b
    - tau model move     ('>>', None) -> skip (invisible transitions do not produce events)
Cases are linked by case id, not by the position in the alignment result. If the alignment fails,
the original anomalous trace is kept and a failure flag is set; only successfully aligned cases
are guaranteed to be model-conformant.

The ground truth is not used here. This script stores alignment diagnostics (cost, fitness, number
of moves), which are not repair accuracy.

Tutorial 3 (Optimal Alignment) imports align_case / repair_from_alignment / ALIGNMENT_VARIANT from this file.
These functions are dataset-independent and are used for every dataset (Small ... BPIC 2017).
The settings block below (DATA_NAME, input / split / output paths) applies only to the standalone
Small run. Output folder names (outputs/exp061_method_b_alignment) are kept as in the official run.

Standalone usage (Small, from the repository root, PM4Py environment .venv):
    python src/alignment/exact_alignment.py --dry_run       (model / input checks only)
    python src/alignment/exact_alignment.py --run_id 1      (official repaired log B)
    python src/alignment/exact_alignment.py --run_id 2      (reproducibility run)
    python src/alignment/exact_alignment.py --compare_runs  (run 1 vs run 2)
"""

import argparse
import filecmp
import json
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo
from pm4py.objects.log.obj import Event, EventLog, Trace
from pm4py.objects.process_tree.obj import Operator

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/alignment/<file>.py -> repository root


# ============================================================
# 1. Experiment settings
# ============================================================

DATA_NAME = "Small"
ANOMALY_LOG_PATH = PROJECT_ROOT / "data" / "raw" / "Small_1.00.csv"
NOISE_THRESHOLD = 0.0

# PM4Py default variant for single-trace alignment (optimal alignment, A* + state equation).
# Which of several equally optimal alignments is returned can depend on the variant, so it is fixed and recorded.
ALIGNMENT_VARIANT = alignments_algo.Variants.VERSION_STATE_EQUATION_A_STAR

# train/test split of the official Small run
METHOD_A_RUN1_DIR = PROJECT_ROOT / "outputs" / "exp060_repaired_log_a" / "run1"
TEST_IDS_PATH = METHOD_A_RUN1_DIR / "Small_1000_ppm0.0_valratio0.5_seed42_run1_test_case_ids.txt"
VAL_IDS_PATH = METHOD_A_RUN1_DIR / "Small_1000_ppm0.0_valratio0.5_seed42_run1_val_case_ids.txt"
EXPECTED_N_TEST = 524
EXPECTED_N_VAL = 523

EXP_DIR = PROJECT_ROOT / "outputs" / "exp061_method_b_alignment"
FILE_PREFIX = f"{DATA_NAME}_IM_noise0_alignment"


def run_dir(run_id):
    return EXP_DIR / f"run{run_id}"


def run_file(run_id, suffix):
    return run_dir(run_id) / f"{FILE_PREFIX}_run{run_id}_{suffix}"


# ============================================================
# 2. Data loading
# ============================================================
# PM4Py dataframe conversion may re-sort events by timestamp, so the EventLog is built directly
# to keep the event order of the file (the order RP-BEAR sees).

def load_case_traces():
    df = pd.read_csv(ANOMALY_LOG_PATH, usecols=["Case", "Activity", "type_res_trace"])
    traces = df.groupby("Case", sort=False)["Activity"].agg(list)
    labels = df.groupby("Case", sort=False)["type_res_trace"].first()
    clean_ids = labels[labels.isna()].index.tolist()
    anomalous_ids = labels[labels.notna()].index.tolist()
    return traces, labels, clean_ids, anomalous_ids


def build_event_log(case_ids, traces):
    log = EventLog()
    for cid in case_ids:
        trace = Trace()
        trace.attributes["concept:name"] = cid
        for act in traces[cid]:
            trace.append(Event({"concept:name": act}))
        log.append(trace)
    return log


def read_ids(path):
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ============================================================
# 3. Process-tree language enumeration (checks that the model reproduces the clean variants)
# ============================================================
# Trees with only SEQUENCE / XOR / PARALLEL / tau have a finite language, which is enumerated in full.
# Other operators (e.g. LOOP) return None.

def interleave(a, b):
    if not a:
        return [b]
    if not b:
        return [a]
    return [[a[0]] + rest for rest in interleave(a[1:], b)] + [[b[0]] + rest for rest in interleave(a, b[1:])]


def enumerate_tree_language(node):
    if node.operator is None:
        return [[]] if node.label is None else [[node.label]]

    child_langs = [enumerate_tree_language(c) for c in node.children]
    if any(lang is None for lang in child_langs):
        return None

    if node.operator == Operator.XOR:
        return [t for lang in child_langs for t in lang]

    if node.operator == Operator.SEQUENCE:
        result = [[]]
        for lang in child_langs:
            result = [prefix + suffix for prefix in result for suffix in lang]
        return result

    if node.operator == Operator.PARALLEL:
        result = [[]]
        for lang in child_langs:
            result = [mix for prefix in result for suffix in lang for mix in interleave(prefix, suffix)]
        return result

    return None


def count_operators(node, counter=None):
    counter = Counter() if counter is None else counter
    counter[str(node.operator) if node.operator is not None else ("tau" if node.label is None else "activity")] += 1
    for c in node.children:
        count_operators(c, counter)
    return counter


# ============================================================
# 4. One alignment -> repaired trace
# ============================================================

def repair_from_alignment(alignment):
    repaired = []
    n_sync = n_log = n_model_visible = n_tau = 0
    for log_label, model_label in alignment:
        if log_label != ">>" and model_label != ">>":
            # synchronous move: keep
            repaired.append(model_label)
            n_sync += 1
        elif model_label == ">>":
            # log move: delete
            n_log += 1
        elif model_label is None:
            # invisible (tau) model move: skip
            n_tau += 1
        else:
            # visible model move: insert
            repaired.append(model_label)
            n_model_visible += 1
    return repaired, {"n_sync_moves": n_sync, "n_log_moves": n_log, "n_model_moves_visible": n_model_visible, "n_tau_moves": n_tau}


def align_case(trace_obj, net, im, fm):
    result = alignments_algo.apply(trace_obj, net, im, fm, variant=ALIGNMENT_VARIANT)
    if result is None or result.get("alignment") is None:
        raise RuntimeError("alignment returned None")
    return result


# ============================================================
# 5. Discovery + model checks (shared by the dry run and the official run)
# ============================================================

def discover_and_check(traces, clean_ids):
    clean_log = build_event_log(clean_ids, traces)

    t0 = time.time()
    tree = pm4py.discover_process_tree_inductive(clean_log, noise_threshold=NOISE_THRESHOLD)
    net, im, fm = pm4py.convert_to_petri_net(tree)
    discovery_sec = time.time() - t0

    clean_variants = Counter(tuple(traces[cid]) for cid in clean_ids)

    language = enumerate_tree_language(tree)
    language_set = None if language is None else set(tuple(t) for t in language)

    # clean-log fitness: identical variants have identical alignments, so each variant is aligned once
    # and the result is weighted by its number of cases.
    variant_rows = []
    for variant, count in clean_variants.most_common():
        trace_obj = Trace()
        for act in variant:
            trace_obj.append(Event({"concept:name": act}))
        res = align_case(trace_obj, net, im, fm)
        variant_rows.append(
            {
                "variant": ">>".join(variant),
                "n_cases": count,
                "length": len(variant),
                "alignment_cost": res["cost"],
                "fitness": res["fitness"],
                "in_model_language": None if language_set is None else variant in language_set,
            }
        )
    n_clean = sum(clean_variants.values())
    weighted_fitness = sum(r["fitness"] * r["n_cases"] for r in variant_rows) / n_clean

    visible = [t for t in net.transitions if t.label is not None]
    model_check = {
        "process_tree": str(tree),
        "tree_node_counts": dict(count_operators(tree)),
        "n_places": len(net.places),
        "n_transitions": len(net.transitions),
        "n_visible_transitions": len(visible),
        "n_invisible_transitions": len(net.transitions) - len(visible),
        "n_arcs": len(net.arcs),
        "visible_activities": sorted(t.label for t in visible),
        "n_clean_cases": n_clean,
        "n_clean_variants": len(clean_variants),
        "model_language_size": None if language_set is None else len(language_set),
        "model_language_equals_clean_variants": None if language_set is None else language_set == set(clean_variants),
        "clean_variants_not_in_model": None if language_set is None else [">>".join(v) for v in clean_variants if v not in language_set],
        "model_traces_not_in_clean": None if language_set is None else sorted(">>".join(t) for t in language_set - set(clean_variants)),
        "clean_log_fitness_alignment_weighted": weighted_fitness,
        "clean_variants": variant_rows,
        "discovery_sec": round(discovery_sec, 4),
    }
    return tree, net, im, fm, language_set, model_check


# ============================================================
# 6. Pre-run checks
# ============================================================

def check_split(anomalous_ids):
    test_ids, val_ids = read_ids(TEST_IDS_PATH), read_ids(VAL_IDS_PATH)
    anomalous_set = set(anomalous_ids)
    return {
        "test_ids_file": str(TEST_IDS_PATH),
        "val_ids_file": str(VAL_IDS_PATH),
        "n_test": len(test_ids),
        "n_val": len(val_ids),
        "n_test_unique": len(set(test_ids)),
        "test_expected_524": len(set(test_ids)) == EXPECTED_N_TEST,
        "val_expected_523": len(set(val_ids)) == EXPECTED_N_VAL,
        "test_val_disjoint": not (set(test_ids) & set(val_ids)),
        "test_subset_of_anomalous": set(test_ids) <= anomalous_set,
        "val_subset_of_anomalous": set(val_ids) <= anomalous_set,
        "test_plus_val_equals_all_anomalous": (set(test_ids) | set(val_ids)) == anomalous_set,
    }, test_ids, val_ids


def preflight(run_id=None):
    problems = []
    if not hasattr(pm4py, "discover_process_tree_inductive"):
        problems.append('pm4py is missing or has the wrong version')
    for p in [ANOMALY_LOG_PATH, TEST_IDS_PATH, VAL_IDS_PATH]:
        if not p.exists():
            problems.append(f"input file not found: {p}")
    if run_id is not None and run_dir(run_id).exists():
        problems.append(f"run directory already exists: {run_dir(run_id)}")
    return problems


def environment_text():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
        except Exception as e:
            return f"(failed: {e})"

    return "\n".join(
        [
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"python_executable: {sys.executable}",
            f"python_version: {sys.version}",
            f"platform: {platform.platform()}",
            f"pm4py: {pm4py.__version__}",
            f"pandas: {pd.__version__}",
            f"alignment_variant: {ALIGNMENT_VARIANT}",
            f"git_commit: {run(['git', 'rev-parse', 'HEAD'])}",
            "git_status_short:",
            run(["git", "status", "--short"]) or "(clean)",
            "",
            "pip freeze:",
            run([sys.executable, "-m", "pip", "freeze"]),
        ]
    ) + "\n"


# ============================================================
# 7. Dry run: model / input checks only (no alignment repair)
# ============================================================

def dry_run():
    problems = preflight()
    if problems:
        print("[ABORT]", *problems, sep="\n  - ")
        sys.exit(1)

    traces, labels, clean_ids, anomalous_ids = load_case_traces()
    split, _, _ = check_split(anomalous_ids)
    tree, net, im, fm, _, model_check = discover_and_check(traces, clean_ids)

    out = EXP_DIR / "dry_run"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "dry_run",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pm4py": pm4py.__version__,
        "pandas": pd.__version__,
        "python_executable": sys.executable,
        "alignment_variant": str(ALIGNMENT_VARIANT),
        "n_cases_total": len(labels),
        "n_clean_cases": len(clean_ids),
        "n_anomalous_cases": len(anomalous_ids),
        "split_check": split,
        "model_check": model_check,
    }
    (out / f"{FILE_PREFIX}_dry_run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"{FILE_PREFIX}_process_tree.txt").write_text(str(tree) + "\n", encoding="utf-8")
    pm4py.write_pnml(net, im, fm, str(out / f"{FILE_PREFIX}_petri_net.pnml"))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[DRY RUN] output: {out}")


# ============================================================
# 8. Official run: discovery + alignment repair
# ============================================================

def execute_run(run_id):
    problems = preflight(run_id)
    if problems:
        print("[ABORT]", *problems, sep="\n  - ")
        sys.exit(1)

    out = run_dir(run_id)
    out.mkdir(parents=True)
    run_file(run_id, "environment.txt").write_text(environment_text(), encoding="utf-8")
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    wall_start = time.time()
    started_at = datetime.now()
    log(f"Method B Run {run_id} start {started_at.isoformat(timespec='seconds')}")

    traces, labels, clean_ids, anomalous_ids = load_case_traces()
    split, test_ids, val_ids = check_split(anomalous_ids)
    log(f"split check: {split}")
    if not (split["test_expected_524"] and split["val_expected_523"] and split["test_val_disjoint"] and split["test_subset_of_anomalous"]):
        log("[ABORT] split check failed")
        sys.exit(1)

    tree, net, im, fm, language_set, model_check = discover_and_check(traces, clean_ids)
    log(f"model: places={model_check['n_places']} transitions={model_check['n_transitions']} "
        f"language={model_check['model_language_size']} equals_clean_variants={model_check['model_language_equals_clean_variants']} "
        f"clean_fitness={model_check['clean_log_fitness_alignment_weighted']}")
    run_file(run_id, "model_check.json").write_text(json.dumps(model_check, indent=2, ensure_ascii=False), encoding="utf-8")
    pm4py.write_pnml(net, im, fm, str(run_file(run_id, "petri_net.pnml")))

    # ----- alignment repair of test + val cases, linked by case id -----
    rows_log, rows_case = [], []
    align_start = time.time()
    for case_type, ids in [("test", test_ids), ("val", val_ids)]:
        for cid in ids:
            original = traces[cid]
            trace_obj = build_event_log([cid], traces)[0]
            t0 = time.time()
            try:
                res = align_case(trace_obj, net, im, fm)
                repaired, moves = repair_from_alignment(res["alignment"])
                failed, error = False, ""
                diag = {"alignment_cost": res["cost"], "fitness": res["fitness"],
                        "visited_states": res.get("visited_states"), "queued_states": res.get("queued_states")}
            except Exception as e:
                # fallback: keep the original anomalous trace and set the failure flag
                repaired, failed, error = list(original), True, repr(e)
                moves = {"n_sync_moves": None, "n_log_moves": None, "n_model_moves_visible": None, "n_tau_moves": None}
                diag = {"alignment_cost": None, "fitness": None, "visited_states": None, "queued_states": None}
            case_ms = (time.time() - t0) * 1000

            for i, act in enumerate(repaired, start=1):
                rows_log.append({"case:concept:name": cid, "concept:name": act, "order": i, "type": case_type, "alignment_failed": failed})
            rows_case.append(
                {
                    "case:concept:name": cid,
                    "type": case_type,
                    "anomaly_label": labels[cid],
                    "original_trace": ">>".join(original),
                    "repaired_trace": ">>".join(repaired),
                    "alignment_failed": failed,
                    "error": error,
                    "model_conformant": (None if language_set is None else tuple(repaired) in language_set) if not failed else False,
                    **diag,
                    **moves,
                    "align_ms": round(case_ms, 3),
                }
            )
    align_sec = time.time() - align_start

    repaired_df = pd.DataFrame(rows_log)
    per_case = pd.DataFrame(rows_case)
    repaired_df.to_csv(run_file(run_id, "repaired_log_b.csv"), index=False)
    # align_ms differs between runs, so it is stored separately from the reproducibility files
    per_case.drop(columns=["align_ms"]).to_csv(run_file(run_id, "alignment_per_case.csv"), index=False)
    per_case[["case:concept:name", "type", "align_ms"]].to_csv(run_file(run_id, "alignment_timing.csv"), index=False)
    run_file(run_id, "test_case_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    run_file(run_id, "val_case_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    test_rows = per_case[per_case["type"] == "test"]
    stats = {
        "experiment": 'Method B alignment repair',
        "run_id": run_id,
        "official": run_id == 1,
        "data": DATA_NAME,
        "noise_threshold": NOISE_THRESHOLD,
        "alignment_variant": str(ALIGNMENT_VARIANT),
        "n_test": int((per_case["type"] == "test").sum()),
        "n_val": int((per_case["type"] == "val").sum()),
        "n_alignment_failed_test": int(test_rows["alignment_failed"].sum()),
        "n_alignment_failed_val": int(per_case.loc[per_case["type"] == "val", "alignment_failed"].sum()),
        "n_model_conformant_test": int((test_rows["model_conformant"] == True).sum()),  # noqa: E712
        # Method B diagnostics below (not repair accuracy)
        "diag_test_mean_alignment_cost": float(test_rows["alignment_cost"].mean()),
        "diag_test_mean_fitness": float(test_rows["fitness"].mean()),
        "diag_test_total_log_moves": float(test_rows["n_log_moves"].sum()),
        "diag_test_total_model_moves_visible": float(test_rows["n_model_moves_visible"].sum()),
        "diag_test_n_cost_zero": int((test_rows["alignment_cost"] == 0).sum()),
        "repaired_rows": len(repaired_df),
        "repaired_rows_test": int((repaired_df["type"] == "test").sum()),
        "discovery_sec": model_check["discovery_sec"],
        "alignment_sec": round(align_sec, 4),
        "wall_sec": round(time.time() - wall_start, 4),
        "started_at": started_at.isoformat(timespec="seconds"),
        "pm4py": pm4py.__version__,
    }
    run_file(run_id, "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    log(json.dumps(stats, indent=2))
    run_file(run_id, "execution_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\nMethod B Run {run_id} finished -> {out}")


# ============================================================
# 9. Reproducibility comparison: run 1 vs run 2
# ============================================================

DETERMINISTIC_FILES = ["repaired_log_b.csv", "alignment_per_case.csv", "model_check.json", "petri_net.pnml",
                       "test_case_ids.txt", "val_case_ids.txt"]
TIMING_KEYS = {"wall_sec", "alignment_sec", "discovery_sec", "started_at", "run_id", "official"}


def compare_runs():
    for a, b in combinations((1, 2), 2):
        print(f"=== Run {a} vs Run {b} ===")
        for suffix in DETERMINISTIC_FILES:
            fa, fb = run_file(a, suffix), run_file(b, suffix)
            if suffix == "model_check.json":
                ja = {k: v for k, v in json.loads(fa.read_text(encoding="utf-8")).items() if k != "discovery_sec"}
                jb = {k: v for k, v in json.loads(fb.read_text(encoding="utf-8")).items() if k != "discovery_sec"}
                same = ja == jb
            else:
                same = filecmp.cmp(fa, fb, shallow=False)
            print(f"  {suffix:28s} identical: {same}")
        sa = {k: v for k, v in json.loads(run_file(a, "stats.json").read_text()).items() if k not in TIMING_KEYS}
        sb = {k: v for k, v in json.loads(run_file(b, "stats.json").read_text()).items() if k not in TIMING_KEYS}
        print(f"  stats.json (without timing)   identical: {sa == sb}")


# ============================================================
# 10. Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry_run", action="store_true")
    group.add_argument("--run_id", type=int, choices=(1, 2))
    group.add_argument("--compare_runs", action="store_true")
    args = parser.parse_args()

    try:
        if args.dry_run:
            dry_run()
        elif args.compare_runs:
            compare_runs()
        else:
            execute_run(args.run_id)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
