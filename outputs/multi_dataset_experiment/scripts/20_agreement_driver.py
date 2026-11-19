"""Run the 07 all-optimal agreement as a single process with a memory guard and automatic resume (operational tool).

Some cases need more than about 10 GB RSS on their own before reaching the state limit (5M). So that the run does
not stop repeatedly at the same case, a case that reaches the memory guard is recorded as a resource-limit case
(07 --resource-skip; counted as disagreement in all-optimal agreement, i.e. lower bound) and the remaining cases
continue. The state-limit definition (--max-states) is unchanged.

Behaviour:
  1. run 07 with --skip-gen-check --resource-skip <file> (07 skips cases already in the checkpoint)
  2. every --interval seconds: log system available memory and the RSS / peak working set of the 07 process tree to a csv
  3. if available < --min-avail-gb: stop 07 -> record the running case (the first case that is neither finished nor
     skipped) in the skip file -> go back to 1
  4. finish when 07 exits with rc 0 (all cases done, agreement_summary.json written)
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[3]
S07 = PROJECT_ROOT / "outputs/multi_dataset_experiment/scripts/07_pub_all_optimal_agreement.py"


def done_ids(ck_dir):
    out = set()
    for f in ck_dir.glob("*.jsonl"):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                out.add(json.loads(ln)["row"]["case_id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--c-log", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-avail-gb", type=float, default=3.0)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--log-dir", required=True)
    a = ap.parse_args()
    out = PROJECT_ROOT / a.out
    ck_dir = out / "checkpoint"
    skip_file = out / "resource_limit_cases.tsv"
    mem_csv = out.parent / "memory_driver.csv"
    alg = pd.read_csv(next((PROJECT_ROOT / a.b_dir).glob("*_alignment_per_case.csv")), dtype={"case:concept:name": str})
    alg = alg[alg.type == "test"]
    order = list(alg["case:concept:name"])
    trace_len = {c: len(t.split(">>")) for c, t in zip(alg["case:concept:name"], alg["original_trace"])}
    b_cost = dict(zip(alg["case:concept:name"], alg["alignment_cost"]))
    new_csv = not mem_csv.exists()
    fh = mem_csv.open("a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new_csv:
        w.writerow(["time", "attempt", "avail_gb", "used_percent", "rss_gb", "peak_wset_gb", "done_cases", "event"])
    attempt = 0
    while True:
        attempt += 1
        cmd = [sys.executable, str(S07), "--b-dir", a.b_dir, "--c-log", a.c_log, "--gt", a.gt, "--out", a.out,
               "--skip-gen-check", "--resource-skip", str(skip_file)]
        lo = open(Path(a.log_dir) / f"bpic_agree_driver_attempt{attempt}.log", "w", encoding="utf-8")
        le = open(Path(a.log_dir) / f"bpic_agree_driver_attempt{attempt}.err", "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=lo, stderr=le)
        pp = psutil.Process(proc.pid)
        peak, killed = 0.0, False
        last_done = len(done_ids(ck_dir))
        case_start = time.time()
        while proc.poll() is None:
            time.sleep(a.interval)
            vm = psutil.virtual_memory()
            rss = pw = 0.0
            try:
                for q in [pp] + pp.children(recursive=True):
                    mi = q.memory_info()
                    rss += mi.rss
                    pw = max(pw, getattr(mi, "peak_wset", 0))
            except psutil.NoSuchProcess:
                pass
            peak = max(peak, pw / 2**30)
            nd = len(done_ids(ck_dir))
            if nd != last_done:
                last_done, case_start = nd, time.time()
            event = ""
            if vm.available / 2**30 < a.min_avail_gb and proc.poll() is None:
                d, sk = done_ids(ck_dir), set()
                if skip_file.exists():
                    sk = {ln.split("\t")[0] for ln in skip_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
                cur = next((c for c in order if c not in d and c not in sk), None)
                try:
                    for q in pp.children(recursive=True) + [pp]:
                        q.kill()
                except psutil.NoSuchProcess:
                    pass
                proc.wait()
                killed = True
                event = f"KILL case {cur}: available {vm.available / 2**30:.2f} GB < {a.min_avail_gb} GB"
                if cur is not None:
                    with skip_file.open("a", encoding="utf-8") as sf:
                        sf.write(f"{cur}\trss_gb={rss / 2**30:.2f}\tsec_in_case={time.time() - case_start:.0f}\t"
                                 f"trace_len={trace_len.get(cur)}\tB_cost={b_cost.get(cur)}\t"
                                 f"explored_states=N/A (killed before max_states)\t{datetime.now().isoformat(timespec='seconds')}\n")
            w.writerow([datetime.now().isoformat(timespec="seconds"), attempt, round(vm.available / 2**30, 2), vm.percent,
                        round(rss / 2**30, 2), round(pw / 2**30, 2), nd, event])
            fh.flush()
        lo.close()
        le.close()
        if killed:
            continue
        rc = proc.returncode
        w.writerow([datetime.now().isoformat(timespec="seconds"), attempt, "", "", "", round(peak, 2), len(done_ids(ck_dir)),
                    f"07 exited rc={rc}"])
        fh.flush()
        sys.exit(rc)


if __name__ == "__main__":
    main()
