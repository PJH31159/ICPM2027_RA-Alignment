"""20_agreement_driver.py — 07 all-optimal agreement를 1 process로 실행하며 메모리를 감시하고, 위험 시 resume (운영 도구)

왜 (BPIC 2017, 2026-11-16):
  - 4 shard 동시 실행 -> PC 가상 메모리 부족 (Windows Event 2004).
  - 2 shard -> 두 shard가 동시에 무거운 case에 걸려 watchdog 종료.
  - 1 process -> case 하나가 단독으로 RSS ~10 GB 이상 (state 상한 5M에 닿기 전) -> watchdog 종료.
  같은 case에서 계속 멈추지 않도록, 메모리 위험 수준에 닿은 case는 resource-limit case로 기록하고
  (07 --resource-skip, all-optimal에서 불일치 = 하한) 나머지를 이어서 계산한다.
  state-limit 정의(--max-states)는 바꾸지 않는다.

동작:
  1. 07을 --skip-gen-check --resource-skip <file> 로 실행 (checkpoint의 완료 case는 07이 건너뜀)
  2. --interval 초마다 시스템 available, 07 process tree RSS / peak working set을 csv에 기록
  3. available < --min-avail-gb 이면 07 종료 -> 진행 중이던 case(완료도 skip도 아닌 첫 case)를 skip 파일에 기록 -> 1부터 다시
  4. 07이 rc 0으로 끝나면 (전체 case 완료 + agreement_summary.json 작성) 종료
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
