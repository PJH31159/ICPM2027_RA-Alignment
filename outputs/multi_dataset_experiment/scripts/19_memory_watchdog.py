"""19_memory_watchdog.py — agreement shard 실행 중 메모리 기록 + 위험 시 중단 (실험 정의와 무관한 운영 도구)

왜: BPIC 2017 all-optimal agreement 4-shard 실행이 가상 메모리 부족(Windows Event 2004, python.exe 하나가 ~59 GB)으로
    PC 전체를 멈추게 했다 (2026-11-16). 같은 일이 다시 생기지 않도록 다음을 한다.
    - 60초마다 시스템 available / used 메모리와, 감시 대상 PID(및 자식)의 RSS / peak working set을 csv에 기록
    - available이 --min-avail-gb 아래로 떨어지면 감시 대상 process를 종료 (완료 case는 07의 checkpoint에 이미 저장됨)
    - 감시 대상이 모두 끝나면 종료

실행: python 19_memory_watchdog.py --pids 123,456 --labels shard0,shard1 --out <csv> [--min-avail-gb 4]
"""
import argparse
import csv
import time
from datetime import datetime

import psutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-avail-gb", type=float, default=4.0)
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    pids = [int(x) for x in a.pids.split(",")]
    labels = a.labels.split(",")
    procs = {}
    for pid, lab in zip(pids, labels):
        try:
            procs[lab] = psutil.Process(pid)
        except psutil.NoSuchProcess:
            pass
    peak = {lab: 0.0 for lab in labels}
    with open(a.out, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["time", "avail_gb", "used_percent"] + [f"{lab}_rss_gb" for lab in labels]
                       + [f"{lab}_peak_wset_gb" for lab in labels] + ["event"])
        while True:
            vm = psutil.virtual_memory()
            rss, pk, alive = {}, {}, 0
            for lab in labels:
                p = procs.get(lab)
                r = pw = 0.0
                if p is not None:
                    try:
                        for q in [p] + p.children(recursive=True):
                            mi = q.memory_info()
                            r += mi.rss
                            pw += getattr(mi, "peak_wset", 0)
                        alive += 1
                    except psutil.NoSuchProcess:
                        procs[lab] = None
                rss[lab], pk[lab] = r / 2**30, pw / 2**30
                peak[lab] = max(peak[lab], pk[lab])
            event = ""
            if vm.available / 2**30 < a.min_avail_gb and alive:
                event = f"KILL: available {vm.available / 2**30:.1f} GB < {a.min_avail_gb} GB"
                for p in procs.values():
                    if p is None:
                        continue
                    try:
                        for q in p.children(recursive=True) + [p]:
                            q.kill()
                    except psutil.NoSuchProcess:
                        pass
            w.writerow([datetime.now().isoformat(timespec="seconds"), round(vm.available / 2**30, 2), vm.percent]
                       + [round(rss[lab], 2) for lab in labels] + [round(peak[lab], 2) for lab in labels] + [event])
            fh.flush()
            if event or not alive:
                break
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
