"""Memory watchdog for all-optimal agreement runs (operational tool; does not affect the experiment definition).

The exhaustive all-optimal search can exhaust system memory when several heavy cases run at the same time.
    - every 60 s: log system available / used memory and the RSS / peak working set of the monitored PIDs
      (and their children) to a csv
    - if available memory drops below --min-avail-gb: terminate the monitored processes
      (finished cases are already stored in the checkpoint of 07)
    - exit when all monitored processes have finished

Usage: python 19_memory_watchdog.py --pids 123,456 --labels shard0,shard1 --out <csv> [--min-avail-gb 4]
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
