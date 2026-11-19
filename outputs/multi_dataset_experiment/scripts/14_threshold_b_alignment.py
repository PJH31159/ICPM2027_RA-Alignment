"""Method B (alignment) test repair time per noise_threshold (sequential, timing only).

A simpler Petri net (higher threshold) may also speed up alignment. For every threshold the Method B repair
time is measured on the same test cases under the same conditions (sequential, one process, scipy LP shim).
Timed section = same as the official run-b: after building the trace object, align_case (PM4Py state-equation A*)
+ repair_from_alignment. Discovery, PNML loading and data loading are excluded.

Subcommands:
  split DS            (.venv-rpbear) same split as sim (seed 42, val_ratio 0.5) -> threshold_sweep/split_<DS>.json
  align DS TH [N] [T] (.venv)        align the test cases with sweep/petri_net.pnml.
                                     N = fixed random sample size (seed 42; omitted = all test cases)
                                     T = per-case alignment time limit in seconds (PM4Py PARAM_MAX_ALIGN_TIME_TRACE; omitted = none)
Output: outputs/simulation/<DS>/threshold_0pXX/b_alignment_timing/{case_timing.csv, summary.json}
"""
import json
import random
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]


def _data_root(cli_root=None):
    """Data root: --data-root argument > environment variable ICPM2027_DATA_ROOT > <repo>/data/raw.

    Same priority as src/utils/data_utils.resolve_data_root; it must point to the P-BEAR-RL Full_input folder.
    """
    import os
    if cli_root is None and "--data-root" in sys.argv:
        cli_root = sys.argv[sys.argv.index("--data-root") + 1]
    if cli_root:
        return Path(cli_root).resolve()
    env = os.environ.get("ICPM2027_DATA_ROOT")
    return Path(env).resolve() if env else (PROJECT_ROOT / "data" / "raw").resolve()


def _strip_data_root(argv):
    """Remove '--data-root X' from argv before positional arguments are read (existing argument handling unchanged)."""
    if "--data-root" in argv:
        i = argv.index("--data-root")
        del argv[i:i + 2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
OUT = PROJECT_ROOT / "outputs/multi_dataset_experiment/threshold_sweep"
SIM = PROJECT_ROOT / "outputs/simulation"
FULL = _data_root()
INPUTS = {
    "RoadTraffic": {"anomaly_log": SIM / "RoadTraffic/exploratory_dedup/derived_data/Road_Traffic_1.00_dedup.csv",
                    "ground_truth_log": FULL / "encoded_normal/Road_Traffic.csv", "name": "Road_Traffic_dedup"},
    "BPIC2017": {"anomaly_log": FULL / "encoded_anomaly/b17_1.00.csv", "ground_truth_log": FULL / "encoded_normal/b17.csv",
                 "name": "b17"},
}


def tag(th):
    return f"threshold_{th:.2f}".replace(".", "p")


def split(ds):
    from utils.run_utils import expected_split

    inp = INPUTS[ds]
    val, test = expected_split(inp["name"], {k: inp[k] for k in ("anomaly_log", "ground_truth_log")}, 0.5, 42)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"split_{ds}.json").write_text(json.dumps({"seed": 42, "val_ratio": 0.5, "val": val, "test": test}), encoding="utf-8")
    print(ds, "val", len(val), "test", len(test))


def align(ds, th, n_sample=None, timeout=None):
    sys.modules["cvxopt"] = None  # scipy LP shim (same as shim_simulation.py)
    import lxml
    from lxml import etree as _etree

    _obj = types.ModuleType("lxml.objectify")
    _obj.parse, _obj.fromstring = _etree.parse, _etree.fromstring
    sys.modules["lxml.objectify"] = _obj
    lxml.objectify = _obj
    import numpy as np
    import pandas as pd
    import pm4py
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo

    from utils.alignment_utils import load_exp061
    from utils.data_utils import build_event_log, load_case_traces

    m = load_exp061()
    d = SIM / ds / tag(th) / "b_alignment_timing"
    d.mkdir(parents=True, exist_ok=True)
    net, im, fm = pm4py.read_pnml(str(SIM / ds / tag(th) / "sweep" / "petri_net.pnml"))
    traces, _l, _c, _a = load_case_traces(INPUTS[ds]["anomaly_log"])
    traces.index = traces.index.map(str)
    test = json.loads((OUT / f"split_{ds}.json").read_text(encoding="utf-8"))["test"]
    ids = test if n_sample is None else sorted(random.Random(42).sample(sorted(test), n_sample))
    params = {} if timeout is None else {alignments_algo.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: timeout}
    rows = []
    t_phase = time.perf_counter()
    for i, cid in enumerate(ids):
        trace_obj = build_event_log([cid], traces)[0]
        t0 = time.perf_counter()
        status = "ok"
        try:
            res = alignments_algo.apply(trace_obj, net, im, fm, variant=m.ALIGNMENT_VARIANT, parameters=params)
            if res is None or res.get("alignment") is None:
                raise RuntimeError("alignment returned None")
            repaired, _mv = m.repair_from_alignment(res["alignment"])
            cost = res["cost"]
        except Exception as e:  # noqa: BLE001  same fallback as the official run (keep the original trace)
            repaired, cost = list(traces[cid]), None
            status = "timeout_or_failed: " + repr(e)[:80]
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"case_id": cid, "repair_time_ms": round(ms, 4), "status": status, "alignment_cost": cost,
                     "trace_length": len(traces[cid]), "repaired_trace": ">>".join(map(str, repaired))})
        if (i + 1) % 200 == 0 or (n_sample and (i + 1) % 10 == 0):
            print(f"  {i + 1}/{len(ids)} {(time.perf_counter() - t_phase):.1f}s", flush=True)
    wall = (time.perf_counter() - t_phase) * 1000
    df = pd.DataFrame(rows)
    df.to_csv(d / "case_timing.csv", index=False)
    ok = df[df.status == "ok"]["repair_time_ms"]
    allv = df["repair_time_ms"]
    summ = {"dataset": ds, "noise_threshold": th, "net": {"places": len(net.places), "transitions": len(net.transitions),
                                                          "arcs": len(net.arcs)},
            "scope": "per case: PM4Py state-equation A* alignment + repair_from_alignment (EXP-061 code); "
                     "sequential, 1 process, scipy LP (shim). excludes discovery / loading / PNML read",
            "cases": "all test cases" if n_sample is None else f"fixed random sample {n_sample} of {len(test)} test cases (seed 42)",
            "per_case_timeout_sec": timeout, "n": len(df), "n_ok": int(len(ok)), "n_timeout_or_failed": int(len(df) - len(ok)),
            "mean_ms": round(float(allv.mean()), 4), "median_ms": round(float(allv.median()), 4),
            "p95_ms": round(float(np.percentile(allv, 95)), 4), "total_ms": round(float(allv.sum()), 4),
            "max_ms": round(float(allv.max()), 4),
            "mean_ms_ok_only": round(float(ok.mean()), 4) if len(ok) else None, "phase_wall_ms": round(wall, 4),
            "note": "timeout case는 timeout까지 걸린 시간으로 mean에 포함 (실제 alignment 시간보다 작게 잡힘 = 하한)"}
    (d / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _strip_data_root(sys.argv)
    cmd = sys.argv[1]
    if cmd == "split":
        split(sys.argv[2])
    elif cmd == "align":
        a = sys.argv[2:]
        align(a[0], float(a[1]), int(a[2]) if len(a) > 2 and a[2] != "all" else None, float(a[3]) if len(a) > 3 else None)
