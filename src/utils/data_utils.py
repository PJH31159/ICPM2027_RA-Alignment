"""Data root resolution, input paths / checksums, and log reading / conversion."""

import os
from pathlib import Path

from utils.validation_utils import PROJECT_ROOT, file_record

ENV_DATA_ROOT = "ICPM2027_DATA_ROOT"


# ------------------------------------------------------------
# 1. Data root / input files
# ------------------------------------------------------------

def resolve_data_root(cli_root=None):
    """Priority: CLI --data-root > environment variable ICPM2027_DATA_ROOT > <repo>/data/raw."""
    if cli_root:
        return Path(cli_root).resolve(), "cli --data-root"
    env = os.environ.get(ENV_DATA_ROOT)
    if env:
        return Path(env).resolve(), f"env {ENV_DATA_ROOT}"
    return (PROJECT_ROOT / "data" / "raw").resolve(), "default data/raw"


def resolve_input_paths(cfg, data_root, anomaly_level):
    """Dataset config + data root -> anomaly log / ground-truth log paths.

    Detects the P-BEAR-RL layout (encoded_anomaly/, encoded_normal/) and the flat layout (data/raw).
    """
    stem = cfg["file_stem"]
    root = Path(data_root)
    if (root / "encoded_anomaly").is_dir():
        layout = "p-bear-rl (encoded_anomaly / encoded_normal)"
        anomaly = root / "encoded_anomaly" / f"{stem}_{anomaly_level}.csv"
        normal = root / "encoded_normal" / f"{stem}.csv"
    else:
        layout = "flat"
        anomaly = root / f"{stem}_{anomaly_level}.csv"
        normal = root / f"{stem}.csv"
    return {"anomaly_log": anomaly, "ground_truth_log": normal}, layout


def describe_inputs(input_paths):
    """Resolved path, size and SHA-256 of each input file."""
    return [file_record(role, path) for role, path in input_paths.items()]


# ------------------------------------------------------------
# 2. Observed log (PM4Py, file event order preserved)
# ------------------------------------------------------------

def load_case_traces(anomaly_log_path):
    """Per-case activity list, label, and clean / anomalous case ids."""
    import pandas as pd

    df = pd.read_csv(anomaly_log_path, usecols=["Case", "Activity", "type_res_trace"])
    traces = df.groupby("Case", sort=False)["Activity"].agg(list)
    labels = df.groupby("Case", sort=False)["type_res_trace"].first()
    clean_ids = labels[labels.isna()].index.tolist()
    anomalous_ids = labels[labels.notna()].index.tolist()
    return traces, labels, clean_ids, anomalous_ids


def build_event_log(case_ids, traces):
    """Build the EventLog directly: PM4Py dataframe conversion may re-sort events by timestamp."""
    from pm4py.objects.log.obj import Event, EventLog, Trace

    log = EventLog()
    for cid in case_ids:
        trace = Trace()
        trace.attributes["concept:name"] = cid
        for act in traces[cid]:
            trace.append(Event({"concept:name": act}))
        log.append(trace)
    return log


# ------------------------------------------------------------
# 3. Generated traces -> event log (RP-BEAR input format)
# ------------------------------------------------------------

def write_generated_log(generated_traces, path):
    """Write each generated trace once (weight 1) as a Case/Activity CSV. Case ids: gen_0001, ..."""
    import pandas as pd

    rows = []
    for i, trace in enumerate(generated_traces, start=1):
        for act in trace:
            rows.append({"Case": f"gen_{i:04d}", "Activity": act})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_log_like_sim(path, has_label):
    """Read a CSV with the same dtypes / column names as sim._load_and_prepare_data."""
    import numpy as np
    import pandas as pd

    usecols = ["Case", "Activity"] + (["Timestamp", "type_res_trace"] if has_label else [])
    dtype = {"Case": "category", "Activity": "category"}
    if has_label:
        dtype["type_res_trace"] = "category"
    df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    df = df.rename(columns={"Case": "case:concept:name", "Activity": "concept:name",
                            "Timestamp": "time:timestamp", "type_res_trace": "label"})
    if not has_label:
        df["time:timestamp"] = np.nan
        df["label"] = np.nan
    return df[["case:concept:name", "concept:name", "time:timestamp", "label"]]


def add_start_end_like_sim(data):
    """Same Start/End/order handling as sim._load_and_prepare_data() (different input only).

    Returns the rows of cases with an empty label (same format as `clean` in sim).
    """
    import pandas as pd

    data = data.copy()
    data["order"] = data.groupby("case:concept:name", observed=True).cumcount() + 1

    start_df = data[data["order"] == 1].copy()
    start_df.loc[:, "order"] = 0
    start_df.loc[:, "concept:name"] = "Start"

    end_indices = data.groupby("case:concept:name", observed=True)["order"].idxmax()
    end_df = data.loc[end_indices].copy()
    end_df.loc[:, "order"] = end_df["order"] + 1
    end_df.loc[:, "concept:name"] = "End"

    data2 = (
        pd.concat([data, start_df, end_df], ignore_index=True)
        .sort_values(by=["case:concept:name", "order"], kind="stable")
        .reset_index(drop=True)
    )
    return data2[data2.label.isna()].reset_index(drop=True)
