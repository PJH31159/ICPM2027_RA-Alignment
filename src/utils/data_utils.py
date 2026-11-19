"""
data_utils — 데이터 root 결정, 입력 파일 경로/checksum, 로그 읽기와 변환 (REF-001)

출처:
    load_case_traces, build_event_log         <- src/61_method_b_alignment_repair.py
    read_log_like_sim, add_start_end_like_sim  <- src/63_generated_nbg_rpbear.py
    write_generated_log                        <- src/63 stage_generate
"""

import os
from pathlib import Path

from utils.validation_utils import PROJECT_ROOT, file_record

ENV_DATA_ROOT = "ICPM2027_DATA_ROOT"


# ------------------------------------------------------------
# 1. 데이터 root / 입력 파일
# ------------------------------------------------------------

def resolve_data_root(cli_root=None):
    """우선순위: CLI --data-root > 환경변수 ICPM2027_DATA_ROOT > 프로젝트 data/raw."""
    if cli_root:
        return Path(cli_root).resolve(), "cli --data-root"
    env = os.environ.get(ENV_DATA_ROOT)
    if env:
        return Path(env).resolve(), f"env {ENV_DATA_ROOT}"
    return (PROJECT_ROOT / "data" / "raw").resolve(), "default data/raw"


def resolve_input_paths(cfg, data_root, anomaly_level):
    """dataset 설정 + data root -> 이상 log / ground-truth log 경로.

    P-BEAR-RL 형식(encoded_anomaly/, encoded_normal/)과 flat 형식(data/raw)을 자동 인식한다.
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
    """각 입력 파일의 resolved path / size / SHA-256."""
    return [file_record(role, path) for role, path in input_paths.items()]


# ------------------------------------------------------------
# 2. 관측 log 읽기 (pm4py용, 파일 event 순서 유지)
# ------------------------------------------------------------

def load_case_traces(anomaly_log_path):
    """case별 activity list, label, clean / anomalous case id (EXP-061과 동일)."""
    import pandas as pd

    df = pd.read_csv(anomaly_log_path, usecols=["Case", "Activity", "type_res_trace"])
    traces = df.groupby("Case", sort=False)["Activity"].agg(list)
    labels = df.groupby("Case", sort=False)["type_res_trace"].first()
    clean_ids = labels[labels.isna()].index.tolist()
    anomalous_ids = labels[labels.notna()].index.tolist()
    return traces, labels, clean_ids, anomalous_ids


def build_event_log(case_ids, traces):
    """pm4py dataframe 변환은 timestamp로 정렬할 수 있으므로 EventLog를 직접 만든다."""
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
# 3. 생성 trace -> event log (RP-BEAR 입력 형식)
# ------------------------------------------------------------

def write_generated_log(generated_traces, path):
    """생성 trace를 1회씩(weight 1) Case/Activity CSV로 쓴다. case id = gen_0001, ..."""
    import pandas as pd

    rows = []
    for i, trace in enumerate(generated_traces, start=1):
        for act in trace:
            rows.append({"Case": f"gen_{i:04d}", "Activity": act})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_log_like_sim(path, has_label):
    """sim._load_and_prepare_data와 같은 dtype/열 이름으로 CSV를 읽는다."""
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
    """sim._load_and_prepare_data()의 Start/End/order 처리와 같은 코드 (입력만 다름).

    return: label이 비어 있는 case의 rows (= sim의 `clean`과 같은 형식)
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
