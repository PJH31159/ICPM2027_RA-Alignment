"""
dataset_config.py — dataset별 "논리 정보"를 한 곳에 모은 설정 파일 (REF-001)

여기에는 PC마다 달라지는 절대경로를 넣지 않는다. 데이터 root는 실행할 때
다음 우선순위로 정한다 (utils/data_utils.resolve_data_root):
    1. CLI --data-root
    2. 환경변수 ICPM2027_DATA_ROOT
    3. 프로젝트 내부 data/raw

데이터 root 아래 파일 위치 (두 가지 layout을 자동 인식):
    - P-BEAR-RL 형식: <root>/encoded_anomaly/<file_stem>_<level>.csv,
                      <root>/encoded_normal/<file_stem>.csv
    - flat 형식(data/raw): <root>/<file_stem>_<level>.csv, <root>/<file_stem>.csv

아직 확인되지 않은 값은 확정하지 않고 TODO / identity_confirmed=False로 남긴다.
"""

from copy import deepcopy

# ------------------------------------------------------------
# 1. Dataset 순서 (교수님 언급, 2026-09-28)
# ------------------------------------------------------------
# 기록용 순서이다. "연달아 학습"이 하나의 Q-table을 이어서 학습한다는 뜻인지는
# 아직 교수님께 재확인하지 않았다 (STATE.md Section 26). 따라서 현재는 각 dataset을
# 독립적으로 실행하며, continual / cumulative Q-table learning은 구현하지 않는다.
DATASET_ORDER = ["Small", "Medium", "Large", "Wide", "Credit", "Road"]
CONTINUAL_LEARNING = None  # TODO: 교수님 확인 후 결정 (None = 사용하지 않음)

# ------------------------------------------------------------
# 2. 공통 실행 조건
# ------------------------------------------------------------
ANOMALY_LEVEL = "1.00"                       # <file_stem>_1.00.csv
ENV_DIRS = {"pm4py": ".venv", "rpbear": ".venv-rpbear"}   # 프로젝트 기준 상대경로

RPBEAR_PARAMS = {  # D031 (EXP-060과 같은 조건)
    "num_epi": 1000,
    "alpha": 0.0,
    "val_ratio": 0.5,
    "seed": 42,
    "max_step": 8,
    "use_multiprocessing": True,
    "num_cores": None,  # None -> cpu_count() - 1
}
SMOKE_NUM_EPI = 5

DISCOVERY = {"noise_threshold": 0.0}   # Inductive Miner (D032)

# Trace generation 기본값.
# safety_* 값은 무한 탐색을 막는 안전장치일 뿐이며, 연구에서 정의한
# loop_num / loop_total과 같은 개념이 아니다 (D004, D034).
# strategy=None이면 generate 단계는 모델 구조만 보고하고 trace를 만들지 않는다.
DEFAULT_TRACE_GENERATION = {
    "strategy": None,  # None | "exhaustive_acyclic" | "bounded_loop" (loop 모델: loop_num / loop_total, D004 / D038)
    "safety_max_firings_per_transition": 5,
    "safety_max_trace_length": 100,
    "safety_max_traces": 1_000_000,
}

# ------------------------------------------------------------
# 3. Dataset별 정보
# ------------------------------------------------------------
DATASETS = {
    "Small": {
        "file_stem": "Small",
        "identity_confirmed": True,
        "identity_note": "EXP-060~063에서 사용한 Small (data/raw와 P-BEAR-RL 파일이 byte 동일)",
        "trace_generation": {"strategy": "exhaustive_acyclic"},  # EXP-063에서 검증한 전략
        "expected_model": {  # 회귀 검증용 (EXP-061/063)
            "n_places": 16, "n_transitions": 20, "n_invisible_transitions": 0, "n_arcs": 40,
            "n_clean_cases": 11453, "n_clean_variants": 6,
        },
        # Small 회귀 검증에 쓰는 기존 결과 (프로젝트 기준 상대경로)
        "regression": {
            "exp063_model_check": "outputs/exp063_generated_nbg_rpbear/generated_traces/model_check.json",
            "exp063_generated_log": "outputs/exp063_generated_nbg_rpbear/generated_traces/generated_normal_log.csv",
            "exp063_case_level": "outputs/exp063_generated_nbg_rpbear/summary/exp063_case_level.csv",
            "v2_seq_reference": "outputs/reference_runs/validation_v012/results/v2_seq_1/repaired_log.csv",
            "split_val_ids": "outputs/exp060_repaired_log_a/run1/Small_1000_ppm0.0_valratio0.5_seed42_run1_val_case_ids.txt",
            "split_test_ids": "outputs/exp060_repaired_log_a/run1/Small_1000_ppm0.0_valratio0.5_seed42_run1_test_case_ids.txt",
            "edge_digest": "c2a535271bbdec62",
            "likelihood_diff_edges": 64,
            "likelihood_diff_graphs": 10,
            "anchor_match": 22,
            "adm_equal_alpha0": 1047,
        },
        # 기존 공식 결과 (evaluate --with-reference, 회귀 검증용)
        "reference_runs": {
            "A_run1": "outputs/exp060_repaired_log_a/run1/Small_1000_ppm0.0_valratio0.5_seed42_run1_repaired_log_a.csv",
            "A_run2": "outputs/exp060_repaired_log_a/run2/Small_1000_ppm0.0_valratio0.5_seed42_run2_repaired_log_a.csv",
            "A_run3": "outputs/exp060_repaired_log_a/run3/Small_1000_ppm0.0_valratio0.5_seed42_run3_repaired_log_a.csv",
            "B_run1": "outputs/exp061_method_b_alignment/run1/Small_IM_noise0_alignment_run1_repaired_log_b.csv",
            "B_run2": "outputs/exp061_method_b_alignment/run2/Small_IM_noise0_alignment_run2_repaired_log_b.csv",
            "C_run1": "outputs/exp063_generated_nbg_rpbear/run1/Small_1000_ppm0.0_valratio0.5_seed42_generatedNBG_run1_repaired_log_c.csv",
        },
        "reference_expected_exact": {"A_run1": 501, "A_run2": 502, "A_run3": 507,
                                     "B_run1": 521, "B_run2": 522, "C_run1": 501},
    },
    "Medium": {
        "file_stem": "Medium",
        "identity_confirmed": True,
        "identity_note": "P-BEAR-RL Full_input의 Medium (아직 이 프로젝트에서 사용한 적 없음)",
        # 2026-10-12: Small(EXP-063)과 같은 생성 정의. screening에서 LOOP 0 / PARALLEL 0 / cycle 없음,
        # 모델 언어 = 관측 variant 수를 확인했으므로 exhaustive enumeration 허용 (generate 단계가 다시 검사함)
        "trace_generation": {"strategy": "exhaustive_acyclic"},
        "expected_model": None,  # 처음 보는 dataset: 기대값 없이 관찰만 기록
    },
    "Large": {
        "file_stem": "Large",
        "identity_confirmed": True,
        "identity_note": "정상 log Large.csv는 data/raw와 P-BEAR-RL이 byte 동일 (Phase 2에서 사용). "
                         "이상 log Large_1.00.csv는 P-BEAR-RL에만 있음",
        # 2026-10-12: Small(EXP-063)과 같은 생성 정의 (Medium / Wide와 같은 이유)
        "trace_generation": {"strategy": "exhaustive_acyclic"},
        "expected_model": None,
    },
    "Wide": {
        "file_stem": "Wide",
        "identity_confirmed": True,
        "identity_note": "P-BEAR-RL Full_input의 Wide (아직 이 프로젝트에서 사용한 적 없음)",
        # 2026-10-12: Small(EXP-063)과 같은 생성 정의. screening에서 LOOP 0 / PARALLEL 0 / cycle 없음,
        # 모델 언어 = 관측 variant 수를 확인했으므로 exhaustive enumeration 허용 (generate 단계가 다시 검사함)
        "trace_generation": {"strategy": "exhaustive_acyclic"},
        "expected_model": None,
    },
    "Credit": {
        "file_stem": "credit-card",
        "identity_confirmed": False,
        "identity_note": "TODO: 교수님이 말한 'Credit'이 P-BEAR-RL의 credit-card인지 재확인 필요 "
                         "(README에서는 Credit으로 표기)",
        # EXP-066 / D038: loop 모델이므로 공식 loop bound (5,7)로 bounded generation (D004 semantics)
        "trace_generation": {
            "strategy": "bounded_loop", "loop_num": 5, "loop_total": 7,
            "reference_traces": "outputs/credit_loop_bound_sensitivity/credit_loop_bound_final_traces.csv",
        },
        # D039: Method C New NBG = observed clean log + model-only generated traces (W2, Expanded New NBG)
        "new_nbg_input": "original_plus_model_only",
        # EXP-067 준비 검증 기준 (EXP-066 / credit_nbg_analysis 탐색 결과, alpha 0)
        "expected_preparation": {
            "generated_unique": 5000, "observed_in_generated": 19, "model_only": 4981,
            "original_nbg": {"graphs": 14, "nodes": 3750, "edges": 4107, "digest": "ffe91bf916897aa6"},
            "expanded_nbg": {"graphs": 14, "nodes": 4824, "edges": 7815, "digest": "657d7b11b2126d50"},
            "clean_false_positive": 0, "anomaly_miss": 41, "newly_accepted_anomalies": 15,
            "uncovered_clean_cases": 90, "uncovered_clean_normal": 90,
        },
        "expected_model": None,
    },
    "Pub": {
        "file_stem": "mccloud",
        "identity_confirmed": True,
        "identity_note": "P-BEAR-RL README 'credit-card, mccloud (named as Credit, Pub in our paper)' + 교수님 확인 "
                         "(2026-11-10). normal = encoded_normal/mccloud.csv, anomaly = encoded_anomaly/mccloud_1.00.csv",
        # 2026-11-10 multi-dataset 확장 (탐색 단계 결정, 연구 문서에는 아직 확정 반영하지 않음):
        # LOOP 1개 -> loop_total = loop_num. main (4,4): (4,4)->(5,5)에서 생성 6,339 -> 14,907, coverage 99.98% -> 100%
        # (elbow). (3,3) / (5,5)는 sensitivity. 근거: outputs/simulation/Pub/dataset_audit/generation_feasibility.csv
        "trace_generation": {"strategy": "bounded_loop", "loop_num": 4, "loop_total": 4, "reference_traces": None},
        "new_nbg_input": "original_plus_model_only",  # Credit과 같은 W2 (D039)
        "expected_preparation": None,  # 처음 보는 dataset: 기대값 없이 관찰만 기록
        "expected_model": None,
    },
    "Road": {
        "file_stem": "Road_Traffic",
        "identity_confirmed": False,
        "identity_note": "TODO: 교수님이 말한 'Road'가 P-BEAR-RL의 Road_Traffic인지 재확인 필요. "
                         "또한 이 Road_Traffic CSV가 EXP-039~058에서 사용한 Road Traffic XES와 "
                         "같은 source/process인지 이름만으로 가정하지 않는다. Road 단계 전에 "
                         "dataset identity 검증이 먼저 필요하다.",
        "expected_model": None,
    },
    # 2026-11-12 (교수님: Road / BPIC은 기존 Method C 그대로, Process Discovery noise_threshold로 Petri Net 경량화).
    # threshold별 실행: --noise-threshold X --loop-bound n,t  (출력: outputs/simulation/<name>/threshold_0pXX/)
    "RoadTraffic": {
        "file_stem": "Road_Traffic",
        "run_name": "Road_Traffic_dedup",
        "identity_confirmed": True,
        "identity_note": "P-BEAR-RL Full_input Road_Traffic. Road exploratory preprocessing: duplicate-event derived copy "
                         "(08_road_raw_vs_dedup.py: clean case 중 모든 event가 정확히 두 번씩 들어간 475건을 한 번씩으로 복원, "
                         "anomaly case는 그대로). 원본 raw 파일은 수정하지 않는다. 교수님의 명시적 dedup 승인은 아직 없음",
        "anomaly_log_override": "outputs/simulation/RoadTraffic/exploratory_dedup/derived_data/Road_Traffic_1.00_dedup.csv",
        # loop bound는 threshold마다 다르다 (13_noise_threshold_sweep.py 규칙). CLI --loop-bound로 지정
        "trace_generation": {"strategy": "bounded_loop", "loop_num": 1, "loop_total": 1, "reference_traces": None},
        "new_nbg_input": "original_plus_model_only",  # W2 (D039), Credit / Pub과 같음
        "expected_preparation": None,
        "expected_model": None,
    },
    "BPIC2017": {
        "file_stem": "b17",
        "identity_confirmed": True,
        "identity_note": "P-BEAR-RL Full_input b17 (BPI Challenge 2017). normal = encoded_normal/b17.csv, "
                         "anomaly = encoded_anomaly/b17_1.00.csv",
        "trace_generation": {"strategy": "bounded_loop", "loop_num": 1, "loop_total": 4, "reference_traces": None},
        "new_nbg_input": "original_plus_model_only",
        "expected_preparation": None,
        "expected_model": None,
    },
}


def get_dataset(name):
    """dataset 설정을 기본값과 합쳐서 돌려준다."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. choose from {list(DATASETS)}")
    cfg = deepcopy(DATASETS[name])
    cfg["name"] = name
    tg = deepcopy(DEFAULT_TRACE_GENERATION)
    tg.update(cfg.get("trace_generation", {}))
    cfg["trace_generation"] = tg
    cfg.setdefault("expected_model", None)
    cfg.setdefault("regression", None)
    cfg.setdefault("reference_runs", {})
    cfg.setdefault("reference_expected_exact", {})
    cfg.setdefault("new_nbg_input", "generated_only")  # D034 (Small~Wide). Credit: "original_plus_model_only" (D039)
    cfg.setdefault("expected_preparation", None)
    return cfg
