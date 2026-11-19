"""Logical per-dataset settings in one place.

No machine-specific absolute paths are stored here. The data root is resolved at run time
(utils/data_utils.resolve_data_root):
    1. CLI --data-root
    2. environment variable ICPM2027_DATA_ROOT
    3. <repo>/data/raw

File layout under the data root (detected automatically):
    - P-BEAR-RL layout: <root>/encoded_anomaly/<file_stem>_<level>.csv,
                        <root>/encoded_normal/<file_stem>.csv
    - flat layout (data/raw): <root>/<file_stem>_<level>.csv, <root>/<file_stem>.csv
"""

from copy import deepcopy

# ------------------------------------------------------------
# 1. Dataset order
# ------------------------------------------------------------
# Each dataset is run independently; continual / cumulative Q-table learning is not implemented.
DATASET_ORDER = ["Small", "Medium", "Large", "Wide", "Credit", "Road"]
CONTINUAL_LEARNING = None  # None = not used

# ------------------------------------------------------------
# 2. Common run settings
# ------------------------------------------------------------
ANOMALY_LEVEL = "1.00"
ENV_DIRS = {"pm4py": ".venv", "rpbear": ".venv-rpbear"}   # relative to the repository root

RPBEAR_PARAMS = {  # official RP-BEAR parameters
    "num_epi": 1000,
    "alpha": 0.0,
    "val_ratio": 0.5,
    "seed": 42,
    "max_step": 8,
    "use_multiprocessing": True,
    "num_cores": None,  # None -> cpu_count() - 1
}
SMOKE_NUM_EPI = 5

DISCOVERY = {"noise_threshold": 0.0}   # Inductive Miner

# Trace generation defaults.
# safety_* values only prevent unbounded search; they are not the research parameters
# loop_num / loop_total. With strategy=None the generate stage only reports the model structure.
DEFAULT_TRACE_GENERATION = {
    "strategy": None,  # None | "exhaustive_acyclic" | "bounded_loop" (loop models: loop_num / loop_total)
    "safety_max_firings_per_transition": 5,
    "safety_max_trace_length": 100,
    "safety_max_traces": 1_000_000,
}

# ------------------------------------------------------------
# 3. Per-dataset settings
# ------------------------------------------------------------
DATASETS = {
    "Small": {
        "file_stem": "Small",
        "identity_confirmed": True,
        "identity_note": "EXP-060~063에서 사용한 Small (data/raw와 P-BEAR-RL 파일이 byte 동일)",
        "trace_generation": {"strategy": "exhaustive_acyclic"},  # acyclic model: enumerate the full model language
        "expected_model": {  # regression check of the Small pipeline
            "n_places": 16, "n_transitions": 20, "n_invisible_transitions": 0, "n_arcs": 40,
            "n_clean_cases": 11453, "n_clean_variants": 6,
        },
        # existing Small results used by the regression stage (relative to the repository root)
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
        # existing official results (evaluate --with-reference, regression check)
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
        # No LOOP / PARALLEL / cycle and model language = observed variants, so exhaustive enumeration is
        # allowed (the generate stage checks this again).
        "trace_generation": {"strategy": "exhaustive_acyclic"},
        "expected_model": None,  # first run on this dataset: observations only, no expected values
    },
    "Large": {
        "file_stem": "Large",
        "identity_confirmed": True,
        "identity_note": "정상 log Large.csv는 data/raw와 P-BEAR-RL이 byte 동일 (Phase 2에서 사용). "
                         "이상 log Large_1.00.csv는 P-BEAR-RL에만 있음",
        # same generation definition as Small (acyclic model)
        "trace_generation": {"strategy": "exhaustive_acyclic"},
        "expected_model": None,
    },
    "Wide": {
        "file_stem": "Wide",
        "identity_confirmed": True,
        "identity_note": "P-BEAR-RL Full_input의 Wide (아직 이 프로젝트에서 사용한 적 없음)",
        # No LOOP / PARALLEL / cycle and model language = observed variants, so exhaustive enumeration is
        # allowed (the generate stage checks this again).
        "trace_generation": {"strategy": "exhaustive_acyclic"},
        "expected_model": None,
    },
    "Credit": {
        "file_stem": "credit-card",
        "identity_confirmed": False,
        "identity_note": "TODO: 교수님이 말한 'Credit'이 P-BEAR-RL의 credit-card인지 재확인 필요 "
                         "(README에서는 Credit으로 표기)",
        # loop model: bounded generation with the official loop bound (5,7)
        "trace_generation": {
            "strategy": "bounded_loop", "loop_num": 5, "loop_total": 7,
            "reference_traces": "outputs/credit_loop_bound_sensitivity/credit_loop_bound_final_traces.csv",
        },
        # Method C New NBG = observed clean log + model-only generated traces (W2, Expanded New NBG)
        "new_nbg_input": "original_plus_model_only",
        # expected values of the preparation check (alpha 0)
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
        # One LOOP -> loop_total = loop_num. Main bound (4,4): from (4,4) to (5,5) the generated traces grow
        # from 6,339 to 14,907 while clean coverage only goes from 99.98% to 100% (elbow). (3,3) / (5,5) are sensitivity bounds.
        "trace_generation": {"strategy": "bounded_loop", "loop_num": 4, "loop_total": 4, "reference_traces": None},
        "new_nbg_input": "original_plus_model_only",  # W2, same as Credit
        "expected_preparation": None,  # first run on this dataset: observations only, no expected values
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
    # Road / BPIC: Method C unchanged; the Petri net is simplified via the discovery noise_threshold.
    # Per-threshold runs: --noise-threshold X --loop-bound n,t  (output: outputs/simulation/<name>/threshold_0pXX/)
    "RoadTraffic": {
        "file_stem": "Road_Traffic",
        "run_name": "Road_Traffic_dedup",
        "identity_confirmed": True,
        "identity_note": "P-BEAR-RL Full_input Road_Traffic. Road exploratory preprocessing: duplicate-event derived copy "
                         "(08_road_raw_vs_dedup.py: clean case 중 모든 event가 정확히 두 번씩 들어간 475건을 한 번씩으로 복원, "
                         "anomaly case는 그대로). 원본 raw 파일은 수정하지 않는다. 교수님의 명시적 dedup 승인은 아직 없음",
        "anomaly_log_override": "outputs/simulation/RoadTraffic/exploratory_dedup/derived_data/Road_Traffic_1.00_dedup.csv",
        # the loop bound depends on the threshold (13_noise_threshold_sweep.py); set it with --loop-bound
        "trace_generation": {"strategy": "bounded_loop", "loop_num": 1, "loop_total": 1, "reference_traces": None},
        "new_nbg_input": "original_plus_model_only",  # W2, same as Credit / Pub
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
    """Return the dataset settings merged with the defaults."""
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
    cfg.setdefault("new_nbg_input", "generated_only")  # Small-Wide: "generated_only"; loop datasets: "original_plus_model_only" (W2)
    cfg.setdefault("expected_preparation", None)
    return cfg
