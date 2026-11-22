# ICPM-BPI27

This repository contains the implementation and experimental materials for our paper submitted to ICPM-BPI 2027.

> **Status: snapshot for code reference (private).** All official runs reported here are finished (BPIC 2017 C+PN
> included). Items marked **TODO** are not final yet. Raw event logs and trained models are **not** included
> (see [Data](#5-data) and [Pretrained models](#11-pretrained-models)).

---

## 1. Research goal

Event-log repair with reinforcement learning (RP-BEAR) normally learns what "normal" behaviour looks like
from the observed clean log only, encoded as a set of next-behaviour graphs (NBGs). We study whether a
**process model** can make that knowledge more complete:

- We discover a Petri net from the clean log.
- We generate a **bounded trace language** from the model, with a loop bound `(loop_num, loop_total)`, because a model
  with loops has an infinite language.
- We use those model traces to extend the NBGs (W2 "Expanded New NBG").

We then compare the repaired traces with an alignment-based repair (Method B). We measure both
**agreement** and **runtime** (training time and sequential per-case repair time).

The bounded trace language is always reported as *traces under a loop bound*, never as "all possible traces".

## 2. Methods

| Method | Idea | Training |
|---|---|---|
| **B: Alignment repair** | Optimal alignment (PM4Py state-equation A\*) of each anomalous trace against the discovered Petri net. The repaired trace is the model projection of the alignment. | none |
| **C: RP-BEAR + W2 New NBG** | RP-BEAR Q-learning repair. The NBGs are built from the observed clean log **plus** model-only traces from bounded generation (W2, `original_plus_model_only`). Small–Wide use generated traces only (`generated_only`). | RP-BEAR training phase |
| **C+PN: C + Petri-net gate** | Same as C. In addition, an episode may only terminate as "normal" if the repaired trace is accepted by the discovered Petri net (PN gate). | RP-BEAR training phase |

RP-BEAR itself (reward, penalty, Q-learning, ADM) is used **unchanged** (`src/rpbear/func_rpbear_final.py`).

## 3. Pipeline

```
anomaly log (clean + anomalous cases)
 ├─ clean cases ─► Process Discovery (Inductive Miner, noise_threshold) ─► Petri net
 │                   └─► bounded trace generation (loop bound) ─► model traces
 │                         └─► W2 New NBG = observed clean NBG ∪ model-only traces
 ├─ anomalous cases ─► split train / test (val_ratio, seed)
 │     ├─ Method B : alignment repair on test cases
 │     ├─ Method C : RP-BEAR training on train cases, repair of test cases
 │     └─ Method C+PN : as C, with the Petri-net acceptance gate
 └─ agreement evaluation (test cases)
       ├─ exact       : X repaired trace == B repaired trace
       ├─ native all-optimal       : X trace ∈ set of all optimal B repairs (PM4Py cost)
       └─ tau-excluded all-optimal : as above, with the invisible-transition (τ) cost excluded
```

All-optimal searches use a state limit (`--max-states 5,000,000`). Cases that hit the state limit or
the memory limit are counted as **disagreement**, so the reported all-optimal agreement is a **lower bound**.
The upper bound is the lower bound plus the number of limited cases.

## 4. Main files

| Path | Role |
|---|---|
| `src/simulation.py` | Stage orchestrator (`--dataset`, `--stage`). Runs discover / generate / nbg / smoke / run / run-b / run-c-pngate / test-timing\* |
| `src/sim.py` | RP-BEAR execution engine (case split, training / test phases, timing) |
| `src/dataset_config.py` | Dataset settings and official RP-BEAR parameters |
| `src/rpbear/func_rpbear_final.py` | RP-BEAR core (environment, Q-learning agent, NBG discovery). Not modified |
| `src/utils/` | discovery, bounded generation, NBG building, alignment, PN gate, timing, evaluation, validation helpers |
| `src/alignment/exact_alignment.py` | Method B (exact / optimal PM4Py alignment + repair): `align_case`, `repair_from_alignment`, used for every dataset. Loaded by path from `utils/alignment_utils.py`, `utils/timing_utils.py` (via `simulation.py`) and script `24`; keep this path or update those references. Its standalone CLI runs the Small reference run |
| `outputs/simulation/Credit/alignment_all_optimal_analysis/scripts/credit_all_optimal.py` | All-optimal search definitions (`dev_optimal`, `repair_from_alignment`). Loaded by `07_pub_all_optimal_agreement.py` |
| `outputs/credit_loop_bound_sensitivity/credit_loop_bound_final_traces.csv` | Credit reference trace set. The `generate` stage checks that bounded generation `(5,7)` reproduces it |
| `outputs/multi_dataset_experiment/scripts/` | Reproduction and analysis scripts (see below) |
| `outputs/multi_dataset_experiment/scripts/shim_simulation.py` | Runs `simulation.py` with the scipy LP solver (used for every `.venv` / Method B stage) |

Scripts in `outputs/multi_dataset_experiment/scripts/`:

| Script | Purpose |
|---|---|
| `06_credit_scipy_equivalence.py` | Method B: scipy vs cvxopt LP solver, case-by-case equivalence (Credit) |
| `07_pub_all_optimal_agreement.py` | Exact / native / tau-excluded all-optimal agreement (B–C, B–C+PN), with checkpoint and `--max-states` |
| `08_road_raw_vs_dedup.py` | Builds the Road Traffic duplicate-event derived copy used as the RoadTraffic input |
| `09_timing_summary.py` | Collects stored timing values into `timing_summary.csv/json` |
| `12_loop_requirement_analysis.py`, `13_noise_threshold_sweep.py`, `16_bpic_candidate_search.py` | noise_threshold / loop-bound selection for Road Traffic and BPIC 2017 |
| `14_threshold_b_alignment.py` | Method B repair time per noise_threshold |
| `15_build_paper_tables.py`, `17_bpic_feasibility_tables.py` | Build the paper tables from stored outputs |
| `18_exact_agreement.py` | Fast exact agreement from official repaired logs |
| `19_memory_watchdog.py`, `20_agreement_driver.py` | Memory-safe single-process driver for `07` (records resource-limit cases) |
| `22_agreement_from_checkpoint.py` | Re-uses B's optimal sets from a `07` checkpoint to evaluate another method (no search) |
| `24_bpic_complex_trace_timing.py` | BPIC 2017 case study: repeated B / C / C+PN timing on 4 traces with the longest all-optimal search (supplementary) |
| `25_bpic_complex_trace_candidates.py` | BPIC 2017 case study: rule-based selection and repeated verification of 10 further candidates (reuses `24`) |
| `26_bpic_complex_trace_tables.py` | Tables (PNG / Excel) of the case study from its stored CSV files |

Scripts `07`, `15`, `17`, `18`, `22`, `24`, `25` read **full** run outputs (per-case repaired logs, checkpoints, per-case
timings, agents). Those are not part of this snapshot, so the scripts only work after the corresponding stages have been
run locally (see [Pretrained models](#11-pretrained-models) for the planned distribution of these assets).

## 5. Data

The event logs are **not** included. **TODO:** confirm redistribution terms before adding data or links.

The pipeline expects the P-BEAR-RL `Full_input` layout:

```
<DATA_ROOT>/encoded_anomaly/<file_stem>_1.00.csv   # anomaly log (clean + anomalous cases, column type_res_trace)
<DATA_ROOT>/encoded_normal/<file_stem>.csv         # ground-truth normal log (evaluation only)
```

The data root is resolved as `--data-root` > environment variable `ICPM2027_DATA_ROOT` > `data/raw`.
This applies to `src/simulation.py` and to the analysis scripts `07`, `08`, `12`, `13`, `14` and `16`.
If you do not pass `--data-root`, set the environment variable instead, for example `export ICPM2027_DATA_ROOT=<DATA_ROOT>`.
The `data/raw` fallback only works if you place the `encoded_anomaly/` and `encoded_normal/` folders there.

## 6. Environment

There are two separate virtual environments (Python 3.13.9). RP-BEAR needs pandas 2.x; PM4Py runs on pandas 3.x.

```bash
python -m venv .venv            # discovery / generation / Method B / agreement
.venv/Scripts/python -m pip install -r requirements-pm4py.txt
python -m venv .venv-rpbear     # NBG / RP-BEAR (Method C, C+PN)
.venv-rpbear/Scripts/python -m pip install -r requirements-rpbear.txt
```

On Linux/macOS, use `bin/python` instead of `Scripts/python`. Stages that use `.venv` were run through
`shim_simulation.py` (LP solver = scipy). Small–Wide Method B timings in the earlier experiment were measured with cvxopt.

## 7. How to run (one dataset)

### Official arguments per dataset

The official noise threshold and loop bound of Road Traffic and BPIC 2017 are passed on the command line.
The `trace_generation` values stored for these two datasets in `src/dataset_config.py` are exploratory defaults
(threshold 0.0) and are **not** the official conditions. With `--noise-threshold`, outputs go to
`outputs/simulation/<Dataset>/threshold_0pXX/`. A noise threshold > 0 requires `PYTHONHASHSEED=0`, because the PM4Py
IMf result can depend on Python's hash order.

| Dataset | Arguments | Notes |
|---|---|---|
| Small, Medium, Large, Wide | `--dataset <name>` | exhaustive enumeration (acyclic models), threshold 0.0 |
| Credit | `--dataset Credit` | threshold 0.0, loop bound (5,7) from `dataset_config.py` |
| Pub | `--dataset Pub` | threshold 0.0, loop bound (4,4) from `dataset_config.py` |
| Road Traffic | `--dataset RoadTraffic --noise-threshold 0.2 --loop-bound 2,2` | run `08_road_raw_vs_dedup.py` first to create the derived input log; `PYTHONHASHSEED=0` |
| BPIC 2017 | `--dataset BPIC2017 --noise-threshold 0.1 --loop-bound 1,9` | `PYTHONHASHSEED=0` |

Example: BPIC 2017 with the official condition. Run from the repository root.

```bash
export PYTHONHASHSEED=0   # required for --noise-threshold > 0
A="--dataset BPIC2017 --noise-threshold 0.1 --loop-bound 1,9 --data-root <DATA_ROOT>"
VP=.venv/Scripts/python; RP=.venv-rpbear/Scripts/python; SH=outputs/multi_dataset_experiment/scripts/shim_simulation.py

$VP $SH $A --stage discover       # Petri net
$VP $SH $A --stage generate       # bounded trace language
$RP src/simulation.py $A --stage nbg                # Original / W2 NBG + train/test split files
$RP src/simulation.py $A --stage smoke              # 5-episode smoke check
$VP $SH $A --stage run-b --run-id 1                 # Method B
$RP src/simulation.py $A --stage run --method C --run-id 1            # Method C
$VP $SH $A --stage pngate-acceptor-check
$RP src/simulation.py $A --stage pngate-smoke
$RP src/simulation.py $A --stage run-c-pngate --run-id 1              # Method C+PN
# sequential test-only timing (replace <...> with the files written by the runs above)
$VP $SH $A --stage test-timing-b --pnml <methodB_run1/*_petri_net.pnml> --repaired-log <methodB_run1/*_repaired_log_b.csv> --label seq_B
$RP src/simulation.py $A --stage test-timing --method C --agent <methodC_run1/*_agent.pkl> --repaired-log <methodC_run1/*_repaired_log_c.csv> --label seq_C
$RP src/simulation.py $A --stage test-timing-pngate --run-id 1 --label seq_CPN
```

`shim_simulation.py` runs `src/simulation.py` itself, so it takes the same arguments.
**TODO:** check that the stage sequence above reproduces the snapshot on a clean machine. It was taken from the orchestration shell scripts
used in the experiments, which are not included because they contain machine-specific paths.

Agreement (after the runs):

```bash
.venv/Scripts/python outputs/multi_dataset_experiment/scripts/18_exact_agreement.py --b-dir <methodB_run1> --x-log <repaired_log_c.csv> --label C --out <dir>
.venv/Scripts/python outputs/multi_dataset_experiment/scripts/20_agreement_driver.py --b-dir <methodB_run1> --c-log <repaired_log_c.csv> --gt <DATA_ROOT>/encoded_normal/<file_stem>.csv --out <dir> --log-dir <dir>   # runs 07 with a memory guard
.venv/Scripts/python outputs/multi_dataset_experiment/scripts/22_agreement_from_checkpoint.py --ckpt-run <run1> --x-log <repaired_log_cpn.csv> --label CPN --out <dir> --check
```

**TODO:** document the remaining options of `07` (see `--help`).

Outputs go to `outputs/simulation/<Dataset>[/threshold_0pXX]/<stage>/`. A stage stops if its folder already exists, unless you pass `--overwrite`.

## 8. Official experiment conditions

| Parameter | Value |
|---|---|
| `val_ratio` | **0.5**. Anomalous cases are shuffled with the seed and split 50 % train ("validation" phase in the code) / 50 % test. Clean cases are not split. Discovery and NBGs use all clean cases |
| `seed` | 42 |
| `num_epi` | 1000 (training episodes per case) |
| `alpha` | 0 (ADM threshold) |
| `max_step` | 8 |
| multiprocessing | on, 15 workers (`cpu_count() - 1` on the experiment PC) |
| all-optimal `--max-states` | 5,000,000 |

These are the official conditions for every reported result: `val_ratio = 0.5`, `seed = 42`, `num_epi = 1000`,
`alpha = 0`, `max_step = 8`. They are set in `src/dataset_config.py` (`RPBEAR_PARAMS`).

In the code, "validation" means the RP-BEAR **training** phase: the Q-table is updated there. "test" is the held-out
set: the Q-table is not updated. Test cases whose anomalous trace is identical to a training trace reuse the repair
from RP-BEAR's repair memory.

Test inference as implemented in the official runs (`src/rpbear/func_rpbear_final.py`, unchanged RP-BEAR code):

- Actions are chosen epsilon-greedily (`select_action(state, greedy_only=0)`) with the exploration rate stored in the
  trained agent. For the BPIC 2017 C and C+PN agents the stored value is recorded as 0.5 (the initial value): during the
  multiprocessing training the decay is applied to the agent copies in the worker processes, not to the saved agent.
  The other agents were trained through the same code path, but their stored value was not recorded. The rate is not
  decayed during test inference.
- The repair location is drawn at random among the anomalous positions (`deterministic=0`).
- Up to 100 episodes are run per test case; the first episode that ends with a normal trace is kept.
- Results therefore depend on the random seed; the case study in `complex_trace_timing/` repeats runs with 5 seeds.

Whether this matches the description of test inference in the paper text is still being checked (see
[Known open issues](#known-open-issues)).

## 9. Dataset conditions

| Dataset | file_stem | Discovery noise_threshold | Loop bound `(loop_num, loop_total)` | NBG input (Method C) | Test cases |
|---|---|---|---|---|---|
| Small | Small | 0.0 | acyclic (exhaustive) | generated_only | 524 |
| Medium | Medium | 0.0 | acyclic (exhaustive) | generated_only | 333 |
| Large | Large | 0.0 | acyclic (exhaustive) | generated_only | 643 |
| Wide | Wide | 0.0 | acyclic (exhaustive) | generated_only | 337 |
| Credit | credit-card | 0.0 | (5,7) | W2 original_plus_model_only | 210 |
| Pub | mccloud | 0.0 | (4,4) | W2 original_plus_model_only | 269 |
| Road Traffic | Road_Traffic (duplicate-event derived copy) | 0.20 | (2,2) | W2 original_plus_model_only | 2,752 |
| BPIC 2017 | b17 | 0.10 | (1,9) | W2 original_plus_model_only | 2,360 |

Notes:

- **Road Traffic:** a derived copy is used; the source log is not modified. In 475 clean cases of the source log every
  event occurs exactly twice in a row; this pattern strongly affected process discovery (11 self-loops instead of 1) and
  trace-generation complexity. In the derived copy these 475 clean cases are restored to one occurrence per event
  (`08_road_raw_vs_dedup.py`). Anomalous cases are unchanged.
- **BPIC 2017:** with `(0.10, (1,9))` the bounded language covers only 43.49 % of clean cases. This is the best executable
  trade-off found, not a coverage-≥ 90 % condition. Its results are not directly comparable with the other datasets.
- **Small–Wide:** results come from an earlier experiment. All-optimal agreement and C+PN were not evaluated (N/A).

## 10. Results in this snapshot

- **Paper tables:** `outputs/multi_dataset_experiment/paper_*.csv|md`, `bpic_candidate_feasibility.*`,
  `threshold_sweep_summary.*`, `timing_summary.*`, `table_validation.json`, and
  `threshold_sweep/sweep_structure_language.*`.
- **Per-dataset summaries** for the selected condition, under `outputs/simulation/<Dataset>/...`:
  - `generate/generate_summary.json`
  - `nbg/nbg_comparison_summary.json`
  - `preparation/*_nbg_summary.json`
  - `runs/method{B,C}_run1/*_{stats,timing}.json`
  - `C_PN_Gate/run1/*_{stats,timing}.json`
  - `test_timing/*/timing_summary.json`
  - `alignment_all_optimal_analysis/**/agreement_summary.json` and `bc_*agreement.json`

Per-case repaired logs, smoke outputs, execution logs, checkpoints and agents are not included.

### BPIC 2017 final results (official run, `(0.10, (1,9))`, 2,360 test cases)

The BPIC 2017 C+PN run is finished and the paper tables contain its measured values
(`paper_result_table.*`, `paper_threshold_agreement_table.*`, `paper_threshold_timing_table.*`).

| Metric | Method C | Method C+PN |
|---|---:|---:|
| Exact agreement with B | 926 / 2360 (39.24 %) | 1041 / 2360 (44.11 %) |
| Native all-optimal agreement (lower bound) | 983 / 2360 (41.65 %) | 1104 / 2360 (46.78 %) |
| Tau-excluded all-optimal agreement (lower bound) | 998 / 2360 (42.29 %) | 1124 / 2360 (47.63 %) |
| Training (official run, 15 workers) | 154,322,800.54 ms (≈ 42.9 h) | 152,211,578.58 ms (≈ 42.3 h) |
| Sequential test repair, mean over all test cases | 6,015.30 ms/case | 17,953.16 ms/case |

- Method B sequential test repair (alignment + repair, all 2,360 test cases): 4,887.86 ms/case.
- The all-optimal search did not finish for 81 test cases (49 hit `--max-states`, 32 stopped at the memory limit;
  `alignment_all_optimal_analysis/run1/resource_limit_cases.tsv` lists the 32). They are counted as disagreement, so both
  all-optimal values are lower bounds. B–C+PN reuses the B optimal sets of the B–C run
  (`22_agreement_from_checkpoint.py`, `alignment_all_optimal_analysis/bcpn_from_checkpoint/agreement_summary.json`).
- Clean-case coverage of the bounded language is 43.49 % (best executable trade-off, see Section 9).
- Training time, sequential test repair time, the multiprocessing test throughput of the official runs and the
  all-optimal evaluation time are different measurements; the tables state which one each column uses.

### Supplementary: BPIC 2017 complex-trace case study

`outputs/simulation/BPIC2017/threshold_0p10/complex_trace_timing/` contains a case study of test traces whose
all-optimal search was especially hard: 4 traces with the longest search, and 10 further candidates selected by a rule
fixed in advance, each repaired 5 times (seeds 42–46) by B, C and C+PN with the official models. It is a case study of
selected traces, **not** an average over BPIC 2017; see its README for the selection rule, all runs (including the
failed candidates), the tables and the limitations.

## 11. Pretrained models

The official trained agents (`*_agent.pkl`, RP-BEAR `QLearningAgent` objects saved with pickle) exist for 12 runs but
are not included in this repository because of their size (1.41 GB in total). **TODO:** they will be published
separately (for example as release assets) together with a manifest and SHA-256 checksums.

| Dataset | Method C | Method C+PN |
|---|---|---|
| Small | 69.0 MB (EXP-063 run 1) | not evaluated |
| Medium | 43.4 MB | not evaluated |
| Large | 109.9 MB | not evaluated |
| Wide | 43.1 MB | not evaluated |
| Credit | 29.1 MB | 29.1 MB |
| Pub | 43.3 MB | 43.4 MB |
| Road Traffic (0.20, (2,2)) | 50.2 MB | 50.0 MB |
| BPIC 2017 (0.10, (1,9)) | 451.5 MB | 450.0 MB |

A model file alone is not enough to run inference without retraining. The test-only stages
(`--stage test-timing`, `--stage test-timing-pngate`) also need:

- the data root (anomaly and normal logs, P-BEAR-RL `Full_input` layout; for Road Traffic the derived copy made by `08`);
- the official repaired log of the run (`*_repaired_log_c.csv` / `*_repaired_log_cpn.csv`), from which the train / test
  split and the validation repair memory are rebuilt;
- the generate-stage output (`generate/model_only_generated_log.csv`) to rebuild the W2 NBG (its edge digest is stored in
  `*_stats.json` as `nbg_edge_set_digest`);
- for C+PN, the discovered Petri net (`discover/petri_net.pnml`);
- the `.venv-rpbear` environment (Python 3.13.9, pandas 2.3.3, numpy 2.3.5): the pickles reference
  `rpbear.func_rpbear_final.QLearningAgent` and pandas / numpy 2.x internals, so `src/` must stay importable as is;
- `PYTHONHASHSEED=0` and the official parameters of Section 8.

Pickle files can execute code when they are loaded. Only load model files from the official release and check their
SHA-256 checksum first.

## 12. License and third-party code

**TODO:** license not decided yet.

- `src/rpbear/` is the RP-BEAR implementation. Its provenance and license must be confirmed before any public release.
- This project uses PM4Py (AGPL-3.0) as a dependency.

## Known open issues

- **Test inference vs. paper text (to be confirmed):** the official runs use epsilon-greedy test inference with the
  stored exploration rate (recorded as 0.5 for BPIC 2017), a random repair location and up to 100 episodes (Section 8). If the paper describes
  greedy inference (epsilon = 0), the two descriptions differ; the code and the reported results are not changed until
  this is decided.
- **Method naming:** the mapping between the method names used here (B, C, C+PN) and the names used in the paper is not
  fixed in this repository yet; the tables keep the names B, C and C+PN.
- **License and data redistribution:** see Sections 5 and 12.
