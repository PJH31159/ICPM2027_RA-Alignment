# RA-Alignment: A Reinforcement Learning Framework for Approximate Conformance Checking

Implementation, paper results and pre-trained models of the paper
*"A Reinforcement Learning Framework for Approximate Conformance Checking"* (ICPM 2027 workshops).

> **Status:** paper under review; this repository is being prepared for public release.
> The pre-trained models are **not yet published** (see [Pre-trained models](#5-pre-trained-models)).
> The license is **not decided yet** (see [License](#12-license-and-citation)).

---

## 1. Overview

RA-Alignment approximates optimal Petri-net trace alignment with a pre-trained reinforcement-learning repair policy
instead of a runtime state-space search:

1. discover a reference Petri net from the conforming (clean) traces (PM4Py Inductive Miner, noise threshold NT);
2. sample a **bounded** trace language from the model with a loop bound `(num, tot)` (a model with loops has an
   infinite language, so we always speak of *traces under a loop bound*);
3. build Normal Behavior Graphs (NBGs) from the observed clean traces plus the model-only sampled traces;
4. train an RP-BEAR Q-learning policy [P-BEAR-RL] on half of the deviating traces and align the unseen half.

Method names used in the code and in the result files:

| Code name | Meaning | Training |
|---|---|---|
| **B** | Optimal alignment (PM4Py state-equation A\*, `src/alignment/exact_alignment.py`) + repair to the model projection. Evaluation reference ("Optimal-Alignment" in the paper) | none |
| **C** | RP-BEAR repair with the model-extended NBG (Small–Wide: generated traces only; the other datasets: observed + model-only traces, "W2") | Q-learning |
| **C+PN** | C, plus a Petri-net acceptance gate: an episode may end as "normal" only if the repaired trace is accepted by the reference Petri net (W2 datasets only) | Q-learning |

"RA-Alignment" in the paper corresponds to **C** for Small–Wide and to **C+PN** (accuracy) / **C** (training and
alignment time) for the other datasets; see [Paper results](#10-paper-results) — this mix is reported, not hidden.

## 2. Repository structure

```text
├── README.md
├── requirements-pm4py.txt        # .venv        : discovery, generation, Method B, agreement
├── requirements-rpbear.txt       # .venv-rpbear : NBG, RP-BEAR training / test (Methods C, C+PN)
├── src/                          # algorithm + pipeline
│   ├── simulation.py             #   entry point: --dataset <name> --stage <stage>
│   ├── sim.py                    #   RP-BEAR execution engine (split, training / test phases, timing)
│   ├── dataset_config.py         #   dataset settings and the official RP-BEAR parameters
│   ├── rpbear/func_rpbear_final.py   # RP-BEAR core: environment, QLearningAgent, NBG discovery (unchanged)
│   ├── alignment/exact_alignment.py  # Method B: optimal alignment + repair
│   └── utils/                    #   discovery, bounded generation, NBG, PN gate, timing, evaluation, validation
├── scripts/                      # tools outside the stage pipeline
│   ├── run_with_scipy_lp.py      #   runs src/simulation.py with PM4Py's scipy LP solver (used for all .venv stages)
│   ├── prepare_road_traffic.py   #   Road Traffic derived input copy
│   ├── exact_agreement.py        #   ACC (exact) from repaired logs
│   ├── all_optimal_agreement.py  #   ACC (native / tau-excluded all-optimal) with --max-states
│   ├── run_all_optimal_agreement.py   # memory-guarded driver of all_optimal_agreement.py
│   ├── agreement_from_checkpoint.py   # reuse B's optimal sets for another method (no search)
│   ├── credit_all_optimal_analysis.py # all-optimal search definitions (read by all_optimal_agreement.py)
│   ├── bpic2017_case_study.py         # Table 3 measurement (B / C / C+PN, 5 runs per trace)
│   ├── bpic2017_case_study_candidates.py  # Table 3 case selection (rule fixed in advance)
│   └── make_paper_tables.py      #   rebuilds results/table{1,2,3}_*.csv from results/supporting/
├── results/                      # paper Tables 1-3 and the official result files behind them
│   ├── README.md
│   ├── table1_dataset_summary.csv, table2_alignment_results.csv, table3_bpic2017_case_study.csv
│   └── supporting/               #   copies of the official result files + MANIFEST.csv (SHA-256, source keys)
├── models/                       # pre-trained models: manifest only (files = GitHub Release assets)
│   ├── README.md
│   └── model_manifest.csv
└── outputs/                      # written by the pipeline at run time
    └── credit_loop_bound_sensitivity/credit_loop_bound_final_traces.csv   # Credit reference trace set (input of `generate`)
```

`outputs/` is otherwise empty on purpose: every stage creates its own folder and **stops if that folder already
exists** (unless `--overwrite`). Stored results therefore live in `results/`, not in `outputs/`.

## 3. Requirements

- Python 3.13.9, two virtual environments (RP-BEAR needs pandas 2.x; PM4Py 2.7.23.6 runs on pandas 3.x):
  `requirements-pm4py.txt` (`.venv`) and `requirements-rpbear.txt` (`.venv-rpbear`).
- Windows 11 was used for all official runs (15 worker processes). On Linux / macOS use `bin/python` instead of
  `Scripts/python`.
- LP solver: all `.venv` stages of the Credit, Pub, Road Traffic and BPIC 2017 runs used PM4Py's **scipy** solver via
  `scripts/run_with_scipy_lp.py` (cvxopt could not be loaded on the experiment PC). The earlier Small–Wide Method B
  timings used **cvxopt**. Alignment times are solver-dependent (Credit: 3.46 ms/case with cvxopt vs 122.91 with scipy).

## 4. Datasets

The event logs are **not included**.

| Group | Logs | Source |
|---|---|---|
| Synthetic-1 | Small, Medium, Large, Wide | P-BEAR-RL benchmark logs [P-BEAR-RL]; the P-BEAR-RL repository links a download of its `Full_input` data. **TODO:** confirm with the authors that this download is the version used here and that it may be referenced |
| Synthetic-2 | Credit (`credit-card`), Pub (`mccloud`) | same as above |
| Real-life | Road Traffic Fine Management, BPIC 2017 | 4TU.ResearchData: [10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5), [10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b); the deviation-injected versions come from the same `Full_input` data set |

Expected layout (P-BEAR-RL `Full_input`):

```text
<DATA_ROOT>/encoded_anomaly/<file_stem>_1.00.csv   # clean + deviating cases (column type_res_trace)
<DATA_ROOT>/encoded_normal/<file_stem>.csv         # ground-truth normal log
```

The data root is `--data-root` > environment variable `ICPM2027_DATA_ROOT` > `data/raw`.
**Road Traffic** uses a derived copy in which 475 clean cases whose events all occur exactly twice are restored to
single occurrences (anomalous cases unchanged); create it once with
`.venv/Scripts/python scripts/prepare_road_traffic.py --data-root <DATA_ROOT> --derive-only`.

## 5. Pre-trained models

The 12 official trained agents (Methods C and C+PN, 1.41 GB) will be published as assets of a GitHub Release.
**Status: Not yet published.** `models/model_manifest.csv` lists every file with its size, SHA-256, training
configuration and the run metadata it needs; `models/README.md` explains download, placement and loading.

A model file alone is not enough: the test stages also need the run's stats file and official repaired log
(distributed together with the agents), the dataset, and a local `discover` / `generate` / `nbg` run.

## 6. Installation

```bash
python -m venv .venv          && .venv/Scripts/python -m pip install -r requirements-pm4py.txt
python -m venv .venv-rpbear   && .venv-rpbear/Scripts/python -m pip install -r requirements-rpbear.txt
```

## 7. Quick start

Check the paper numbers from the stored results (no data, no model, any Python with pandas):

```bash
python scripts/make_paper_tables.py
```

It verifies the SHA-256 of every file in `results/supporting/`, rebuilds `results/table{1,2,3}_*.csv` and prints, for
every value, the paper value next to the recomputed one with a status (MATCH / ROUNDING / MISMATCH / UNKNOWN).

## 8. Training (Scenario A: from scratch)

Official arguments per dataset (Road Traffic and BPIC 2017 pass the official noise threshold and loop bound on the
command line; their `trace_generation` entries in `dataset_config.py` are exploratory defaults):

| Dataset | Arguments |
|---|---|
| Small, Medium, Large, Wide | `--dataset <name>` (acyclic models, exhaustive enumeration) |
| Credit | `--dataset Credit` (loop bound (5,7); `generate` checks the result against `outputs/credit_loop_bound_sensitivity/`) |
| Pub | `--dataset Pub` (loop bound (4,4)) |
| Road Traffic | `--dataset RoadTraffic --noise-threshold 0.2 --loop-bound 2,2` (derived copy first, see Section 4) |
| BPIC 2017 | `--dataset BPIC2017 --noise-threshold 0.1 --loop-bound 1,9` |

Example (BPIC 2017), from the repository root. `PYTHONHASHSEED=0` is required for a noise threshold > 0.

```bash
export PYTHONHASHSEED=0
A="--dataset BPIC2017 --noise-threshold 0.1 --loop-bound 1,9 --data-root <DATA_ROOT>"
VP=.venv/Scripts/python; RP=.venv-rpbear/Scripts/python; SH=scripts/run_with_scipy_lp.py

$VP $SH $A --stage discover                                    # Petri net
$VP $SH $A --stage generate                                    # bounded trace language
$RP src/simulation.py $A --stage nbg                           # NBGs + train / test split
$RP src/simulation.py $A --stage smoke                         # short check, required before `run`
$RP src/simulation.py $A --stage run --method C --run-id 1     # Method C (training + test)
$VP $SH $A --stage pngate-acceptor-check                       # C+PN only (W2 datasets)
$RP src/simulation.py $A --stage pngate-smoke                  # required before run-c-pngate
$RP src/simulation.py $A --stage run-c-pngate --run-id 1       # Method C+PN
$VP $SH $A --stage run-b --run-id 1                            # Method B
```

Official parameters (`src/dataset_config.py`, `RPBEAR_PARAMS`): `val_ratio = 0.5`, `seed = 42`, `num_epi = 1000`,
`alpha = 0`, `max_step = 8`, 15 worker processes; Q-learning rate 0.1, discount 0.9, initial exploration rate 0.5,
decay 0.0001. In the code, "validation" is the training phase (Q-table updated) and "test" the unseen half (no update).

Test inference as implemented (unchanged RP-BEAR code): actions are epsilon-greedy with the exploration rate stored in
the agent (0.5 in the checked agents: BPIC 2017 C / C+PN and Credit C / C+PN), the repair location is drawn at random
and up to 100 episodes are run per test case. Results therefore depend on the random seed.

**Not yet verified:** that this stage sequence reproduces the official runs on a clean machine. Training is expensive
(BPIC 2017 Method C: 42.9 h with 15 workers).

## 9. Evaluation

Accuracy (ACC) compares the RL repaired trace of each test case with Method B:

```bash
# exact: X repaired trace == B repaired trace (Small–Wide criterion)
.venv/Scripts/python scripts/exact_agreement.py --b-dir <runs/methodB_run1> --x-log <repaired_log_c.csv> --label C --out <file.json>
# tau-excluded all-optimal: X in the set of optimal B repairs (cost log/model move 1, tau 0); Credit–BPIC criterion
.venv/Scripts/python scripts/all_optimal_agreement.py --b-dir <runs/methodB_run1> --c-log <repaired_log_c.csv> \
    --cpn-log <repaired_log_cpn.csv> --gt <DATA_ROOT>/encoded_normal/<file_stem>.csv --out <dir> [--max-states 5000000]
# large logs (BPIC 2017): B-C with a memory guard and resume, then B-C+PN from the stored B optimal sets
.venv/Scripts/python scripts/run_all_optimal_agreement.py --b-dir <runs/methodB_run1> --c-log <repaired_log_c.csv> \
    --gt <DATA_ROOT>/encoded_normal/<file_stem>.csv --out <dir> --log-dir <dir>
.venv/Scripts/python scripts/agreement_from_checkpoint.py --ckpt-run <dir> --x-log <repaired_log_cpn.csv> --label CPN --out <dir> --check
```

The all-optimal search is bounded (`--max-states 5000000`); cases that hit the state or memory limit count as
disagreement, so the reported value is a lower bound. (The `agreement` stage of `simulation.py` compares Methods A, B
and C and needs a Method A run, which is not part of the paper.)

Sequential per-case timing with a trained agent (Scenario B below uses the same stages):

```bash
$VP $SH $A --stage test-timing-b --pnml <methodB_run1/*_petri_net.pnml> --repaired-log <methodB_run1/*_repaired_log_b.csv> --label seq_B
$RP src/simulation.py $A --stage test-timing --method C --agent <methodC_run1/*_agent.pkl> --repaired-log <methodC_run1/*_repaired_log_c.csv> --label seq_C
$RP src/simulation.py $A --stage test-timing-pngate --run-id 1 --label seq_CPN
```

**Scenario B: evaluation with the pre-trained models** (once they are published):
prepare the dataset → download an agent and its run metadata → place them as listed in `models/README.md` →
run `discover`, `generate` and `nbg` locally (needed to rebuild the NBG and, for C+PN, the Petri-net gate) →
`test-timing` / `test-timing-pngate` → ACC with the scripts above, using the official repaired log of the bundle and a
local `run-b`. What this reproduces and what it does not is listed in `models/README.md` (the replay re-measures time;
RL-path repairs may differ from the official ones because the random streams differ, so ACC is computed from the
official repaired logs).

The BPIC 2017 case study (Table 3) is measured with `scripts/bpic2017_case_study_candidates.py --select` (case
selection), `--method B|C|CPN` and `--summarize`; it needs the official models and the all-optimal checkpoint.

## 10. Paper results

`results/` holds Tables 1–3 rebuilt from the official result files (`results/README.md` explains every column).
Findings that are reported rather than corrected:

- **Table 2, method:** for Credit, Pub, Road Traffic and BPIC 2017 the ACC is that of **C+PN**, while the training and
  RA-Alignment times are those of **C** (the C+PN values are in the same rows as `*_cpn_reference`).
- **Table 2, ACC criterion:** Small–Wide use exact agreement with the single B repair; Credit–Road use tau-excluded
  all-optimal agreement; the BPIC 2017 row is the 10-trace case study of Table 3 (the full test set gives
  1,124 / 2,360 = 47.63 %, lower bound).
- **Table 2, LP solver:** Optimal-Alignment times are cvxopt for Small–Wide and scipy for the other datasets.
- **Table 2, rounding:** RA-Alignment times of Medium, Large and Wide differ by 0.01 ms/case.
- **Table 3:** the paper lists 10 of the 11 measured traces; Application_128623883 was measured with the same protocol
  but is not in the paper (with it: 39 / 55, 18.01 s vs 3.76 s).
- **Table 1:** `#Var` has no source in the stored results; `#Act` of Credit, Pub, Road Traffic and BPIC 2017 differs
  from the number of visible activities of the discovered model (15/15/14/26 vs 12/12/11/25).

## 11. Limitations

- Datasets are not redistributed (Section 4); the pre-trained models are not yet published (Section 5).
- A full run from scratch has not yet been verified on a clean machine.
- Results depend on the LP solver (timing) and on the random seed (epsilon-greedy test inference).
- Open issue: the official test inference is epsilon-greedy with the stored exploration rate (0.5 where checked) and a
  random repair location (Section 8). If the paper text describes greedy inference (epsilon = 0), the two descriptions
  differ; the code and the reported results are not changed until this is decided.
- `simulation.py` still contains the research-only `regression` stage (and the Small-specific branches it uses); it
  needs files of the research repository and does not run here.
- BPIC 2017: the bounded language covers only 43.49 % of clean cases (best executable trade-off).
- The noise-threshold / loop-bound selection procedure, solver-equivalence checks and other exploratory analyses are
  kept in the research repository and are not part of this release.

## 12. License and citation

**License: TODO (not decided).**

- `src/rpbear/func_rpbear_final.py` extends the RP-BEAR implementation of P-BEAR-RL
  (<https://github.com/jonghyeonk/P-BEAR-RL>; Ko, Wynn, Comuzzi, Maggi, *A reinforcement learning framework for event
  log anomaly detection and repair*, ICPM 2025). That repository has no license file; reuse terms are being confirmed
  with its authors.
- PM4Py (AGPL-3.0) is used as a dependency and is not redistributed here.

Citation: **TODO** (after publication).
