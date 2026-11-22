# RA-Alignment: A Reinforcement Learning Framework for Approximate Conformance Checking

Code of the paper *A Reinforcement Learning Framework for Approximate Conformance Checking* (J. Park, J. Ko,
H.-G. Jo, M. Comuzzi; submitted to an ICPM 2027 workshop). RA-Alignment learns to repair deviating traces with
Q-learning, using Normal Behavior Graphs (NBGs) built from a bounded set of model traces. The repaired traces are
compared with Optimal Alignment (PM4Py).

This repository contains five tutorials that run end to end on the **Credit** dataset (the running example of the
paper), the pre-trained Credit model of the paper run, and the 10 representative BPIC 2017 test traces of paper
Table 3. The implementation of RA-Alignment builds upon the P-BEAR-RL framework
(see [Original implementation and citation](#original-implementation-and-citation)).

> Software license: **not decided yet** (see [License](#license)).

## Method in one paragraph

1. Discover a process model (Inductive Miner, noise threshold 0.00) from the conforming cases and convert it into an
   accepting Petri net.
2. Sample the bounded model trace set L_N, which is the exhaustive model language under the loop bounds
   (b_num, b_tot) = (5, 7).
3. Build the NBGs from L_N only. Each unique trace counts once, and no observed traces are added.
4. Train a Q-learning agent on half of the deviating traces (L_train). An episode can end as "aligned" only if the
   Petri net accepts the repaired trace.
5. Repair the unseen half (L_test) with the frozen Q-table, and compare it with Optimal Alignment.

## Repository structure

```
tutorials/                   five Credit tutorials (+ _common.py: paths and official settings)
src/rpbear/func_rpbear_final.py   RP-BEAR core (environment, Q-learning agent, NBG discovery) [P-BEAR-RL]
src/sim.py                   RP-BEAR execution engine (data loading, split, per-case repair)
src/utils/                   discovery, bounded sampling, L_N-only NBG, Petri-net gate, sequential timing
src/alignment/               Optimal Alignment repair (PM4Py) and the tau-excluded all-optimal search (ACC)
data/credit/                 Credit event logs (see Data source)
data/bpic2017/               the 10 BPIC 2017 test traces of paper Table 3
artifacts/reference/         official files of the paper run (used only for the comparison checks)
artifacts/pretrained_models/ pre-trained Credit RA-Alignment model of the paper run
artifacts/MANIFEST.csv       SHA-256, size and origin of every shipped artifact
```

The tutorials write to `artifacts/petri_net/`, `artifacts/sampling/`, `artifacts/models/` and `results/`. These
folders are not tracked.

## Installation

Python 3.13. RP-BEAR needs pandas 2.x and PM4Py runs on pandas 3.x, so two virtual environments are used:

```bash
python -m venv .venv            # PM4Py environment: tutorials 1, 3, 5
.venv/Scripts/python -m pip install -r requirements-pm4py.txt          # Linux/macOS: .venv/bin/python
python -m venv .venv-rpbear     # RP-BEAR environment: tutorials 2, 4
.venv-rpbear/Scripts/python -m pip install -r requirements-rpbear.txt
```

The tutorials use PM4Py's scipy LP solver (cvxopt is hidden inside the tutorial process), as in the paper runs.

## Data source

- **Credit** (`data/credit/`): `credit-card_1.00.csv` is the Credit log with injected deviations (5,000 cases, 420
  deviating; column `type_res_trace`), `credit-card.csv` the corresponding normal log. Both come from the benchmark
  data of P-BEAR-RL [1] and are published here with the permission of J. Ko (P-BEAR-RL author).
- **BPIC 2017** (`data/bpic2017/`): 10 traces of the deviation-injected BPIC 2017 log of the P-BEAR-RL benchmark.
  The original log is *BPI Challenge 2017* (B. van Dongen, 4TU.ResearchData, 2017,
  <https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b>), distributed under the 4TU General Terms of
  Use.

The terms of the original data sets apply to the data files; they are separate from the license of the code.

## Quick start (pre-trained model, no training)

```bash
.venv/Scripts/python tutorials/01_discovery_sampling.py           # Petri net + L_N (a few seconds)
PYTHONHASHSEED=0 .venv-rpbear/Scripts/python tutorials/04_ra_alignment.py
.venv/Scripts/python tutorials/03_optimal_alignment.py
.venv/Scripts/python tutorials/05_compare_performance.py
```

On Windows PowerShell, set the variable first with `$env:PYTHONHASHSEED = "0"`. `PYTHONHASHSEED=0` is needed only to
reproduce the reference output exactly.

## Tutorials

| # | Script | Env | Input | Output |
|---|---|---|---|---|
| 1 | `01_discovery_sampling.py` | PM4Py | Credit log | `artifacts/petri_net/` (PNML, process tree), `artifacts/sampling/` (L_N, metadata) |
| 2 | `02_train_ra_alignment.py` | RP-BEAR | Credit log, L_N, PNML | `artifacts/models/<run>/` (agent, training statistics, repaired log, split) |
| 3 | `03_optimal_alignment.py` | PM4Py | PNML, L_test | `results/optimal_alignment/` (repaired traces, time per case) |
| 4 | `04_ra_alignment.py` | RP-BEAR | model, L_N, PNML | `results/ra_alignment/` (repaired traces, time per case) |
| 5 | `05_compare_performance.py` | PM4Py | outputs of 3 and 4 | `results/comparison/` (ACC, times) |

**1. Discovery and sampling.** This tutorial discovers the process tree and the Petri net, then enumerates L_N
with (b_num, b_tot) = (5, 7). The enumeration has safety bounds: trace length 100, 1,000,000 intermediate
traces, 600 s and 16 GB. If a bound is hit, the run stops; the traces are not truncated. Every trace is replayed on
the Petri net. Finally, the outputs are compared with the official files: same process tree, same Petri net
structure, and the same 5,000 traces.

**2. Training (optional).** This tutorial trains a new agent with the official settings:

- 1,000 episodes per case, at most 8 steps;
- learning rate 0.1, discount 0.9, exploration 0.5 (decay 0.0001);
- alpha 0, train/test split 50/50 with seed 42;
- the L_N-only NBG and the Petri-net gate.

Use `--workers N` for the number of processes (the default is the CPU count − 1, at most 15). `--smoke` checks the
code path in a few seconds and does not produce a usable model. The paper run took 1,489 s with 15 workers. Because
the workers update the shared Q-table in parallel, a new training does not reproduce the paper model bit for bit.

**3. Optimal Alignment.** This tutorial aligns each test trace with PM4Py (A* with the state equation, standard
costs) and repairs it from the alignment: a log move deletes the event, a visible model move inserts it, and a tau
move produces nothing. The timed section is the alignment plus the repair, case by case. The repaired traces are
compared with the official output.

**4. RA-Alignment inference.** This tutorial loads the agent and checks that the NBG, the split and the Petri net
match its training run. Each test case is then repaired one by one with the frozen Q-table: a trace already
repaired during training is reused, otherwise the agent runs up to 100 episodes with the Petri-net gate. The
tutorial also checks that the Q-table is unchanged. With the pre-trained model, the output is compared with the
reference run. `--model-dir artifacts/models/<run>` uses a model from tutorial 2 instead.

**5. Performance comparison.** **ACC** counts a repaired trace as correct if it belongs to the set of *all* optimal
repairs of the test trace. A trace can have several optimal alignments, so this set is computed by an exhaustive
search on the Petri net that counts visible deviations only: a log move or a visible model move costs 1, and a
silent transition costs 0. The sets are recomputed and checked against the official sets; `--use-reference` skips
the search. Agreement with a single Optimal Alignment output is reported separately and is not used as ACC.

## Results (Credit, 210 test cases)

| Method | Source | ACC | Mean time per case |
|---|---|---|---|
| Optimal Alignment | tutorial 3 | 210/210 (100.00 %) | 124 ms |
| RA-Alignment | tutorial 4 (pre-trained, sequential inference) | 161/210 (76.67 %) | 391 ms |
| RA-Alignment | paper run (test phase of the training run) | 163/210 (77.62 %) | – |

The times were measured on one machine, in a single process, and will differ on other hardware. Tutorial 4 does not
reproduce the paper run's test phase exactly. During that phase, the random (epsilon-greedy) actions came from the
random streams of the parallel workers, and those streams were not saved. Instead, tutorial 4 reproduces a saved
sequential reference run of the same model (210/210 identical traces).

## Pre-trained models

The Credit model used by the tutorials is included in this repository (see below). The pre-trained models of all
paper runs (Table 2) are distributed as assets of the GitHub Release **`models-v1.0`** of this repository (Releases
page), one zip file per dataset, plus `SHA256SUMS.txt`.

| Dataset | Release asset | NBG source | Petri-net gate |
|---|---|---|---|
| Small | `RA-Alignment_small.zip` | L_N only | no |
| Medium | `RA-Alignment_medium.zip` | L_N only | no |
| Large | `RA-Alignment_large.zip` | L_N only | no |
| Wide | `RA-Alignment_wide.zip` | L_N only | no |
| Credit | `RA-Alignment_credit.zip` (same model as in this repository) | L_N only | yes |
| Pub | `RA-Alignment_pub.zip` | L_N only | yes |
| Road Traffic | `RA-Alignment_road_traffic.zip` | L_N only | yes |
| BPIC 2017 | not yet available | – | – |

- For the acyclic Synthetic-1 models (Small–Wide), L_N is the complete model language. These models were trained
  without the Petri-net gate.
- **Install:** download a zip, check it against `SHA256SUMS.txt`, and extract it into `artifacts/pretrained_models/`.
  This gives `artifacts/pretrained_models/<dataset>_ra_alignment/` with the same files as the Credit folder.
- **`MODEL_INFO.json`:** each zip contains this file, which records:
  - the model type, the NBG source and the gate;
  - the settings (1,000 episodes, alpha 0, seed 42, at most 8 steps, learning rate 0.1, discount 0.9, exploration 0.5);
  - the activity order;
  - the SHA-256 of the input logs and of every file.
- **To use a model other than Credit,** you also need:
  - the dataset log (P-BEAR-RL benchmark data [1], not included here);
  - its Petri net and L_N, which are not included either.

The Credit folder `artifacts/pretrained_models/credit_ra_alignment/` contains:

- `RA-Alignment_credit_agent.pkl`: the Q-table (490,039 entries), 29.6 MB;
- the training statistics, the training-run repaired log (its training rows are the repair memory), and the
  train/test split.

The SHA-256 of each file is listed in `artifacts/MANIFEST.csv`. `EXP-068` in file names and metadata is the internal
label of the paper run.

> **Security note:** pickle files can execute code when they are loaded. Do not load pickle files from untrusted
> sources. Tutorial 4 loads the agent with an allow-list unpickler, which accepts only the RA-Alignment agent class
> and plain numpy/pandas objects. It also checks the pre-trained file against its SHA-256 first.

## BPIC 2017 representative test cases

`data/bpic2017/bpic2017_test_10.csv` contains the 10 test cases of paper Table 3 (269 events), with the exact event
sequences of the official test split. `artifacts/reference/bpic2017_petri_net.pnml` is the reference Petri net
(Inductive Miner, noise threshold 0.10), and `artifacts/reference/bpic2017_table3_reference.csv` gives, per case,
the Optimal Alignment repair, the set of optimal repairs used for ACC and the Table 3 values.

These cases are **not a random sample** of the BPIC 2017 test set. They were selected for the comparison because
exact alignment on the full test set ran out of memory; for these 10 cases it completed. The paper result
(36/50 = 72.00 %: 10 cases × 5 independent runs; median 19.02 s per case for Optimal Alignment and 3.31 s for
RA-Alignment) was produced in the research project. **No pre-trained BPIC 2017 model is included**, so this
result cannot be re-run from this repository.

## Original implementation and citation

The implementation of RA-Alignment builds upon the P-BEAR-RL framework. The original P-BEAR-RL implementation is
available at <https://github.com/jonghyeonk/P-BEAR-RL>.

- `src/rpbear/func_rpbear_final.py`: RP-BEAR core (environment, Q-learning agent, NBG discovery), an extended version
  of the P-BEAR-RL implementation (`func.py`); in this repository only its comments were translated.
- `src/sim.py`: RP-BEAR execution engine, follows the behaviour of the RP-BEAR run script.
- Developed for RA-Alignment: bounded sampling, L_N-only NBG, Petri-net gate (`src/utils/`), Optimal Alignment repair
  and the all-optimal search (`src/alignment/`), and the tutorials.

If you use this code, please cite P-BEAR-RL [1]:

```bibtex
@inproceedings{ko2025reinforcement,
  title={A reinforcement learning framework for event log anomaly detection and repair},
  author={Ko, Jonghyeong and Wynn, Moe Thandar and Comuzzi, Marco and Maggi, Fabrizio Maria},
  booktitle={2025 7th International Conference on Process Mining (ICPM)},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
```

Citation of the RA-Alignment paper: **TODO** (after publication).

## License

**Software license: not decided yet.** PM4Py (AGPL-3.0) is used as a dependency and is not redistributed here.
