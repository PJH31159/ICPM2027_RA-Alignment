# Pre-trained models

**Status: Not yet published.** The files below are the official agents of the paper runs. They are distributed as
assets of a GitHub Release of this repository (not in git). `release_url` in `model_manifest.csv` is empty and
`status` is `NOT_PUBLISHED` until the release exists.

## Models

12 RP-BEAR `QLearningAgent` objects saved with Python `pickle` (protocol 4), 1,412,022,714 bytes in total.
Method **C**: RP-BEAR with the model-extended NBG. Method **C+PN**: C plus the Petri-net acceptance gate
(datasets with the W2 NBG only). Small–Wide have no C+PN model.

| Dataset | Method | File | Size | SHA-256 | Used for |
|---|---|---|---:|---|---|
| Small | C | `Small_1000_ppm0.0_valratio0.5_seed42_generatedNBG_run1_agent.pkl` | 69.0 MB | `ca61bcbf215254c565a885d7eab2185927117e4d2163956282aa531faa546c41` | Table 2 (ACC, training, RA-Alignment time) |
| Medium | C | `Medium_1000_ppm0.0_valratio0.5_seed42_methodC_run1_agent.pkl` | 43.4 MB | `2994caa205780af596ff7bc3376f8a6767aace4bbd43874f2486bf738e991058` | Table 2 (ACC, training, RA-Alignment time) |
| Large | C | `Large_1000_ppm0.0_valratio0.5_seed42_methodC_run1_agent.pkl` | 109.9 MB | `bf6945de4b3c68fe15de22b279bcc1658bd956b35d36631447a917f2eaa149a6` | Table 2 (ACC, training, RA-Alignment time) |
| Wide | C | `Wide_1000_ppm0.0_valratio0.5_seed42_methodC_run1_agent.pkl` | 43.1 MB | `f3233fc9357bac07b96a1eaca511e571159cc6832b2b9ee22bdec554d3e62bd8` | Table 2 (ACC, training, RA-Alignment time) |
| Credit | C | `credit-card_1000_ppm0.0_valratio0.5_seed42_methodC_run1_agent.pkl` | 29.1 MB | `230018958710a674d18928488961039a5175e817783207e43f8d4ab4d46bb4b8` | Table 2 (training, RA-Alignment time) |
| Credit | C+PN | `credit-card_1000_ppm0.0_valratio0.5_seed42_methodCPN_run1_agent.pkl` | 29.1 MB | `b9066b46e82e942a1dc937964114759cda072b9645a3301d4d8ab9bbfb4847e5` | Table 2 (ACC) |
| Pub | C | `mccloud_1000_ppm0.0_valratio0.5_seed42_methodC_run1_agent.pkl` | 43.3 MB | `0e1e5b549c3a874d96fa7ae61bc27f154b1585c9007b9b771df856e3a75b95f7` | Table 2 (training, RA-Alignment time) |
| Pub | C+PN | `mccloud_1000_ppm0.0_valratio0.5_seed42_methodCPN_run1_agent.pkl` | 43.4 MB | `ac899792e9fe06aa26e9fa20cff742c2bd4e45c7bc3c33d876ffb4c49a135f83` | Table 2 (ACC) |
| RoadTraffic | C | `Road_Traffic_1000_ppm0.0_valratio0.5_seed42_methodC_run1_agent.pkl` | 50.2 MB | `2e18783225bbd10aace8eb021bcfcbcccabe4b7527fb654eea14bfbaf811cca4` | Table 2 (training, RA-Alignment time) |
| RoadTraffic | C+PN | `Road_Traffic_1000_ppm0.0_valratio0.5_seed42_methodCPN_run1_agent.pkl` | 50.0 MB | `e3fa8ed7dc4072cda604ba4f93a9301c5cb0be8f62d588fe5be68cb77189e1ba` | Table 2 (ACC) |
| BPIC2017 | C | `b17_1000_ppm0.0_valratio0.5_seed42_methodC_run1_agent.pkl` | 451.5 MB | `f2f60f2c35dbaedf4c009b58bedd569a8b3af214a7a3b27c4d4983a76ba07753` | Table 2 (training); Table 3 (Method C reference) |
| BPIC2017 | C+PN | `b17_1000_ppm0.0_valratio0.5_seed42_methodCPN_run1_agent.pkl` | 450.0 MB | `1b9636886c9c872a9bc25c4d7cd41db661698c6be3782b452bc5628eb95c7bd0` | Table 2 / Table 3 (ACC, RA-Alignment time) |

Which method gives which paper number is explained in `../results/README.md` (Table 2 uses C+PN for the accuracy of
Credit–BPIC 2017 and C for their training / alignment times).

## Training configuration

All 12 runs: `alpha = 0`, `num_epi = 1000`, `max_step = 8`, `seed = 42`, `val_ratio = 0.5`, 15 worker processes,
learning rate 0.1, discount 0.9, initial exploration rate 0.5, decay 0.0001 (`src/dataset_config.py`,
`src/rpbear/func_rpbear_final.py`). Dataset conditions (noise threshold, loop bound, NBG input) are in the main README
and `src/dataset_config.py`. The Small model comes from an earlier run of the same RP-BEAR code with the generated-trace
NBG (its stats file records `method = "C (Petri Net generated traces -> New NBG -> RP-BEAR)"`).

## Python environment

The pickles reference `rpbear.func_rpbear_final.QLearningAgent`, numpy 2.x internals (`numpy._core`) and pandas 2.x
internals. Use the `.venv-rpbear` environment of `../requirements-rpbear.txt` (Python 3.13.9, numpy 2.3.5,
pandas 2.3.3) with `src/` on `sys.path`, and keep `src/rpbear/func_rpbear_final.py` at its path and name.
A read-only load of the Credit C and C+PN agents with this repository's `src/` succeeded (class
`rpbear.func_rpbear_final.QLearningAgent`; Q-table sizes 482,266 and 481,405; stored exploration rate 0.5).
The other 10 agents were checked by a pickle opcode scan only (same class and module references), not loaded.

## Release assets (planned)

| Asset | Content | Size |
|---|---|---:|
| the 12 `*_agent.pkl` files above | byte-identical official agents (original file names) | 1,412,022,714 B |
| `run_metadata_v1.zip` | per run: `*_stats.json`, `*_repaired_log_c.csv` / `*_repaired_log_cpn.csv`, `*_test_case_ids.txt`, `*_val_case_ids.txt`, stored under the paths of the next section | 22,485,471 B before compression |
| `SHA256SUMS.txt` | SHA-256 of every asset and of every file in the zip | small |

Every asset is below GitHub's 2 GiB per-file limit for release assets. **Not created yet.**

## Required file structure

Place every agent and its metadata in the run folder the pipeline expects (repository-relative):

| Dataset | Method | Folder |
|---|---|---|
| Small, Medium, Large, Wide | C | `outputs/simulation/<Dataset>/runs/methodC_run1/` |
| Credit, Pub | C | `outputs/simulation/<Dataset>/runs/methodC_run1/` |
| Credit, Pub | C+PN | `outputs/simulation/<Dataset>/C_PN_Gate/run1/` |
| RoadTraffic | C / C+PN | `outputs/simulation/RoadTraffic/threshold_0p20/runs/methodC_run1/`, `.../threshold_0p20/C_PN_Gate/run1/` |
| BPIC2017 | C / C+PN | `outputs/simulation/BPIC2017/threshold_0p10/runs/methodC_run1/`, `.../threshold_0p10/C_PN_Gate/run1/` |

(The official Small run was stored under a different folder name in the research project; it is placed with the other
datasets here.)

## Download and verification

Download from the Release page of this repository once it is published (**Not yet published**), then check every file
before loading it:

```bash
sha256sum -c SHA256SUMS.txt                                   # Linux / macOS / Git Bash
Get-FileHash .\<file>_agent.pkl -Algorithm SHA256             # PowerShell, compare with the table above
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>
```

## Loading

```python
import pickle, sys
sys.path.insert(0, "src")                      # run from the repository root, .venv-rpbear
with open("outputs/simulation/Credit/C_PN_Gate/run1/credit-card_1000_ppm0.0_valratio0.5_seed42_methodCPN_run1_agent.pkl", "rb") as f:
    agent = pickle.load(f)                     # rpbear.func_rpbear_final.QLearningAgent
print(len(agent.q_table), agent.exploration_rate)
```

**Security:** unpickling can execute code. Load only files from the official release whose SHA-256 matches.

The pipeline loads agents in `src/simulation.py` (stages `test-timing` and `test-timing-pngate`) and in
`scripts/bpic2017_case_study.py`.

## Evaluation with the models (Scenario B)

1. Prepare the dataset (main README, Section 4; Road Traffic needs the derived copy).
2. Run `discover`, `generate` and `nbg` for the dataset (main README, Section 8). The test stages rebuild the NBG from
   the `generate` output and, for C+PN, the gate from the `discover` Petri net; the split is rebuilt from the seed and
   checked against the official repaired log.
3. Put the agent and its metadata in the folder above.
4. Method C: `--stage test-timing --method C --agent <agent.pkl> --repaired-log <repaired_log_c.csv> --label <name>`.
   Method C+PN: `--stage test-timing-pngate --run-id 1 --label <name>`.
5. ACC: `scripts/exact_agreement.py` or `scripts/all_optimal_agreement.py` with the **official** repaired log and a
   local `run-b`.

## Reproduction limitations

- The test stages re-measure the sequential per-case time with the official agent; they do not update the Q-table.
- Test inference is epsilon-greedy with a random repair location (unchanged RP-BEAR code), so repairs on the RL path
  can differ from the official repaired log (for example Pub, Method C: 216 / 269 test traces identical in the official
  replay). The paper's ACC is computed from the official repaired logs.
- Timing depends on the machine and on the LP solver used for Method B.
- Not yet verified on a machine other than the experiment PC.

## Provenance

Trained in the research project of the authors on Windows 11 with Python 3.13.9 (`.venv-rpbear`), 15 worker processes.
The files are byte-identical to the research originals (SHA-256 above). The run statistics behind the paper values are
also in `../results/supporting/` (see its `MANIFEST.csv`).

## License and third-party notice

**License: TODO (not decided).** The agents were trained with an extension of the RP-BEAR implementation of
P-BEAR-RL (<https://github.com/jonghyeonk/P-BEAR-RL>, no license file); publication of code and models is being confirmed
with its authors. The models were trained on the event logs listed in the main README; their redistribution terms
apply to the derived artifacts as far as they contain log content (the repaired logs in `run_metadata_v1.zip` do).

## Release notes (draft)

> **RA-Alignment pre-trained models v1**
>
> Official RP-BEAR agents used in the paper "A Reinforcement Learning Framework for Approximate Conformance
> Checking": Method C for Small, Medium, Large, Wide, Credit, Pub, Road Traffic and BPIC 2017, and Method C+PN for
> Credit, Pub, Road Traffic and BPIC 2017 (12 files, 1.41 GB). `run_metadata_v1.zip` contains the stats, official
> repaired logs and train / test case ids of each run, needed by the test stages. Verify every file with
> `SHA256SUMS.txt` before loading (pickle). Placement and usage: `models/README.md`. Requires Python 3.13.9 with
> `requirements-rpbear.txt`.
