# Threshold → timing (ms)

speedup = B mean / C mean (1.0 = equal, > 1.0 = C faster, < 1.0 = B faster).
B, C, C+PN repair = all sequential test-only measurements. The C variants run only at the selected threshold.
The B column is measured for every threshold in the same way with the sweep Petri net (14_threshold_b_alignment.py) for comparison across thresholds (some values were measured concurrently with other jobs, about 10% error; Road 0.20 / 0.30, the same model, give 55.92 / 61.29). The Road B value of the main result table is a sequential replay with the official run-b Petri net without concurrent jobs (seq_B, 48.83 ms/case).
**The BPIC 2017 B column is a random sample of 30 test cases (seed 42, 60 s timeout per case; B Cases = 30).** The C / C+PN columns are sequential measurements over all 2,360 test cases, so the BPIC speedups in this table compare different measurement scopes and are not used for the main comparison. The main BPIC timing is in paper_result_table.md (seq_B, all 2,360 test cases, sequential, 4,887.86 ms/case).
Training = RP-BEAR training (validation) phase of the official run (multiprocessing, 15 workers). Repair = sequential test inference. The multiprocessing test time (throughput) of the official run is not used in this table.

| Dataset | Threshold | B Cases | B Alignment Mean ms/case | C Training ms | C Repair Mean ms/case | C+PN Training ms | C+PN Repair Mean ms/case | Status | C Speedup vs B | C+PN Speedup vs B |
|---|---|---|---|---|---|---|---|---|---|---|
| RoadTraffic | 0.00 | 2,752 | 230.47 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| RoadTraffic | 0.05 | 2,752 | 70.78 | N/A | N/A | N/A | N/A | NOT_RUN_C | N/A | N/A |
| RoadTraffic | 0.10 | 2,752 | 50.64 | N/A | N/A | N/A | N/A | COVERAGE_LOSS | N/A | N/A |
| RoadTraffic | 0.15 | 2,752 | 45.06 | N/A | N/A | N/A | N/A | COVERAGE_LOSS | N/A | N/A |
| RoadTraffic | 0.20 | 2,752 | 55.92 | 5,691,823.21 | 100.76 | 5,622,976.89 | 304.39 | SELECTED | 0.56 | 0.18 |
| RoadTraffic | 0.30 | 2,752 | 61.29 | N/A | N/A | N/A | N/A | SAME_AS_0.20 | N/A | N/A |
| RoadTraffic | 0.40 | 2,752 | 46.49 | N/A | N/A | N/A | N/A | COVERAGE_LOSS | N/A | N/A |
| BPIC2017 | 0.00 | 30 | 15,582.53 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| BPIC2017 | 0.05 | 30 | 1,797.77 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| BPIC2017 | 0.10 | 30 | 1,716.25 | 154,322,800.54 | 6,015.30 | 152,211,578.58 | 17,953.16 | SELECTED | 0.28 | 0.10 |
| BPIC2017 | 0.15 | 30 | 1,536.66 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| BPIC2017 | 0.20 | 30 | 1,828.92 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| BPIC2017 | 0.30 | 30 | 1,828.92 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| BPIC2017 | 0.40 | 30 | 3,273.21 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
