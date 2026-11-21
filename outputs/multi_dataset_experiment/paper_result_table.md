# Main result table (selected condition per dataset)

- Agreement = B–C+PN tau-excluded all-optimal agreement (count / test cases). BPIC 2017: B–C+PN computed from the B optimal sets of the B–C all-optimal run (22_agreement_from_checkpoint.py); B–C agreement is in Notes and paper_threshold_agreement_table.md.
- Training = RP-BEAR training ('validation') phase ms (official run, 15 workers). Repair = sequential test-only mean ms/case.
- Repair ms/case is a sequential (one process) measurement over all test cases for every dataset (B seq_B, C seq_C, C+PN seq_CPN; Credit: mean of 3 replays). The multiprocessing test time (throughput) of the official runs and the 30-case BPIC sweep sample are not used.
- B = Alignment (PM4Py state-equation A*), sequential, scipy LP (same solver for every dataset). Training N/A.
- speedup = B mean / C mean (> 1.0 = C faster).

| Dataset | Selected Noise Threshold | Loop Bound | Clean Coverage (%) | Generated Traces | Test Cases | B-C+PN Agreement | B-C+PN Agreement (%) | B Alignment Repair (ms/case) | C Training (ms) | C Repair (ms/case) | C+PN Training (ms) | C+PN Repair (ms/case) | C Speedup vs B | C+PN Speedup vs B | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Credit | 0.00 | (5,7) | 98.03 | 5,000 | 210 | 162 | 77.14 | 122.91 | 2,138,269.43 | 305.81 | 1,996,337.35 | 409.75 | 0.40 | 0.30 | OK | N/A |
| Pub | 0.00 | (4,4) | 99.98 | 6,339 | 269 | 240 | 89.22 | 50.23 | 2,333,283.01 | 665.75 | 2,342,414.71 | 865.05 | 0.07 | 0.06 | OK | N/A |
| RoadTraffic | 0.20 | (2,2) | 99.16 | 36,174 | 2,752 | 2,354 | 85.54 | 48.83 | 5,691,823.21 | 100.76 | 5,622,976.89 | 304.39 | 0.48 | 0.16 | SELECTED | Road exploratory preprocessing: duplicate-event derived copy; all-optimal search hit max_states for 11/2752 cases (counted as disagreement; all-optimal agreement is a lower bound, upper bound +11) |
| BPIC2017 | 0.10 | (1,9) | 43.49 | 31,440 | 2,360 | 1,124 | 47.63 | 4,887.86 | 154,322,800.54 | 6,015.30 | 152,211,578.58 | 17,953.16 | 0.81 | 0.27 | SELECTED | best executable trade-off (coverage >= 90% abandoned); clean coverage < 50%; B-C exact 926 / 2360 (39.24%), B-C tau-excluded all-optimal 998 / 2360 (42.29%) (for comparison; B-C+PN exact 1041 / 2360 (44.11%)); all-optimal search hit max_states for 49/2360 cases and stopped at the memory limit for 32 cases (resource-limit) (counted as disagreement; all-optimal agreement is a lower bound, upper bound +81) |
