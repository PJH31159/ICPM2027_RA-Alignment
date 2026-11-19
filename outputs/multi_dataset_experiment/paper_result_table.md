# Main result table (selected condition per dataset)

- Agreement = B–C+PN tau-excluded all-optimal agreement (count / test cases). BPIC 2017: C+PN in progress -> N/A; B–C agreement is in Notes and paper_threshold_agreement_table.md.
- Training = RP-BEAR validation(학습) phase ms (official run, 15 workers). Repair = 순차 Test-only mean ms/case.
- B = Alignment (PM4Py state-equation A*), 순차, scipy LP (모든 dataset 동일 solver). Training N/A.
- speedup = B mean / C mean (> 1.0 = C가 더 빠름).

| Dataset | Selected Noise Threshold | Loop Bound | Clean Coverage (%) | Generated Traces | Test Cases | B-C+PN Agreement | B-C+PN Agreement (%) | B Alignment Repair (ms/case) | C Training (ms) | C Repair (ms/case) | C+PN Training (ms) | C+PN Repair (ms/case) | C Speedup vs B | C+PN Speedup vs B | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Credit | 0.00 | (5,7) | 98.03 | 5,000 | 210 | 162 | 77.14 | 122.91 | 2,138,269.43 | 305.81 | 1,996,337.35 | 409.75 | 0.40 | 0.30 | OK | N/A |
| Pub | 0.00 | (4,4) | 99.98 | 6,339 | 269 | 240 | 89.22 | 50.23 | 2,333,283.01 | 665.75 | 2,342,414.71 | 865.05 | 0.07 | 0.06 | OK | N/A |
| RoadTraffic | 0.20 | (2,2) | 99.16 | 36,174 | 2,752 | 2,354 | 85.54 | 48.83 | 5,691,823.21 | 100.76 | 5,622,976.89 | 304.39 | 0.48 | 0.16 | SELECTED | Road exploratory preprocessing: duplicate-event derived copy; all-optimal search hit max_states for 11/2752 cases (counted as disagreement; all-optimal agreement is a lower bound, upper bound +11) |
| BPIC2017 | 0.10 | (1,9) | 43.49 | 31,440 | 2,360 | N/A | N/A | 4,887.86 | 154,322,800.54 | 6,015.30 | N/A | N/A | 0.81 | N/A | SELECTED | best executable trade-off (coverage >= 90% abandoned); clean coverage < 50%; C+PN: In progress (pending final C+PN result); B-C exact 926 / 2360 (39.24%), B-C tau-excluded all-optimal 998 / 2360 (42.29%) (C+PN in progress, so B-C is reported instead); all-optimal search hit max_states for 49/2360 cases and stopped at the memory limit for 32 cases (resource-limit) (counted as disagreement; all-optimal agreement is a lower bound, upper bound +81) |
