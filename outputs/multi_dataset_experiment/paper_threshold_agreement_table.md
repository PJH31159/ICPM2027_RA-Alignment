# Threshold → agreement

repair accuracy는 main 표에서 제외 (교수님 지시).
- RoadTraffic 0.20: all-optimal search hit max_states for 11/2752 cases (counted as disagreement; all-optimal agreement is a lower bound, upper bound +11)
- BPIC2017 0.10: all-optimal search hit max_states for 49/2360 cases and stopped at the memory limit for 32 cases (resource-limit) (counted as disagreement; all-optimal agreement is a lower bound, upper bound +81)

| Dataset | Threshold | TestCases | B-C Exact | B-C Tau-Excluded All-Optimal | B-C+PN Exact | B-C+PN Native All-Optimal | B-C+PN Tau-Excluded All-Optimal | Status |
|---|---|---|---|---|---|---|---|---|
| RoadTraffic | 0.00 | N/A | N/A | N/A | N/A | N/A | N/A | INFEASIBLE |
| RoadTraffic | 0.05 | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN_C |
| RoadTraffic | 0.10 | N/A | N/A | N/A | N/A | N/A | N/A | COVERAGE_LOSS |
| RoadTraffic | 0.15 | N/A | N/A | N/A | N/A | N/A | N/A | COVERAGE_LOSS |
| RoadTraffic | 0.20 | 2,752 | 2029 / 2752 (73.73%) | 2073 / 2752 (75.33%) | 2293 / 2752 (83.32%) | 2297 / 2752 (83.47%) | 2354 / 2752 (85.54%) | SELECTED |
| RoadTraffic | 0.30 | N/A | N/A | N/A | N/A | N/A | N/A | SAME_AS_0.20 |
| RoadTraffic | 0.40 | N/A | N/A | N/A | N/A | N/A | N/A | COVERAGE_LOSS |
| BPIC2017 | 0.00 | N/A | N/A | N/A | N/A | N/A | N/A | INFEASIBLE |
| BPIC2017 | 0.05 | N/A | N/A | N/A | N/A | N/A | N/A | INFEASIBLE |
| BPIC2017 | 0.10 | 2,360 | 926 / 2360 (39.24%) | 998 / 2360 (42.29%) | N/A | N/A | N/A | SELECTED |
| BPIC2017 | 0.15 | N/A | N/A | N/A | N/A | N/A | N/A | INFEASIBLE |
| BPIC2017 | 0.20 | N/A | N/A | N/A | N/A | N/A | N/A | INFEASIBLE |
| BPIC2017 | 0.30 | N/A | N/A | N/A | N/A | N/A | N/A | INFEASIBLE |
| BPIC2017 | 0.40 | N/A | N/A | N/A | N/A | N/A | N/A | INFEASIBLE |
