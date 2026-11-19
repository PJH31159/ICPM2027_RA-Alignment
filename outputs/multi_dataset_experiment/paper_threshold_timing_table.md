# Threshold → timing (ms)

speedup = B mean / C mean (1.0 = 동일, > 1.0 = C가 더 빠름, < 1.0 = B가 더 빠름).
B, C, C+PN repair = 모두 순차 Test-only 측정. C 계열은 선택 threshold에서만 실행.
B 열은 threshold 간 비교를 위해 모두 sweep Petri Net으로 같은 방식(14_threshold_b_alignment.py)으로 잰 값이다 (일부는 다른 작업과 동시에 측정되어 약 10% 오차; 같은 모델인 Road 0.20 / 0.30이 55.92 / 61.29). main result table의 Road B 값은 공식 run-b Petri Net으로 다른 작업 없이 다시 잰 순차 replay(seq_B, 48.83 ms/case)이다.

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
| BPIC2017 | 0.10 | 30 | 1,716.25 | 154,322,800.54 | 6,015.30 | N/A | N/A | SELECTED | 0.28 | N/A |
| BPIC2017 | 0.15 | 30 | 1,536.66 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| BPIC2017 | 0.20 | 30 | 1,828.92 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| BPIC2017 | 0.30 | 30 | 1,828.92 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
| BPIC2017 | 0.40 | 30 | 3,273.21 | N/A | N/A | N/A | N/A | INFEASIBLE | N/A | N/A |
