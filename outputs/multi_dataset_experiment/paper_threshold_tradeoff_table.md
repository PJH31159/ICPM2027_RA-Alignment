# Threshold trade-off

| Dataset | Threshold | Loop Bound | Coverage (%) | Generated Traces | Petri Net Transitions | Alignment ms/case | C ms/case | Agreement (B-C+PN tau-excl. all-opt., %) | Decision |
|---|---|---|---|---|---|---|---|---|---|
| RoadTraffic | 0.00 | (1,1) | 95.11 | 1,344,244 | 33 | 230.47 | N/A | N/A | Baseline (Too large) |
| RoadTraffic | 0.05 | (2,2) | 99.42 | 241,528 | 30 | 70.78 | N/A | N/A | Candidate (larger language) |
| RoadTraffic | 0.10 | (10,10) | 94.13 | 4,598 | 28 | 50.64 | N/A | N/A | Coverage loss |
| RoadTraffic | 0.15 | (10,10) | 94.13 | 3,586 | 27 | 45.06 | N/A | N/A | Coverage loss |
| RoadTraffic | 0.20 | (2,2) | 99.16 | 36,174 | 27 | 55.92 | 100.76 | 85.54 | **SELECTED** |
| RoadTraffic | 0.30 | (2,2) | 99.16 | 36,174 | 27 | 61.29 | N/A | N/A | Same model as 0.20 |
| RoadTraffic | 0.40 | (12,12) | 81.03 | 700,362 | 21 | 46.49 | N/A | N/A | Coverage loss |
| BPIC2017 | 0.00 | (10,14) | 95.11 | (est. 1e+21) | 75 | 15,582.53 | N/A | N/A | TOO_LARGE (BPIC rule; 95%-rule status INFEASIBLE) |
| BPIC2017 | 0.05 | (8,39) | 87.80 | (est. 9.73e+27) | 78 | 1,797.77 | N/A | N/A | dominated (lower coverage at same budget) (BPIC rule; 95%-rule status INFEASIBLE) |
| BPIC2017 | 0.10 | (1,9) | 43.49 | 31,440 | 76 | 1,716.25 | 6,015.30 | N/A | **SELECTED** |
| BPIC2017 | 0.15 | (8,33) | 58.57 | (est. 1.06e+22) | 71 | 1,536.66 | N/A | N/A | dominated (lower coverage at same budget) (BPIC rule; 95%-rule status INFEASIBLE) |
| BPIC2017 | 0.20 | (8,33) | 58.57 | (est. 1.15e+22) | 71 | 1,828.92 | N/A | N/A | dominated (lower coverage at same budget) (BPIC rule; 95%-rule status INFEASIBLE) |
| BPIC2017 | 0.30 | (8,33) | 58.57 | (est. 1.15e+22) | 71 | 1,828.92 | N/A | N/A | dominated (lower coverage at same budget) (BPIC rule; 95%-rule status INFEASIBLE) |
| BPIC2017 | 0.40 | (8,37) | 2.05 | (est. 5.51e+29) | 67 | 3,273.21 | N/A | N/A | TOO_LARGE (BPIC rule; 95%-rule status INFEASIBLE) |

## Threshold selection rationale

규칙 (모든 threshold 공통): 같은 loop-bound 규칙(clean coverage ≥ 95%인 Pareto-minimal bound 중 language 최소)을 적용한 뒤, Method C를 실행할 수 있는(생성 language ≤ 2M) threshold 중 coverage ≥ 90%(가능하면 ≥ 95%)를 유지하면서 language가 가장 작은 지점을 고른다. Credit / Pub는 0.00에서 이미 실행 가능하므로 0.00 그대로 쓴다.

**Road Traffic → 0.20 (2,2).** 0.00은 1,344,244 traces (95.11%)로 W2 NBG가 1시간 안에 끝나지 않았다. 0.05는 241,528 traces / 99.42%로 실행은 가능하지만 language가 0.20보다 6.7배 크다. 0.10 / 0.15는 최대 coverage 94.13%로 95%에 못 미친다(language는 작음). 0.20은 36,174 traces로 99.16%를 유지해 language가 급격히 줄어드는 elbow이며, 0.30은 0.20과 같은 모델이다. 0.40은 activity h, i, j가 모델에서 빠지고 coverage가 81.03%로 떨어진다.

**BPIC 2017 → 0.10 (1,9), best executable trade-off.** 위 95% 규칙으로는 어떤 threshold도 실행 가능한 bound가 없다 (0.00은 95%에 약 10^21 traces; threshold > 0은 unbounded 최대 coverage도 90% 미만: 0.05 87.80%, 0.15–0.30 58.57%, 0.40 2.05%). 교수님 방향 수정(2026-11-12)에 따라 BPIC는 coverage 기준을 버리고, 현재 PC에서 기존 B / C / C+PN을 수정 없이 실행할 수 있는 조건 중 clean coverage가 가장 높은 조건을 고른다 (paper_bpic_threshold_tradeoff.md, bpic_candidate_feasibility.md). threshold 0.05는 language를 키우면 coverage가 더 높지만((1,10) 97,056 traces / 47.82%), 같은 50k budget에서는 (1,8) 20,032 traces / 40.09%로 0.10 (1,9) 31,440 traces / 43.49%보다 낮다. 0.15–0.30은 39.71%에서 포화된다. 따라서 0.10 (1,9)가 trade-off에서 가장 적절하다. coverage 43.49%는 clean case의 절반 미만이므로 Road와 같은 수준의 모델 coverage로 해석하면 안 된다. BPIC 2017 C+PN: In progress (pending final C+PN result).
