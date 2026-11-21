# Threshold trade-off

- Alignment ms/case = threshold-sweep measurement (14_threshold_b_alignment.py, sequential). BPIC 2017 uses a 30-case test sample, Road all test cases. Not the main timing (main = paper_result_table.md).
- C ms/case = sequential measurement over all test cases (run only at the selected threshold).

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
| BPIC2017 | 0.10 | (1,9) | 43.49 | 31,440 | 76 | 1,716.25 | 6,015.30 | 47.63 | **SELECTED** |
| BPIC2017 | 0.15 | (8,33) | 58.57 | (est. 1.06e+22) | 71 | 1,536.66 | N/A | N/A | dominated (lower coverage at same budget) (BPIC rule; 95%-rule status INFEASIBLE) |
| BPIC2017 | 0.20 | (8,33) | 58.57 | (est. 1.15e+22) | 71 | 1,828.92 | N/A | N/A | dominated (lower coverage at same budget) (BPIC rule; 95%-rule status INFEASIBLE) |
| BPIC2017 | 0.30 | (8,33) | 58.57 | (est. 1.15e+22) | 71 | 1,828.92 | N/A | N/A | dominated (lower coverage at same budget) (BPIC rule; 95%-rule status INFEASIBLE) |
| BPIC2017 | 0.40 | (8,37) | 2.05 | (est. 5.51e+29) | 67 | 3,273.21 | N/A | N/A | TOO_LARGE (BPIC rule; 95%-rule status INFEASIBLE) |

## Threshold selection rationale

Rule (same for every threshold): apply the same loop-bound rule (Pareto-minimal bound with the smallest language among those with clean coverage ≥ 95%), then choose, among the thresholds at which Method C can run (generated language ≤ 2M), the point with the smallest language that keeps coverage ≥ 90% (≥ 95% if possible). Credit / Pub are already executable at 0.00, so 0.00 is kept.

**Road Traffic → 0.20 (2,2).** 0.00 gives 1,344,244 traces (95.11%) and the W2 NBG did not finish within one hour. 0.05 gives 241,528 traces / 99.42%, executable but with a language 6.7 times larger than 0.20. 0.10 / 0.15 reach at most 94.13%, below 95% (small language). 0.20 keeps 99.16% with 36,174 traces, the elbow where the language drops sharply; 0.30 is the same model as 0.20. At 0.40 the activities h, i, j disappear from the model and coverage drops to 81.03%.

**BPIC 2017 → 0.10 (1,9), best executable trade-off.** With the 95% rule above no threshold has an executable bound (0.00 needs about 10^21 traces for 95%; for thresholds > 0 even the unbounded maximal coverage is below 90%: 0.05 87.80%, 0.15–0.30 58.57%, 0.40 2.05%). BPIC 2017 therefore drops the coverage criterion and uses the condition with the highest clean coverage among those under which B / C / C+PN run unchanged (paper_bpic_threshold_tradeoff.md, bpic_candidate_feasibility.md). Threshold 0.05 reaches higher coverage with a larger language ((1,10) 97,056 traces / 47.82%), but within the same 50k budget (1,8) gives 20,032 traces / 40.09%, below 0.10 (1,9) with 31,440 traces / 43.49%. 0.15–0.30 saturate at 39.71%. Hence 0.10 (1,9) is the best trade-off. A coverage of 43.49% is below half of the clean cases and must not be read as model coverage comparable to Road. BPIC 2017 C+PN was run afterwards (measured training time: see the timing table).
