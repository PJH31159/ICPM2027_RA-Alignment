# BPIC 2017: representative test cases (paper Table 3)

`bpic2017_test_10.csv` contains the 10 test cases of Table 3 of the paper, 269 events in total. Each row is one event.

| Column | Meaning |
|---|---|
| `Case` | case id (e.g. `Application_1210575946`) |
| `Activity` | encoded activity label, as in the P-BEAR-RL benchmark |
| `Timestamp` | event timestamp |
| `type_res_trace` | injected deviation pattern(s) of the case |

- The rows are copied unchanged from the deviation-injected BPIC 2017 log of the P-BEAR-RL benchmark, in file order.
  The file order is the event order used by the loaders of this repository.
- All 10 cases belong to the official test split of the paper run.
- Each trace is identical to the input of the official Optimal Alignment run.
- Only the columns read by the loaders are kept.
- The 10 cases were selected for the paper's comparison and are not a random sample of the test set (see the main
  README).

Source: *BPI Challenge 2017*, B. van Dongen, 4TU.ResearchData (2017),
<https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b> (4TU General Terms of Use). Deviations were
injected as in P-BEAR-RL (<https://github.com/jonghyeonk/P-BEAR-RL>).
