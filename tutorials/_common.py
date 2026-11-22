"""Shared settings and helpers of the five Credit tutorials (paths, official settings, checks)."""
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------- data (Credit running example)
DATA_NAME = "credit-card"
DATA_DIR = ROOT / "data" / "credit"
ANOMALY_LOG = DATA_DIR / "credit-card_1.00.csv"   # clean + deviating cases (column type_res_trace)
NORMAL_LOG = DATA_DIR / "credit-card.csv"         # ground-truth normal log (progress printout only)
INPUT_PATHS = {"anomaly_log": ANOMALY_LOG, "ground_truth_log": NORMAL_LOG}

# ---------------------------------------------------------------- official settings of the paper run (Credit)
NOISE_THRESHOLD = 0.0              # Inductive Miner noise threshold (NT)
B_NUM, B_TOT = 5, 7                # (b_num, b_tot) of Definition 2
SAFETY = {"max_trace_length": 100, "max_intermediate": 1_000_000, "max_seconds": 600, "max_rss_gb": 16}
RL = {"num_epi": 1000, "alpha": 0.0, "val_ratio": 0.5, "seed": 42, "max_step": 8}
RL_AGENT = {"learning_rate": 0.1, "discount_factor": 0.9, "exploration_rate": 0.5, "exploration_decay_rate": 0.0001}

# ---------------------------------------------------------------- artifacts and results
ART = ROOT / "artifacts"
REF = ART / "reference"
PNML = ART / "petri_net" / "credit_petri_net.pnml"          # tutorial 1 output
TREE = ART / "petri_net" / "credit_process_tree.txt"        # tutorial 1 output
LN_TRACES = ART / "sampling" / "credit_LN.csv"              # tutorial 1 output (one row per unique trace)
LN_LOG = ART / "sampling" / "credit_LN_log.csv"             # tutorial 1 output (Case / Activity, one case per trace)
PRETRAINED = ART / "pretrained_models" / "credit_ra_alignment"
MODELS = ART / "models"                                     # tutorial 2 output (your own trained models)
RES = ROOT / "results"
OA_RESULTS = RES / "optimal_alignment"
RA_RESULTS = RES / "ra_alignment"
CMP_RESULTS = RES / "comparison"
REF_TEST_IDS = REF / "credit_test_case_ids.txt"

ALLOWED_PICKLE_GLOBALS = {  # what an RA-Alignment agent pickle may reference
    ("rpbear.func_rpbear_final", "QLearningAgent"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"), ("numpy", "ndarray"), ("numpy", "dtype"), ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"), ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"), ("builtins", "slice")}


def scipy_lp_shim():
    """Use PM4Py's scipy LP solver (as in the paper runs) by hiding cvxopt; stub lxml.objectify (only the PTML importer
    needs it). Call before importing pm4py. Affects only this process."""
    sys.modules["cvxopt"] = None
    import lxml
    from lxml import etree as _etree

    obj = types.ModuleType("lxml.objectify")
    obj.parse, obj.fromstring = _etree.parse, _etree.fromstring
    sys.modules["lxml.objectify"] = obj
    lxml.objectify = obj


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_ids(path):
    return [x for x in Path(path).read_text(encoding="utf-8").split() if x]


def check(ok, msg, strict=True):
    print(f"  [{'OK' if ok else 'FAILED'}] {msg}", flush=True)
    if strict and not ok:
        raise SystemExit(f"stopped: {msg}")
    return ok


def need(path, hint):
    if not Path(path).is_file():
        raise SystemExit(f"[missing] {Path(path).relative_to(ROOT)} - {hint}")
    return Path(path)


def manifest_sha(rel):
    """Expected SHA-256 of a shipped artifact (artifacts/MANIFEST.csv)."""
    import csv

    with open(ART / "MANIFEST.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["file"] == rel:
                return r["sha256"]
    return None


def load_agent(path):
    """Load an RA-Alignment agent; only the globals in ALLOWED_PICKLE_GLOBALS may be referenced by the pickle."""
    class AllowList(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in ALLOWED_PICKLE_GLOBALS:
                raise pickle.UnpicklingError(f"blocked global {module}.{name}")
            return super().find_class(module, name)

    with open(path, "rb") as fh:
        return AllowList(fh).load()


def net_signature(net, im, fm):
    """Structure of a Petri net independent of element names (labels, arc pattern per transition, markings)."""
    from collections import Counter

    def tsig(t):
        pre = sorted((len(a.source.in_arcs), len(a.source.out_arcs)) for a in t.in_arcs)
        post = sorted((len(a.target.in_arcs), len(a.target.out_arcs)) for a in t.out_arcs)
        return str((t.label, tuple(pre), tuple(post)))
    return {"places": len(net.places), "transitions": len(net.transitions), "arcs": len(net.arcs),
            "invisible": sum(t.label is None for t in net.transitions),
            "transition_signatures": sorted(Counter(map(tsig, net.transitions)).items()),
            "im": sorted(len(p.in_arcs) for p in im), "fm": sorted(len(p.out_arcs) for p in fm)}


def model_files(model_dir):
    """Files of an RA-Alignment model folder (pre-trained or written by tutorial 2)."""
    model_dir = Path(model_dir)
    pick = lambda pat: next(iter(sorted(model_dir.glob(pat))), None)  # noqa: E731
    return {"agent": pick("*_agent.pkl"), "repaired_log": pick("*_repaired_log.csv"), "stats": pick("*_training_stats.json"),
            "train_ids": pick("*_train_case_ids.txt"), "test_ids": pick("*_test_case_ids.txt")}


def space(trace):
    return " ".join(map(str, trace))
