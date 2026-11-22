"""Run src/simulation.py inside an LP-solver shim (.venv stages).

Inside this process only, cvxopt is hidden (PM4Py LP solver -> scipy) and lxml.objectify is replaced by
lxml.etree; then src/simulation.py runs unchanged. This is needed where the cvxopt / lxml.objectify
binaries cannot be loaded. The installation and the src/ code are not changed.
Credit Method B equivalence (research-repository check, not included here): on 210/210 test and 210/210 val cases the
alignment cost, repaired trace and repaired_log_b equal the official cvxopt run.

Usage: .venv/Scripts/python scripts/run_with_scipy_lp.py --dataset Pub --stage discover --data-root <DATA_ROOT>
"""
import runpy
import sys
import types
from pathlib import Path

sys.modules["cvxopt"] = None
import lxml  # noqa: E402
from lxml import etree as _etree  # noqa: E402

_obj = types.ModuleType("lxml.objectify")
_obj.parse, _obj.fromstring = _etree.parse, _etree.fromstring
sys.modules["lxml.objectify"] = _obj
lxml.objectify = _obj

SIM = Path(__file__).resolve().parents[1] / "src" / "simulation.py"
sys.path.insert(0, str(SIM.parent))

# Optional (ICPM_REUSE_MODEL_CHECK=1): replace utils.discovery_utils.model_report by the model_check.json
# that the discover stage already computed for the same model.
# Why: model_report computes alignment fitness for every clean variant (BPIC 2017: 3,431 variants, about 8.5 h).
# The generate / run-b stages rediscover the same tree and would repeat the same deterministic computation.
# Reuse only if the process-tree string and places / transitions / invisible / arcs are identical and the
# tree has a LOOP (language not enumerated, language_set=None); otherwise the original function is used.
# Off by default; definitions of B / C / C+PN are unchanged.
import os  # noqa: E402

if os.environ.get("ICPM_REUSE_MODEL_CHECK") == "1":
    import json  # noqa: E402

    import utils.discovery_utils as _du  # noqa: E402

    _orig_model_report = _du.model_report
    _sim_root = SIM.parents[1] / "outputs" / "simulation"

    def _cached_model_report(tree, net, im, fm, traces, clean_ids, discovery_sec, safety_max_traces):
        key = str(tree)
        visible = [t for t in net.transitions if t.label is not None]
        cur = {"n_places": len(net.places), "n_transitions": len(net.transitions),
               "n_invisible_transitions": len(net.transitions) - len(visible), "n_arcs": len(net.arcs)}
        for p in sorted(_sim_root.glob("*/threshold_*/discover/model_check.json")):
            mc = json.loads(p.read_text(encoding="utf-8"))
            if mc.get("process_tree") != key or any(mc.get(k) != v for k, v in cur.items()):
                continue
            if mc.get("structure", {}).get("estimated_raw_tree_language_size") is not None:
                continue  # models whose language is enumerated are not reused
            n_clean = len(clean_ids)
            if mc.get("n_clean_cases") != n_clean:
                continue
            mc = dict(mc, discovery_sec=round(discovery_sec, 4))
            print(f"[shim] model_report reused from {p.relative_to(SIM.parents[1])} "
                  f"(same tree / places / transitions / arcs / clean cases)", flush=True)
            return mc, None
        print("[shim] model_report: no matching discover model_check -> original computation", flush=True)
        return _orig_model_report(tree, net, im, fm, traces, clean_ids, discovery_sec, safety_max_traces)

    _du.model_report = _cached_model_report
sys.argv = [str(SIM)] + sys.argv[1:]
print("[shim] cvxopt hidden -> pm4py LP solver = scipy; lxml.objectify -> lxml.etree", flush=True)
runpy.run_path(str(SIM), run_name="__main__")
