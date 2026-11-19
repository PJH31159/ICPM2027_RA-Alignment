"""shim_simulation.py — src/simulation.py를 scipy LP solver shim 안에서 실행 (.venv 단계 전용)

왜: 이 PC는 Windows Smart App Control이 .venv의 cvxopt / lxml.objectify DLL을 차단한다 (사용자: SAC는 끄지 않음).
    process 안에서만 cvxopt를 숨기고(pm4py LP solver -> scipy) lxml.objectify를 lxml.etree로 대체한 뒤
    src/simulation.py를 그대로 실행한다. 설치 / 보안 설정 / src 코드는 바꾸지 않는다.
    Credit Method B equivalence (06_credit_scipy_equivalence.py): test 210/210, val 210/210에서
    alignment cost / repaired trace / repaired_log_b가 official(cvxopt)과 동일 -> "official-equivalent for this experiment".

실행: .venv\\Scripts\\python outputs\\multi_dataset_experiment\\scripts\\shim_simulation.py --dataset Pub --stage discover --data-root ...
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

SIM = Path(__file__).resolve().parents[3] / "src" / "simulation.py"
sys.path.insert(0, str(SIM.parent))

# 선택 옵션 (BPIC 2017, 2026-11-13): ICPM_REUSE_MODEL_CHECK=1 이면 utils.discovery_utils.model_report를
# 같은 모델에 대해 discover stage가 이미 계산한 model_check.json으로 대체한다.
# 왜: model_report는 clean variant마다 alignment fitness를 계산한다 (BPIC 3,431 variants, discover에서 약 8.5시간).
#     generate / run-b stage는 같은 tree를 다시 발견하고 같은 계산을 반복한다 (결과는 discover와 같은 net에 대한
#     deterministic alignment 이므로 동일). 재사용 조건: process_tree 문자열, places / transitions / invisible / arcs가
#     모두 같고 tree에 LOOP가 있어 language를 나열하지 않는 경우 (language_set=None). 조건이 하나라도 다르면 원래 함수를 쓴다.
#     src/ 코드, discover 결과, B/C/C+PN 정의는 바꾸지 않는다. 기본값은 꺼짐.
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
                continue  # language를 나열하는 모델은 재사용하지 않음
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
