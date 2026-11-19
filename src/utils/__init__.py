"""
utils — multi-dataset research pipeline의 공통 기능 (REF-001)

simulation.py(orchestrator)가 이 모듈들을 단계별로 호출한다.
pm4py가 필요한 모듈(discovery_utils, trace_generation_utils의 일부)은 .venv에서,
RP-BEAR가 필요한 모듈(nbg_utils, run_utils)은 .venv-rpbear에서 쓴다.
그래서 이 파일에서는 하위 모듈을 미리 import하지 않는다.
"""
