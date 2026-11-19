"""Shared helpers of the multi-dataset pipeline.

src/simulation.py calls these modules stage by stage. Modules that need PM4Py
(discovery_utils, parts of trace_generation_utils) run in .venv; modules that need
RP-BEAR (nbg_utils, run_utils) run in .venv-rpbear. Submodules are therefore not
imported here.
"""
