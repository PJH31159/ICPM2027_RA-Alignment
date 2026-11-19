"""Multi-dataset research pipeline orchestrator.

simulation.py applies the same research pipeline, stage by stage, to every dataset. It manages the
stage order, checks that the previous stage passed, records the input files and sets the output
locations; the computations are in utils/. sim.py is the RP-BEAR engine called by the run stages.

Pipeline:
    Method A: observed clean log -> Original NBG -> RP-BEAR
    Method B: clean cases -> Inductive Miner -> Petri net -> alignment repair
    Method C: observed clean log -> Inductive Miner -> Petri net -> generated (bounded) traces
              -> New NBG -> RP-BEAR
    C+PN    : Method C + Petri-net conformance gate on episode termination

Stages and environments:
    discover  (.venv)        IM -> process tree -> Petri net, structure checks (loop / parallel / cycle)
    generate  (.venv)        generation strategy -> exhaustive or bounded generation -> observed vs generated
    nbg       (.venv-rpbear) New NBG (weight 1, actset order, anchor check), comparison with Original, ADM check
    smoke     (.venv-rpbear) num_epi=5 end-to-end run + sequential A' vs C'
    run       (.venv-rpbear) official run (--method A|C --run-id N)
    evaluate  (any)          common scoring (--with-reference: also the configured reference results)
    regression (any)         Small only: comparison with the existing Small results
    test-timing          (.venv-rpbear) sequential test-only replay with a trained agent -> per-case repair latency
                         (--agent, --repaired-log, --label, --method; timing only, not an accuracy experiment)
    test-timing-validate (.venv-rpbear) checks that a test-only replay is byte-identical to the original test phase
    agreement            (any) agreement of the official Method A / B / C repaired logs
    run-b                (.venv) Method B official run (--run-id N) or smoke (--limit N)
    test-timing-b        (.venv) Method B: sequential alignment + repair of the test cases with the official Petri net
                         (--pnml, --repaired-log, --label; timing only)
    loop-bound-screening (.venv) loop bound selection for loop models: bounded trace set per (loop_num, loop_total)
                         -> smallest bound with clean-case coverage >= 0.90
    credit-loop-bound-sensitivity (.venv) Credit only: coverage vs number of generated traces per bound, final bound (5,7)
    pngate-* / run-c-pngate / test-timing-pngate  C+PN (W2 + Petri-net conformance gate), utils/pn_gate_utils.py;
                         outputs in <dataset>/C_PN_Gate/ and <dataset>/pn_gate_analysis/
    all       runs discover -> generate -> nbg -> smoke, each in its own environment (run not included)

Examples:
    .venv/Scripts/python src/simulation.py --dataset Small --stage discover
    .venv-rpbear/Scripts/python src/simulation.py --dataset Small --stage nbg
    python src/simulation.py --dataset Medium --stage discover --data-root <DATA_ROOT>

Data root priority: --data-root > environment variable ICPM2027_DATA_ROOT > <repo>/data/raw
Output: outputs/simulation/<Dataset>/<stage>/ (an existing stage folder stops the run unless --overwrite;
        official run folders under runs/ are never overwritten)
Exit codes: 0 PASSED, 1 FAILED, 2 BLOCKED (needs a human decision)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import dataset_config as dc
from utils.data_utils import describe_inputs, resolve_data_root, resolve_input_paths
from utils.validation_utils import (PROJECT_ROOT, CheckRecorder, Tee, read_ids, read_json,
                                    read_stage_status, write_json, write_stage_status)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "simulation"
STAGE_REQUIRES = {"discover": [], "generate": ["discover"], "nbg": ["generate"], "smoke": ["nbg"],
                  "run": ["smoke"], "evaluate": [], "regression": []}


# ============================================================
# 1. Run context (dataset, data root, input files, output location)
# ============================================================

class Context:
    def __init__(self, args):
        self.args = args
        self.cfg = dc.get_dataset(args.dataset)
        self.name = args.dataset
        self.data_root, self.root_source = resolve_data_root(args.data_root)
        self.input_paths, self.layout = resolve_input_paths(self.cfg, self.data_root, dc.ANOMALY_LEVEL)
        if self.cfg.get("anomaly_log_override"):  # RoadTraffic: derived copy (the raw file is not modified)
            self.input_paths["anomaly_log"] = PROJECT_ROOT / self.cfg["anomaly_log_override"]
        self.out = OUTPUT_ROOT / self.name
        # noise_threshold (Inductive Miner): if given, outputs go to threshold_0pXX/.
        # Otherwise dc.DISCOVERY (0.0) and the default output paths are used.
        th = getattr(args, "noise_threshold", None)
        self.noise = dc.DISCOVERY["noise_threshold"] if th is None else float(th)
        if th is not None:
            self.out = self.out / threshold_tag(self.noise)
        lb = getattr(args, "loop_bound", None)
        if lb:  # --loop-bound n,t : loop bound for bounded_loop generation
            n, t = (int(x) for x in lb.split(","))
            self.cfg["trace_generation"].update({"strategy": "bounded_loop", "loop_num": n, "loop_total": t,
                                                 "reference_traces": None})
            self.cfg["expected_preparation"] = None
        self.params = dict(dc.RPBEAR_PARAMS)

    def stage_dir(self, stage):
        return self.out / stage

    def data_name(self):
        """data_name for sim.run_rpbear (used in output file names) = file_stem, or run_name if set."""
        return self.cfg.get("run_name", self.cfg["file_stem"])


def threshold_tag(th):
    """noise_threshold -> output folder name (0.2 -> threshold_0p20)."""
    return f"threshold_{th:.2f}".replace(".", "p")


def record_inputs(ctx, rec):
    """Record resolved path / size / SHA-256 of the input files; fail if they differ from the previous record."""
    print(f"  dataset={ctx.name} (file_stem={ctx.cfg['file_stem']}) | data root={ctx.data_root} "
          f"[{ctx.root_source}] | layout={ctx.layout}")
    if not ctx.cfg["identity_confirmed"]:
        print(f"  [NOTE] dataset identity not confirmed: {ctx.cfg['identity_note']}")
    missing = [str(p) for p in ctx.input_paths.values() if not Path(p).exists()]
    if not rec.check("input files exist", not missing, missing or ""):
        return False
    records = describe_inputs(ctx.input_paths)
    path = ctx.out / "inputs.json"
    if path.exists():
        before = {r["role"]: r["sha256"] for r in read_json(path)["files"]}
        now = {r["role"]: r["sha256"] for r in records}
        return rec.check("input checksums unchanged since first use", before == now)
    write_json(path, {"dataset": ctx.name, "data_root": str(ctx.data_root), "data_root_source": ctx.root_source,
                      "layout": ctx.layout, "identity_confirmed": ctx.cfg["identity_confirmed"],
                      "identity_note": ctx.cfg["identity_note"], "files": records})
    for r in records:
        print(f"  [INPUT] {r['role']}: {r['resolved_path']} ({r['size_bytes']} bytes, sha256 {r['sha256'][:16]}…)")
    return True


def open_stage(ctx, stage):
    """Prepare the stage folder and check that the previous stage passed. Returns the stage dir or None."""
    for req in STAGE_REQUIRES[stage]:
        st = read_stage_status(ctx.stage_dir(req))
        if not st or st["status"] != "PASSED":
            print(f"[ABORT] previous stage '{req}' is not PASSED ({st['status'] if st else 'not run'})")
            return None
    d = ctx.stage_dir(stage)
    if d.exists():
        if not ctx.args.overwrite:
            print(f"[ABORT] {d} already exists. Use --overwrite to run again")
            return None
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def finish(stage_dir, stage, rec, blocked=False, extra=None):
    status = "BLOCKED" if blocked else ("PASSED" if rec.ok else "FAILED")
    write_stage_status(stage_dir, stage, status, rec, extra)
    return {"PASSED": 0, "FAILED": 1, "BLOCKED": 2}[status]


def need_pm4py():
    try:
        import pm4py  # noqa: F401
        return True
    except ModuleNotFoundError:
        print('[ERROR] this stage must run in .venv (with pm4py).')
        return False


TIMING_SCOPE = {  # training time and preprocessing time are reported separately
    "training_ms": "RP-BEAR training ('validation') phase wall-clock = sim.py val_total_ms (excludes data load / NBG construction / agent saving / test / evaluation). Method B = N/A",
    "test_repair_mean_ms_per_case / median": 'A / C / C+PN: per-case timer (perf_counter inside the worker, includes multiprocessing CPU sharing; C+PN includes the Petri-net gate). B: per-case align_case + repair_from_alignment (sequential)',
    "test_wall_ms_per_case": 'test phase wall-clock / test cases (A / C / C+PN: multiprocessing throughput)',
    "preprocessing": 'discovery_ms / generation_ms / nbg_build_ms = values recorded by the discover / generate / nbg stages (if present)',
}


def write_standard_timing(ctx, run_dir, method):
    """Write <prefix>_timing.json (common schema) into an official run folder; never overwrites."""
    import pandas as pd

    stats_f = next(Path(run_dir).glob("*_stats.json"), None)
    if stats_f is None:
        print('[timing] no stats.json -> timing.json skipped')
        return
    out = stats_f.with_name(stats_f.name.replace("_stats.json", "_timing.json"))
    if out.exists():
        print(f"[timing] {out.name} already exists -> not overwritten")
        return
    st = read_json(stats_f)
    row = {"dataset": ctx.name, "method": method, "run_id": ctx.args.run_id, "timing_scope": TIMING_SCOPE}
    if method == "B":
        tm = pd.read_csv(next(Path(run_dir).glob("*_alignment_timing.csv")))
        te = tm[tm["type"] == "test"]["align_ms"]
        try:
            from pm4py.util.lp import solver as _lp
            solver = _lp.DEFAULT_LP_SOLVER_VARIANT
        except Exception:  # noqa: BLE001
            solver = None
        row.update({"train_cases": None, "test_cases": int(len(te)), "training_ms": None,
                    "test_repair_total_ms": round(float(te.sum()), 4),
                    "test_repair_mean_ms_per_case": round(float(te.mean()), 4) if len(te) else None,
                    "test_repair_median_ms_per_case": round(float(te.median()), 4) if len(te) else None,
                    "test_wall_ms_per_case": round(float(te.sum()) / len(te), 4) if len(te) else None,
                    "lp_solver_in_process": solver})
    else:
        ct = pd.read_csv(next(Path(run_dir).glob("*_case_timing.csv")))
        te = ct[ct["phase"] == "test"]
        rl = te[te["path"] == "RL"]["repair_time_ms"]
        n = int(st["n_test_cases"])
        row.update({"train_cases": int(st["n_val_cases"]), "test_cases": n, "training_ms": st["val_total_ms"],
                    "test_repair_total_ms": st["test_total_ms"],
                    "test_repair_mean_ms_per_case": round(float(te["repair_time_ms"].mean()), 4),
                    "test_repair_median_ms_per_case": round(float(te["repair_time_ms"].median()), 4),
                    "test_wall_ms_per_case": round(st["test_total_ms"] / n, 4) if n else None,
                    "test_rl_path_cases": int(len(rl)),
                    "test_rl_path_mean_ms": round(float(rl.mean()), 4) if len(rl) else None,
                    "test_rl_path_median_ms": round(float(rl.median()), 4) if len(rl) else None,
                    "num_cores": st.get("num_cores"), "use_multiprocessing": st.get("use_multiprocessing")})
    pre = {"discovery_ms": None, "generation_ms": None, "nbg_build_ms": None}
    mc = ctx.stage_dir("discover") / "model_check.json"
    if mc.exists() and read_json(mc).get("discovery_sec") is not None:
        pre["discovery_ms"] = round(read_json(mc)["discovery_sec"] * 1000, 4)
    gs = ctx.stage_dir("generate") / "generate_summary.json"
    if gs.exists() and read_json(gs).get("generation_time_sec") is not None:
        pre["generation_ms"] = round(read_json(gs)["generation_time_sec"] * 1000, 4)
    ns = ctx.stage_dir("nbg") / "nbg_comparison_summary.json"
    if ns.exists():
        pre["nbg_build_ms"] = read_json(ns).get("timing_ms")  # only present if the nbg stage recorded its timing
    row["preprocessing"] = pre
    row["source"] = {"stats": stats_f.name}
    write_json(out, row)
    print(f"[timing] {out.name}: training_ms={row['training_ms']} test_repair_mean_ms_per_case={row['test_repair_mean_ms_per_case']}")


# ============================================================
# 2. Stages
# ============================================================

def stage_discover(ctx):
    if not need_pm4py():
        return 1
    d = open_stage(ctx, "discover")
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print('\n[discover] Discover the process model from clean cases and report loop / parallel / cycle and the expected language size.')
        if not record_inputs(ctx, rec):
            return finish(d, "discover", rec)
        import pm4py

        from utils.data_utils import load_case_traces
        from utils.discovery_utils import discover_model, model_report

        traces, _labels, clean_ids, anomalous_ids = load_case_traces(ctx.input_paths["anomaly_log"])
        tree, net, im, fm, sec = discover_model(traces, clean_ids, ctx.noise)
        model_check, _lang = model_report(tree, net, im, fm, traces, clean_ids, sec,
                                          ctx.cfg["trace_generation"]["safety_max_traces"])
        model_check["n_anomalous_cases"] = len(anomalous_ids)
        write_json(d / "model_check.json", model_check)
        (d / "process_tree.txt").write_text(model_check["process_tree"] + "\n", encoding="utf-8")
        pm4py.write_pnml(net, im, fm, str(d / "petri_net.pnml"))

        print(f"  process tree: {model_check['process_tree'][:300]}")
        for k in ["n_places", "n_transitions", "n_invisible_transitions", "n_arcs", "n_clean_cases",
                  "n_clean_variants", "clean_log_fitness_alignment_weighted"]:
            rec.info(k, model_check[k])
        for k, v in model_check["structure"].items():
            rec.info(f"structure.{k}", v)
        if ctx.cfg["expected_model"]:
            for k, v in ctx.cfg["expected_model"].items():
                rec.check(f"expected {k} == {v}", model_check[k] == v, f"(actual {model_check[k]})")
    return finish(d, "discover", rec)


def stage_generate(ctx):
    if not need_pm4py():
        return 1
    d = open_stage(ctx, "generate")
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print('\n[generate] Generate the clean trace set allowed by the Petri net (only when this is safe).')
        if not record_inputs(ctx, rec):
            return finish(d, "generate", rec)
        from collections import Counter

        import pandas as pd

        from utils.data_utils import load_case_traces, write_generated_log
        from utils.discovery_utils import discover_model, model_report
        from utils.trace_generation_utils import (bound_was_hit, compare_generated_observed, decide_strategy,
                                                  enumerate_petri_net_language)

        tg = ctx.cfg["trace_generation"]
        traces, _l, clean_ids, _a = load_case_traces(ctx.input_paths["anomaly_log"])
        tree, net, im, fm, sec = discover_model(traces, clean_ids, ctx.noise)
        model_check, tree_lang = model_report(tree, net, im, fm, traces, clean_ids, sec, tg["safety_max_traces"])
        prev = read_json(ctx.stage_dir("discover") / "model_check.json")
        rec.check("re-discovered model == discover stage", model_check["process_tree"] == prev["process_tree"])

        if tg["strategy"] == "bounded_loop":  # loop model: bounded generation with the official loop bound
            return generate_bounded_loop(ctx, d, rec, tree, net, im, fm, traces, clean_ids)

        decision = decide_strategy(tg, model_check)
        write_json(d / "strategy_decision.json", {"decision": decision, "trace_generation_config": tg,
                                                  "structure": model_check["structure"]})
        print(f"  strategy={decision['strategy']} allowed={decision['allowed']}")
        if not decision["allowed"]:
            for r in decision["reasons"]:
                print(f"   - {r}")
            print('  -> no traces generated (BLOCKED); a bounded generation setting is required.')
            return finish(d, "generate", rec, blocked=True, extra={"reasons": decision["reasons"]})

        accepted, enum_stats = enumerate_petri_net_language(
            net, im, fm, tg["safety_max_firings_per_transition"], tg["safety_max_trace_length"],
            tg["safety_max_traces"])
        observed = Counter(tuple(traces[cid]) for cid in clean_ids)
        summary = compare_generated_observed(accepted, observed, tree_lang)
        summary.update({"enumeration_stats": enum_stats, "strategy": decision["strategy"],
                        "safety_bounds": {k: v for k, v in tg.items() if k.startswith("safety_")},
                        "safety_bounds_note": "safety bounds only; not loop_num / loop_total",
                        "weighting": "each unique generated trace used once (model language treated as a set; "
                                     "observed frequencies are NOT restored)"})
        for k, v in summary.items():
            if k not in ("generated_only", "observed_only"):
                print(f"  {k}: {v}")

        rec.check("no safety bound was hit (truncation / pruning)", not bound_was_hit(enum_stats), enum_stats)
        rec.check("generated unique traces > 0", summary["generated_unique_trace_count"] > 0)
        if summary["generated_equals_process_tree_language"] is not None:
            rec.check("generated set == Process Tree language (cross-check)",
                      summary["generated_equals_process_tree_language"])
        rec.info("generated_only_count (structural novelty)", summary["generated_only_count"])
        rec.info("observed_only_count", summary["observed_only_count"])

        generated = sorted(set(accepted))
        write_generated_log(generated, d / "generated_normal_log.csv")
        pd.DataFrame([{"trace_id": f"gen_{i:04d}", "trace": ">>".join(t), "length": len(t),
                       "observed_clean_cases": observed.get(t, 0)} for i, t in enumerate(generated, start=1)]
                     ).to_csv(d / "generated_traces_vs_observed.csv", index=False)
        write_json(d / "generate_summary.json", summary)
    return finish(d, "generate", rec)


def generate_bounded_loop(ctx, d, rec, tree, net, im, fm, traces, clean_ids):
    """Generate stage for loop models with the official loop bound (e.g. Credit (5,7) -> 5,000 traces).

    generated_normal_log.csv (generated-only input) is not written, so that it cannot be used by mistake.
    Instead model_only_generated_log.csv (input of the W2 Expanded New NBG) and generated_bounded_traces.csv
    (record) are written.
    """
    from collections import Counter

    import pandas as pd

    from utils.data_utils import write_generated_log
    from utils.loop_bound_utils import generate_bounded, make_petri_acceptor

    tg = ctx.cfg["trace_generation"]
    exp = ctx.cfg["expected_preparation"] or {}
    prep = ctx.out / "preparation"
    prep.mkdir(exist_ok=True)
    n, t = tg["loop_num"], tg["loop_total"]
    print(f"  strategy=bounded_loop (loop_num={n}, loop_total={t})")
    lang, gst = generate_bounded(tree, n, t, tg["safety_max_trace_length"], tg["safety_max_traces"], 600, 16)
    gen = set(lang)
    if tg.get("reference_traces"):  # check against the reference trace set if configured (Credit); otherwise no reference
        ref_path = PROJECT_ROOT / tg["reference_traces"]
        ref = {tuple(x.split(">>")) for x in pd.read_csv(ref_path)["trace"]}
        rec.check(f"bounded generation ({n},{t}) == reference traces ({tg['reference_traces']})", gen == ref,
                  f"(generated {len(gen)}, reference {len(ref)})")
    else:
        ref = None
        rec.info("reference traces", "none (first run for this dataset; observed only)")
    rec.check("bounded generation not truncated (ExplosionAborted not raised)", len(gen) > 0, f"({len(gen)})")
    if exp:
        rec.check(f"generated unique == {exp['generated_unique']}", len(gen) == exp["generated_unique"], f"({len(gen)})")
    accepts = make_petri_acceptor(net, im, fm)
    rec.check("all generated traces accepted by discovered Petri Net", all(accepts(x) for x in gen))

    observed = Counter(tuple(traces[c]) for c in clean_ids)
    obs_in = [v for v in observed if v in gen]
    model_only = sorted(gen - set(observed), key=lambda z: (len(z), z))
    covered = sum(observed[v] for v in obs_in)
    rec.check("generated = observed-in-bound + model-only (disjoint)",
              len(obs_in) + len(model_only) == len(gen) and not set(model_only) & set(observed))
    if exp:
        rec.check(f"observed clean variants in generated set == {exp['observed_in_generated']}",
                  len(obs_in) == exp["observed_in_generated"], f"({len(obs_in)})")
        rec.check(f"model-only traces == {exp['model_only']}", len(model_only) == exp["model_only"], f"({len(model_only)})")
        rec.check(f"uncovered clean cases == {exp['uncovered_clean_cases']}",
                  sum(observed.values()) - covered == exp["uncovered_clean_cases"], f"({sum(observed.values()) - covered})")

    pd.DataFrame([{"trace": ">>".join(x), "length": len(x), "observed_clean_cases": observed.get(x, 0),
                   "model_only": x not in observed} for x in sorted(gen, key=lambda z: (len(z), z))]
                 ).to_csv(d / "generated_bounded_traces.csv", index=False)
    write_generated_log(model_only, d / "model_only_generated_log.csv")
    gen_summary = {"strategy": "bounded_loop", "loop_num": n, "loop_total": t,
                   "generated_unique_trace_count": len(gen), "reference_traces": tg.get("reference_traces"),
                   "equals_reference": None if ref is None else gen == ref,"generation_time_sec": round(gst["generation_time_sec"], 4),
                   "generation_ms": round(gst["generation_time_sec"] * 1000, 2),
                   "observed_clean_variant_count": len(observed), "observed_in_generated": len(obs_in),
                   "model_only_count": len(model_only), "clean_cases": sum(observed.values()),
                   "clean_cases_covered": covered, "clean_case_coverage": round(covered / sum(observed.values()), 6),
                   "max_trace_length": max(map(len, gen)),
                   "safety_bounds": {k: v for k, v in tg.items() if k.startswith("safety_")},
                   "new_nbg_input": ctx.cfg["new_nbg_input"],
                   "files": {"model_only_log (Method C NBG input, D039)": "model_only_generated_log.csv",
                             "all generated traces (record)": "generated_bounded_traces.csv",
                             "generated_normal_log.csv": 'NOT written (the generated-only input is not used)'}}
    write_json(d / "generate_summary.json", gen_summary)
    write_json(prep / "bounded_generation_check.json",
               {k: gen_summary[k] for k in ["strategy", "loop_num", "loop_total", "generated_unique_trace_count",
                                            "reference_traces", "equals_reference", "generation_time_sec",
                                            "clean_case_coverage", "max_trace_length"]}
               | {"all_accepted_by_petri_net": all(accepts(x) for x in gen)})
    write_json(prep / "model_only_check.json",
               {"definition": "generated set - observed clean variant set", "generated": len(gen),
                "observed_clean_variants": len(observed), "observed_in_generated": len(obs_in),
                "model_only": len(model_only), "disjoint_from_observed": not set(model_only) & set(observed),
                "clean_cases_covered": covered, "uncovered_clean_cases": sum(observed.values()) - covered,
                "model_only_log": str((d / "model_only_generated_log.csv").relative_to(PROJECT_ROOT))})
    for k in ["generated_unique_trace_count", "observed_in_generated", "model_only_count", "clean_case_coverage"]:
        print(f"  {k}: {gen_summary[k]}")
    return finish(d, "generate", rec)


def w2_nbg_preparation(ctx, rec, d, clean, anomaly, actset, filtered_actset, original, new):
    """W2 Expanded New NBG preparation checks: structure / edge provenance / alpha-0 ADM (all clean + all anomalous cases)."""
    import pandas as pd

    from utils.nbg_utils import adm_equality, edge_set_digest, nbg_edge_set, trace_edge_set

    exp = ctx.cfg["expected_preparation"]
    prep = ctx.out / "preparation"
    prep.mkdir(exist_ok=True)
    gen = pd.read_csv(ctx.stage_dir("generate") / "generated_bounded_traces.csv")
    generated = [tuple(x.split(">>")) for x in gen["trace"]]
    body = clean[~clean["concept:name"].isin(["Start", "End"])]
    clean_tr = {str(c): tuple(map(str, g["concept:name"])) for c, g in body.groupby("case:concept:name", observed=True)}
    observed = set(clean_tr.values())

    def summ(nbgs, name):
        return {"name": name, "graphs": len(nbgs), "nodes": int(sum(g.number_of_nodes() for g in nbgs)),
                "edges": int(sum(g.number_of_edges() for g in nbgs)), "digest": edge_set_digest(nbgs, actset),
                "per_graph": [{"activity": str(a), "nodes": g.number_of_nodes(), "edges": g.number_of_edges()}
                              for a, g in zip(actset, nbgs)]}

    so, sn = summ(original, "Original NBG (observed clean log)"), summ(new, 'Expanded New NBG (W2)')
    for key, s in (("original_nbg", so), ("expanded_nbg", sn)):
        if exp is None:  # dataset without expected values: record observations only (edge-provenance checks still apply below)
            rec.info(f"{key} graphs / nodes / edges / digest (observed)", (s["graphs"], s["nodes"], s["edges"], s["digest"]))
            continue
        e = exp[key]
        rec.check(f"{key} graphs / nodes / edges / digest == exploratory",
                  (s["graphs"], s["nodes"], s["edges"], s["digest"]) == (e["graphs"], e["nodes"], e["edges"], e["digest"]),
                  f"({s['graphs']}, {s['nodes']}, {s['edges']}, {s['digest']})")
    E_o, E_n = nbg_edge_set(original, actset), nbg_edge_set(new, actset)
    rec.check("Original edge set == independent recount from observed clean variants", E_o == trace_edge_set(observed))
    E_gen = trace_edge_set(generated)  # = generated-only (W1) edge set; W1 NBG itself is not built
    rec.check("W2 edge set == Original ∪ generated-trace edges (= Original ∪ W1)", E_n == E_o | E_gen,
              f"(original {len(E_o)}, generated {len(E_gen)}, model-only added {len(E_n - E_o)})")
    rec.check("no Original edge lost in W2", E_o <= E_n)
    write_json(prep / "original_nbg_summary.json", so)
    write_json(prep / "expanded_w2_nbg_summary.json",
               sn | {"model_only_edges_added": len(E_n - E_o), "original_edges_kept": len(E_o & E_n),
                     "edge_set_equals_original_union_generated": E_n == E_o | E_gen})

    alpha = ctx.params["alpha"]
    anom_ids = [str(c) for c in anomaly.loc[anomaly["label"].notna(), "case:concept:name"].unique()]
    clean_ids = list(clean_tr)
    print(f"  alpha={alpha} ADM: clean {len(clean_ids)} + anomalous {len(anom_ids)} cases (P_BEAR_RL.ADM)")
    _adm, pc = adm_equality(anomaly, {"clean": clean_ids, "anomalous": anom_ids}, actset, filtered_actset,
                            original, new, [alpha])
    pc["normal_original"] = pc["orig_locations"].apply(lambda x: len(x) == 0)
    pc["normal_w2"] = pc["new_locations"].apply(lambda x: len(x) == 0)
    gen_set = set(generated)
    pc["trace_in_generated"] = pc["case"].map(lambda c: clean_tr.get(str(c)) in gen_set if str(c) in clean_tr else None)
    pc.to_csv(d / "alpha0_adm_per_case_original_vs_w2.csv", index=False)
    cl, an = pc[pc.split == "clean"], pc[pc.split == "anomalous"]
    unc = cl[cl.trace_in_generated == False]  # noqa: E712
    clean_chk = {"alpha": alpha, "clean_cases": len(cl), "same_decision": int((cl.normal_original == cl.normal_w2).sum()),
                 "normal_original": int(cl.normal_original.sum()), "normal_w2": int(cl.normal_w2.sum()),
                 "clean_false_positive_w2": int((~cl.normal_w2).sum()),
                 "uncovered_clean_cases": len(unc), "uncovered_clean_normal_w2": int(unc.normal_w2.sum())}
    anom_chk = {"alpha": alpha, "anomalous_cases": len(an), "anomaly_miss_original": int(an.normal_original.sum()),
                "anomaly_miss_w2": int(an.normal_w2.sum()),
                "newly_accepted_anomalies": int((~an.normal_original & an.normal_w2).sum()),
                "original_normal_to_w2_anomaly": int((an.normal_original & ~an.normal_w2).sum()),
                "same_location": int(an.anomaly_locations_equal.sum()),
                "newly_accepted_case_ids": sorted(an.loc[~an.normal_original & an.normal_w2, "case"].astype(str))}
    write_json(prep / "alpha0_clean_adm_check.json", clean_chk)
    write_json(prep / "alpha0_anomaly_adm_check.json", anom_chk)
    if exp is None:  # dataset without expected values: check only structural invariants, record the rest
        rec.check(f"alpha0 clean: same decision Original vs W2 ({len(cl)}/{len(cl)})", clean_chk["same_decision"] == len(cl), clean_chk)
        rec.info("alpha0 uncovered clean cases / W2 normal", (len(unc), clean_chk["uncovered_clean_normal_w2"]))
        rec.info("alpha0 anomaly miss original / W2 / newly accepted",
                 (anom_chk["anomaly_miss_original"], anom_chk["anomaly_miss_w2"], anom_chk["newly_accepted_anomalies"]))
        return
    rec.check(f"alpha0 clean: same decision {len(cl)}/{len(cl)}, false positive {exp['clean_false_positive']}",
              clean_chk["same_decision"] == len(cl) and clean_chk["clean_false_positive_w2"] == exp["clean_false_positive"],
              clean_chk)
    rec.check(f"alpha0 uncovered clean {exp['uncovered_clean_cases']}: W2 normal {exp['uncovered_clean_normal']}",
              (len(unc), clean_chk["uncovered_clean_normal_w2"]) == (exp["uncovered_clean_cases"], exp["uncovered_clean_normal"]))
    rec.check(f"alpha0 anomaly: newly accepted {exp['newly_accepted_anomalies']}, total miss {exp['anomaly_miss']}",
              (anom_chk["newly_accepted_anomalies"], anom_chk["anomaly_miss_w2"]) == (exp["newly_accepted_anomalies"], exp["anomaly_miss"]),
              {k: anom_chk[k] for k in ["anomaly_miss_original", "anomaly_miss_w2", "newly_accepted_anomalies"]})


def w2_smoke_checks(ctx, rec, d, results, n):
    """W2 Method C smoke checks (num_epi = SMOKE_NUM_EPI; not official performance values)."""
    import pandas as pd

    mp, seq_a, seq_c = results["mp_methodC"], results["seq_methodA"], results["seq_methodC"]
    df = pd.read_csv(d / f"smoke_mp_epi{n}_methodC.csv", dtype=str, keep_default_na=False)
    types = df.groupby("case:concept:name")["type"].first().value_counts().to_dict()
    rec.check("Method C (W2) smoke: alpha == official alpha (0)", mp.get("alpha") == ctx.params["alpha"], mp.get("alpha"))
    rec.check("Method C (W2) smoke: Q-table updated (size > 0)", (mp.get("q_table_size") or 0) > 0, mp.get("q_table_size"))
    rec.check("Method C (W2) smoke: repaired log has val and test cases", types.get("val", 0) > 0 and types.get("test", 0) > 0, types)
    rec.check("Method A (Original NBG) sequential smoke ran (nbg_source discover_NBG)",
              seq_a.get("nbg_source") != "override" and (seq_a.get("q_table_size") or 0) > 0, seq_a.get("nbg_source"))
    keep = ["nbg_source", "alpha", "num_epi", "q_table_size", "n_val_cases", "n_test_cases", "val_p_anomaly",
            "test_p_anomaly", "wall_sec"]
    write_json(ctx.out / "preparation" / "smoke_method_c_summary.json", {
        "note": f"smoke only (num_epi={n}); NOT official performance", "new_nbg_input": ctx.cfg["new_nbg_input"],
        "mp_methodC": {k: mp.get(k) for k in keep}, "seq_methodA": {k: seq_a.get(k) for k in keep},
        "seq_methodC": {k: seq_c.get(k) for k in keep}, "repaired_log_case_types_mp_C": types,
        "sequential_A_equals_C": results["sequential_A_equals_C"],
        "checks": [c for c in rec.checks if c["check"].startswith(("Method C (W2)", "Method A ("))]})


def stage_nbg(ctx):
    d = open_stage(ctx, "nbg")
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print('\n[nbg] Build the New NBGs and compare structure / edges / likelihood / ADM with the Original NBGs.')
        if not record_inputs(ctx, rec):
            return finish(d, "nbg", rec)
        import numpy as np

        from rpbear.func_rpbear_final import discover_NBG
        from sim import _load_and_prepare_data
        from utils.data_utils import add_start_end_like_sim, read_log_like_sim
        from utils.nbg_utils import NBGValidationError, adm_equality, build_method_c_nbgs, compare_nbgs, order_mismatches
        from utils.run_utils import expected_split

        clean, anomaly, _gt, actset, filtered_actset, _af = _load_and_prepare_data(ctx.data_name(), ctx.input_paths)
        actset = np.asarray(actset)
        again = add_start_end_like_sim(read_log_like_sim(ctx.input_paths["anomaly_log"], has_label=True))
        cols = ["case:concept:name", "concept:name", "order"]
        rec.check("generated-log converter reproduces sim clean (case, activity, order)",
                  again[cols].astype(str).equals(clean[cols].astype(str)), f"({len(clean)} rows)")

        import time as _t
        _t0 = _t.perf_counter()
        original = discover_NBG(clean, vis=False)
        nbg_timing = {"original_nbg_build_ms": round((_t.perf_counter() - _t0) * 1000, 2)}
        rec.check("Original NBG ordering: actset[i] == anchor(NBG[i])", not order_mismatches(original, actset),
                  f"({len(actset)} activities)")
        w2 = ctx.cfg["new_nbg_input"] == "original_plus_model_only"
        if w2:
            rec.check('generate stage has no generated-only log',
                      not (ctx.stage_dir("generate") / "generated_normal_log.csv").exists())
        try:
            _t0 = _t.perf_counter()
            new, info = build_method_c_nbgs(ctx.cfg["new_nbg_input"], ctx.stage_dir("generate"), clean, actset)
            nbg_timing["new_nbg_build_ms"] = round((_t.perf_counter() - _t0) * 1000, 2)
        except NBGValidationError as e:
            rec.check(f"New NBG built ({ctx.cfg['new_nbg_input']}, activity set, actset ordering)", False, e)
            return finish(d, "nbg", rec)
        rec.check("New NBG ordering: actset[i] == anchor(NBG[i]) for all i", True, info["anchor_match"])
        for k, v in info.items():
            print(f"  {k}: {v}")

        summary, graphs, edges = compare_nbgs(original, new, actset)
        graphs.to_csv(d / "nbg_per_graph_comparison.csv", index=False)
        edges.to_csv(d / "nbg_edge_likelihood_comparison.csv", index=False)
        for k, v in summary.items():
            rec.info(k, v)

        val_ids, test_ids = expected_split(ctx.data_name(), ctx.input_paths, ctx.params["val_ratio"],
                                           ctx.params["seed"])
        (d / "split_val_case_ids.txt").write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")
        (d / "split_test_case_ids.txt").write_text("\n".join(sorted(test_ids)) + "\n", encoding="utf-8")
        # generated_only (Small-Wide): diagnostics at alpha 0 / 0.01 / 0.05; W2: official alpha 0 only
        alphas = [ctx.params["alpha"]] if w2 else [ctx.params["alpha"], 0.01, 0.05]
        adm, per_case = adm_equality(anomaly, {"test": test_ids, "val": val_ids}, actset, filtered_actset,
                                     original, new, alphas)
        per_case.to_csv(d / "adm_equality_per_case_alpha0.csv", index=False)
        for a, st in adm.items():
            rec.info(f"ADM equality {a}", st)
        write_json(d / "nbg_comparison_summary.json", {"new_nbg_info": info, "comparison": summary, "timing_ms": nbg_timing,
                                                       "adm_functional_equality": adm,
                                                       "split_sizes": {"val": len(val_ids), "test": len(test_ids)}})
        if w2:
            w2_nbg_preparation(ctx, rec, d, clean, anomaly, actset, filtered_actset, original, new)
    return finish(d, "nbg", rec)


def stage_smoke(ctx):
    d = open_stage(ctx, "smoke")
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print('\n[smoke] Before the long official run: check that New NBG -> RP-BEAR runs end to end, that the split is identical,')
        print("        and that under deterministic (sequential) execution Original NBG (A') and New NBG (C') give the expected results.")
        if not record_inputs(ctx, rec):
            return finish(d, "smoke", rec)
        import numpy as np

        from sim import _load_and_prepare_data
        from utils.nbg_utils import build_method_c_nbgs, edge_set_digest
        from utils.run_utils import quick_run, split_ids_from_repaired

        _c, _a, _g, actset, _f, _af = _load_and_prepare_data(ctx.data_name(), ctx.input_paths)
        actset = np.asarray(actset)
        new, _info = build_method_c_nbgs(ctx.cfg["new_nbg_input"], ctx.stage_dir("generate"), _c, actset)
        rec.info("Method C New NBG input", ctx.cfg["new_nbg_input"])
        prev = read_json(ctx.stage_dir("nbg") / "nbg_comparison_summary.json")
        rec.check("New NBG digest == nbg stage", edge_set_digest(new, actset) == prev["comparison"]["edge_set_digest"]["new"])
        exp_val = set(read_ids(ctx.stage_dir("nbg") / "split_val_case_ids.txt"))
        exp_test = set(read_ids(ctx.stage_dir("nbg") / "split_test_case_ids.txt"))
        n = dc.SMOKE_NUM_EPI

        print(f"\n  [1] multiprocessing, num_epi={n}, Method C end-to-end")
        df, st, timing = quick_run(ctx.data_name(), ctx.input_paths, ctx.params, n, True, new, return_case_timing=True)
        df.to_csv(d / f"smoke_mp_epi{n}_methodC.csv", index=False)
        timing.to_csv(d / f"smoke_mp_epi{n}_methodC_case_timing.csv", index=False)
        val, test = split_ids_from_repaired(df)
        rec.check("run finished with nbg_source=override", st.get("nbg_source") == "override")
        rec.check("val/test split == nbg stage split", set(val) == exp_val and set(test) == exp_test,
                  f"(val {len(val)}, test {len(test)})")
        print(f"  val {st['val_p_anomaly']:.4f} | test {st['test_p_anomaly']:.4f} | Q-table {st['q_table_size']} | {st['wall_sec']} s")

        print(f"\n  [2] sequential (deterministic) run, num_epi={n}: A' (Original NBG) vs C' (New NBG)")
        results = {"mp_methodC": st}
        for label, nbgs in (("A", None), ("C", new)):
            df_s, st_s, timing_s = quick_run(ctx.data_name(), ctx.input_paths, ctx.params, n, False, nbgs,
                                             return_case_timing=True)
            df_s.to_csv(d / f"smoke_seq_epi{n}_method{label}.csv", index=False)
            timing_s.to_csv(d / f"smoke_seq_epi{n}_method{label}_case_timing.csv", index=False)
            results[f"seq_method{label}"] = st_s
            print(f"  {label}': val {st_s['val_p_anomaly']:.4f} | test {st_s['test_p_anomaly']:.4f} | "
                  f"Q-table {st_s['q_table_size']} | {st_s['wall_sec']} s")
        identical = (d / f"smoke_seq_epi{n}_methodA.csv").read_bytes() == (d / f"smoke_seq_epi{n}_methodC.csv").read_bytes()
        rec.info("sequential A' == C' (byte-identical)", identical)
        results["sequential_A_equals_C"] = identical
        write_json(d / "smoke_summary.json", results)
        if ctx.cfg["new_nbg_input"] == "original_plus_model_only":
            w2_smoke_checks(ctx, rec, d, results, n)
    return finish(d, "smoke", rec)


def stage_run(ctx):
    from utils.nbg_utils import edge_set_digest  # noqa: F401  (used for Method C)

    method, run_id = ctx.args.method, ctx.args.run_id
    if method is None or run_id is None:
        print('[ERROR] the run stage needs --method A|C and --run-id N.')
        return 1
    for req in STAGE_REQUIRES["run"]:
        st = read_stage_status(ctx.stage_dir(req))
        if not st or st["status"] != "PASSED":
            print(f"[ABORT] previous stage '{req}' is not PASSED")
            return 1
    rec = CheckRecorder()
    if not record_inputs(ctx, rec):
        return 1
    import numpy as np

    from sim import _load_and_prepare_data
    from utils.nbg_utils import build_method_c_nbgs
    from utils.run_utils import execute_official_run

    nbgs, extra = None, {"inputs": read_json(ctx.out / "inputs.json")["files"]}
    if method == "C":
        _c, _a, _g, actset, _f, _af = _load_and_prepare_data(ctx.data_name(), ctx.input_paths)
        actset = np.asarray(actset)
        nbgs, info = build_method_c_nbgs(ctx.cfg["new_nbg_input"], ctx.stage_dir("generate"), _c, actset)
        digest = edge_set_digest(nbgs, actset)
        nbg_digest = read_json(ctx.stage_dir("nbg") / "nbg_comparison_summary.json")["comparison"]["edge_set_digest"]["new"]
        if digest != nbg_digest:
            print(f"[ABORT] Method C New NBG digest {digest} != nbg stage {nbg_digest}")
            return 1
        extra.update({"nbg_edge_set_digest": digest, "new_nbg_info": info, "new_nbg_input": ctx.cfg["new_nbg_input"],
                      "actset_act_freq_source": "observed clean log (same as Method A)"})
    p = ctx.params
    prefix = (f"{ctx.cfg['file_stem']}_{p['num_epi']}_ppm{p['alpha']}_valratio{p['val_ratio']}_seed{p['seed']}"
              f"_method{method}_run{run_id}")
    code = execute_official_run(ctx.data_name(), ctx.input_paths, p, method,
                                ctx.out / "runs" / f"method{method}_run{run_id}", prefix, nbgs, extra)
    if code == 0:
        write_standard_timing(ctx, ctx.out / "runs" / f"method{method}_run{run_id}", method)
    return code


def collect_runs(ctx, with_reference):
    """Repaired logs to evaluate {label: path}."""
    runs = {}
    runs_dir = ctx.out / "runs"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            files = list(run_dir.glob("*_repaired_log_*.csv"))
            if files:
                runs[run_dir.name] = files[0]
    if with_reference:
        for label, rel in ctx.cfg["reference_runs"].items():
            runs[f"ref_{label}"] = PROJECT_ROOT / rel
    return runs


COST_FIELDS = ["num_epi", "alpha", "use_multiprocessing", "num_cores", "n_val_cases", "n_test_cases",
               "training_time_sec", "total_repair_time_sec", "total_run_time_sec", "test_per_case_ms",
               "val_total_ms", "test_total_ms", "total_sec"]


def rpbear_cost_summary(runs):
    """Collect cost / training-time fields from *_stats.json of each run folder (runs without stats.json are skipped).

    Runs made before per-case timing was added have no per-case fields; those stay empty.
    """
    import pandas as pd

    rows = []
    for label, path in runs.items():
        stats_files = list(Path(path).parent.glob("*_stats.json"))
        if not stats_files:
            continue
        s = read_json(stats_files[0])
        row = {"run": label}
        row.update({k: s.get(k) for k in COST_FIELDS})
        row.update({k: v for k, v in s.items() if k.startswith(("test_case_ms_", "val_case_ms_"))})
        timing = list(Path(path).parent.glob("*_case_timing.csv"))
        row["case_timing_csv"] = str(timing[0]) if timing else None
        rows.append(row)
    return pd.DataFrame(rows)


def stage_evaluate(ctx):
    d = open_stage(ctx, "evaluate")
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print('\n[evaluate] Score every run with the same evaluation functions (ground truth is used only here).')
        if not record_inputs(ctx, rec):
            return finish(d, "evaluate", rec)
        from utils import evaluation_utils as ev

        runs = collect_runs(ctx, ctx.args.with_reference)
        if not rec.check("at least one run to evaluate", bool(runs), list(runs)):
            return finish(d, "evaluate", rec)
        split_file = ctx.stage_dir("nbg") / "split_test_case_ids.txt"
        if split_file.exists():
            test_ids = read_ids(split_file)
        elif ctx.cfg.get("regression"):
            test_ids = read_ids(PROJECT_ROOT / ctx.cfg["regression"]["split_test_ids"])
        else:
            rec.check("test split available (run nbg stage first)", False)
            return finish(d, "evaluate", rec)

        gt, anomalous, labels, normal_variants = ev.load_evaluation_basis(ctx.input_paths["anomaly_log"],
                                                                          ctx.input_paths["ground_truth_log"])
        traces = {}
        for label, path in runs.items():
            traces[label] = ev.canonical_traces(path, "case:concept:name", "concept:name", "test")
            rec.check(f"{label}: test case set == split ({len(test_ids)})",
                      set(traces[label].index) == set(test_ids) and len(traces[label]) == len(test_ids))
        if not rec.ok:
            return finish(d, "evaluate", rec)

        cl = ev.case_level_table(test_ids, gt, anomalous, labels, normal_variants, traces)
        official = ev.official_metrics(cl, list(traces))
        pattern = ev.pattern_table(cl, list(traces))
        cl.to_csv(d / "case_level.csv", index=False)
        official.to_csv(d / "official_metrics.csv")
        pattern.to_csv(d / "exact_match_by_pattern.csv", index=False)
        for label, expected in ctx.cfg["reference_expected_exact"].items():
            key = f"ref_{label}"
            if key in official.index:
                got = int(official.loc[key, "n_exact_match"])
                rec.check(f"{key} Exact Match == {expected}", got == expected, f"(got {got})")
        pairs = {}
        for a in [k for k in traces if "A_run1" in k or k.startswith("methodA")]:
            for c in [k for k in traces if "C_run1" in k or k.startswith("methodC")]:
                if a.startswith("ref_") == c.startswith("ref_"):
                    pairs[f"{a} vs {c}"] = ev.contingency(cl, a, c)
        cost = rpbear_cost_summary(runs)
        if len(cost):
            cost.to_csv(d / "rpbear_cost_summary.csv", index=False)
        write_json(d / "evaluation_summary.json", {"runs": {k: str(v) for k, v in runs.items()},
                                                   "official": official.reset_index().to_dict(orient="records"),
                                                   "contingency": pairs})
        print(official[["n_exact_match", "exact_match_rate", "recon_err_after", "normal_variant_rate"]]
              .to_string(float_format=lambda x: f"{x:.4f}"))
        for k, v in pairs.items():
            print(f"  contingency {k}: {v}")
    return finish(d, "evaluate", rec)


def stage_regression(ctx):
    """Small only: check that the results of the common pipeline equal the existing Small results."""
    reg = ctx.cfg.get("regression")
    if not reg:
        print(f"[ERROR] {ctx.name} has no regression reference (Small only).")
        return 1
    d = open_stage(ctx, "regression")
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print('\n[regression] Check that the Small results are unchanged with respect to the reference results.')
        import pandas as pd

        P = lambda rel: PROJECT_ROOT / rel  # noqa: E731
        for stage in ["discover", "generate", "nbg", "smoke", "evaluate"]:
            st = read_stage_status(ctx.stage_dir(stage))
            rec.check(f"stage {stage} PASSED", bool(st) and st["status"] == "PASSED", st["status"] if st else "missing")

        new_mc = read_json(ctx.stage_dir("discover") / "model_check.json")
        old_mc = read_json(P(reg["exp063_model_check"]))
        keys = [k for k in old_mc if k != "discovery_sec"]
        diff = [k for k in keys if new_mc.get(k) != old_mc[k]]
        rec.check('model_check == reference (all fields except discovery_sec)', not diff, diff or f"({len(keys)} fields)")
        s = new_mc["structure"]
        rec.check("structure: no LOOP / PARALLEL / cycle (enum count)",
                  s["n_loop_operators"] == 0 and s["n_parallel_operators"] == 0 and not s["petri_net_has_cycle"], s)

        gs = read_json(ctx.stage_dir("generate") / "generate_summary.json")
        rec.check("generated unique traces == 6", gs["generated_unique_trace_count"] == 6)
        rec.check("observed clean variants == 6", gs["observed_clean_variant_count"] == 6)
        rec.check("generated == observed == Process Tree language",
                  gs["generated_equals_observed"] and gs["generated_equals_process_tree_language"])
        rec.check('generated_normal_log.csv byte-identical to the reference',
                  (ctx.stage_dir("generate") / "generated_normal_log.csv").read_bytes()
                  == P(reg["exp063_generated_log"]).read_bytes())

        nb = read_json(ctx.stage_dir("nbg") / "nbg_comparison_summary.json")
        cmp_ = nb["comparison"]
        rec.check(f"edge digest (original == new == {reg['edge_digest']})",
                  cmp_["edge_set_digest"]["original"] == cmp_["edge_set_digest"]["new"] == reg["edge_digest"])
        rec.check("structure equal / edge set equal",
                  cmp_["structure_equal (all node sets incl. activity/level)"] and cmp_["edge_set_equal (all graphs)"])
        rec.check(f"likelihood differs: {reg['likelihood_diff_edges']} edges / {reg['likelihood_diff_graphs']} graphs",
                  cmp_["edges_with_likelihood_difference"] == reg["likelihood_diff_edges"]
                  and cmp_["graphs_with_likelihood_difference"] == reg["likelihood_diff_graphs"],
                  f"({cmp_['edges_with_likelihood_difference']} / {cmp_['graphs_with_likelihood_difference']})")
        rec.check(f"NBG anchor {reg['anchor_match']}/{reg['anchor_match']}",
                  nb["new_nbg_info"]["anchor_match"] == f"{reg['anchor_match']}/{reg['anchor_match']}")
        a0 = nb["adm_functional_equality"][f"alpha={ctx.params['alpha']}"]
        total_m = a0["test"]["adm_matrix_equal"] + a0["val"]["adm_matrix_equal"]
        total_l = a0["test"]["anomaly_locations_equal"] + a0["val"]["anomaly_locations_equal"]
        rec.check(f"ADM equality at alpha=0: {reg['adm_equal_alpha0']}/{reg['adm_equal_alpha0']}",
                  total_m == total_l == reg["adm_equal_alpha0"], f"(matrix {total_m}, locations {total_l})")
        rec.check('split == reference run 1 (val / test ids)',
                  read_ids(ctx.stage_dir("nbg") / "split_val_case_ids.txt") == sorted(read_ids(P(reg["split_val_ids"])))
                  and read_ids(ctx.stage_dir("nbg") / "split_test_case_ids.txt") == sorted(read_ids(P(reg["split_test_ids"]))))

        n = dc.SMOKE_NUM_EPI
        ref = P(reg["v2_seq_reference"]).read_bytes()
        a_seq = (ctx.stage_dir("smoke") / f"smoke_seq_epi{n}_methodA.csv").read_bytes()
        c_seq = (ctx.stage_dir("smoke") / f"smoke_seq_epi{n}_methodC.csv").read_bytes()
        rec.check("sequential A' byte-identical to the reference sequential run", a_seq == ref)
        rec.check("sequential C' byte-identical to A'", c_seq == a_seq)

        off = pd.read_csv(ctx.stage_dir("evaluate") / "official_metrics.csv").set_index("run")
        for label, expected in ctx.cfg["reference_expected_exact"].items():
            got = int(off.loc[f"ref_{label}", "n_exact_match"]) if f"ref_{label}" in off.index else None
            rec.check(f"evaluation reproduces {label} == {expected}/524", got == expected, f"(got {got})")
        new_cl = pd.read_csv(ctx.stage_dir("evaluate") / "case_level.csv").set_index("case:concept:name")
        old_cl = pd.read_csv(P(reg["exp063_case_level"])).set_index("case:concept:name")
        mismatch = []
        base = ["anomaly_label", "pattern_group", "anomalous_trace", "ground_truth", "changed_by_anomaly", "lev_before"]
        for col in base:
            if not new_cl.loc[old_cl.index, col].equals(old_cl[col]):
                mismatch.append(col)
        for old_m, new_m in (("A", "ref_A_run1"), ("C", "ref_C_run1"), ("B", "ref_B_run1")):
            for suf in ["repaired", "correct", "lev_after", "normal_variant", "error_type"]:
                if not new_cl.loc[old_cl.index, f"{new_m}_{suf}"].equals(old_cl[f"{old_m}_{suf}"]):
                    mismatch.append(f"{old_m}_{suf}")
        for suf in ["repaired", "correct"]:
            for r in (2, 3):
                if not new_cl.loc[old_cl.index, f"ref_A_run{r}_{suf}"].equals(old_cl[f"A_run{r}_{suf}"]):
                    mismatch.append(f"A_run{r}_{suf}")
        rec.check('case-level table == reference case-level table (A, C, B, A run2/3)', not mismatch,
                  mismatch or f"({len(old_cl)} cases)")

        # The historical Small scripts live in src/experiments/p6_small_repaired_log_comparison/.
        # They are compared with HEAD; only the allowed path lines may differ.
        from utils.validation_utils import historical_script_drift

        p6 = "src/experiments/p6_small_repaired_log_comparison"
        for name in ["60_generate_repaired_log_a.py", "61_method_b_alignment_repair.py",
                     "62_compare_method_a_b.py", "63_generated_nbg_rpbear.py"]:
            drift = historical_script_drift(f"{p6}/{name}", f"src/{name}")
            rec.check(f"{name} preserved (only path lines differ)", not drift, drift or "")
        st = subprocess.run(["git", "status", "--porcelain", "--", "src/rpbear"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True).stdout.strip()
        rec.check("rpbear core unchanged (git status clean)", st == "", st or "")
    return finish(d, "regression", rec)


def _open_dir(ctx, d):
    """Prepare a per-label folder (same overwrite rule as open_stage)."""
    if d.exists():
        if not ctx.args.overwrite:
            print(f"[ABORT] {d} already exists. Use --overwrite to run again")
            return None
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def stage_test_timing_validate(ctx):
    """Check that a sequential test-only replay is byte-identical to the original test phase (smoke settings, A and C)."""
    d = _open_dir(ctx, ctx.stage_dir("test_timing") / "validation")
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print('\n[test-timing-validate] Starting from the same state (Q-table, exploration rate, repair memory, RNG state),')
        print('        a test-only replay must be byte-identical to the original test phase (logic-invariance check).')
        if not record_inputs(ctx, rec):
            return finish(d, "test-timing-validate", rec)
        import numpy as np
        import pandas as pd

        from sim import _load_and_prepare_data
        from utils.nbg_utils import build_method_c_nbgs
        from utils.timing_utils import summarize, validate_test_only_equivalence

        _c, _a, _g, actset, _f, _af = _load_and_prepare_data(ctx.data_name(), ctx.input_paths)
        new, _info = build_method_c_nbgs(ctx.cfg["new_nbg_input"], ctx.stage_dir("generate"), _c, np.asarray(actset))
        p, n = ctx.params, dc.SMOKE_NUM_EPI
        reg = ctx.cfg.get("regression") or {}
        out = {}
        for label, nbgs in (("A", None), ("C", new)):
            r = validate_test_only_equivalence(ctx.data_name(), ctx.input_paths, n, p["alpha"], p["val_ratio"],
                                               p["seed"], p["max_step"], nbgs)
            ref_csv = r["test_reference"].to_csv(index=False)
            rep_csv = r["test_replay"].to_csv(index=False)
            rec.check(f"{label}: Test-only replay byte-identical to original sequential Test phase", ref_csv == rep_csv,
                      f"({len(r['test_reference'])} rows)")
            tmp = d / f"_full_sequential_{label}.csv"  # written like the smoke stage (df.to_csv(path)) for a byte comparison
            pd.concat([r["val"], r["test_reference"]], ignore_index=True).to_csv(tmp, index=False)
            full = tmp.read_bytes()
            tmp.unlink()
            smoke = (ctx.stage_dir("smoke") / f"smoke_seq_epi{n}_method{label}.csv").read_bytes()
            rec.check(f"{label}: setup reproduces run_rpbear sequential output (smoke_seq_epi{n}_method{label}.csv)",
                      full == smoke)
            if label == "A" and reg.get("v2_seq_reference"):
                rec.check('A: setup reproduces the reference sequential run',
                          full == (PROJECT_ROOT / reg["v2_seq_reference"]).read_bytes())
            pd.DataFrame(r["timing"]).to_csv(d / f"replay_case_timing_{label}.csv", index=False)
            out[label] = summarize(r["timing"], r["wall_ms"])
        write_json(d / "validation_summary.json", {"num_epi": n, "timing_replay_from_snapshot": out,
                                                  "checks": rec.checks})
    return finish(d, "test-timing-validate", rec)


LOOP_BOUND_SCREENING = {  # safety guards are not research parameters
    "loop_num_range": (1, 5),
    "loop_total_range": (1, 20),
    "coverage_threshold": 0.90,          # clean-case coverage >= 0.90
    "explosion_unique_traces": 100_000,  # safety: EXPLOSION if the final number of unique traces exceeds this
    "max_seconds_per_combo": 600,        # safety
    "max_rss_gb": 16,                    # safety
}


def stage_loop_bound_screening(ctx):
    """Official loop-bound selection for models with loops: build bounded trace sets from small
    (loop_num, loop_total) upwards and select the first bound with clean-case coverage >= 0.90.
    NBG / RP-BEAR are not run.
    """
    if not need_pm4py():
        return 1
    STAGE_REQUIRES.setdefault("loop-bound-screening", ["discover"])
    out = PROJECT_ROOT / "outputs" / f"{ctx.name.lower()}_loop_bound_screening"
    if out.exists() and not ctx.args.overwrite:
        print(f"[ABORT] {out} already exists. Use --overwrite to run again")
        return 1
    for req in STAGE_REQUIRES["loop-bound-screening"]:
        st = read_stage_status(ctx.stage_dir(req))
        if not st or st["status"] != "PASSED":
            print(f"[ABORT] previous stage '{req}' is not PASSED")
            return 1
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rec = CheckRecorder()
    with Tee(out / "log.txt"):
        print('\n[loop-bound-screening] The model language is infinite because of loops; select the official loop bound by clean-case coverage.')
        if not record_inputs(ctx, rec):
            return finish(out, "loop-bound-screening", rec)
        from collections import Counter

        import pandas as pd
        import pm4py

        from utils.data_utils import load_case_traces
        from utils.discovery_utils import discover_model
        from utils.loop_bound_utils import ExplosionAborted, generate_bounded, make_petri_acceptor

        cfg = LOOP_BOUND_SCREENING
        tg = ctx.cfg["trace_generation"]
        max_len, max_inter = tg["safety_max_trace_length"], tg["safety_max_traces"]
        traces, _l, clean_ids, _a = load_case_traces(ctx.input_paths["anomaly_log"])
        tree, _n, _i, _f, _s = discover_model(traces, clean_ids, ctx.noise)
        prev = read_json(ctx.stage_dir("discover") / "model_check.json")
        rec.check("re-discovered Process Tree == discover stage", str(tree) == prev["process_tree"], str(tree))
        net, im, fm = pm4py.read_pnml(str(ctx.stage_dir("discover") / "petri_net.pnml"))
        accepts = make_petri_acceptor(net, im, fm)

        observed = Counter(tuple(traces[cid]) for cid in clean_ids)
        n_cases, n_var = sum(observed.values()), len(observed)
        rec.check("clean cases / variants", True, f"{n_cases} cases, {n_var} variants")
        rec.check("Petri Net acceptor accepts all observed clean variants (acceptor sanity)",
                  all(accepts(v) for v in observed), "")
        n_anom_acc = sum(1 for cid in _a if accepts(tuple(traces[cid])))
        rec.check("Petri Net acceptor rejects some anomalous traces (acceptor negative sanity)",
                  n_anom_acc < len(_a), f"(accepted anomalous {n_anom_acc}/{len(_a)})")

        rows, sets, first_cov = [], {}, {}
        exploded = []  # (n, t): supersets of this bound are not run

        def run_combo(n, t, phase):
            if any(n >= en and t >= et for en, et in exploded):
                rows.append({"loop_num": n, "loop_total": t, "status": "SKIPPED_SUPERSET_OF_EXPLOSION",
                             "phase": phase, "explosion_flag": True})
                return None
            try:
                lang, st = generate_bounded(tree, n, t, max_len, max_inter, cfg["max_seconds_per_combo"],
                                            cfg["max_rss_gb"])
            except ExplosionAborted as e:
                exploded.append((n, t))
                rows.append({"loop_num": n, "loop_total": t, "status": "ABORTED", "phase": phase,
                             "explosion_flag": True, "notes": str(e)})
                return None
            gen = set(lang)
            cov_var = [v for v in observed if v in gen]
            cov_cases = sum(observed[v] for v in cov_var)
            for v in cov_var:
                first_cov.setdefault(v, (n, t))
            explosion = len(gen) > cfg["explosion_unique_traces"]
            if explosion:
                exploded.append((n, t))
            prev_same_n = [r for r in rows if r["loop_num"] == n and r.get("generated_unique") is not None
                           and r["loop_total"] == t - 1]
            row = {"loop_num": n, "loop_total": t,
                   "status": ("EXPLOSION" if explosion else ("EMPTY" if not gen else "COMPLETED")),
                   "phase": phase, "generated_unique": len(gen),
                   "model_only": len(gen) - len(cov_var),
                   "observed_variant_covered": len(cov_var), "observed_variant_total": n_var,
                   "variant_coverage": round(len(cov_var) / n_var, 6),
                   "clean_case_covered": cov_cases, "clean_case_total": n_cases,
                   "case_coverage": round(cov_cases / n_cases, 6),
                   "max_trace_length": max((len(x) for x in gen), default=0),
                   "generation_time_sec": round(st["generation_time_sec"], 4),
                   "peak_intermediate": st["peak_intermediate"],
                   "growth_vs_prev_total": (round(len(gen) / prev_same_n[0]["generated_unique"], 3)
                                            if prev_same_n and prev_same_n[0]["generated_unique"] else None),
                   "explosion_flag": explosion, "notes": ""}
            rows.append(row)
            sets[(n, t)] = gen
            print(f"  ({n},{t:>2}) {row['status']:<9} unique={len(gen):>9,} case_cov={row['case_coverage']:.4f} "
                  f"var_cov={len(cov_var)}/{n_var} maxlen={row['max_trace_length']} t={row['generation_time_sec']}s")
            return row

        thr = cfg["coverage_threshold"]
        selected = None
        lo_n, hi_n = cfg["loop_num_range"]
        lo_t, hi_t = cfg["loop_total_range"]
        for n in range(lo_n, hi_n + 1):
            for t in range(lo_t, hi_t + 1):
                r = run_combo(n, t, "screening")
                if r and r["status"] == "COMPLETED" and r["case_coverage"] >= thr:
                    selected = r
                    break
            if selected:
                break

        checks_extra = {}
        if selected is None:
            print('  -> no bound with clean-case coverage >= 0.90 within the search range (BLOCKED)')
            pd.DataFrame(rows).to_csv(out / f"{ctx.name.lower()}_loop_bound_screening.csv", index=False)
            return finish(out, "loop-bound-screening", rec, blocked=True)

        n, t = selected["loop_num"], selected["loop_total"]
        print(f"\n  first combo with case coverage >= {thr}: ({n},{t}) -> neighbor checks")
        for nn, tt in [(n, t + 1), (n, t + 2), (n + 1, t)]:
            run_combo(nn, tt, "neighbor_check")
        df = pd.DataFrame(rows)

        # --- checks ---
        S = sets[(n, t)]
        rec.check("selected: all generated traces accepted by discovered Petri Net",
                  all(accepts(x) for x in S), f"({len(S)} traces)")
        rec.check("selected: generated traces unique (set size == list size)", len(S) == len(list(S)))
        recount = sum(1 for cid in clean_ids if tuple(traces[cid]) in S)
        rec.check("selected: case coverage recount by case_id (exact full sequence)",
                  recount == selected["clean_case_covered"], f"({recount})")
        rec.check("threshold uses >= 0.90", selected["case_coverage"] >= thr)
        earlier = df[(df["phase"] == "screening") & ((df["loop_num"] < n) | ((df["loop_num"] == n) & (df["loop_total"] < t)))]
        rec.check("minimality: every earlier combo in search order has case coverage < 0.90 or no result",
                  bool((earlier["case_coverage"].fillna(0) < thr).all()), f"({len(earlier)} earlier combos)")
        incl = []
        for a, b in [((n, t - 1), (n, t)), ((n, t), (n, t + 1)), ((n, t + 1), (n, t + 2)), ((n, t), (n + 1, t)),
                     ((n - 1, hi_t), (n, hi_t))]:
            if a in sets and b in sets:
                incl.append((a, b, sets[a] <= sets[b]))
        rec.check("monotonicity: L(smaller bound) ⊆ L(larger bound) for checked neighbors",
                  all(x[2] for x in incl), [f"{a}⊆{b}:{ok}" for a, b, ok in incl])
        mono = True
        for nn, g in df[df["generated_unique"].notna()].groupby("loop_num"):
            g = g.sort_values("loop_total")
            mono &= bool((g["case_coverage"].diff().dropna() >= 0).all())
        rec.check("case coverage non-decreasing in loop_total for each loop_num", mono)

        df.to_csv(out / f"{ctx.name.lower()}_loop_bound_screening.csv", index=False)
        write_json(out / f"{ctx.name.lower()}_loop_bound_selected.json", {
            "criterion": "minimum loop bound with clean case coverage >= 0.90",
            "search_order": "loop_num ascending, then loop_total ascending",
            "coverage_type": "clean_case_coverage", "threshold": thr,
            "selected_loop_num": n, "selected_loop_total": t,
            "case_coverage": selected["case_coverage"], "clean_case_covered": selected["clean_case_covered"],
            "clean_case_total": n_cases, "variant_coverage": selected["variant_coverage"],
            "observed_variant_covered": selected["observed_variant_covered"], "observed_variant_total": n_var,
            "generated_unique": selected["generated_unique"], "model_only": selected["model_only"],
            "max_trace_length": selected["max_trace_length"],
            "generation_time_sec": selected["generation_time_sec"],
            "loop_semantics": {
                "loop_num": "instance-local; max do-part executions per loop instance",
                "loop_total": "sum of do-part executions over all loop instances in one trace (not redo count)",
                "nested_loop": "instance-local; a re-entered inner loop is a new instance with its own counter",
                "trace_length": f"not a research parameter; max length {max_len} is a safety guard only",
                "reference": 'instance-local loop_num; loop_total = sum of do executions'},
            "safety_guards": {"max_trace_length": max_len, "max_intermediate_set": max_inter,
                              "explosion_unique_traces": cfg["explosion_unique_traces"],
                              "max_seconds_per_combo": cfg["max_seconds_per_combo"], "max_rss_gb": cfg["max_rss_gb"],
                              "note": "safety only; not research parameters"},
            "status": ('THRESHOLD-RULE MINIMUM. Not the official bound: the final Credit bound is chosen by the coverage-complexity elbow in stage credit-loop-bound-sensitivity'),
        })
        unc = [{"variant": ">>".join(v), "frequency": round(c / n_cases, 6), "case_count": c, "trace_length": len(v),
                "first_covered_in_screened": (f"({first_cov[v][0]},{first_cov[v][1]})" if v in first_cov else "not covered")}
               for v, c in observed.items() if v not in S]
        pd.DataFrame(sorted(unc, key=lambda r: -r["case_count"]),
                     columns=["variant", "frequency", "case_count", "trace_length", "first_covered_in_screened"]
                     ).to_csv(out / f"{ctx.name.lower()}_loop_bound_uncovered_variants.csv", index=False)
        pd.DataFrame([{"trace": ">>".join(x), "length": len(x), "observed_clean_cases": observed.get(x, 0)}
                      for x in sorted(S, key=lambda z: (len(z), z))]
                     ).to_csv(out / f"{ctx.name.lower()}_loop_bound_selected_traces.csv", index=False)
        checks_extra = {"selected": [n, t], "n_rows": len(df)}
        print(f"  selected ({n},{t}): unique={selected['generated_unique']:,} case_cov={selected['case_coverage']} "
              f"var_cov={selected['observed_variant_covered']}/{n_var}; uncovered variants={len(unc)} "
              f"({sum(r['case_count'] for r in unc)} cases)")
    return finish(out, "loop-bound-screening", rec, extra=checks_extra)


CREDIT_LOOP_BOUND_SENSITIVITY = {  # safety guards are not research parameters
    # bounds larger than (3,7), run in order; supersets of a bound that hit EXPLOSION are not run
    "bounds": [(2, 7), (3, 7), (3, 8), (3, 9), (3, 10), (4, 7), (4, 8), (4, 9), (4, 10), (5, 7), (5, 8)],
    "analytic_only_bounds": [(6, 7), (7, 7)],  # coverage from the loop counts of observed variants, without generating traces
    "final_bound": (5, 7),                    # final bound at the coverage-complexity elbow
    "explosion_unique_traces": 100_000,
    "max_seconds_per_combo": 600,
    "max_rss_gb": 16,
}


def stage_credit_loop_bound_sensitivity(ctx):
    """Credit loop-bound sensitivity: compare coverage vs number of generated traces per bound and store the
    trace set / uncovered variants of the final bound (5,7). NBG / RP-BEAR are not run.
    """
    if ctx.name != "Credit":
        print('[ERROR] this stage is Credit only (the loop counting is specific to the Credit tree).')
        return 1
    if not need_pm4py():
        return 1
    out = PROJECT_ROOT / "outputs" / "credit_loop_bound_sensitivity"
    if out.exists() and not ctx.args.overwrite:
        print(f"[ABORT] {out} already exists. Use --overwrite to run again")
        return 1
    st = read_stage_status(ctx.stage_dir("discover"))
    if not st or st["status"] != "PASSED":
        print("[ABORT] previous stage 'discover' is not PASSED")
        return 1
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rec = CheckRecorder()
    stage = "credit-loop-bound-sensitivity"
    with Tee(out / "log.txt"):
        print('\n[credit-loop-bound-sensitivity] Compare the coverage / trace-count trade-off per bound to use the bound before the generated trace count grows sharply (coverage-complexity elbow) as the official bound.')
        if not record_inputs(ctx, rec):
            return finish(out, stage, rec)
        from collections import Counter

        import pandas as pd
        import pm4py

        from utils.data_utils import load_case_traces
        from utils.discovery_utils import discover_model
        from utils.loop_bound_utils import (CREDIT_TREE, ExplosionAborted, credit_loop_requirements, generate_bounded,
                                            make_petri_acceptor)

        cfg = CREDIT_LOOP_BOUND_SENSITIVITY
        max_len = ctx.cfg["trace_generation"]["safety_max_trace_length"]
        max_inter = ctx.cfg["trace_generation"]["safety_max_traces"]
        traces, _l, clean_ids, _a = load_case_traces(ctx.input_paths["anomaly_log"])
        tree, _n, _i, _f, _s = discover_model(traces, clean_ids, ctx.noise)
        prev = read_json(ctx.stage_dir("discover") / "model_check.json")
        rec.check("re-discovered Process Tree == discover stage == Credit tree in loop_bound_utils",
                  str(tree) == prev["process_tree"] == CREDIT_TREE)
        net, im, fm = pm4py.read_pnml(str(ctx.stage_dir("discover") / "petri_net.pnml"))
        accepts = make_petri_acceptor(net, im, fm)
        observed = Counter(tuple(traces[cid]) for cid in clean_ids)
        n_cases, n_var = sum(observed.values()), len(observed)
        req = {v: credit_loop_requirements(v) for v in observed}

        def analytic_cov(n, t):
            vs = [v for v in observed if req[v]["need_loop_num"] <= n and req[v]["need_loop_total"] <= t]
            return vs, sum(observed[v] for v in vs)

        rows, sets, exploded = [], {}, []
        analytic_match = True
        for n, t in cfg["bounds"]:
            if any(n >= en and t >= et for en, et in exploded):
                rows.append({"loop_num": n, "loop_total": t, "status": "SKIPPED_SUPERSET_OF_EXPLOSION",
                             "explosion_flag": True})
                continue
            try:
                lang, gst = generate_bounded(tree, n, t, max_len, max_inter, cfg["max_seconds_per_combo"],
                                             cfg["max_rss_gb"])
            except ExplosionAborted as e:
                exploded.append((n, t))
                rows.append({"loop_num": n, "loop_total": t, "status": "ABORTED", "explosion_flag": True,
                             "notes": str(e)})
                continue
            gen = set(lang)
            cov = [v for v in observed if v in gen]
            cc = sum(observed[v] for v in cov)
            analytic_match &= set(analytic_cov(n, t)[0]) == set(cov)
            explosion = len(gen) > cfg["explosion_unique_traces"]
            if explosion:
                exploded.append((n, t))
            rows.append({"loop_num": n, "loop_total": t, "status": "EXPLOSION" if explosion else "COMPLETED",
                         "generated_unique": len(gen), "model_only": len(gen) - len(cov),
                         "clean_case_covered": cc, "case_coverage": round(cc / n_cases, 6),
                         "uncovered_clean_cases": n_cases - cc, "observed_variant_covered": len(cov),
                         "variant_coverage": round(len(cov) / n_var, 6), "uncovered_variants": n_var - len(cov),
                         "max_trace_length": max(map(len, gen)),
                         "generation_time_sec": round(gst["generation_time_sec"], 4),
                         "peak_intermediate": gst["peak_intermediate"], "explosion_flag": explosion, "notes": ""})
            sets[(n, t)] = gen
            print(f"  ({n},{t:>2}) {rows[-1]['status']:<9} unique={len(gen):>9,} case_cov={cc / n_cases:.4f} "
                  f"uncovered={n_cases - cc} var={len(cov)}/{n_var} maxlen={rows[-1]['max_trace_length']}")
        for n, t in cfg["analytic_only_bounds"]:
            vs, cc = analytic_cov(n, t)
            rows.append({"loop_num": n, "loop_total": t, "status": "ANALYTIC_COVERAGE_ONLY",
                         "clean_case_covered": cc, "case_coverage": round(cc / n_cases, 6),
                         "uncovered_clean_cases": n_cases - cc, "observed_variant_covered": len(vs),
                         "variant_coverage": round(len(vs) / n_var, 6), "uncovered_variants": n_var - len(vs),
                         "explosion_flag": False,
                         "notes": "traces not generated; coverage from observed loop counts (credit_loop_requirements)"})
            print(f"  ({n},{t:>2}) ANALYTIC  case_cov={cc / n_cases:.4f} (traces not generated)")

        df = pd.DataFrame(rows)
        gen_rows = {(r["loop_num"], r["loop_total"]): r for r in rows if r.get("generated_unique") is not None}

        def prev_of(n, t):  # previous bound: same loop_num with loop_total - 1; at loop_total 7, loop_num - 1
            return (n, t - 1) if t > 7 else (n - 1, 7)

        base = gen_rows.get((3, 7))
        for col in ["prev_bound", "coverage_gain_vs_prev_pp", "trace_ratio_vs_prev", "coverage_gain_vs_3_7_pp",
                    "trace_ratio_vs_3_7"]:
            df[col] = None
        for i, r in df.iterrows():
            k = (r["loop_num"], r["loop_total"])
            p = prev_of(*k)
            if p in gen_rows:
                df.at[i, "prev_bound"] = f"({p[0]},{p[1]})"
                df.at[i, "coverage_gain_vs_prev_pp"] = round((r["case_coverage"] - gen_rows[p]["case_coverage"]) * 100, 2)
                if k in gen_rows:
                    df.at[i, "trace_ratio_vs_prev"] = round(r["generated_unique"] / gen_rows[p]["generated_unique"], 3)
            if base and k in gen_rows:
                df.at[i, "coverage_gain_vs_3_7_pp"] = round((r["case_coverage"] - base["case_coverage"]) * 100, 2)
                df.at[i, "trace_ratio_vs_3_7"] = round(r["generated_unique"] / base["generated_unique"], 3)
        df.to_csv(out / "credit_loop_bound_sensitivity.csv", index=False)

        # --- checks ---
        rec.check("all generated traces accepted by discovered Petri Net (every generated bound)",
                  all(accepts(x) for g in sets.values() for x in g), f"({sum(map(len, sets.values())):,} traces)")
        rec.check("coverage from observed loop counts == coverage from generated sets (every generated bound)",
                  analytic_match)
        incl = [(a, b, sets[a] <= sets[b]) for a, b in [
            ((2, 7), (3, 7)), ((3, 7), (3, 8)), ((3, 8), (3, 9)), ((3, 9), (3, 10)), ((3, 7), (4, 7)), ((4, 7), (4, 8)),
            ((4, 8), (4, 9)), ((4, 9), (4, 10)), ((3, 8), (4, 8)), ((4, 7), (5, 7)), ((5, 7), (5, 8)), ((4, 8), (5, 8))]
            if a in sets and b in sets]
        rec.check("monotonicity: L(smaller) ⊆ L(larger) for nested bounds", all(x[2] for x in incl), f"({len(incl)} pairs)")
        scr = PROJECT_ROOT / "outputs" / "credit_loop_bound_screening" / "credit_loop_bound_screening.csv"
        if scr.exists():
            s = pd.read_csv(scr)
            same = all(int(s[(s.loop_num == n) & (s.loop_total == t)].iloc[0]["generated_unique"]) == gen_rows[(n, t)]["generated_unique"]
                       for n, t in [(2, 7), (3, 7)])
            rec.check("(2,7) and (3,7) == loop-bound-screening stage", same)

        fn, ft = cfg["final_bound"]
        if not rec.check(f"final bound ({fn},{ft}) was generated", (fn, ft) in sets):
            return finish(out, stage, rec)
        S, fr = sets[(fn, ft)], gen_rows[(fn, ft)]
        rec.check("final: case coverage recount by case_id (exact full sequence)",
                  sum(1 for cid in clean_ids if tuple(traces[cid]) in S) == fr["clean_case_covered"])
        write_json(out / "credit_loop_bound_final.json", {
            "dataset": "Credit (credit-card; identity not yet confirmed)",
            "final_loop_num": fn, "final_loop_total": ft,
            "criterion": ('coverage-complexity elbow: keep clean case coverage as high as possible, choose the bound before generated trace count starts to grow sharply'),
            "evidence": {f"({r['loop_num']},{r['loop_total']})": {"case_coverage": r["case_coverage"],
                                                                  "generated_unique": r.get("generated_unique")}
                         for r in df.where(df.notna(), None).to_dict("records")},
            "case_coverage": fr["case_coverage"], "clean_case_covered": fr["clean_case_covered"],
            "clean_case_total": n_cases, "uncovered_clean_cases": fr["uncovered_clean_cases"],
            "variant_coverage": fr["variant_coverage"], "observed_variant_covered": fr["observed_variant_covered"],
            "observed_variant_total": n_var, "generated_unique": fr["generated_unique"], "model_only": fr["model_only"],
            "max_trace_length": fr["max_trace_length"],
            "superseded": '(2,7) = minimum bound with clean case coverage >= 0.90 (threshold rule) is NOT the official bound',
            "loop_semantics": 'instance-local loop_num; loop_total = sum of do executions',
            "status": "OFFICIAL CREDIT LOOP BOUND (no NBG / RP-BEAR run yet)"})

        var_rows = []
        gen_order = [b for b in cfg["bounds"] if b in sets]
        for v, c in observed.most_common():
            r = req[v]
            drivers = [name for name, val in [("inner b-loop", r["inner_do_max"]), ("outer loop (l)", r["outer_do"]),
                                              ("e-loop", r["e_do"])] if val == r["need_loop_num"]]
            var_rows.append({"variant": ">>".join(v), "case_count": c, "frequency": round(c / n_cases, 6),
                             "trace_length": len(v), "e_do": r["e_do"], "outer_do": r["outer_do"],
                             "inner_do_per_instance": "/".join(map(str, r["inner_do_list"])),
                             "need_loop_num": r["need_loop_num"], "need_loop_total": r["need_loop_total"],
                             "loop_num_driven_by": "+".join(drivers),
                             "first_covered_in_generated_bounds": next((f"({n},{t})" for n, t in gen_order if v in sets[(n, t)]),
                                                                       "not in generated bounds"),
                             "covered_by_final": v in S})
        vdf = pd.DataFrame(var_rows)
        vdf.to_csv(out / "credit_clean_variant_loop_requirements.csv", index=False)
        unc = vdf[~vdf["covered_by_final"]].copy()
        unc["excluded_by"] = [("loop_num" if a > fn else "") + ("+" if a > fn and b > ft else "") + ("loop_total" if b > ft else "")
                              for a, b in zip(unc["need_loop_num"], unc["need_loop_total"])]
        unc.drop(columns=["covered_by_final"]).to_csv(out / "credit_loop_bound_final_uncovered_variants.csv", index=False)
        pd.DataFrame([{"trace": ">>".join(x), "length": len(x), "observed_clean_cases": observed.get(x, 0)}
                      for x in sorted(S, key=lambda z: (len(z), z))]).to_csv(out / "credit_loop_bound_final_traces.csv", index=False)
        rec.info("final bound", f"({fn},{ft}) unique={fr['generated_unique']:,} case_cov={fr['case_coverage']} "
                                f"uncovered={fr['uncovered_clean_cases']} cases / {len(unc)} variants; "
                                f"100% coverage needs ({vdf.need_loop_num.max()},{vdf.need_loop_total.max()})")
    return finish(out, stage, rec, extra={"final_bound": [fn, ft]})


def stage_run_b(ctx):
    """Method B (alignment) official run: clean cases -> IM -> Petri net -> state-equation A* alignment -> repaired log.

    Same alignment / repair code as the Small Method B script (utils.alignment_utils). The split is the nbg-stage split
    (same as A / C). With --limit N: smoke run on N test cases, stored in outputs/simulation/<dataset>/smoke_b/.
    """
    if not need_pm4py():
        return 1
    a = ctx.args
    smoke = a.limit is not None
    if not smoke and a.run_id is None:
        print('[ERROR] the run-b stage needs --run-id N (or --limit N for a smoke run).')
        return 1
    st = read_stage_status(ctx.stage_dir("nbg"))
    if not st or st["status"] != "PASSED":
        print('[ABORT] the nbg stage (split files) is not PASSED')
        return 1
    if smoke:
        d = _open_dir(ctx, ctx.stage_dir("smoke_b"))
        prefix = f"{ctx.cfg['file_stem']}_IM_noise{ctx.noise:g}_alignment_smoke"
    else:
        d = ctx.out / "runs" / f"methodB_run{a.run_id}"
        if d.exists():
            print(f"[ABORT] run directory already exists (not overwritten): {d}")
            return 1
        d.mkdir(parents=True)
        prefix = f"{ctx.cfg['file_stem']}_IM_noise{ctx.noise:g}_alignment_run{a.run_id}"
    if d is None:
        return 1
    rec = CheckRecorder()

    def f(suffix):
        return d / f"{prefix}_{suffix}"

    with Tee(f("execution_log.txt")):
        print(f"\n[run-b] {ctx.name} Method B {'smoke (limit ' + str(a.limit) + ')' if smoke else 'official run ' + str(a.run_id)}")
        if not record_inputs(ctx, rec):
            return finish(d, "run-b", rec)
        import time as _time
        from utils.validation_utils import environment_info
        from datetime import datetime

        import pandas as pd
        import pm4py

        from utils.alignment_utils import alignment_repair
        from utils.data_utils import load_case_traces
        from utils.discovery_utils import discover_model, model_report

        wall0, started = _time.perf_counter(), datetime.now()
        write_json(f("environment.json"), environment_info())
        traces, labels, clean_ids, anomalous_ids = load_case_traces(ctx.input_paths["anomaly_log"])
        # case ids as strings (split files are strings; Credit has numeric case ids that pandas reads as int)
        traces.index, labels.index = traces.index.map(str), labels.index.map(str)
        clean_ids, anomalous_ids = [str(c) for c in clean_ids], [str(c) for c in anomalous_ids]
        test_ids = read_ids(ctx.stage_dir("nbg") / "split_test_case_ids.txt")
        val_ids = read_ids(ctx.stage_dir("nbg") / "split_val_case_ids.txt")
        rec.check("split from nbg stage covers all anomalous cases (disjoint)",
                  set(test_ids) | set(val_ids) == set(map(str, anomalous_ids)) and not set(test_ids) & set(val_ids),
                  f"(val {len(val_ids)}, test {len(test_ids)})")
        if smoke:
            test_ids, val_ids = test_ids[:a.limit], []
        tree, net, im, fm, disc_sec = discover_model(traces, clean_ids, ctx.noise)
        mc, lang = model_report(tree, net, im, fm, traces, clean_ids, disc_sec,
                                ctx.cfg["trace_generation"]["safety_max_traces"])
        ref = read_json(ctx.stage_dir("discover") / "model_check.json")
        same = all(mc[k] == ref[k] for k in ["n_places", "n_transitions", "n_invisible_transitions", "n_arcs",
                                             "process_tree"])
        rec.check("re-discovered model == discover stage (places / transitions / arcs / tree)", same,
                  {k: mc[k] for k in ["n_places", "n_transitions", "n_arcs"]})
        write_json(f("model_check.json"), mc)
        pm4py.write_pnml(net, im, fm, str(f("petri_net.pnml")))

        t_align = _time.perf_counter()
        rows_log, rows_case = alignment_repair(traces, labels, [("test", test_ids), ("val", val_ids)], net, im, fm,
                                               set(map(tuple, lang)) if lang is not None else None)
        align_sec = _time.perf_counter() - t_align
        repaired = pd.DataFrame(rows_log)
        per_case = pd.DataFrame(rows_case)
        repaired.to_csv(f("repaired_log_b.csv"), index=False)
        per_case.drop(columns=["align_ms"]).to_csv(f("alignment_per_case.csv"), index=False)
        per_case[["case:concept:name", "type", "align_ms"]].to_csv(f("alignment_timing.csv"), index=False)
        f("test_case_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
        f("val_case_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
        tr = per_case[per_case["type"] == "test"]
        n_fail = int(tr["alignment_failed"].sum())
        rec.info("alignment failures (test / val)", (n_fail, int(per_case.loc[per_case["type"] == "val", "alignment_failed"].sum())))
        stats = {"dataset": ctx.name, "method": "B", "run_id": a.run_id, "official": not smoke, "smoke_limit": a.limit,
                 "noise_threshold": ctx.noise, "alignment_variant": "VERSION_STATE_EQUATION_A_STAR",
                 "n_test": len(test_ids), "n_val": len(val_ids), "n_alignment_failed_test": n_fail,
                 "n_alignment_failed_val": int(per_case.loc[per_case["type"] == "val", "alignment_failed"].sum()),
                 "test_align_ms_mean": float(tr["align_ms"].mean()) if len(tr) else None,
                 "test_align_ms_median": float(tr["align_ms"].median()) if len(tr) else None,
                 "discovery_sec": round(disc_sec, 4), "alignment_sec": round(align_sec, 4),
                 "wall_sec": round(_time.perf_counter() - wall0, 4), "started_at": started.isoformat(timespec="seconds"),
                 "pm4py": pm4py.__version__, "repair_rule": "sync keep / log move delete / visible model move insert / tau skip",
                 "code": 'utils.alignment_utils (Method B align_case + repair_from_alignment)'}
        write_json(f("stats.json"), stats)
        print({k: stats[k] for k in ["n_test", "n_val", "n_alignment_failed_test", "test_align_ms_mean",
                                                 "alignment_sec", "wall_sec"]})
        if not smoke:
            write_standard_timing(ctx, f("stats.json").parent, "B")
    return finish(d, "run-b", rec)


def stage_agreement(ctx):
    """Agreement of the Method A / B / C repaired logs (existing official repaired logs only, no new repair)."""
    refs = ctx.cfg.get("reference_runs", {})
    need = {"A": "A_run1", "B": "B_run1", "C": "C_run1"}
    if all(v in refs for v in need.values()):  # Small: existing official results
        paths = {m: PROJECT_ROOT / refs[r] for m, r in need.items()}
        split_file = PROJECT_ROOT / ctx.cfg["regression"]["split_test_ids"]
    else:  # other datasets: official pipeline runs (runs/method{A,B,C}_run1)
        paths = {}
        for m in "ABC":
            fs = list((ctx.out / "runs" / f"method{m}_run1").glob("*_repaired_log_*.csv"))
            if len(fs) != 1:
                print(f"[ERROR] {ctx.name}: runs/method{m}_run1 repaired log not found.")
                return 1
            paths[m] = fs[0]
        split_file = ctx.stage_dir("nbg") / "split_test_case_ids.txt"
    d = PROJECT_ROOT / "outputs" / "repaired_log_agreement"
    d.mkdir(parents=True, exist_ok=True)
    pc_path, sm_path = d / f"{ctx.name}_abc_agreement_per_case.csv", d / f"{ctx.name}_abc_agreement_summary.csv"
    if (pc_path.exists() or sm_path.exists()) and not ctx.args.overwrite:
        print(f"[ABORT] {pc_path.name} already exists. Use --overwrite to rebuild")
        return 1
    rec = CheckRecorder()
    from utils.agreement_utils import abc_agreement

    test_ids = read_ids(split_file)
    pc, summ = abc_agreement(paths, test_ids, ctx.input_paths["anomaly_log"], ctx.input_paths["ground_truth_log"])
    pc.to_csv(pc_path, index=False)
    summ.to_csv(sm_path, index=False)
    rec.check("per-case rows == test cases", len(pc) == len(test_ids), f"({len(pc)})")
    print(summ.to_string(index=False))
    write_json(d / f"{ctx.name}_abc_agreement_sources.json", {"runs": {m: str(p) for m, p in paths.items()},
                                                               "test_split": str(split_file),
                                                               "definition": 'per case_id, full activity sequence identical',
                                                               "checks": rec.checks})
    return 0 if rec.ok else 1


def stage_test_timing_b(ctx):
    """Method B (alignment): sequential test-only alignment + repair timing with the saved official Petri net (.venv)."""
    a = ctx.args
    if not (a.pnml and a.label and a.repaired_log):
        print('[ERROR] --pnml, --repaired-log (repaired log of the run folder that contains the official alignment_per_case.csv) and --label are required.')
        return 1
    if not need_pm4py():
        return 1
    d = _open_dir(ctx, ctx.stage_dir("test_timing") / a.label)
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print(f"\n[test-timing-b] {a.label}: sequential test-only alignment + repair with the official Petri net (perf_counter)")
        print('        Purpose: per-case latency. Accuracy comes from the official Method B run 1; this replay is not a performance result.')
        import pandas as pd

        from utils.timing_utils import replay_alignment_timing, summarize_values

        run_dir = Path(a.repaired_log).parent
        test_ids_f = list(run_dir.glob("*_test_case_ids.txt"))
        per_case_f = list(run_dir.glob("*_alignment_per_case.csv"))
        model_f = list(run_dir.glob("*_model_check.json"))
        if not rec.check("official run files found (test ids, per-case, model_check)",
                         len(test_ids_f) == len(per_case_f) == len(model_f) == 1):
            return finish(d, "test-timing-b", rec)
        test_ids = read_ids(test_ids_f[0])
        expected_n = len(read_ids(ctx.stage_dir("nbg") / "split_test_case_ids.txt")) if ctx.name != "Small" else 524
        rec.check(f"test cases == {expected_n}", len(test_ids) == expected_n, f"({len(test_ids)})")
        script = PROJECT_ROOT / "src/experiments/p6_small_repaired_log_comparison/61_method_b_alignment_repair.py"
        from utils.data_utils import load_case_traces

        traces = load_case_traces(ctx.input_paths["anomaly_log"])[0]
        traces.index = traces.index.map(str)  # split ids are strings (numeric Credit case ids)
        rows, net_info, wall = replay_alignment_timing(script, a.pnml, test_ids, traces)
        mc = read_json(model_f[0])
        rec.check("Petri Net (PNML) == official model_check (places / transitions)",
                  net_info["n_places"] == mc["n_places"] and net_info["n_transitions"] == mc["n_transitions"],
                  net_info)
        rec.check("no alignment failure", not any(r["failed"] for r in rows))

        df = pd.DataFrame(rows)
        df.insert(1, "path", "Alignment")
        df[["case_id", "path", "repair_time_ms"]].to_csv(d / "case_timing.csv", index=False)
        df[["case_id", "repaired_trace", "alignment_cost", "failed"]].to_csv(d / "replay_test_repaired_traces.csv", index=False)

        off = pd.read_csv(per_case_f[0], dtype=str, keep_default_na=False)
        off = off[off["type"] == "test"].set_index("case:concept:name")
        agree = sum(off.loc[r["case_id"], "repaired_trace"] == r["repaired_trace"] for r in rows)
        rec.info("repaired trace agreement with official Run 1 (timing replay, not an accuracy result)",
                 f"{agree}/{len(rows)}")
        summ = summarize_values([r["repair_time_ms"] for r in rows])
        summary = {"label": a.label, "method": "B (Alignment)", "purpose": "timing only (not an accuracy experiment)",
                   "boundary": "per case: align_case (PM4Py alignment) + repair_from_alignment; excludes loading, "
                               "discovery, Petri Net creation (PNML read), trace object build, CSV save, evaluation",
                   "pnml": str(a.pnml), "net": net_info, "test_case_ms": summ,
                   "test_phase_wall_ms": round(wall, 4), "test_phase_wall_ms_per_case": round(wall / len(rows), 4),
                   "trace_agreement_with_official_run1": f"{agree}/{len(rows)}", "checks": rec.checks}
        write_json(d / "timing_summary.json", summary)
        print(f"  mean {summ['mean']} ms | median {summ['median']} ms | std {summ['std']} | min {summ['min']} | max {summ['max']}"
              f" | agreement {agree}/{len(rows)}")
    return finish(d, "test-timing-b", rec)


def stage_test_timing(ctx):
    """Sequential test-only replay with a trained agent (.pkl) to measure per-case repair latency (timing only)."""
    a = ctx.args
    if not (a.agent and a.repaired_log and a.label and a.method):
        print('[ERROR] --agent, --repaired-log, --label and --method are required.')
        return 1
    d = _open_dir(ctx, ctx.stage_dir("test_timing") / a.label)
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print(f"\n[test-timing] {a.label}: sequential test-only replay with the trained agent (no Q-table update, unchanged action selection)")
        print('        Purpose: per-case repair latency without multiprocessing effects. Not an accuracy experiment.')
        if not record_inputs(ctx, rec):
            return finish(d, "test-timing", rec)
        import pickle

        import numpy as np
        import pandas as pd

        from sim import _load_and_prepare_data
        from utils import evaluation_utils as ev
        from utils.nbg_utils import build_method_c_nbgs
        from utils.timing_utils import prepare_rpbear_inputs, rebuild_repair_memory, replay_test_phase, summarize, \
            test_result_frame

        repaired_log = Path(a.repaired_log)
        stats_files = list(repaired_log.parent.glob("*_stats.json"))
        if not rec.check("official stats.json found", len(stats_files) == 1, [str(s) for s in stats_files]):
            return finish(d, "test-timing", rec)
        st = read_json(stats_files[0])
        rec.check("method matches official nbg_source", (st.get("nbg_source") == "override") == (a.method == "C"),
                  st.get("nbg_source"))
        nbgs = None
        if a.method == "C":
            _c, _a, _g, actset, _f, _af = _load_and_prepare_data(ctx.data_name(), ctx.input_paths)
            nbgs, _info = build_method_c_nbgs(ctx.cfg["new_nbg_input"], ctx.stage_dir("generate"), _c, np.asarray(actset))
        prep = prepare_rpbear_inputs(ctx.data_name(), ctx.input_paths, st["val_ratio"], st["seed"], nbgs)
        off = pd.read_csv(repaired_log, dtype=str, keep_default_na=False)
        types = off.groupby("case:concept:name", sort=False)["type"].first()
        rec.check("split == official run split",
                  set(types[types == "val"].index) == set(map(str, prep["val_ids"]))
                  and set(types[types == "test"].index) == set(map(str, prep["test_ids"])),
                  f"(val {len(prep['val_ids'])}, test {len(prep['test_ids'])})")
        memory, minfo = rebuild_repair_memory(repaired_log, prep["case_to_trace"], prep["val_ids"])
        rec.info("repair memory rebuilt from official validation rows", minfo)

        with open(a.agent, "rb") as fh:
            agent = pickle.load(fh)
        q_before, eps_before = dict(agent.q_table), agent.exploration_rate
        rec.info("agent", {"q_table_size": len(q_before), "exploration_rate": eps_before})
        results, timing, wall = replay_test_phase(prep, agent, memory, float(st["alpha"]), int(st["num_epi"]),
                                                  int(st["max_step"]), seed=int(st["seed"]))
        rec.check("Q-table unchanged by Test replay (no Q-value update)", agent.q_table == q_before)
        rec.check("exploration_rate unchanged by Test replay", agent.exploration_rate == eps_before)

        rep = test_result_frame(results)
        rep.to_csv(d / "replay_test_repaired.csv", index=False)
        pd.DataFrame(timing)[["case_id", "path", "repair_time_ms"]].to_csv(d / "case_timing.csv", index=False)

        # per-case comparison with the official results; RL-path results may differ because the random
        # streams differ (see timing_utils)
        off_test = off[off["type"] == "test"]
        rep_s = pd.read_csv(d / "replay_test_repaired.csv", dtype=str, keep_default_na=False)
        cols = list(off.columns)
        same_rows, same_trace = {}, {}
        for cid, g in off_test.groupby("case:concept:name", sort=False):
            r = rep_s[rep_s["case:concept:name"] == cid]
            same_rows[cid] = g[cols].reset_index(drop=True).equals(r[cols].reset_index(drop=True))
            same_trace[cid] = tuple(g["concept:name"]) == tuple(r["concept:name"])
        path_of = {t["case_id"]: t["path"] for t in timing}
        agree = {}
        for path in ("all", "RL", "Mem"):
            ids = [c for c in same_rows if path == "all" or path_of.get(c) == path]
            agree[path] = {"n": len(ids), "rows_identical": int(sum(same_rows[c] for c in ids)),
                           "trace_identical": int(sum(same_trace[c] for c in ids))}
        rec.info("agreement with official test output (rows identical / trace identical)", agree)
        off_rl = int(st.get("test_inference_runs", -1))
        rec.info("RL-path count: replay vs official", {"replay": sum(t["path"] == "RL" for t in timing),
                                                       "official": off_rl})
        gt = ev.canonical_traces(ctx.input_paths["ground_truth_log"], "Case", "Activity")
        rep_tr = ev.canonical_traces(d / "replay_test_repaired.csv", "case:concept:name", "concept:name", "test")
        replay_exact = int(sum(rep_tr[c] == gt[c] for c in rep_tr.index))

        summ = summarize(timing, wall)
        summary = {
            "label": a.label, "method": a.method, "purpose": "timing only (not an accuracy experiment)",
            "mode": "sequential Test-only replay with the official trained agent",
            "official_run": {"repaired_log": str(repaired_log), "agent": str(a.agent),
                             **{k: st.get(k) for k in ["alpha", "num_epi", "seed", "max_step", "val_ratio",
                                                       "num_cores", "use_multiprocessing", "test_per_case_ms",
                                                       "test_total_ms", "test_inference_runs", "test_p_anomaly"]}},
            "main_metric_candidates": {"test_rl_mean_ms": summ.get("test_case_ms_rl_mean"),
                                       "test_rl_median_ms": summ.get("test_case_ms_rl_median")},
            "supplementary_paper_metric": {
                "official_test_phase_wall_per_case_ms (multiprocessing)": st.get("test_per_case_ms"),
                "replay_test_phase_wall_per_case_ms (sequential)": summ["test_phase_wall_ms_per_case"]},
            "timing_summary": summ, "repair_memory": minfo, "agreement_with_official": agree,
            "replay_exact_match_test (timing replay, not an accuracy result)": f"{replay_exact}/{len(rep_tr)}",
            "checks": rec.checks}
        write_json(d / "timing_summary.json", summary)
        print(f"  RL mean {summ['test_case_ms_rl_mean']} ms | RL median {summ['test_case_ms_rl_median']} ms | "
              f"Mem mean {summ['test_case_ms_mem_mean']} ms | wall/case {summ['test_phase_wall_ms_per_case']} ms")
    return finish(d, "test-timing", rec)


def stage_all(ctx):
    """Run discover -> generate -> nbg -> smoke with each environment's python (run not included)."""
    plan = [("discover", "pm4py"), ("generate", "pm4py"), ("nbg", "rpbear"), ("smoke", "rpbear")]
    for stage, env in plan:
        py = PROJECT_ROOT / dc.ENV_DIRS[env] / "Scripts" / "python.exe"
        if not py.exists():
            py = PROJECT_ROOT / dc.ENV_DIRS[env] / "bin" / "python"
        cmd = [str(py), str(Path(__file__).resolve()), "--dataset", ctx.name, "--stage", stage]
        if ctx.args.data_root:
            cmd += ["--data-root", ctx.args.data_root]
        if ctx.args.overwrite:
            cmd.append("--overwrite")
        print(f"\n>>> {' '.join(cmd)}")
        code = subprocess.call(cmd)
        if code != 0:
            print(f"[STOP] stage {stage} exit={code} ({'BLOCKED' if code == 2 else 'FAILED'})")
            return code
    print('\n[all] discover .. smoke passed. Run the official run separately with --stage run --method A|C --run-id N.')
    return 0


# ============================================================
# 3. Main
# ============================================================

# ============================================================
# 2b. Method C + Petri-net conformance gate (C+PN)
#     The Method C (W2) / A / B paths are unchanged. Outputs: <dataset>/C_PN_Gate/, <dataset>/pn_gate_analysis/
# ============================================================

PNGATE_DIR, PNGATE_ANALYSIS = "C_PN_Gate", "pn_gate_analysis"


def _pngate_require_w2(ctx):
    if ctx.cfg["new_nbg_input"] != "original_plus_model_only":
        print('[ERROR] the C+PN gate is defined only for datasets with the W2 Expanded New NBG.')
        return False
    return True


def _pngate_inputs(ctx, rec):
    """W2 NBG (same input as official Method C) + gate from the official discovered Petri net + basic checks."""
    import numpy as np

    from sim import _load_and_prepare_data
    from utils.nbg_utils import build_method_c_nbgs, edge_set_digest
    from utils.pn_gate_utils import PNMLAcceptor

    clean, anomaly, gt, actset, filtered, _af = _load_and_prepare_data(ctx.data_name(), ctx.input_paths)
    actset = np.asarray(actset)
    nbgs, _info = build_method_c_nbgs(ctx.cfg["new_nbg_input"], ctx.stage_dir("generate"), clean, actset)
    digest = edge_set_digest(nbgs, actset)
    ref = read_json(ctx.stage_dir("nbg") / "nbg_comparison_summary.json")["comparison"]["edge_set_digest"]["new"]
    rec.check("W2 Expanded New NBG digest == nbg stage (official Method C input)", digest == ref, digest)
    gate = PNMLAcceptor(ctx.stage_dir("discover") / "petri_net.pnml")
    gate.accepts(("a",))  # parse the PNML once
    mc = read_json(ctx.stage_dir("discover") / "model_check.json")
    rec.check("gate Petri Net == official discovered model (places / transitions / invisible / arcs)",
              gate.info == {"places": mc["n_places"], "transitions": mc["n_transitions"],
                            "invisible": mc["n_invisible_transitions"], "arcs": mc["n_arcs"]}, gate.info)
    return clean, anomaly, gt, actset, filtered, nbgs, gate, digest


def stage_pngate_acceptor_check(ctx):
    """(.venv) Check PNMLAcceptor (for .venv-rpbear, no PM4Py) == loop_bound_utils.make_petri_acceptor (PM4Py)."""
    if not need_pm4py() or not _pngate_require_w2(ctx):
        return 1
    out = ctx.out / PNGATE_ANALYSIS
    out.mkdir(exist_ok=True)
    rec = CheckRecorder()
    import pandas as pd
    import pm4py

    from utils.data_utils import load_case_traces
    from utils.loop_bound_utils import make_petri_acceptor
    from utils.pn_gate_utils import PNMLAcceptor

    pnml = ctx.stage_dir("discover") / "petri_net.pnml"
    net, im, fm = pm4py.read_pnml(str(pnml))
    ref, mine = make_petri_acceptor(net, im, fm), PNMLAcceptor(pnml)
    traces = load_case_traces(ctx.input_paths["anomaly_log"])[0]
    pool = {"case traces": {tuple(t) for t in traces},
            f"generated ({ctx.cfg['trace_generation']['loop_num']},{ctx.cfg['trace_generation']['loop_total']})": {tuple(t.split(">>")) for t in pd.read_csv(ctx.stage_dir("generate") / "generated_bounded_traces.csv")["trace"]}}
    rep = set()
    for m in "ABC":
        for f in (ctx.out / "runs" / f"method{m}_run1").glob("*_repaired_log_*.csv"):
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            for _cid, g in df.groupby("case:concept:name", sort=False):
                rep.add(tuple(a for a in g["concept:name"] if a not in ("Start", "End")))
    pool["official A/B/C repaired traces"] = rep
    res = {}
    for name, ts in pool.items():
        mism = [t for t in ts if ref(t) != mine.accepts(t)]
        res[name] = {"n": len(ts), "accepted_ref": sum(ref(t) for t in ts), "mismatch": len(mism)}
        rec.check(f"PNMLAcceptor == pm4py acceptor on {name}", not mism, res[name])
    write_json(out / "acceptor_equivalence.json", {"pnml": str(pnml), "results": res, "checks": rec.checks})
    return 0 if rec.ok else 1


def stage_pngate_offline(ctx):
    """(.venv-rpbear) Which official C repaired traces the gate would reject at termination (no RL run)."""
    if not _pngate_require_w2(ctx):
        return 1
    out = ctx.out / PNGATE_ANALYSIS
    out.mkdir(exist_ok=True)
    rec = CheckRecorder()
    with Tee(out / "offline_log.txt"):
        print('\n[pngate-offline] Apply the Petri-net gate offline to the existing official Method C (W2) results')
        if not record_inputs(ctx, rec):
            return 1
        import pandas as pd

        from utils.pn_gate_utils import adm_normal

        clean, anomaly, _gt, actset, filtered, nbgs, gate, _dg = _pngate_inputs(ctx, rec)
        test_ids = read_ids(ctx.stage_dir("nbg") / "split_test_case_ids.txt")
        rc = pd.read_csv(next((ctx.out / "runs" / "methodC_run1").glob("*_repaired_log_c.csv")), dtype=str, keep_default_na=False)
        rc = rc[rc["type"] == "test"]
        an = {str(c): tuple(a for a in g["concept:name"].astype(str) if a not in ("Start", "End"))
              for c, g in anomaly.groupby("case:concept:name", observed=True)}
        d40 = pd.read_csv(ctx.out / "concurrency_analysis" / "bc_disagreement_cases.csv", dtype={"case_id": str})
        d40 = set(d40.loc[d40.primary_cause == "D_W2_edge_level_overgeneralization", "case_id"])
        rows = []
        for cid, g in rc.groupby("case:concept:name", sort=False):
            fin = tuple(a for a in g["concept:name"] if a not in ("Start", "End"))
            term = g["rl_success"].iloc[0] == "1"
            w2n, pn = adm_normal(fin, actset, filtered, nbgs), gate.accepts(fin)
            rows.append({"case_id": cid, "C_final_trace": " ".join(fin), "anomaly_trace": " ".join(an[cid]),
                         "C_unrepaired": fin == an[cid], "rl_success (terminated by ADM normal)": term,
                         "rl_steps": int(g["rl_steps"].iloc[0]), "W2_ADM_normal": w2n, "petri_conformant": pn,
                         "gate": ("allow (same as baseline)" if term and pn else
                                  "reject termination" if term and not pn else "n/a (no successful episode)"),
                         "in_D40": cid in d40})
        df = pd.DataFrame(rows)
        rec.check("test cases == 210 split", set(df.case_id) == set(test_ids), len(df))
        df.to_csv(out / "offline_gate_diagnostic.csv", index=False)
        rej = df[df.gate == "reject termination"]
        dd = df[df.in_D40]
        summ = {"test_cases": len(df), "terminated_by_ADM_normal": int(df["rl_success (terminated by ADM normal)"].sum()),
                "W2_ADM_normal_final": int(df.W2_ADM_normal.sum()), "petri_conformant_final": int(df.petri_conformant.sum()),
                "gate_allow": int((df.gate == "allow (same as baseline)").sum()), "gate_reject_termination": len(rej),
                "gate_na": int(df.gate.str.startswith("n/a").sum()),
                "D40_found": len(dd), "D40_rejected_by_gate": int((dd.gate == "reject termination").sum()),
                "rejected_not_in_D40": sorted(set(rej.case_id) - set(dd.case_id)),
                "note": "offline: final traces of the existing official run; actual C+PN behaviour requires a new RL run"}
        rec.check("D40: all 40 Petri-rejected C final traces are rejected by the gate",
                  summ["D40_found"] == 40 and summ["D40_rejected_by_gate"] == 40, f"({summ['D40_rejected_by_gate']}/40)")
        write_json(out / "offline_gate_summary.json", summ | {"checks": rec.checks})
        for k, v in summ.items():
            print(f"  {k}: {v}")
    return 0 if rec.ok else 1


def stage_pngate_smoke(ctx):
    """(.venv-rpbear) C+PN smoke: gate off == sim (byte), accept-all gate == gate off, PN gate behaviour / invariants."""
    if not _pngate_require_w2(ctx):
        return 1
    d = _open_dir(ctx, ctx.out / PNGATE_ANALYSIS / "smoke")
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print('\n[pngate-smoke] C+PN gate smoke (not official performance values)')
        if not record_inputs(ctx, rec):
            return finish(d, "pngate-smoke", rec)
        import pandas as pd

        from utils.pn_gate_utils import AcceptAllGate, run_rpbear_gated

        clean, anomaly, _gt, actset, filtered, nbgs, gate, _dg = _pngate_inputs(ctx, rec)
        p, n, lim = ctx.params, dc.SMOKE_NUM_EPI, 40

        def run(nbg, g, mp, limit):
            return run_rpbear_gated(ctx.data_name(), n, p["alpha"], p["val_ratio"], p["seed"], p["max_step"],
                                    p["num_cores"], mp, nbg, ctx.input_paths, pn_gate=g, limit=limit)

        # 1) gate off == sim sequential smoke (A, C), byte-identical
        for label, nbg in (("A", None), ("C", nbgs)):
            df, st, _t, _ag = run(nbg, None, False, None)
            df.to_csv(d / f"gateoff_seq_epi{n}_method{label}.csv", index=False)
            same = (d / f"gateoff_seq_epi{n}_method{label}.csv").read_bytes() == \
                (ctx.stage_dir("smoke") / f"smoke_seq_epi{n}_method{label}.csv").read_bytes()
            rec.check(f"gate off (pn_gate_utils runner) == sim sequential smoke Method {label} (byte-identical)", same)
        # 2) accept-all gate == gate off (limit, sequential): conformant traces terminate as before
        off, _s, _t, _a = run(nbgs, None, False, lim)
        allg, _s, _t, _a = run(nbgs, AcceptAllGate(), False, lim)
        rec.check(f"accept-all gate == gate off (W2, sequential, val/test first {lim})",
                  off.equals(allg.drop(columns=["pn_gate_rejected_episodes"])))
        # 3) PN gate, sequential + multiprocessing
        res = {}
        for mp in (False, True):
            df, st, _t, ag = run(nbgs, gate, mp, lim)
            df.to_csv(d / f"pngate_{'mp' if mp else 'seq'}_epi{n}_limit{lim}.csv", index=False)
            per = df.groupby("case:concept:name", sort=False, observed=True).agg(
                rl_success=("rl_success", "first"), rl_steps=("rl_steps", "first"),
                rejected=("pn_gate_rejected_episodes", "first"), type=("type", "first"))
            fin = {c: tuple(a for a in g["concept:name"] if a not in ("Start", "End"))
                   for c, g in df.groupby("case:concept:name", sort=False, observed=True)}
            succ = per[per.rl_success == 1]
            rej = per[per.rejected > 0]
            r = {"cases": len(per), "q_table_size": st["q_table_size"], "cases_with_gate_rejection": len(rej),
                 "gate_rejected_episodes_total": int(per.rejected.sum()),
                 "rejected_then_success": int((rej.rl_success == 1).sum()), "rejected_no_success": int((rej.rl_success == 0).sum()),
                 "success_cases": len(succ), "max_rl_steps": int(per.rl_steps.max())}
            res["mp" if mp else "seq"] = r
            tag = "multiprocessing" if mp else "sequential"
            rec.check(f"[{tag}] every successful case ends on a Petri-conformant trace",
                      all(gate.accepts(fin[c]) for c in succ.index), r)
            rec.check(f"[{tag}] gate was invoked and rejected at least one ADM-normal termination", r["gate_rejected_episodes_total"] > 0)
            rec.check(f"[{tag}] max_step semantics kept (rl_steps <= max_step)", r["max_rl_steps"] <= p["max_step"])
            rec.check(f"[{tag}] repaired log has val and test cases, Q-table updated",
                      set(per.type) == {"val", "test"} and st["q_table_size"] > 0)
            rec.check(f"[{tag}] alpha == official alpha", st["alpha"] == p["alpha"], st["alpha"])
        write_json(d / "smoke_summary.json", {"num_epi": n, "limit": lim, "pn_gate": res, "checks": rec.checks})
    return finish(d, "pngate-smoke", rec)


def stage_run_c_pngate(ctx):
    """(.venv-rpbear) C+PN official run: W2 Expanded New NBG + Petri-net gate, official settings. The C run is not touched."""
    if not _pngate_require_w2(ctx) or ctx.args.run_id is None:
        print('[ERROR] --run-id N is required (W2 datasets only).')
        return 1
    st = read_stage_status(ctx.out / PNGATE_ANALYSIS / "smoke")
    if not st or st["status"] != "PASSED":
        print('[ABORT] pngate-smoke is not PASSED')
        return 1
    d = ctx.out / PNGATE_DIR / f"run{ctx.args.run_id}"
    if d.exists():
        print(f"[ABORT] run directory already exists (not overwritten): {d}")
        return 1
    d.mkdir(parents=True)
    rec = CheckRecorder()
    p = dict(ctx.params)
    from multiprocessing import cpu_count
    if p.get("num_cores") is None:
        p["num_cores"] = max(1, cpu_count() - 1)
    prefix = f"{ctx.cfg['file_stem']}_{p['num_epi']}_ppm{p['alpha']}_valratio{p['val_ratio']}_seed{p['seed']}_methodCPN_run{ctx.args.run_id}"

    def f(suffix):
        return d / f"{prefix}_{suffix}"

    with Tee(f("execution_log.txt")):
        print(f"\n[run-c-pngate] {ctx.name} Method C + Petri Net gate official run {ctx.args.run_id} | {p}")
        if not record_inputs(ctx, rec):
            return finish(d, "run-c-pngate", rec)
        import pickle
        import time
        from datetime import datetime

        import pandas as pd

        from utils.pn_gate_utils import run_rpbear_gated
        from utils.validation_utils import environment_info

        tg = ctx.cfg["trace_generation"]
        rec.check("official conditions (num_epi 1000 / alpha 0 / val_ratio 0.5 / seed 42 / max_step 8 / multiprocessing)",
                  (p["num_epi"], p["alpha"], p["val_ratio"], p["seed"], p["max_step"], p["use_multiprocessing"]) ==
                  (1000, 0.0, 0.5, 42, 8, True), p)
        official_bound = {"Credit": (5, 7), "Pub": (4, 4)}.get(ctx.name)  # fixed official bounds for Credit / Pub
        if official_bound is None:  # threshold runs (e.g. RoadTraffic): must equal the bound actually used by the generate stage
            gs = read_json(ctx.stage_dir("generate") / "generate_summary.json")
            official_bound = (gs["loop_num"], gs["loop_total"])
        rec.check(f"loop bound == {official_bound}", (tg["loop_num"], tg["loop_total"]) == official_bound)
        _c, _an, _g, _act, _fil, nbgs, gate, digest = _pngate_inputs(ctx, rec)
        if not rec.ok:
            return finish(d, "run-c-pngate", rec)
        write_json(f("environment.json"), environment_info())
        started, t0 = datetime.now(), time.time()
        df, stats, timing, agent = run_rpbear_gated(ctx.data_name(), p["num_epi"], p["alpha"], p["val_ratio"], p["seed"],
                                                    p["max_step"], p["num_cores"], True, nbgs, ctx.input_paths,
                                                    pn_gate=gate, limit=None, verbose=True)
        df.to_csv(f("repaired_log_cpn.csv"), index=False)
        timing.to_csv(f("case_timing.csv"), index=False)
        with open(f("agent.pkl"), "wb") as fh:
            pickle.dump(agent, fh)
        types = df.groupby("case:concept:name", sort=False, observed=True)["type"].first()
        val_ids, test_ids = sorted(map(str, types[types == "val"].index)), sorted(map(str, types[types == "test"].index))
        f("val_case_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
        f("test_case_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
        rec.check("split == nbg stage split (same as official A / B / C)",
                  set(val_ids) == set(read_ids(ctx.stage_dir("nbg") / "split_val_case_ids.txt"))
                  and set(test_ids) == set(read_ids(ctx.stage_dir("nbg") / "split_test_case_ids.txt")))
        per = df.groupby("case:concept:name", sort=False, observed=True)["pn_gate_rejected_episodes"].first()
        stats.update({"method": "C+PN (W2 Expanded New NBG + Petri Net conformance gate)", "run_id": ctx.args.run_id,
                      "nbg_edge_set_digest": digest, "new_nbg_input": ctx.cfg["new_nbg_input"],
                      "loop_bound": [tg["loop_num"], tg["loop_total"]], "started_at": started.isoformat(timespec="seconds"),
                      "wall_sec": round(time.time() - t0, 2), "cases_with_gate_rejection": int((per > 0).sum()),
                      "gate_rejected_episodes_total": int(per.sum()), "checks": rec.checks})
        write_json(f("stats.json"), stats)
        pd.DataFrame([{k: v for k, v in stats.items() if k != "checks"}]).to_csv(f("stats.csv"), index=False)
        write_standard_timing(ctx, f("stats.json").parent, "C+PN")
        print({k: stats[k] for k in ["n_val_cases", "n_test_cases", "training_time_sec", "q_table_size",
                                     "cases_with_gate_rejection", "gate_rejected_episodes_total", "wall_sec"]})
    return finish(d, "run-c-pngate", rec)


def stage_test_timing_pngate(ctx):
    """(.venv-rpbear) Sequential test-only replay with the trained C+PN agent (with gate; timing only). Same definition as test-timing."""
    a = ctx.args
    if not _pngate_require_w2(ctx) or not (a.label and a.run_id):
        print('[ERROR] --label and --run-id are required.')
        return 1
    run = ctx.out / PNGATE_DIR / f"run{a.run_id}"
    d = _open_dir(ctx, ctx.out / PNGATE_DIR / "test_timing" / a.label)
    if d is None:
        return 1
    rec = CheckRecorder()
    with Tee(d / "log.txt"):
        print(f"\n[test-timing-pngate] {a.label}: sequential test-only replay with the C+PN agent (timing only, not an accuracy experiment)")
        if not record_inputs(ctx, rec):
            return finish(d, "test-timing-pngate", rec)
        import pickle

        import pandas as pd

        from utils import evaluation_utils as ev
        from utils.pn_gate_utils import replay_test_phase_gated
        from utils.timing_utils import prepare_rpbear_inputs, rebuild_repair_memory, summarize, test_result_frame

        rep_log = next(run.glob("*_repaired_log_cpn.csv"))
        st = read_json(next(run.glob("*_stats.json")))
        _c, _an, _g, _act, _fil, nbgs, gate, _dg = _pngate_inputs(ctx, rec)
        prep = prepare_rpbear_inputs(ctx.data_name(), ctx.input_paths, st["val_ratio"], st["seed"], nbgs)
        memory, minfo = rebuild_repair_memory(rep_log, prep["case_to_trace"], prep["val_ids"])
        with open(next(run.glob("*_agent.pkl")), "rb") as fh:
            agent = pickle.load(fh)
        q_before = dict(agent.q_table)
        results, timing, wall = replay_test_phase_gated(prep, agent, memory, float(st["alpha"]), int(st["num_epi"]),
                                                        int(st["max_step"]), gate, int(st["seed"]))
        rec.check("Q-table unchanged by Test replay", agent.q_table == q_before)
        rep = test_result_frame(results)
        rep.to_csv(d / "replay_test_repaired.csv", index=False)
        pd.DataFrame(timing)[["case_id", "path", "repair_time_ms"]].to_csv(d / "case_timing.csv", index=False)
        off = pd.read_csv(rep_log, dtype=str, keep_default_na=False)
        off = off[off["type"] == "test"]
        rep_s = pd.read_csv(d / "replay_test_repaired.csv", dtype=str, keep_default_na=False)
        same = sum(tuple(g["concept:name"]) == tuple(rep_s.loc[rep_s["case:concept:name"] == c, "concept:name"])
                   for c, g in off.groupby("case:concept:name", sort=False))
        summ = summarize(timing, wall)
        write_json(d / "timing_summary.json", {
            "label": a.label, "method": "C+PN", "purpose": "timing only (not an accuracy experiment)",
            "mode": "sequential Test-only replay with the official C+PN agent (gate included)",
            "official_run": {"repaired_log": str(rep_log), "test_per_case_ms": st.get("test_per_case_ms")},
            "timing_summary": summ, "repair_memory": minfo, "trace_agreement_with_official": f"{same}/{len(rep_s['case:concept:name'].unique())}",
            "checks": rec.checks})
        print(f"  RL mean {summ['test_case_ms_rl_mean']} | median {summ['test_case_ms_rl_median']} | n {summ['test_case_ms_rl_n']}")
    return finish(d, "test-timing-pngate", rec)


def stage_pngate_evaluate(ctx):
    """(.venv-rpbear) Evaluate the official C+PN results: EMR / error types / agreement / case subsets / timing summary."""
    if not _pngate_require_w2(ctx) or ctx.args.run_id is None:
        print('[ERROR] --run-id N is required.')
        return 1
    out = ctx.out / PNGATE_ANALYSIS
    rec = CheckRecorder()
    with Tee(out / f"evaluate_run{ctx.args.run_id}_log.txt"):
        print('\n[pngate-evaluate] Evaluate the official C+PN results (official EMR = repaired log vs ground truth exact match)')
        import statistics as stt
        from itertools import combinations

        import pandas as pd

        from utils import evaluation_utils as ev
        from utils.pn_gate_utils import adm_normal

        clean, anomaly, _gt, actset, filtered, nbgs, gate, _dg = _pngate_inputs(ctx, rec)
        run = ctx.out / PNGATE_DIR / f"run{ctx.args.run_id}"
        logs = {"A": next((ctx.out / "runs/methodA_run1").glob("*_repaired_log_a.csv")),
                "B": next((ctx.out / "runs/methodB_run1").glob("*_repaired_log_b.csv")),
                "C": next((ctx.out / "runs/methodC_run1").glob("*_repaired_log_c.csv")),
                "CPN": next(run.glob("*_repaired_log_cpn.csv"))}
        gt, an, labels, nv = ev.load_evaluation_basis(ctx.input_paths["anomaly_log"], ctx.input_paths["ground_truth_log"])
        test_ids = read_ids(ctx.stage_dir("nbg") / "split_test_case_ids.txt")
        tr = {m: ev.canonical_traces(p, "case:concept:name", "concept:name", "test") for m, p in logs.items()}
        rec.check("all four repaired logs cover the same 210 test cases", all(set(t.index) == set(test_ids) for t in tr.values()))
        cl = ev.case_level_table(test_ids, gt, an, labels, nv, tr)
        om = ev.official_metrics(cl, list(tr))
        cl.to_csv(out / "cpn_case_level.csv", index=False)
        om.to_csv(out / "cpn_official_metrics.csv")
        obs = {tuple(v) for v in nv}
        gen = pd.read_csv(ctx.stage_dir("generate") / "generated_bounded_traces.csv")
        model_only = {tuple(t.split(">>")) for t, mo in zip(gen.trace, gen.model_only) if mo}
        T = lambda m, c: tuple(tr[m][c])  # noqa: E731

        def kind(t, c):
            return ("correct" if t == tuple(gt[c]) else "unrepaired" if t == tuple(an[c]) else
                    "observed normal variant (other)" if t in obs else "model-only generated trace" if t in model_only
                    else "Petri Net accepts (other)" if gate.accepts(t) else "Petri Net rejects")

        kinds = {m: pd.Series([kind(T(m, c), c) for c in test_ids]).value_counts().to_dict() for m in tr}
        emr = {m: {"correct": int(om.loc[m, "n_exact_match"]), "n": int(om.loc[m, "n_cases"]),
                   "emr": round(float(om.loc[m, "exact_match_rate"]), 4), "kinds": kinds[m]} for m in tr}
        # agreement
        pairs = {f"{x}-{y}": sum(T(x, c) == T(y, c) for c in test_ids) for x, y in combinations(["A", "B", "C", "CPN"], 2)}
        cats = {}
        for c in test_ids:
            a_, b_, p_ = T("A", c), T("B", c), T("CPN", c)
            cat = ("A=B=CPN" if a_ == b_ == p_ else "A=B only" if a_ == b_ else "A=CPN only" if a_ == p_ else
                   "B=CPN only" if b_ == p_ else "all different")
            ok = [t == tuple(gt[c]) for t in (a_, b_, p_)]
            cats.setdefault(cat, {"n": 0, "all_correct": 0, "some_correct": 0, "all_wrong": 0})
            cats[cat]["n"] += 1
            cats[cat]["all_correct" if all(ok) else "all_wrong" if not any(ok) else "some_correct"] += 1
        agr = {"pairs_exact": {k: f"{v}/210 = {v / 210:.4f}" for k, v in pairs.items()}, "A_B_CPN_categories": cats,
               "target_80pct": 168, "B-CPN": pairs["B-CPN"], "B-C baseline": pairs["B-C"],
               "B-CPN_reaches_80pct": pairs["B-CPN"] >= 168, "shortfall_to_80pct": max(0, 168 - pairs["B-CPN"])}
        # B-C disagreement cases caused by W2 edge-level over-generalisation
        dis = pd.read_csv(ctx.out / "concurrency_analysis" / "bc_disagreement_cases.csv", dtype={"case_id": str})
        d40 = list(dis.loc[dis.primary_cause == "D_W2_edge_level_overgeneralization", "case_id"])
        rows40 = []
        for c in d40:
            b_, c0, p_, g_ = T("B", c), T("C", c), T("CPN", c), tuple(gt[c])
            rows40.append({"case_id": c, "anomaly_trace": " ".join(an[c]), "ground_truth": " ".join(g_), "B_repaired": " ".join(b_),
                           "C_baseline_repaired": " ".join(c0), "CPN_repaired": " ".join(p_), "B_correct": b_ == g_,
                           "C_baseline_correct": c0 == g_, "CPN_correct": p_ == g_, "C_baseline_petri": gate.accepts(c0),
                           "CPN_petri": gate.accepts(p_), "CPN_eq_B": p_ == b_, "CPN_eq_ground_truth": p_ == g_,
                           "CPN_unrepaired": p_ == tuple(an[c]), "CPN_kind": kind(p_, c)})
        t40 = pd.DataFrame(rows40)
        t40.to_csv(out / "gate_40_case_tracking.csv", index=False)
        s40 = {"n": len(t40), "CPN_eq_B": int(t40.CPN_eq_B.sum()), "CPN_correct": int(t40.CPN_correct.sum()),
               "CPN_petri_conformant": int(t40.CPN_petri.sum()), "CPN_unrepaired": int(t40.CPN_unrepaired.sum()),
               "CPN_other_wrong": int((~t40.CPN_correct & ~t40.CPN_unrepaired).sum()), "CPN_kinds": t40.CPN_kind.value_counts().to_dict()}
        # cases newly accepted by W2 (all splits)
        ids15 = read_json(ctx.out / "preparation" / "alpha0_anomaly_adm_check.json")["newly_accepted_case_ids"]
        allrep = {m: ev.canonical_traces(p, "case:concept:name", "concept:name") for m, p in logs.items()}
        cpn_df = pd.read_csv(logs["CPN"], dtype=str, keep_default_na=False)
        meta = cpn_df.groupby("case:concept:name", sort=False).agg(rl_success=("rl_success", "first"), rl_steps=("rl_steps", "first"),
                                                                    rejected=("pn_gate_rejected_episodes", "first"), type=("type", "first"))
        rows15 = []
        for c in ids15:
            g_ = tuple(gt[c])
            rows15.append({"case_id": c, "split": meta.loc[c, "type"], "anomaly_trace": " ".join(an[c]), "ground_truth": " ".join(g_),
                           "A_repaired": " ".join(allrep["A"][c]), "C_baseline_repaired": " ".join(allrep["C"][c]),
                           "CPN_repaired": " ".join(allrep["CPN"][c]), "A_correct": tuple(allrep["A"][c]) == g_,
                           "C_baseline_correct": tuple(allrep["C"][c]) == g_, "CPN_correct": tuple(allrep["CPN"][c]) == g_,
                           "CPN_unrepaired": tuple(allrep["CPN"][c]) == tuple(an[c]), "CPN_rl_steps": meta.loc[c, "rl_steps"],
                           "CPN_gate_rejected_episodes": meta.loc[c, "rejected"], "anomaly_petri": gate.accepts(tuple(an[c])),
                           "anomaly_W2_ADM_normal": adm_normal(tuple(an[c]), actset, filtered, nbgs)})
        t15 = pd.DataFrame(rows15)
        t15.to_csv(out / "newly_accepted_15_gate_check.csv", index=False)
        s15 = {"n": len(t15), "A_correct": int(t15.A_correct.sum()), "CPN_correct": int(t15.CPN_correct.sum()),
               "CPN_unrepaired": int(t15.CPN_unrepaired.sum()), "anomaly_petri_accepts": int(t15.anomaly_petri.sum()),
               "gate_rejections_on_these": int(t15.CPN_gate_rejected_episodes.astype(int).sum())}
        # re-classification of B-CPN disagreements (same definition as concurrency_analysis)
        ind = {frozenset(x) for x in read_json(ctx.out / "concurrency_analysis" / "concurrency_analysis_summary.json")
               ["independence"]["used_independence (both)"]}

        def indep(x, y):
            return x != y and frozenset((x, y)) in ind

        def lnf(w):
            rem, o = list(w), []
            while rem:
                x, i = min((x, i) for i, x in enumerate(rem) if all(indep(rem[j], x) for j in range(i)))
                o.append(x)
                rem.pop(i)
            return tuple(o)

        def lcs(a, b):
            prev = [0] * (len(b) + 1)
            for x in a:
                cur = [0]
                for j, y in enumerate(b, 1):
                    cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
                prev = cur
            return prev[-1]

        rowsd = []
        for c in test_ids:
            b_, p_ = T("B", c), T("CPN", c)
            if b_ == p_:
                continue
            a_ = tuple(an[c])
            acc_p = gate.accepts(p_)
            conc = lnf(b_) == lnf(p_) and acc_p
            cost = lambda t: len(a_) + len(t) - 2 * lcs(a_, t)  # noqa: E731
            cause = ("concurrency-related (ordering only)" if conc else
                     "PN-rejected premature termination" if not acc_p else
                     "alignment tie candidate" if cost(p_) == cost(b_) else
                     "model-only behavior" if (b_ not in obs or p_ not in obs) else "observed normal variant difference")
            rowsd.append({"case_id": c, "anomaly_trace": " ".join(a_), "ground_truth": " ".join(gt[c]), "B_repaired": " ".join(b_),
                          "CPN_repaired": " ".join(p_), "B_correct": b_ == tuple(gt[c]), "CPN_correct": p_ == tuple(gt[c]),
                          "CPN_petri": acc_p, "CPN_W2_ADM_normal": adm_normal(p_, actset, filtered, nbgs),
                          "CPN_kind": kind(p_, c), "in_previous_77": c in set(dis.case_id), "in_previous_D40": c in set(d40),
                          "primary_cause": cause})
        dd = pd.DataFrame(rowsd)
        dd.to_csv(out / "bc_disagreement_after_gate.csv", index=False)
        recl = {"disagreements": len(dd), "previous": 77, "causes": dd.primary_cause.value_counts().to_dict() if len(dd) else {},
                "still_in_previous_D40": int(dd.in_previous_D40.sum()) if len(dd) else 0,
                "new_disagreements_not_in_previous_77": int((~dd.in_previous_77).sum()) if len(dd) else 0}
        # timing (3 reps) / training
        st = read_json(next(run.glob("*_stats.json")))
        reps = sorted((ctx.out / PNGATE_DIR / "test_timing").glob("rep*_CPN"))
        tim = None
        if reps:
            sm = [read_json(r / "timing_summary.json")["timing_summary"] for r in reps]
            means = [s["test_case_ms_rl_mean"] for s in sm]
            meds = [s["test_case_ms_rl_median"] for s in sm]
            sd = stt.stdev(means) if len(means) > 1 else 0.0
            tim = {"reps": len(sm), "rl_n": sm[0]["test_case_ms_rl_n"], "rl_mean_avg": round(stt.mean(means), 2),
                   "rl_mean_sd": round(sd, 2), "rl_mean_cv_pct": round(sd / stt.mean(means) * 100, 2),
                   "rl_mean_min": round(min(means), 2), "rl_mean_max": round(max(means), 2),
                   "rl_median_avg": round(stt.mean(meds), 2), "per_rep_rl_mean": means, "per_rep_rl_median": meds,
                   "seq_wall_per_case_avg": round(stt.mean(s["test_phase_wall_ms_per_case"] for s in sm), 2),
                   "official_mp_test_per_case_ms": st.get("test_per_case_ms"),
                   "replay_trace_agreement": [read_json(r / "timing_summary.json")["trace_agreement_with_official"] for r in reps]}
            pd.DataFrame([{k: v for k, v in tim.items()}]).to_csv(ctx.out / PNGATE_DIR / "test_timing" / "repeat_summary_3reps.csv", index=False)
        evals = {"emr": emr, "training_min": {"CPN": round(st["training_time_sec"] / 60, 2)},
                 "gate_stats_official": {k: st.get(k) for k in ["cases_with_gate_rejection", "gate_rejected_episodes_total"]},
                 "timing_CPN": tim, "D40": s40, "newly_accepted_15": s15, "unchanged_rule": "official EMR = evaluate (exact match)",
                 "checks": rec.checks}
        write_json(out / "cpn_evaluation_summary.json", evals)
        write_json(out / "cpn_agreement_summary.json", agr | {"reclassification_B_CPN": recl})
        print(json.dumps({"emr": {m: v["emr"] for m, v in emr.items()}, "pairs": agr["pairs_exact"], "D40": s40, "15": s15,
                          "recl": recl, "timing": tim}, ensure_ascii=False, indent=1, default=str))
    return 0 if rec.ok else 1


STAGES = {"discover": stage_discover, "generate": stage_generate, "nbg": stage_nbg, "smoke": stage_smoke,
          "run": stage_run, "evaluate": stage_evaluate, "regression": stage_regression, "all": stage_all,
          "test-timing": stage_test_timing, "test-timing-validate": stage_test_timing_validate,
          "test-timing-b": stage_test_timing_b, "agreement": stage_agreement, "run-b": stage_run_b,
          "loop-bound-screening": stage_loop_bound_screening,
          "credit-loop-bound-sensitivity": stage_credit_loop_bound_sensitivity,
          "pngate-acceptor-check": stage_pngate_acceptor_check, "pngate-offline": stage_pngate_offline,
          "pngate-smoke": stage_pngate_smoke, "run-c-pngate": stage_run_c_pngate,
          "test-timing-pngate": stage_test_timing_pngate, "pngate-evaluate": stage_pngate_evaluate}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICPM-BPI 2027 multi-dataset research pipeline')
    parser.add_argument("--dataset", required=True, choices=list(dc.DATASETS))
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--data-root", dest="data_root", default=None)
    parser.add_argument("--method", choices=["A", "C"])
    parser.add_argument("--run-id", dest="run_id", type=int)
    parser.add_argument("--with-reference", dest="with_reference", action="store_true",
                        help='evaluate: also evaluate the configured reference results (reference_runs)')
    parser.add_argument("--overwrite", action="store_true", help='remove an existing stage folder and run again (runs/ excluded)')
    parser.add_argument("--agent", default=None, help='test-timing: trained agent .pkl (official run)')
    parser.add_argument("--pnml", default=None, help='test-timing-b: Petri net (.pnml) of the official Method B run')
    parser.add_argument("--limit", type=int, default=None, help='run-b smoke: only N test cases (not an official run)')
    parser.add_argument("--repaired-log", dest="repaired_log", default=None,
                        help='test-timing: repaired log CSV of the same official run (*_stats.json required in the same folder)')
    parser.add_argument("--label", default=None, help='test-timing: output folder name (outputs/simulation/<dataset>/test_timing/<label>)')
    parser.add_argument("--noise-threshold", dest="noise_threshold", type=float, default=None,
                        help='Inductive Miner noise_threshold (> 0: PM4Py IMf). If given, outputs go to outputs/simulation/<dataset>/threshold_0pXX/. Not the ADM alpha')
    parser.add_argument("--loop-bound", dest="loop_bound", default=None,
                        help="loop bound 'loop_num,loop_total' for bounded_loop generation (e.g. 2,2)")
    args = parser.parse_args()
    if args.noise_threshold is not None and args.noise_threshold > 0 and os.environ.get("PYTHONHASHSEED") != "0":
        # PM4Py IMf (2.7.23.6) can return different trees in different processes depending on Python's hash
        # order, so PYTHONHASHSEED=0 is required for reproducibility.
        print('[ERROR] --noise-threshold > 0 requires the environment variable PYTHONHASHSEED=0 for reproducibility.')
        sys.exit(1)
    ctx = Context(args)
    sys.exit(STAGES[args.stage](ctx))
