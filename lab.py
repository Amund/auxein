"""Deterministic laboratory for the Auxein v0.5.0 canon.

Experiment JSON describes only external worlds, material parameters and whether a
phase is one explicit causal sequence. The lab never feeds diagnostic truth into
Auxein. A small built-in semantic battery additionally checks the v0.5 boundary
contracts directly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
from typing import Mapping

from auxein import Auxein, Kernel
from worlds import build_world


def read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _readout_counts(readout: object) -> tuple[int, int, int, int]:
    if not isinstance(readout, Mapping):
        raise ValueError("readout must be an object")
    present = readout.get("present")
    future = readout.get("future", [])
    if not isinstance(present, list) or not isinstance(future, list):
        raise ValueError("readout present/future must be lists")
    present_atoms = sum(len(p) for p in present if isinstance(p, list))
    future_atoms = sum(len(p) for p in future if isinstance(p, list))
    return len(present), present_atoms, len(future), future_atoms


def run_experiment(spec: Mapping[str, object]) -> dict[str, object]:
    model = spec.get("model")
    phases = spec.get("phases")
    if not isinstance(model, Mapping):
        raise ValueError("model must be an object")
    if not isinstance(phases, list) or not phases:
        raise ValueError("phases must be a nonempty list")

    network = Auxein(
        dimension=int(model["dimension"]),
        memory=float(model.get("memory", 10.0)),
        eta=float(model.get("eta", 1.0)),
        scalar=str(model.get("scalar", "f64")),
        mode=str(model.get("mode", "geometry")),
        budget=Decimal(str(model.get("budget", 1000))),
    )

    transformations: Counter[str] = Counter()
    context_emissions = 0
    recognised_atoms = 0
    unknown_atoms = 0
    temporal_recognised_atoms = 0
    temporal_unknown_atoms = 0
    present_presentations = 0
    present_atoms = 0
    future_presentations = 0
    future_atoms = 0
    phase_results: list[dict[str, object]] = []
    seed_base = int(spec.get("seed", 0))

    for phase_index, raw_phase in enumerate(phases):
        if not isinstance(raw_phase, Mapping):
            raise ValueError("each phase must be an object")
        if "eta" in raw_phase:
            network.set_eta(float(raw_phase["eta"]))
        if "budget" in raw_phase:
            network.set_budget(budget=Decimal(str(raw_phase["budget"])))
        steps = int(raw_phase["steps"])
        if steps <= 0:
            raise ValueError("phase.steps must be positive")
        world_spec = raw_phase.get("world")
        if not isinstance(world_spec, Mapping):
            raise ValueError("phase.world must be an object")
        world = build_world(world_spec, seed=seed_base + 1009 * phase_index)
        causal = bool(raw_phase.get("causal", False))

        start_summary = network.summary()
        local = Counter()
        last_readout: object = None
        if causal:
            network.begin_sequence()
        try:
            for step in range(steps):
                sample = world.sample(step)
                report = (
                    network.sequence_step(sample.presentation, detailed_report=True)
                    if causal
                    else network.step(sample.presentation, detailed_report=True)
                )
                last_readout = report["readout"]
                pp, pa, fp, fa = _readout_counts(last_readout)
                present_presentations += pp
                present_atoms += pa
                future_presentations += fp
                future_atoms += fa
                local.update(
                    present_presentations=pp,
                    present_atoms=pa,
                    future_presentations=fp,
                    future_atoms=fa,
                )
                for transform in report["transformations"]:
                    transformations[f"{transform['phase']}:{transform['type']}"] += 1
                for layer in report["layer_reports"]:
                    if layer["context_emitted"]:
                        context_emissions += 1
                        local["context_emissions"] += 1
                    recognised_atoms += int(layer["recognised_atom_count"])
                    unknown_atoms += int(layer["unknown_atom_count"])
                for temporal in report["temporal_reports"]:
                    temporal_recognised_atoms += int(temporal["recognised_atom_count"])
                    temporal_unknown_atoms += int(temporal["unknown_atom_count"])
        finally:
            if causal:
                network.end_sequence()

        phase_results.append(
            {
                "name": str(raw_phase.get("name", f"phase-{phase_index}")),
                "steps": steps,
                "causal_sequence": causal,
                **dict(local),
                "before": start_summary,
                "after": network.summary(),
                "last_readout": last_readout,
            }
        )

    return {
        "name": str(spec.get("name", "unnamed")),
        "description": str(spec.get("description", "")),
        "mode": network.mode,
        "context_emissions": context_emissions,
        "present_presentations": present_presentations,
        "present_atoms": present_atoms,
        "future_presentations": future_presentations,
        "future_atoms": future_atoms,
        "recognised_atoms": recognised_atoms,
        "unknown_atoms": unknown_atoms,
        "temporal_recognised_atoms": temporal_recognised_atoms,
        "temporal_unknown_atoms": temporal_unknown_atoms,
        "transformations": dict(sorted(transformations.items())),
        "summary": network.summary(),
        "phases": phase_results,
        "state": network.export_state(),
    }


def semantic_checks() -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # Gain-weighted vertical knowledge: gains 5 and 8 at x=3.
    g = Auxein(dimension=1, memory=10, eta=0, budget=100)
    g.layers[0].cells = [Kernel(1, (1,), 10), Kernel(1, (2,), 10)]
    r = g.step([[3.0]], detailed_report=True)
    center = r["layer_reports"][0]["context_center"][0]
    record("gain-weighted-context", abs(center - 21 / 13) < 1e-12, center)

    # Atomic calls never create private temporal continuity.
    a = Auxein(dimension=1, memory=10, mode="predictive", budget=1000)
    a.layers[0].cells = [Kernel(1, (1,), 0), Kernel(1, (10,), 0)]
    a.step([[1.0]])
    a.step([[10.0]])
    record(
        "atomic-boundary",
        not a.layers[0].temporal_sigma and a.layers[0].previous is None,
        a.export_state()["layers"][0],
    )

    # The same observations inside one explicit sequence do create a relation.
    s = Auxein(dimension=1, memory=10, mode="predictive", budget=1000)
    s.layers[0].cells = [Kernel(1, (1,), 0), Kernel(1, (10,), 0)]
    s.sequence([[[1.0]], [[10.0]]])
    learned = [k.C for k in s.layers[0].temporal_sigma]
    record("explicit-sequence", learned == [(1.0, 10.0)], learned)

    # An atomic singleton may use old temporal knowledge without retaining P.
    p = Auxein(dimension=1, memory=10, eta=0, mode="predictive", budget=100)
    p.layers[0].cells = [Kernel(1, (1,), 0)]
    p.layers[0].temporal_cells = [Kernel(1, (1, 2), 0)]
    future = p.step([[1.0]])["readout"]["future"]
    record("singleton-prediction", future == [[[1.0, [2], 0.0]]] and p.layers[0].previous is None, future)

    # Predictive authority follows relative source gain; same-target paths use max.
    q = Auxein(dimension=1, memory=10, eta=0, mode="predictive", budget=100)
    q.layers[0].cells = [Kernel(1, (2,), 0)]
    q.layers[0].temporal_cells = [
        Kernel(1, (2, 7), 0),
        Kernel(999, (3.9, 8), 77),
        Kernel(1, (3.8, 8), 0),
    ]
    q_future = q.step([[0.25, [2], 0], [0.75, [99], 0]])["readout"]["future"]
    q_weights = {
        atom[1][0]: atom[0]
        for presentation in q_future
        for atom in presentation
        if atom[1][0] != 0
    }
    record(
        "predictive-relative-gain",
        abs(q_weights.get(7.0, 0.0) - 0.25) < 1e-15
        and abs(q_weights.get(8.0, 0.0) - 0.0475) < 1e-15,
        q_future,
    )

    # Present output is directly admissible as a weighted downstream presentation.
    u = Auxein(dimension=1, memory=10, eta=0, budget=100)
    u.layers[0].cells = [Kernel(1, (1,), 0), Kernel(1, (3,), 0)]
    family = u.step([[1.0], [3.0]])["readout"]["present"]
    d = Auxein(dimension=1, memory=10, eta=0, budget=100)
    parsed = d._presentation(family[0])
    record("composable-present", abs(sum(k.W for k in parsed) - 1.0) < 1e-15, family)

    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="experiment JSON files")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    paths = [Path(p) for p in args.paths] or sorted(Path("experiments").glob("*.json"))
    results = [run_experiment(read_json(path)) for path in paths]
    checks = semantic_checks()
    checks_passed = all(bool(item["passed"]) for item in checks)
    out = {
        "canon": "0.5.0",
        "experiment_count": len(results),
        "semantic_checks_passed": checks_passed,
        "semantic_checks": checks,
        "experiments": results,
    }
    Path(args.output).write_text(
        json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "canon": "0.5.0",
                "experiments": len(results),
                "output": args.output,
                "semantic_checks": {"count": len(checks), "passed": checks_passed},
                "final_cells": {
                    r["name"]: {
                        "geometry": r["summary"]["cells_per_layer"],
                        "predictive_private": r["summary"]["temporal_cells_per_layer"],
                    }
                    for r in results
                },
            },
            indent=2,
        )
    )
    if not checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
