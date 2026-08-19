"""Small deterministic laboratory for the true Auxein v0.2.0 canon.

Experiment JSON files describe only external worlds and execution parameters.
The laboratory records contextual emission, horizontal recurrence, readout and
material growth.  It measures only the current contextual-recursion contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
from typing import Mapping

from auxein import Auxein
from worlds import build_world


def read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


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
        universe=str(model.get("universe", "auxein")),
        budget=Decimal(str(model.get("budget", 1000))),
    )
    transformations: Counter[str] = Counter()
    context_emissions = 0
    recognised_atoms = 0
    unknown_atoms = 0
    readout_items = 0
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
        world_spec = raw_phase.get("world")
        if not isinstance(world_spec, Mapping):
            raise ValueError("phase.world must be an object")
        world = build_world(world_spec, seed=seed_base + 1009 * phase_index)
        start_summary = network.summary()
        local_context = 0
        local_readout = 0
        for step in range(steps):
            sample = world.sample(step)
            report = network.step(sample.presentation, detailed_report=True)
            readout_items += len(report["readout"])
            local_readout += len(report["readout"])
            for transform in report["transformations"]:
                transformations[f"{transform['phase']}:{transform['type']}"] += 1
            for layer in report["layer_reports"]:
                if layer["context_emitted"]:
                    context_emissions += 1
                    local_context += 1
                recognised_atoms += int(layer["recognised_atom_count"])
                unknown_atoms += int(layer["unknown_atom_count"])
        phase_results.append({
            "name": str(raw_phase.get("name", f"phase-{phase_index}")),
            "steps": steps,
            "context_emissions": local_context,
            "readout_items": local_readout,
            "before": start_summary,
            "after": network.summary(),
        })

    return {
        "name": str(spec.get("name", "unnamed")),
        "description": str(spec.get("description", "")),
        "context_emissions": context_emissions,
        "readout_items": readout_items,
        "recognised_atoms": recognised_atoms,
        "unknown_atoms": unknown_atoms,
        "transformations": dict(sorted(transformations.items())),
        "summary": network.summary(),
        "phases": phase_results,
        "state": network.export_state(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="experiment JSON files")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    paths = [Path(p) for p in args.paths]
    if not paths:
        paths = sorted(Path("experiments").glob("*.json"))
    results = [run_experiment(read_json(path)) for path in paths]
    out = {"canon": "0.2.0", "experiment_count": len(results), "experiments": results}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "experiments": len(results),
        "output": args.output,
        "final_cells": {r["name"]: r["summary"]["cells_per_layer"] for r in results},
    }, indent=2))


if __name__ == "__main__":
    main()
