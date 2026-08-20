"""Small deterministic laboratory for the Auxein v0.3.0 canon.

Experiment JSON files describe only external worlds and execution parameters.
The laboratory records geometric context, local recurrence, temporal recurrence,
readout and material growth. Diagnostic truth never enters Auxein.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
from typing import Mapping, Sequence

from auxein import Auxein
from worlds import build_world


def read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _readout_counts(readout: object) -> tuple[int, int]:
    if isinstance(readout, list):
        return len(readout), 0
    if isinstance(readout, Mapping):
        concepts = readout.get("concepts")
        sequences = readout.get("sequences")
        if not isinstance(concepts, list) or not isinstance(sequences, list):
            raise ValueError("temporal readout must contain concepts and sequences lists")
        return len(concepts), len(sequences)
    raise ValueError("invalid readout")



def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("oracle vectors have incompatible dimensions")
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


class OracleDecoder:
    """External diagnostic decoder for labelled toy worlds.

    Labels and expectations are laboratory truth only.  They are never passed
    to Auxein and therefore cannot influence cognition or growth.
    """

    def __init__(self, spec: Mapping[str, object], *, dimension: int) -> None:
        raw_labels = spec.get("labels")
        if not isinstance(raw_labels, Mapping) or not raw_labels:
            raise ValueError("oracle.labels must be a nonempty object")
        self.labels: dict[str, tuple[float, ...]] = {}
        for name, raw_vector in raw_labels.items():
            if not isinstance(name, str) or not name:
                raise ValueError("oracle label names must be nonempty strings")
            if not isinstance(raw_vector, list) or len(raw_vector) != dimension:
                raise ValueError(f"oracle.labels[{name!r}] must have dimension {dimension}")
            self.labels[name] = tuple(float(x) for x in raw_vector)
        self.max_distance2 = float(spec.get("max_distance2", float("inf")))
        if self.max_distance2 < 0.0:
            raise ValueError("oracle.max_distance2 must be nonnegative")

    def decode_vector(self, vector: Sequence[float]) -> dict[str, object]:
        distances = sorted(
            (_dist2(vector, anchor), name) for name, anchor in self.labels.items()
        )
        best_distance, best_name = distances[0]
        ambiguous = len(distances) > 1 and distances[1][0] == best_distance
        label = None if ambiguous or best_distance > self.max_distance2 else best_name
        return {"label": label, "distance2": best_distance}

    def decode_readout(self, readout: object) -> dict[str, object]:
        if not isinstance(readout, Mapping):
            raise ValueError("oracle decoder requires temporal readout")
        raw_concepts = readout.get("concepts")
        raw_sequences = readout.get("sequences")
        if not isinstance(raw_concepts, list) or not isinstance(raw_sequences, list):
            raise ValueError("temporal readout must contain concepts and sequences lists")

        concepts: list[dict[str, object]] = []
        for item in raw_concepts:
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError("invalid concept readout item")
            input_decoded = self.decode_vector(item[1])
            recognised_decoded = self.decode_vector(item[2])
            concepts.append({
                "input": input_decoded["label"],
                "recognised": recognised_decoded["label"],
                "input_distance2": input_decoded["distance2"],
                "recognised_distance2": recognised_decoded["distance2"],
            })

        sequences: list[dict[str, object]] = []
        for item in raw_sequences:
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError("invalid sequence readout item")
            inputs = item[1]
            recognised = item[2]
            if (
                not isinstance(inputs, list)
                or len(inputs) != 2
                or not isinstance(recognised, list)
                or len(recognised) != 2
            ):
                raise ValueError("invalid temporal endpoints")
            input_decoded = [self.decode_vector(v) for v in inputs]
            recognised_decoded = [self.decode_vector(v) for v in recognised]
            sequences.append({
                "input": [v["label"] for v in input_decoded],
                "recognised": [v["label"] for v in recognised_decoded],
                "input_distance2": [v["distance2"] for v in input_decoded],
                "recognised_distance2": [v["distance2"] for v in recognised_decoded],
            })

        concepts.sort(key=lambda item: (str(item["input"]), str(item["recognised"])))
        sequences.sort(
            key=lambda item: (
                tuple(str(x) for x in item["input"]),
                tuple(str(x) for x in item["recognised"]),
            )
        )
        return {"concepts": concepts, "sequences": sequences}

    @staticmethod
    def semantic(decoded: Mapping[str, object]) -> dict[str, object]:
        concepts = decoded.get("concepts")
        sequences = decoded.get("sequences")
        if not isinstance(concepts, list) or not isinstance(sequences, list):
            raise ValueError("invalid decoded oracle readout")
        return {
            "concepts": [
                {"input": item["input"], "recognised": item["recognised"]}
                for item in concepts
            ],
            "sequences": [
                {"input": item["input"], "recognised": item["recognised"]}
                for item in sequences
            ],
        }

    def decode_final_state(
        self, state: Mapping[str, object], *, dimension: int, layer_index: int
    ) -> dict[str, object]:
        layers = state.get("layers")
        if not isinstance(layers, list) or layer_index >= len(layers):
            raise ValueError("oracle final-state layer is missing")
        layer = layers[layer_index]
        if not isinstance(layer, Mapping):
            raise ValueError("invalid layer state")
        raw_cells = layer.get("cells")
        raw_temporal = layer.get("temporal_cells")
        if not isinstance(raw_cells, list) or not isinstance(raw_temporal, list):
            raise ValueError("oracle requires v0.3 temporal layer state")

        concepts: list[str | None] = []
        for cell in raw_cells:
            if not isinstance(cell, Mapping) or not isinstance(cell.get("C"), list):
                raise ValueError("invalid geometric CELL state")
            concepts.append(self.decode_vector(cell["C"])["label"])

        sequences: list[list[str | None]] = []
        for cell in raw_temporal:
            if not isinstance(cell, Mapping) or not isinstance(cell.get("C"), list):
                raise ValueError("invalid temporal CELL state")
            center = cell["C"]
            if len(center) != 2 * dimension:
                raise ValueError("invalid temporal CELL dimension")
            sequences.append([
                self.decode_vector(center[:dimension])["label"],
                self.decode_vector(center[dimension:])["label"],
            ])

        concepts.sort(key=str)
        sequences.sort(key=lambda item: tuple(str(x) for x in item))
        return {"concept_cells": concepts, "sequence_cells": sequences}

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
        universe=str(model.get("universe", "auxein")),
        budget=Decimal(str(model.get("budget", 1000))),
    )
    transformations: Counter[str] = Counter()
    context_emissions = 0
    recognised_atoms = 0
    unknown_atoms = 0
    temporal_recognised_atoms = 0
    temporal_unknown_atoms = 0
    concept_readout_items = 0
    sequence_readout_items = 0
    phase_results: list[dict[str, object]] = []
    seed_base = int(spec.get("seed", 0))
    oracle_spec = spec.get("oracle")
    if oracle_spec is not None and not isinstance(oracle_spec, Mapping):
        raise ValueError("oracle must be an object")
    oracle = (
        None
        if oracle_spec is None
        else OracleDecoder(oracle_spec, dimension=network.dimension)
    )
    oracle_checks: list[dict[str, object]] = []
    oracle_passed = True

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
        raw_checks = raw_phase.get("checks", [])
        if not isinstance(raw_checks, list):
            raise ValueError("phase.checks must be a list")
        checks_by_step: dict[int, Mapping[str, object]] = {}
        for raw_check in raw_checks:
            if not isinstance(raw_check, Mapping):
                raise ValueError("each phase check must be an object")
            check_step = int(raw_check["step"])
            if check_step < 0 or check_step >= steps or check_step in checks_by_step:
                raise ValueError("phase check step must be unique and within the phase")
            checks_by_step[check_step] = raw_check
        if checks_by_step and oracle is None:
            raise ValueError("phase checks require a top-level oracle")

        start_summary = network.summary()
        local_context = 0
        local_concepts = 0
        local_sequences = 0
        local_oracle_checks: list[dict[str, object]] = []
        for step in range(steps):
            sample = world.sample(step)
            report = network.step(sample.presentation, detailed_report=True)
            concept_count, sequence_count = _readout_counts(report["readout"])
            concept_readout_items += concept_count
            sequence_readout_items += sequence_count
            local_concepts += concept_count
            local_sequences += sequence_count
            for transform in report["transformations"]:
                transformations[f"{transform['phase']}:{transform['type']}"] += 1
            for layer in report["layer_reports"]:
                if layer["context_emitted"]:
                    context_emissions += 1
                    local_context += 1
                recognised_atoms += int(layer["recognised_atom_count"])
                unknown_atoms += int(layer["unknown_atom_count"])
            for temporal in report["temporal_reports"]:
                temporal_recognised_atoms += int(temporal["recognised_atom_count"])
                temporal_unknown_atoms += int(temporal["unknown_atom_count"])

            if step in checks_by_step:
                assert oracle is not None
                raw_check = checks_by_step[step]
                expected = raw_check.get("expected")
                if not isinstance(expected, Mapping):
                    raise ValueError("phase check expected must be an object")
                decoded = oracle.decode_readout(report["readout"])
                semantic = oracle.semantic(decoded)
                passed = semantic == expected
                oracle_passed = oracle_passed and passed
                check_result = {
                    "phase": str(raw_phase.get("name", f"phase-{phase_index}")),
                    "step": step,
                    "expected": expected,
                    "actual": semantic,
                    "decoded": decoded,
                    "passed": passed,
                }
                oracle_checks.append(check_result)
                local_oracle_checks.append(check_result)
        phase_results.append(
            {
                "name": str(raw_phase.get("name", f"phase-{phase_index}")),
                "steps": steps,
                "context_emissions": local_context,
                "concept_readout_items": local_concepts,
                "sequence_readout_items": local_sequences,
                "before": start_summary,
                "after": network.summary(),
                "oracle_checks": local_oracle_checks,
            }
        )

    final_state = network.export_state()
    oracle_result: dict[str, object] | None = None
    if oracle is not None:
        final_result: dict[str, object] | None = None
        assert isinstance(oracle_spec, Mapping)
        raw_final = oracle_spec.get("final")
        if raw_final is not None:
            if not isinstance(raw_final, Mapping):
                raise ValueError("oracle.final must be an object")
            layer_index = int(raw_final.get("layer", 0))
            expected_final = raw_final.get("expected")
            if not isinstance(expected_final, Mapping):
                raise ValueError("oracle.final.expected must be an object")
            actual_final = oracle.decode_final_state(
                final_state, dimension=network.dimension, layer_index=layer_index
            )
            final_passed = actual_final == expected_final
            oracle_passed = oracle_passed and final_passed
            final_result = {
                "layer": layer_index,
                "expected": expected_final,
                "actual": actual_final,
                "passed": final_passed,
            }
        oracle_result = {
            "passed": oracle_passed,
            "check_count": len(oracle_checks),
            "checks": oracle_checks,
            "final": final_result,
        }

    return {
        "name": str(spec.get("name", "unnamed")),
        "description": str(spec.get("description", "")),
        "mode": network.mode,
        "context_emissions": context_emissions,
        "readout_items": concept_readout_items + sequence_readout_items,
        "concept_readout_items": concept_readout_items,
        "sequence_readout_items": sequence_readout_items,
        "recognised_atoms": recognised_atoms,
        "unknown_atoms": unknown_atoms,
        "temporal_recognised_atoms": temporal_recognised_atoms,
        "temporal_unknown_atoms": temporal_unknown_atoms,
        "transformations": dict(sorted(transformations.items())),
        "summary": network.summary(),
        "phases": phase_results,
        "oracle": oracle_result,
        "state": final_state,
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
    oracle_experiments = [r for r in results if r.get("oracle") is not None]
    oracle_passed = all(bool(r["oracle"]["passed"]) for r in oracle_experiments)
    out = {
        "canon": "0.3.0",
        "experiment_count": len(results),
        "oracle_experiment_count": len(oracle_experiments),
        "oracle_passed": oracle_passed,
        "experiments": results,
    }
    Path(args.output).write_text(
        json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "experiments": len(results),
                "output": args.output,
                "oracle": {
                    "experiments": len(oracle_experiments),
                    "passed": oracle_passed,
                },
                "final_cells": {
                    r["name"]: {
                        "geometry": r["summary"]["cells_per_layer"],
                        "temporal": r["summary"]["temporal_cells_per_layer"],
                    }
                    for r in results
                },
            },
            indent=2,
        )
    )
    if not oracle_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
