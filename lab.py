"""A small experimental laboratory for the Auxein reference engine.

One JSON experiment expands to one or more deterministic trials.  A trial is
an explicit sequence of phases.  Worlds emit external truth; Auxein never sees
it.  The laboratory uses only the public engine API and writes one JSON object
per trial.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

from auxein import Auxein, CellIdentity, ScalarFootprintMaintenance, StepReport
from worlds import Sample, build_world, derive_phase_seed


SWEEP_FIELDS = ("budget", "memory", "scalar", "seed", "dimension")


def _exact_keys(mapping: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return out


def _budget_decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        out = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a decimal number") from exc
    if not out.is_finite() or out < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return out


def read_json_object(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_float=Decimal)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    if not path.name:
        raise ValueError("save path must designate a file")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _jsonable(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ModelSpec:
    dimension: int
    scalar: str
    memory_half_life: float
    budget: Decimal


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    name: str
    steps: int
    world: Mapping[str, object]
    budget: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    name: str
    x: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TrialSpec:
    index: int
    model: ModelSpec
    seed: int
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    cells: int
    layers: int
    maintenance: int
    capital: float


@dataclass(frozen=True, slots=True)
class LayerObservation:
    layer_index: int
    input_relevance: float
    winner: CellIdentity | None
    recognition: float
    emitted_relevance: float
    distance: float | None
    squared_distance: float | None
    normalized_distance: float | None


@dataclass(frozen=True, slots=True)
class StepObservation:
    layers: tuple[LayerObservation, ...]


def snapshot_network(network: Auxein) -> StepSnapshot:
    return StepSnapshot(
        cells=sum(len(layer.cells) for layer in network._layers),
        layers=len(network._layers),
        maintenance=network.maintenance_units(),
        capital=network.geometric_capital,
    )


def observe_network(network: Auxein, input_value: Sequence[float]) -> StepObservation:
    """Read the current hierarchy without mutating it.

    The observation mirrors the public layer-to-layer presentation performed by
    ``Auxein.step``.  It is intentionally computed before the causal mutation so
    distances refer to the prototypes that actually competed for this sample.
    """

    value = list(map(float, input_value))
    relevance = 1.0
    observations: list[LayerObservation] = []
    for layer_index, layer in enumerate(network._layers):
        read = layer.prepare(value, relevance)
        distance: float | None = None
        squared_distance: float | None = None
        normalized_distance: float | None = None
        if read.winner_slot is not None and read.winner_read is not None:
            winner = layer.cells[read.winner_slot]
            distance = math.dist(value, winner.center)
            squared_distance = distance * distance
            normalized_distance = math.sqrt(
                math.fsum(component * component for component in read.winner_read.error)
            )
        observations.append(
            LayerObservation(
                layer_index=layer_index,
                input_relevance=read.input_relevance,
                winner=read.winner_identity,
                recognition=(
                    read.winner_read.recognition if read.winner_read is not None else 0.0
                ),
                emitted_relevance=read.emission.relevance,
                distance=distance,
                squared_distance=squared_distance,
                normalized_distance=normalized_distance,
            )
        )
        value = list(read.emission.value)
        relevance = read.emission.relevance
    return StepObservation(tuple(observations))


def _percentile(sorted_values: Sequence[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def distribution(values: Iterable[float]) -> dict[str, object]:
    data = sorted(float(value) for value in values)
    if not data:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(data),
        "min": data[0],
        "p25": _percentile(data, 0.25),
        "median": _percentile(data, 0.5),
        "p75": _percentile(data, 0.75),
        "max": data[-1],
        "mean": math.fsum(data) / len(data),
    }


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(math.fsum(component * component for component in value))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    denominator = _norm(left) * _norm(right)
    if denominator <= 0.0:
        return None
    return _dot(left, right) / denominator


def _pairwise_geometry(layer: object) -> dict[str, object]:
    cells = list(layer.cells)
    radius = layer.geometry.radius
    distances: list[float] = []
    normalized_distances: list[float] = []
    nearest: list[float] = []
    axes = [list(cell.plus.mean - cell.minus.mean) for cell in cells]
    axis_cosines: list[float] = []
    for left_index, left in enumerate(cells):
        local: list[float] = []
        for right_index in range(left_index + 1, len(cells)):
            distance = math.dist(left.center, cells[right_index].center)
            distances.append(distance)
            local.append(distance)
            if radius > 0.0:
                normalized_distances.append(distance / radius)
            cosine = _cosine(axes[left_index], axes[right_index])
            if cosine is not None:
                axis_cosines.append(cosine)
        if left_index > 0:
            local.extend(math.dist(left.center, cells[index].center) for index in range(left_index))
        if local:
            nearest.append(min(local))
    return {
        "center_distance_distribution": distribution(distances),
        "normalized_center_distance_distribution": distribution(normalized_distances),
        "nearest_center_distance_distribution": distribution(nearest),
        "internal_axis_cosine_distribution": distribution(axis_cosines),
    }


def _terminal_bud_state(layer_index: int, layer: object, registry: "IdentityRegistry") -> dict[str, object] | None:
    bud = layer.bud
    if bud is None:
        return None
    owners: list[dict[str, object]] = []
    active: list[tuple[float, list[float]]] = []
    for identity, owner in bud.owners.items():
        weight, distinction = owner.distinction
        vector = list(distinction)
        label = registry.label(layer_index, identity)
        owners.append({
            "identity": label,
            "weight": weight,
            "distinction": vector,
            "distinction_norm": _norm(vector),
            "power": weight * _dot(vector, vector),
        })
        if weight > 0.0 and _norm(vector) > 0.0:
            active.append((weight, vector))
    dots: list[float] = []
    cosines: list[float] = []
    weighted_terms: list[float] = []
    total_weight = math.fsum(weight for weight, _ in active)
    for left in range(len(active)):
        left_weight, left_vector = active[left]
        for right in range(left + 1, len(active)):
            right_weight, right_vector = active[right]
            dot = _dot(left_vector, right_vector)
            dots.append(dot)
            cosine = _cosine(left_vector, right_vector)
            if cosine is not None:
                cosines.append(cosine)
            if total_weight > 0.0:
                weighted_terms.append(2.0 * left_weight * right_weight * dot / total_weight)
    return {
        "split_gain": bud.split_gain,
        "concordance": bud.concordance,
        "plus_mass": bud.plus.W,
        "minus_mass": bud.minus.W,
        "plus_mean": list(bud.plus.mean),
        "minus_mean": list(bud.minus.mean),
        "owners": owners,
        "active_owner_count": len(active),
        "owner_dot_distribution": distribution(dots),
        "owner_cosine_distribution": distribution(cosines),
        "pair_contribution_distribution": distribution(weighted_terms),
        "pair_contribution_sum": math.fsum(weighted_terms),
    }


@dataclass(slots=True)
class IdentityRegistry:
    labels: dict[tuple[int, CellIdentity], str] = field(default_factory=dict)
    next_by_layer: dict[int, int] = field(default_factory=dict)

    def label(self, layer_index: int, identity: CellIdentity | None) -> str | None:
        if identity is None:
            return None
        key = (layer_index, identity)
        label = self.labels.get(key)
        if label is None:
            serial = self.next_by_layer.get(layer_index, 0) + 1
            self.next_by_layer[layer_index] = serial
            label = f"L{layer_index}:C{serial}"
            self.labels[key] = label
        return label


def summarize_layer_state(
    network: Auxein,
    registry: IdentityRegistry,
    *,
    ages: Mapping[str, int] | None = None,
    include_cells: bool = False,
) -> list[dict[str, object]]:
    layers: list[dict[str, object]] = []
    for layer_index, layer in enumerate(network._layers):
        masses = [cell.mass for cell in layer.cells]
        split_gains = [cell.split_gain for cell in layer.cells]
        state: dict[str, object] = {
            "layer": layer_index,
            "cells": len(layer.cells),
            "geometry_mean": list(layer.geometry.mean),
            "geometry_radius": layer.geometry.radius,
            "capital": layer.capital,
            "mass_distribution": distribution(masses),
            "split_gain_distribution": distribution(split_gains),
            "pairwise_geometry": _pairwise_geometry(layer),
            "terminal_bud": _terminal_bud_state(layer_index, layer, registry),
        }
        if include_cells:
            cells: list[dict[str, object]] = []
            for cell in layer.cells:
                label = registry.label(layer_index, cell.identity)
                assert label is not None
                parent_mean = cell.parent_mean
                delta_plus = cell.plus.mean - parent_mean
                delta_minus = cell.minus.mean - parent_mean
                internal_axis = cell.plus.mean - cell.minus.mean
                item: dict[str, object] = {
                    "identity": label,
                    "center": list(cell.center),
                    "mass": cell.mass,
                    "split_gain": cell.split_gain,
                    "parent_mean": list(parent_mean),
                    "delta_plus": list(delta_plus),
                    "delta_minus": list(delta_minus),
                    "internal_axis": list(internal_axis),
                    "internal_axis_norm": _norm(internal_axis),
                }
                if ages is not None:
                    item["age"] = ages.get(label)
                cells.append(item)
            state["cell_state"] = cells
        layers.append(state)
    return layers


@dataclass(slots=True)
class Metrics:
    registry: IdentityRegistry
    steps: int = 0
    transformations: Counter[str] = field(default_factory=Counter)
    cells_sum: int = 0
    cells_min: int | None = None
    cells_max: int = 0
    layers_sum: int = 0
    layers_min: int | None = None
    layers_max: int = 0
    maintenance_sum: int = 0
    maintenance_min: int | None = None
    maintenance_max: int = 0
    capital_sum: float = 0.0
    no_layer_steps: int = 0
    vertical_gain_sum: float = 0.0
    vertical_concordance_sum: float = 0.0
    layer_input_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    layer_recognition_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    layer_emission_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    layer_samples: Counter[int] = field(default_factory=Counter)
    quantization_distance_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    quantization_squared_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    normalized_distance_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    quantization_samples: Counter[int] = field(default_factory=Counter)
    winner_counts: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    previous_winner: dict[int, str | None] = field(default_factory=dict)
    winner_transitions: Counter[int] = field(default_factory=Counter)
    winner_changes: Counter[int] = field(default_factory=Counter)
    truth_winners: dict[tuple[str, int], dict[str, Counter[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    presented_depth_sum: int = 0
    active_depth_sum: int = 0
    emitted_depth_sum: int = 0
    active_depth_max: int = 0
    active_depth_histogram: Counter[int] = field(default_factory=Counter)
    budgeted_steps: int = 0
    budget_utilization_sum: float = 0.0
    full_budget_steps: int = 0
    near_full_budget_steps: int = 0

    def update(
        self,
        network: Auxein,
        report: StepReport,
        sample: Sample,
        snapshot: StepSnapshot | None = None,
        observation: StepObservation | None = None,
    ) -> None:
        self.steps += 1
        self.transformations.update(record.kind for record in report.transformations)
        snapshot = snapshot_network(network) if snapshot is None else snapshot
        cells = snapshot.cells
        layers = snapshot.layers
        maintenance = snapshot.maintenance
        capital = snapshot.capital
        self.cells_sum += cells
        self.cells_min = cells if self.cells_min is None else min(self.cells_min, cells)
        self.cells_max = max(self.cells_max, cells)
        self.layers_sum += layers
        self.layers_min = layers if self.layers_min is None else min(self.layers_min, layers)
        self.layers_max = max(self.layers_max, layers)
        self.maintenance_sum += maintenance
        self.maintenance_min = maintenance if self.maintenance_min is None else min(self.maintenance_min, maintenance)
        self.maintenance_max = max(self.maintenance_max, maintenance)
        self.capital_sum += capital
        if layers == 0:
            self.no_layer_steps += 1
        self.vertical_gain_sum += report.vertical_gain
        self.vertical_concordance_sum += report.vertical_concordance

        if report.budget_units > 0:
            utilization = report.maintenance_units / report.budget_units
            self.budgeted_steps += 1
            self.budget_utilization_sum += utilization
            if report.maintenance_units == report.budget_units:
                self.full_budget_steps += 1
            if utilization >= 0.99:
                self.near_full_budget_steps += 1

        self.presented_depth_sum += len(report.layer_reports)
        active_depth = sum(item.input_relevance > 0.0 for item in report.layer_reports)
        emitted_depth = sum(item.emitted_relevance > 0.0 for item in report.layer_reports)
        self.active_depth_sum += active_depth
        self.emitted_depth_sum += emitted_depth
        self.active_depth_max = max(self.active_depth_max, active_depth)
        self.active_depth_histogram[active_depth] += 1

        categorical_truth = {
            key: str(value)
            for key, value in sample.truth.items()
            if isinstance(value, (str, int, bool))
        }
        observations = {
            item.layer_index: item for item in observation.layers
        } if observation is not None else {}
        for layer_report in report.layer_reports:
            layer_index = layer_report.layer_index
            self.layer_samples[layer_index] += 1
            self.layer_input_sum[layer_index] += layer_report.input_relevance
            self.layer_recognition_sum[layer_index] += layer_report.recognition
            self.layer_emission_sum[layer_index] += layer_report.emitted_relevance
            winner = self.registry.label(layer_index, layer_report.winner)
            previous = self.previous_winner.get(layer_index)
            if previous is not None and winner is not None:
                self.winner_transitions[layer_index] += 1
                if previous != winner:
                    self.winner_changes[layer_index] += 1
            self.previous_winner[layer_index] = winner
            if winner is None:
                continue
            self.winner_counts[layer_index][winner] += 1
            observed = observations.get(layer_index)
            if observed is not None and observed.distance is not None:
                self.quantization_samples[layer_index] += 1
                self.quantization_distance_sum[layer_index] += observed.distance
                assert observed.squared_distance is not None
                assert observed.normalized_distance is not None
                self.quantization_squared_sum[layer_index] += observed.squared_distance
                self.normalized_distance_sum[layer_index] += observed.normalized_distance
            for field_name, truth_value in categorical_truth.items():
                self.truth_winners[(field_name, layer_index)][truth_value][winner] += 1

    def summary(self, network: Auxein, elapsed: float) -> dict[str, object]:
        steps = self.steps
        truth_purity: list[dict[str, object]] = []
        for (field_name, layer_index), categories in sorted(self.truth_winners.items()):
            samples = sum(sum(winners.values()) for winners in categories.values())
            category_consistent = sum(
                max(winners.values(), default=0) for winners in categories.values()
            )
            by_winner: dict[str, Counter[str]] = defaultdict(Counter)
            for truth_value, winners in categories.items():
                for winner, count in winners.items():
                    by_winner[winner][truth_value] += count
            winner_pure = sum(
                max(truth_counts.values(), default=0)
                for truth_counts in by_winner.values()
            )
            truth_totals = {
                truth_value: sum(winners.values())
                for truth_value, winners in categories.items()
            }
            winner_totals = {
                winner: sum(truth_counts.values())
                for winner, truth_counts in by_winner.items()
            }
            truth_entropy = -math.fsum(
                (count / samples) * math.log(count / samples)
                for count in truth_totals.values() if count > 0
            ) if samples else 0.0
            winner_entropy = -math.fsum(
                (count / samples) * math.log(count / samples)
                for count in winner_totals.values() if count > 0
            ) if samples else 0.0
            mutual_information = 0.0
            if samples:
                for truth_value, winners in categories.items():
                    for winner, count in winners.items():
                        if count <= 0:
                            continue
                        joint = count / samples
                        truth_probability = truth_totals[truth_value] / samples
                        winner_probability = winner_totals[winner] / samples
                        mutual_information += joint * math.log(
                            joint / (truth_probability * winner_probability)
                        )
            normalizer = math.sqrt(truth_entropy * winner_entropy)
            truth_purity.append(
                {
                    "field": field_name,
                    "layer": layer_index,
                    "samples": samples,
                    "categories": len(categories),
                    "winners": len(by_winner),
                    "category_consistency": (
                        category_consistent / samples if samples else None
                    ),
                    "winner_purity": winner_pure / samples if samples else None,
                    "truth_entropy_nats": truth_entropy,
                    "winner_entropy_nats": winner_entropy,
                    "mutual_information_nats": mutual_information,
                    "normalized_mutual_information": (
                        mutual_information / normalizer if normalizer > 0.0 else 0.0
                    ),
                    "contingency": {
                        truth_value: dict(winners)
                        for truth_value, winners in categories.items()
                    },
                }
            )

        layer_activity: list[dict[str, object]] = []
        for layer_index in sorted(self.layer_samples):
            count = self.layer_samples[layer_index]
            winner_counts = self.winner_counts[layer_index]
            winner_samples = sum(winner_counts.values())
            probabilities = [value / winner_samples for value in winner_counts.values()] if winner_samples else []
            entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)
            occupied = len(winner_counts)
            quantization_count = self.quantization_samples[layer_index]
            transitions = self.winner_transitions[layer_index]
            active_labels: set[str] = set()
            if layer_index < len(network._layers):
                for cell in network._layers[layer_index].cells:
                    label = self.registry.label(layer_index, cell.identity)
                    if label is not None:
                        active_labels.add(label)
            active_counts = [winner_counts[label] for label in active_labels if winner_counts[label] > 0]
            active_samples = sum(active_counts)
            active_probabilities = [count / active_samples for count in active_counts] if active_samples else []
            layer_activity.append(
                {
                    "layer": layer_index,
                    "samples": count,
                    "mean_input_relevance": self.layer_input_sum[layer_index] / count,
                    "mean_recognition": self.layer_recognition_sum[layer_index] / count,
                    "mean_emitted_relevance": self.layer_emission_sum[layer_index] / count,
                    "mean_untransmitted_relevance": (
                        self.layer_input_sum[layer_index] - self.layer_emission_sum[layer_index]
                    ) / count,
                    "mean_winner_distance": (
                        self.quantization_distance_sum[layer_index] / quantization_count
                        if quantization_count else None
                    ),
                    "root_mean_squared_winner_distance": (
                        math.sqrt(self.quantization_squared_sum[layer_index] / quantization_count)
                        if quantization_count else None
                    ),
                    "mean_normalized_winner_distance": (
                        self.normalized_distance_sum[layer_index] / quantization_count
                        if quantization_count else None
                    ),
                    "winner_changes": self.winner_changes[layer_index],
                    "winner_transition_rate": (
                        self.winner_changes[layer_index] / transitions if transitions else None
                    ),
                    "occupation": {
                        "winner_samples": winner_samples,
                        "occupied_identities": occupied,
                        "active_final_cells": len(active_labels),
                        "active_final_cells_with_phase_wins": len(active_counts),
                        "active_final_cells_without_phase_wins": len(active_labels) - len(active_counts),
                        "entropy_nats": entropy,
                        "normalized_entropy": (
                            entropy / math.log(occupied) if occupied > 1 else 0.0
                        ),
                        "effective_identities": math.exp(entropy) if probabilities else 0.0,
                        "active_final_winner_share_distribution": distribution(active_probabilities),
                        "largest_share": max(probabilities, default=0.0),
                        "winner_share_distribution": distribution(probabilities),
                        "winner_counts": dict(winner_counts),
                    },
                }
            )

        return {
            "steps": steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps / elapsed if elapsed > 0.0 else None,
            "transformations": dict(self.transformations),
            "transforms_per_step": sum(self.transformations.values()) / steps if steps else 0.0,
            "cells": {
                "final": sum(len(layer.cells) for layer in network._layers),
                "mean": self.cells_sum / steps if steps else 0.0,
                "min": self.cells_min if self.cells_min is not None else 0,
                "max": self.cells_max,
            },
            "layers": {
                "final": len(network._layers),
                "mean": self.layers_sum / steps if steps else 0.0,
                "min": self.layers_min if self.layers_min is not None else 0,
                "max": self.layers_max,
            },
            "maintenance": {
                "final": network.maintenance_units(),
                "mean": self.maintenance_sum / steps if steps else 0.0,
                "min": self.maintenance_min if self.maintenance_min is not None else network.maintenance_units(),
                "max": self.maintenance_max,
            },
            "budget_use": {
                "budgeted_steps": self.budgeted_steps,
                "mean_utilization": (
                    self.budget_utilization_sum / self.budgeted_steps
                    if self.budgeted_steps else None
                ),
                "full_fraction": (
                    self.full_budget_steps / self.budgeted_steps
                    if self.budgeted_steps else None
                ),
                "near_full_fraction": (
                    self.near_full_budget_steps / self.budgeted_steps
                    if self.budgeted_steps else None
                ),
            },
            "depth": {
                "mean_presented_layers": self.presented_depth_sum / steps if steps else 0.0,
                "mean_positive_input_layers": self.active_depth_sum / steps if steps else 0.0,
                "mean_positive_emission_layers": self.emitted_depth_sum / steps if steps else 0.0,
                "max_positive_input_layers": self.active_depth_max,
                "positive_input_histogram": {
                    str(depth): count for depth, count in sorted(self.active_depth_histogram.items())
                },
            },
            "mean_capital": self.capital_sum / steps if steps else 0.0,
            "final_capital_per_layer": [layer.capital for layer in network._layers],
            "no_layer_steps": self.no_layer_steps,
            "mean_vertical_gain": self.vertical_gain_sum / steps if steps else 0.0,
            "mean_vertical_concordance": self.vertical_concordance_sum / steps if steps else 0.0,
            "layer_activity": layer_activity,
            "layer_state": summarize_layer_state(network, self.registry),
            "truth_purity": truth_purity,
        }


@dataclass(slots=True)
class LifetimeTracker:
    registry: IdentityRegistry
    first_seen: dict[str, int] = field(default_factory=dict)
    last_seen: dict[str, int] = field(default_factory=dict)
    live: set[str] = field(default_factory=set)
    completed_lifetimes: list[tuple[str, int]] = field(default_factory=list)

    def update(self, network: Auxein, step_index: int) -> None:
        current: set[str] = set()
        for layer_index, layer in enumerate(network._layers):
            for cell in layer.cells:
                label = self.registry.label(layer_index, cell.identity)
                assert label is not None
                current.add(label)
                self.first_seen.setdefault(label, step_index)
                self.last_seen[label] = step_index
        for vanished in self.live - current:
            self.completed_lifetimes.append(
                (
                    vanished,
                    self.last_seen[vanished] - self.first_seen[vanished] + 1,
                )
            )
        self.live = current

    def active_ages(self, final_step: int) -> dict[str, int]:
        return {
            label: final_step - self.first_seen[label] + 1
            for label in sorted(self.live)
        }

    def summary(self, final_step: int) -> dict[str, object]:
        active = self.active_ages(final_step)
        completed = [lifetime for _, lifetime in self.completed_lifetimes]
        return {
            "identities_seen": len(self.first_seen),
            "deaths_observed": len(completed),
            "completed_lifetimes": distribution(completed),
            "active_identity_ages": distribution(active.values()),
            "active_identities": [
                {"identity": label, "age": age} for label, age in active.items()
            ],
        }

def parse_experiment(data: Mapping[str, object]) -> tuple[
    str,
    ModelSpec,
    int,
    tuple[PhaseSpec, ...],
    Mapping[str, object],
    int,
    tuple[ProbeSpec, ...],
    str | None,
]:
    _exact_keys(data, {"name", "model", "seed", "load", "phases", "sweep", "observe", "probes"}, "experiment")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("experiment.name must be a nonempty string")

    model_data = _mapping(data.get("model"), "experiment.model")
    _exact_keys(model_data, {"dimension", "scalar", "memory", "budget"}, "experiment.model")
    dimension = _positive_int(model_data.get("dimension"), "model.dimension")
    scalar = model_data.get("scalar")
    if scalar not in ("f32", "f64"):
        raise ValueError("model.scalar must be 'f32' or 'f64'")
    model = ModelSpec(
        dimension,
        str(scalar),
        _positive_float(model_data.get("memory"), "model.memory"),
        _budget_decimal(model_data.get("budget"), "model.budget"),
    )

    seed = data.get("seed", 2701)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("experiment.seed must be an integer")

    raw_phases = data.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("experiment.phases must be a nonempty list")
    phases: list[PhaseSpec] = []
    seen_names: set[str] = set()
    for index, raw_phase in enumerate(raw_phases):
        phase = _mapping(raw_phase, f"phases[{index}]")
        _exact_keys(phase, {"name", "steps", "world", "budget"}, f"phases[{index}]")
        phase_name = phase.get("name")
        if not isinstance(phase_name, str) or not phase_name:
            raise ValueError(f"phases[{index}].name must be a nonempty string")
        if phase_name in seen_names:
            raise ValueError(f"duplicate phase name {phase_name!r}")
        seen_names.add(phase_name)
        steps = _nonnegative_int(phase.get("steps"), f"phase {phase_name}.steps")
        world = _mapping(phase.get("world"), f"phase {phase_name}.world")
        phase_budget = (
            None
            if "budget" not in phase
            else _budget_decimal(phase["budget"], f"phase {phase_name}.budget")
        )
        phases.append(PhaseSpec(phase_name, steps, copy.deepcopy(world), phase_budget))

    raw_sweep = data.get("sweep", {})
    sweep = _mapping(raw_sweep, "experiment.sweep")
    _exact_keys(sweep, set(SWEEP_FIELDS), "experiment.sweep")
    for field_name, values in sweep.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{field_name} must be a nonempty list")

    observe_data = _mapping(data.get("observe", {"every": 100}), "experiment.observe")
    _exact_keys(observe_data, {"every"}, "experiment.observe")
    observe_every = _positive_int(observe_data.get("every", 100), "observe.every")

    raw_probes = data.get("probes", [])
    if not isinstance(raw_probes, list):
        raise ValueError("experiment.probes must be a list")
    probes: list[ProbeSpec] = []
    for index, raw_probe in enumerate(raw_probes):
        probe = _mapping(raw_probe, f"probes[{index}]")
        _exact_keys(probe, {"name", "x"}, f"probes[{index}]")
        probe_name = probe.get("name")
        if not isinstance(probe_name, str) or not probe_name:
            raise ValueError(f"probes[{index}].name must be a nonempty string")
        x = probe.get("x")
        if not isinstance(x, list) or len(x) != dimension:
            raise ValueError(f"probe {probe_name!r} must have dimension {dimension}")
        vector = tuple(float(value) for value in x)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"probe {probe_name!r} must contain finite values")
        probes.append(ProbeSpec(probe_name, vector))

    load = data.get("load")
    if load is not None and (not isinstance(load, str) or not load):
        raise ValueError("experiment.load must be a nonempty path")

    return name, model, seed, tuple(phases), sweep, observe_every, tuple(probes), load


def expand_trials(model: ModelSpec, seed: int, sweep: Mapping[str, object]) -> list[TrialSpec]:
    keys = [field_name for field_name in SWEEP_FIELDS if field_name in sweep]
    choices = [sweep[field_name] for field_name in keys]
    combinations: Iterable[tuple[object, ...]] = itertools.product(*choices) if keys else [()]
    trials: list[TrialSpec] = []
    for index, combination in enumerate(combinations):
        overrides = dict(zip(keys, combination, strict=True))
        dimension = _positive_int(overrides.get("dimension", model.dimension), "trial.dimension")
        scalar = overrides.get("scalar", model.scalar)
        if scalar not in ("f32", "f64"):
            raise ValueError("trial.scalar must be 'f32' or 'f64'")
        memory_half_life = _positive_float(
            overrides.get("memory", model.memory_half_life),
            "trial.memory",
        )
        budget = _budget_decimal(overrides.get("budget", model.budget), "trial.budget")
        trial_seed = overrides.get("seed", seed)
        if isinstance(trial_seed, bool) or not isinstance(trial_seed, int):
            raise ValueError("trial.seed must be an integer")
        parameters = {
            "dimension": dimension,
            "scalar": str(scalar),
            "memory": memory_half_life,
            "budget": str(budget),
            "seed": trial_seed,
        }
        trials.append(
            TrialSpec(
                index,
                ModelSpec(dimension, str(scalar), memory_half_life, budget),
                trial_seed,
                parameters,
            )
        )
    return trials


def _load_network(
    trial: TrialSpec,
    *,
    load_path: Path | None,
    maintenance_model: ScalarFootprintMaintenance,
    check_invariants: bool,
) -> Auxein:
    if load_path is None:
        return Auxein.empty(
            trial.model.dimension,
            memory=trial.model.memory_half_life,
            budget=trial.model.budget,
            maintenance_model=maintenance_model,
            scalar=trial.model.scalar,
            check_invariants=check_invariants,
        )
    state = read_json_object(load_path)
    if int(state.get("dimension", -1)) != trial.model.dimension:
        raise ValueError("loaded state dimension conflicts with the trial")
    if str(state.get("scalar")) != trial.model.scalar:
        raise ValueError("loaded state scalar format conflicts with the trial")
    if float(state.get("memory", math.nan)) != trial.model.memory_half_life:
        raise ValueError("loaded state memory law conflicts with the trial")
    return Auxein.from_state_dict(
        state,
        budget=trial.model.budget,
        maintenance_model=maintenance_model,
        check_invariants=check_invariants,
    )


def _run_probes(
    network: Auxein,
    probes: Sequence[ProbeSpec],
    registry: IdentityRegistry,
    maintenance_model: ScalarFootprintMaintenance,
) -> list[dict[str, object]]:
    if not probes:
        return []
    frozen = network.to_state_dict()
    results: list[dict[str, object]] = []
    for probe in probes:
        reader = Auxein.from_state_dict(
            frozen,
            budget_units=network.budget_units,
            maintenance_model=maintenance_model,
            check_invariants=False,
        )
        report = reader.step(probe.x, detailed_report=True)
        results.append(
            {
                "name": probe.name,
                "x": list(probe.x),
                "layers": [
                    {
                        "layer": item.layer_index,
                        "winner": registry.label(item.layer_index, item.winner),
                        "recognition": item.recognition,
                        "input_relevance": item.input_relevance,
                        "emitted_relevance": item.emitted_relevance,
                        "split_value": item.split_value,
                        "capital": item.capital,
                    }
                    for item in report.layer_reports
                ],
                "transformations": [record.kind for record in report.transformations],
                "vertical_gain": report.vertical_gain,
                "vertical_concordance": report.vertical_concordance,
            }
        )
    if network.to_state_dict() != frozen:
        raise RuntimeError("probes mutated the studied network")
    return results


def run_trial(
    experiment_name: str,
    trial: TrialSpec,
    phases: Sequence[PhaseSpec],
    *,
    observe_every: int,
    probes: Sequence[ProbeSpec],
    load_path: Path | None,
    check_invariants: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    maintenance_model = ScalarFootprintMaintenance()
    network = _load_network(
        trial,
        load_path=load_path,
        maintenance_model=maintenance_model,
        check_invariants=check_invariants,
    )
    initial_step = network.step_index
    registry = IdentityRegistry()
    lifetime = LifetimeTracker(registry)
    total_metrics = Metrics(registry)
    phase_results: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    topological_events: list[dict[str, object]] = []
    global_local_step = 0
    trial_started = time.perf_counter()

    for phase_index, phase in enumerate(phases):
        phase_budget = trial.model.budget if phase.budget is None else phase.budget
        network.budget = phase_budget
        phase_seed = derive_phase_seed(trial.seed, phase_index)
        world = build_world(
            phase.world,
            dimension=network.dimension,
            seed=phase_seed,
        )
        phase_metrics = Metrics(registry)
        window_metrics = Metrics(registry)
        phase_started = time.perf_counter()
        window_started = phase_started
        window_from = 1

        for phase_step in range(phase.steps):
            sample = world.sample(phase_step)
            observation = observe_network(network, sample.x)
            report = network.step(sample.x, detailed_report=True)
            for record in report.transformations:
                topological_events.append(
                    {
                        "trial_step": global_local_step + 1,
                        "network_step": network.step_index,
                        "phase": phase.name,
                        "phase_step": phase_step + 1,
                        "kind": record.kind,
                        "layer": record.layer_index,
                        "geometric_value": record.geometric_value,
                        "maintenance_delta_units": record.maintenance_delta_units,
                    }
                )
            snapshot = snapshot_network(network)
            total_metrics.update(network, report, sample, snapshot, observation)
            phase_metrics.update(network, report, sample, snapshot, observation)
            window_metrics.update(network, report, sample, snapshot, observation)
            lifetime.update(network, global_local_step)
            global_local_step += 1

            window_done = (phase_step + 1) % observe_every == 0 or phase_step + 1 == phase.steps
            if window_done:
                now = time.perf_counter()
                summary = window_metrics.summary(network, now - window_started)
                windows.append(
                    {
                        "phase": phase.name,
                        "from_step": window_from,
                        "to_step": phase_step + 1,
                        "trial_from_step": global_local_step - window_metrics.steps + 1,
                        "trial_to_step": global_local_step,
                        **summary,
                    }
                )
                window_metrics = Metrics(registry)
                window_started = now
                window_from = phase_step + 2

        phase_elapsed = time.perf_counter() - phase_started
        phase_results.append(
            {
                "name": phase.name,
                "seed": phase_seed,
                "budget": str(phase_budget),
                "world": _jsonable(phase.world),
                **phase_metrics.summary(network, phase_elapsed),
                "layer_cell_state": summarize_layer_state(
                    network, registry, include_cells=True
                ),
            }
        )

    elapsed = time.perf_counter() - trial_started
    final_state = network.to_state_dict()
    final_step = max(0, global_local_step - 1)
    active_ages = lifetime.active_ages(final_step)
    result = {
        "experiment": experiment_name,
        "trial": trial.index,
        "parameters": dict(trial.parameters),
        "initial_network_step": initial_step,
        "final_network_step": network.step_index,
        "summary": total_metrics.summary(network, elapsed),
        "phases": phase_results,
        "windows": windows,
        "topological_events": topological_events,
        "identities": lifetime.summary(final_step),
        "final_layer_state": summarize_layer_state(
            network, registry, ages=active_ages, include_cells=True
        ),
        "probes": _run_probes(network, probes, registry, maintenance_model),
        "final_network": network.summary(),
        "maintenance_equivalent_cells": str(
            maintenance_model.budget_from_units(
                network.dimension,
                network.scalar,
                network.maintenance_units(),
            )
        ),
        "versions": {
            "state_schema": final_state["schema_version"],
            "model": final_state["model_version"],
        },
    }
    return result, final_state


def run_experiment(
    path: Path,
    *,
    output: Path | None,
    save: Path | None,
    check_invariants: bool,
    quiet: bool,
) -> list[dict[str, object]]:
    raw = read_json_object(path)
    name, model, seed, phases, sweep, observe_every, probes, load = parse_experiment(raw)
    trials = expand_trials(model, seed, sweep)
    if probes and any(trial.model.dimension != len(probes[0].x) for trial in trials):
        raise ValueError("dimension sweeps are incompatible with fixed probes")
    if save is not None and len(trials) != 1:
        raise ValueError("--save is available only for a single-trial experiment")
    load_path = None
    if load is not None:
        load_path = Path(load)
        if not load_path.is_absolute():
            load_path = path.parent / load_path

    output_handle = None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output.open("w", encoding="utf-8")

    results: list[dict[str, object]] = []
    final_state: dict[str, object] | None = None
    try:
        for trial in trials:
            result, final_state = run_trial(
                name,
                trial,
                phases,
                observe_every=observe_every,
                probes=probes,
                load_path=load_path,
                check_invariants=check_invariants,
            )
            result = _jsonable(result)  # type: ignore[assignment]
            assert isinstance(result, dict)
            results.append(result)
            if output_handle is not None:
                json.dump(result, output_handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                output_handle.write("\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())
            if not quiet:
                summary = result["summary"]
                assert isinstance(summary, Mapping)
                cells = summary["cells"]
                layers = summary["layers"]
                assert isinstance(cells, Mapping) and isinstance(layers, Mapping)
                print(
                    f"trial {trial.index + 1}/{len(trials)} "
                    f"budget={trial.parameters['budget']} scalar={trial.model.scalar} "
                    f"seed={trial.seed} steps={summary['steps']} "
                    f"cells={cells['final']} layers={layers['final']} "
                    f"step/s={float(summary['steps_per_second'] or 0.0):.2f}"
                )
    finally:
        if output_handle is not None:
            output_handle.close()

    if save is not None:
        assert final_state is not None
        write_json_atomic(save, final_state)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="JSON experiment file")
    parser.add_argument("--output", type=Path, help="write one JSONL record per trial")
    parser.add_argument("--save", type=Path, help="save final Auxein state (single trial only)")
    parser.add_argument("--check-invariants", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="print complete results as JSON")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        results = run_experiment(
            args.experiment,
            output=args.output,
            save=args.save,
            check_invariants=args.check_invariants,
            quiet=args.quiet or args.json,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
