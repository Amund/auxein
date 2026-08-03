"""Benchmark and persistence laboratory for the isolated Auxein engine.

The engine owns strict JSON-compatible serialization but performs no I/O.
This module owns streams, equivalent-cell budgets, atomic save/load, timing,
and windowed reporting. Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator, Mapping

from auxein import Auxein, ScalarFootprintMaintenance


def gaussian_stream(rng: random.Random, dimension: int) -> Iterator[list[float]]:
    while True:
        yield [rng.gauss(0.0, 1.0) for _ in range(dimension)]


def alternating_stream(rng: random.Random, dimension: int) -> Iterator[list[float]]:
    sign = 1.0
    while True:
        sample = [0.15 * rng.gauss(0.0, 1.0) for _ in range(dimension)]
        sample[0] += sign * 2.0
        yield sample
        sign *= -1.0


def drifting_stream(rng: random.Random, dimension: int) -> Iterator[list[float]]:
    phase = 0
    while True:
        center = [0.0 for _ in range(dimension)]
        center[0] = 2.5 * math.sin(phase / 200.0)
        if dimension > 1:
            center[1] = 2.5 * math.cos(phase / 317.0)
        phase += 1
        yield [value + 0.25 * rng.gauss(0.0, 1.0) for value in center]


def make_stream(
    name: str,
    rng: random.Random,
    dimension: int,
) -> Iterator[list[float]]:
    if name == "gaussian":
        return gaussian_stream(rng, dimension)
    if name == "alternating":
        return alternating_stream(rng, dimension)
    if name == "drifting":
        return drifting_stream(rng, dimension)
    raise ValueError(f"unknown stream {name!r}")


def parse_nonnegative_decimal(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("budget must be a decimal number") from exc
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError("budget must be finite and nonnegative")
    return value


def read_state(path: str) -> Mapping[str, object]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError("saved Auxein state must be a JSON object")
    return state


def write_state_atomic(path: str, state: Mapping[str, object]) -> Path:
    target = Path(path)
    if not target.name:
        raise ValueError("--save must designate a file")
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                state,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def resolve_network(args: argparse.Namespace) -> tuple[Auxein, ScalarFootprintMaintenance]:
    maintenance_model = ScalarFootprintMaintenance()
    saved_state: Mapping[str, object] | None = read_state(args.load) if args.load else None

    if saved_state is None:
        dimension = 8 if args.dimension is None else args.dimension
        scalar_format = "f64" if args.scalar is None else args.scalar
        memory_half_life = 100.0 if args.memory is None else args.memory
    else:
        try:
            dimension = int(saved_state["dimension"])
            scalar_format = str(saved_state["scalar"])
            memory_half_life = float(saved_state["memory"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("saved state lacks valid dimension/format/memory metadata") from exc
        if args.dimension is not None and args.dimension != dimension:
            raise ValueError("--dimension conflicts with the loaded state")
        if args.scalar is not None and args.scalar != scalar_format:
            raise ValueError("--scalar conflicts with the loaded state")
        if args.memory is not None and args.memory != memory_half_life:
            raise ValueError("--memory conflicts with the loaded state")

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not math.isfinite(memory_half_life) or memory_half_life <= 0.0:
        raise ValueError("memory must be positive and finite")

    if args.budget_units is not None:
        budget_arguments: dict[str, object] = {"budget_units": args.budget_units}
        resolved_units = args.budget_units
    else:
        budget = Decimal(1000) if args.budget is None else args.budget
        budget_arguments = {"budget": budget}
        resolved_units = maintenance_model.budget_to_units(dimension, scalar_format, budget)

    minimum = maintenance_model.root_substrate_units(dimension, scalar_format)
    if resolved_units < minimum:
        raise ValueError(
            f"budget {resolved_units} is below the permanent root substrate {minimum}"
        )

    if saved_state is None:
        network = Auxein.empty(
            dimension,
            memory=memory_half_life,
            eta=1.0 if args.eta is None else args.eta,
            maintenance_model=maintenance_model,
            scalar=scalar_format,
            check_invariants=args.check_invariants,
            **budget_arguments,
        )
    else:
        network = Auxein.from_state_dict(
            saved_state,
            maintenance_model=maintenance_model,
            check_invariants=args.check_invariants,
            **budget_arguments,
        )
        if args.eta is not None:
            network.eta = args.eta
    return network, maintenance_model


def run(args: argparse.Namespace) -> dict[str, object]:
    network, maintenance_model = resolve_network(args)
    rng = random.Random(args.seed)
    stream = make_stream(args.stream, rng, network.dimension)
    initial_step = network.step_index

    for _ in range(args.warmup):
        network.step(next(stream), detailed_report=False)

    transformations: Counter[str] = Counter()
    windows: list[dict[str, object]] = []
    window_transformations: Counter[str] = Counter()
    window_started = time.perf_counter()
    start = window_started

    for measured_index in range(args.steps):
        report = network.step(next(stream), detailed_report=False)
        kinds = [record.kind for record in report.transformations]
        transformations.update(kinds)
        window_transformations.update(kinds)

        window_complete = (
            args.window is not None
            and (
                (measured_index + 1) % args.window == 0
                or measured_index + 1 == args.steps
            )
        )
        if window_complete:
            now = time.perf_counter()
            window_steps = (
                args.window
                if (measured_index + 1) % args.window == 0
                else (measured_index + 1) % args.window
            )
            if window_steps == 0:
                window_steps = args.window
            elapsed = now - window_started
            windows.append(
                {
                    "from_step": measured_index + 2 - window_steps,
                    "to_step": measured_index + 1,
                    "steps": window_steps,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": window_steps / elapsed if elapsed > 0.0 else math.inf,
                    "transformations": dict(window_transformations),
                    "layers": len(network._layers),
                    "cells": sum(len(layer.cells) for layer in network._layers),
                }
            )
            window_transformations.clear()
            window_started = now

    elapsed = time.perf_counter() - start
    if args.save:
        write_state_atomic(args.save, network.to_state_dict())

    summary = network.summary()
    maintenance_equivalent = maintenance_model.budget_from_units(
        network.dimension,
        network.scalar,
        network.maintenance_units(),
    )
    return {
        "stream": args.stream,
        "dimension": network.dimension,
        "scalar": network.scalar,
        "steps": args.steps,
        "warmup": args.warmup,
        "initial_network_step": initial_step,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / elapsed if elapsed > 0.0 else math.inf,
        "microseconds_per_step": elapsed * 1_000_000.0 / args.steps,
        "transformations": dict(transformations),
        "windows": windows,
        "budget_equivalent_cells": str(network.budget),
        "maintenance_equivalent_cells": str(maintenance_equivalent),
        "network": summary,
        "loaded_from": args.load,
        "saved_to": args.save,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--memory", type=float, default=None)
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="learning-rate multiplier in [0, 1] (default: 1 or loaded value)",
    )
    parser.add_argument("--scalar", choices=("f32", "f64"), default=None)
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--budget",
        type=parse_nonnegative_decimal,
        default=None,
        metavar="CELLS",
        help="ergonomic budget in equivalent terminal-cell packages (default: 1000)",
    )
    budget_group.add_argument(
        "--budget-units",
        type=int,
        default=None,
        metavar="UNITS",
        help="advanced exact raw footprint budget",
    )
    parser.add_argument(
        "--stream",
        choices=("gaussian", "alternating", "drifting"),
        default="alternating",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window", type=int, default=None, metavar="STEPS")
    parser.add_argument("--load", metavar="FILE")
    parser.add_argument("--save", metavar="FILE")
    parser.add_argument(
        "--check-invariants",
        action="store_true",
        help="validate the full hierarchy after every causal phase",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")
    if args.dimension is not None and args.dimension <= 0:
        parser.error("dimension must be positive")
    if args.window is not None and args.window <= 0:
        parser.error("window must be positive")
    if args.eta is not None and (
        not math.isfinite(args.eta) or not 0.0 <= args.eta <= 1.0
    ):
        parser.error("eta must lie in [0, 1]")
    if args.budget_units is not None and args.budget_units <= 0:
        parser.error("budget units must be positive")

    try:
        result = run(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return

    for window in result["windows"]:
        print(
            "window "
            f"{window['from_step']}-{window['to_step']}: "
            f"{window['elapsed_seconds']:.6f} s, "
            f"{window['steps_per_second']:.2f} steps/s, "
            f"cells={window['cells']}, layers={window['layers']}, "
            f"transformations={window['transformations']}"
        )

    print("Auxein reference benchmark")
    print(f"  stream              : {result['stream']}")
    print(f"  dimension           : {result['dimension']}")
    print(f"  scalar              : {result['scalar']}")
    print(f"  memory half-life    : {result['network']['memory']} presentations")
    print(f"  budget              : {result['budget_equivalent_cells']} equivalent cells")
    print(f"  warmup steps        : {result['warmup']}")
    print(f"  measured steps      : {result['steps']}")
    print(f"  initial network step: {result['initial_network_step']}")
    print(f"  elapsed             : {result['elapsed_seconds']:.6f} s")
    print(f"  throughput          : {result['steps_per_second']:.2f} steps/s")
    print(f"  latency             : {result['microseconds_per_step']:.2f} µs/step")
    print(f"  transformations     : {result['transformations']}")
    measured_transforms = sum(result["transformations"].values())
    print(f"  transforms/step     : {measured_transforms / result['steps']:.4f}")
    network = result["network"]
    print(f"  layers              : {network['layers']}")
    print(f"  cells per layer     : {network['cells_per_layer']}")
    print(f"  capital per layer   : {network['capital_per_layer']}")
    print(
        "  maintenance         : "
        f"{network['maintenance_units']} / {network['budget_units']} raw units "
        f"({result['maintenance_equivalent_cells']} active equivalents)"
    )
    if result["loaded_from"]:
        print(f"  loaded              : {result['loaded_from']}")
    if result["saved_to"]:
        print(f"  saved               : {result['saved_to']}")


if __name__ == "__main__":
    main()
