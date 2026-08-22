"""Stdlib-only benchmark harness for the Auxein v0.5.0 canon."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from typing import Callable

from auxein import Auxein, Kernel, Layer

Presentation = object


def make_network(
    scenario: str, dimension: int, cells: int, mode: str
) -> tuple[Auxein, Callable[[int], Presentation], bool]:
    network = Auxein(
        dimension=dimension,
        memory=50.0,
        eta=0.0,
        mode=mode,
        budget=max(1000, cells * 2),
    )

    def axis(value: float) -> list[float]:
        c = [0.0] * dimension
        c[0] = value
        return c

    if scenario == "singleton":
        center = axis(2.0)
        network.layers[0].cells = [Kernel(1.0, tuple(center), 0.25)]
        return network, lambda _i: [center], False

    if scenario == "weighted-partial":
        center = axis(2.0)
        unknown = axis(20.0)
        network.layers[0].cells = [Kernel(1.0, tuple(center), 0.25)]
        presentation = [[0.25, center, 0.0], [0.75, unknown, 0.0]]
        return network, lambda _i: presentation, False

    if scenario == "predictive-stable":
        if mode != "predictive":
            raise ValueError("predictive-stable requires --mode predictive")
        center = tuple(axis(2.0))
        target = tuple(axis(3.0))
        network.layers[0].cells = [Kernel(1.0, center, 0.25)]
        network.layers[0].temporal_cells = [Kernel(1.0, center + target, 0.5)]
        p = [list(center)]
        return network, lambda _i: p, False

    if scenario == "predictive-sequence":
        if mode != "predictive":
            raise ValueError("predictive-sequence requires --mode predictive")
        a = tuple(axis(1.0))
        b = tuple(axis(10.0))
        network.layers[0].cells = [Kernel(1.0, a, 0.0), Kernel(1.0, b, 0.0)]
        return network, lambda i: [list(a if i % 2 == 0 else b)], True

    if scenario == "pair-context":
        a = axis(1.0)
        b = axis(3.0)
        network.layers[0].cells = [Kernel(1.0, tuple(a), 0.0), Kernel(1.0, tuple(b), 0.0)]
        network.layers.append(
            Layer([], [Kernel(1.0, tuple(axis(2.0)), 1.0)])
        )
        return network, lambda _i: [a, b], False

    if scenario == "sparse":
        rng = random.Random(7)
        population: list[Kernel] = []
        for i in range(cells):
            c = axis(10.0 + i * 0.01)
            for d in range(1, dimension):
                c[d] = 0.01 * rng.random()
            population.append(Kernel(1.0, tuple(c), 0.01))
        target = axis(10.0 + (cells // 2) * 0.01)
        network.layers[0].cells = population
        return network, lambda _i: [target], False

    if scenario == "dense":
        population = []
        for i in range(cells):
            angle = 2.0 * math.pi * i / max(1, cells)
            c = axis(1.0 + 0.01 * math.cos(angle))
            if dimension > 1:
                c[1] = 0.01 * math.sin(angle)
            population.append(Kernel(1.0, tuple(c), 10.0))
        network.layers[0].cells = population
        x = axis(2.0)
        return network, lambda _i: [x], False

    raise ValueError(f"unknown scenario {scenario}")


def run_once(
    scenario: str,
    dimension: int,
    cells: int,
    steps: int,
    warmup: int,
    mode: str,
) -> float:
    network, presentation_at, causal = make_network(scenario, dimension, cells, mode)
    if causal:
        network.begin_sequence()
        try:
            for i in range(warmup):
                network.sequence_step(presentation_at(i))
            start = time.perf_counter()
            for i in range(warmup, warmup + steps):
                network.sequence_step(presentation_at(i))
            return time.perf_counter() - start
        finally:
            network.end_sequence()

    for i in range(warmup):
        network.step(presentation_at(i))
    start = time.perf_counter()
    for i in range(warmup, warmup + steps):
        network.step(presentation_at(i))
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=[
            "singleton",
            "weighted-partial",
            "predictive-stable",
            "predictive-sequence",
            "pair-context",
            "sparse",
            "dense",
        ],
        default="singleton",
    )
    parser.add_argument("--mode", choices=["geometry", "predictive"], default="geometry")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    elapsed = [
        run_once(args.scenario, args.dimension, args.cells, args.steps, args.warmup, args.mode)
        for _ in range(args.runs)
    ]
    median = statistics.median(elapsed)
    print(
        json.dumps(
            {
                "canon": "0.5.0",
                "mode": args.mode,
                "scenario": args.scenario,
                "dimension": args.dimension,
                "cells": args.cells,
                "steps": args.steps,
                "runs": args.runs,
                "median_seconds": median,
                "microseconds_per_presentation": 1e6 * median / args.steps,
                "presentations_per_second": args.steps / median,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
