"""Stdlib-only benchmark harness for the true Auxein v0.2.0 canon."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time

from auxein import Auxein, Kernel, Layer


def make_network(scenario: str, dimension: int, cells: int) -> tuple[Auxein, list[list[float]]]:
    network = Auxein(dimension=dimension, memory=50.0, eta=0.0, budget=max(1000, cells * 2))
    if scenario == "singleton":
        center = [0.0] * dimension
        center[0] = 2.0
        network.layers[0].cells = [Kernel(1.0, tuple(center), 0.25)]
        return network, [center]
    if scenario == "pair-context":
        a = [0.0] * dimension
        b = [0.0] * dimension
        a[0] = 1.0
        b[0] = 3.0
        network.layers[0].cells = [Kernel(1.0, tuple(a), 0.0), Kernel(1.0, tuple(b), 0.0)]
        network.layers.append(Layer([], [Kernel(1.0, tuple((2.0,) + (0.0,) * (dimension - 1)), 1.0)]))
        return network, [a, b]
    if scenario == "sparse":
        rng = random.Random(7)
        population: list[Kernel] = []
        for i in range(cells):
            c = [0.0] * dimension
            c[0] = 10.0 + i * 0.01
            for d in range(1, dimension):
                c[d] = 0.01 * rng.random()
            population.append(Kernel(1.0, tuple(c), 0.01))
        target = [0.0] * dimension
        target[0] = 10.0 + (cells // 2) * 0.01
        network.layers[0].cells = population
        return network, [target]
    if scenario == "dense":
        population = []
        for i in range(cells):
            angle = 2.0 * math.pi * i / max(1, cells)
            c = [0.0] * dimension
            c[0] = 1.0 + 0.01 * math.cos(angle)
            if dimension > 1:
                c[1] = 0.01 * math.sin(angle)
            population.append(Kernel(1.0, tuple(c), 10.0))
        network.layers[0].cells = population
        x = [0.0] * dimension
        x[0] = 2.0
        return network, [x]
    raise ValueError(f"unknown scenario {scenario}")


def run_once(scenario: str, dimension: int, cells: int, steps: int, warmup: int) -> float:
    network, presentation = make_network(scenario, dimension, cells)
    for _ in range(warmup):
        network.step(presentation)
    start = time.perf_counter()
    for _ in range(steps):
        network.step(presentation)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["singleton", "pair-context", "sparse", "dense"], default="singleton")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    elapsed = [run_once(args.scenario, args.dimension, args.cells, args.steps, args.warmup) for _ in range(args.runs)]
    median = statistics.median(elapsed)
    print(json.dumps({
        "scenario": args.scenario,
        "dimension": args.dimension,
        "cells": args.cells,
        "steps": args.steps,
        "runs": args.runs,
        "median_seconds": median,
        "microseconds_per_step": 1e6 * median / args.steps,
        "steps_per_second": args.steps / median,
    }, indent=2))


if __name__ == "__main__":
    main()
