"""Deterministic external worlds for the Auxein v0.4.0 laboratory.

Worlds never inspect Auxein.  They emit one logical external presentation per
step and optional truth used only by diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Protocol


@dataclass(frozen=True, slots=True)
class Sample:
    presentation: list[list[float]]
    truth: dict[str, object]


class World(Protocol):
    def sample(self, step: int) -> Sample: ...


@dataclass(slots=True)
class PresentationCycle:
    presentations: list[list[list[float]]]
    noise: float
    rng: random.Random

    def sample(self, step: int) -> Sample:
        index = step % len(self.presentations)
        base = self.presentations[index]
        if self.noise == 0.0:
            value = [list(v) for v in base]
        else:
            value = [
                [x + self.noise * self.rng.gauss(0.0, 1.0) for x in vector]
                for vector in base
            ]
        return Sample(value, {"presentation": index, "size": len(value)})


@dataclass(slots=True)
class PointCycle:
    points: list[list[float]]
    noise: float
    rng: random.Random

    def sample(self, step: int) -> Sample:
        index = step % len(self.points)
        point = self.points[index]
        if self.noise == 0.0:
            value = list(point)
        else:
            value = [x + self.noise * self.rng.gauss(0.0, 1.0) for x in point]
        return Sample([value], {"point": index})


@dataclass(slots=True)
class GaussianMixture:
    centers: list[list[float]]
    std: float
    rng: random.Random

    def sample(self, step: int) -> Sample:
        index = step % len(self.centers)
        center = self.centers[index]
        point = [x + self.std * self.rng.gauss(0.0, 1.0) for x in center]
        return Sample([point], {"cluster": index})


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _vectors(value: object, label: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    out: list[list[float]] = []
    for i, vector in enumerate(value):
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"{label}[{i}] must be a nonempty vector")
        out.append([float(x) for x in vector])
    return out


def build_world(spec: Mapping[str, object], *, seed: int) -> World:
    name = spec.get("name")
    rng = random.Random(seed)
    if name == "point-cycle":
        points = _vectors(spec.get("points"), "world.points")
        return PointCycle(points, float(spec.get("noise", 0.0)), rng)
    if name == "presentation-cycle":
        raw = spec.get("presentations")
        if not isinstance(raw, list) or not raw:
            raise ValueError("world.presentations must be a nonempty list")
        presentations = [_vectors(item, f"world.presentations[{i}]") for i, item in enumerate(raw)]
        return PresentationCycle(presentations, float(spec.get("noise", 0.0)), rng)
    if name == "gaussian-mixture":
        centers = _vectors(spec.get("centers"), "world.centers")
        std = float(spec.get("std", 0.1))
        if std < 0.0:
            raise ValueError("world.std must be nonnegative")
        return GaussianMixture(centers, std, rng)
    raise ValueError(f"unknown world {name!r}")
