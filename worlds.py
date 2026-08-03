"""Synthetic worlds for the Auxein experimental laboratory.

Worlds know nothing about Auxein.  They emit an input vector and optional
external truth used only by laboratory metrics.  Every phase owns its own
world instance and pseudo-random generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Number
import random
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class Sample:
    x: list[float]
    truth: dict[str, object]


class World(Protocol):
    def sample(self, step: int) -> Sample: ...


def derive_phase_seed(trial_seed: int, phase_index: int) -> int:
    """Derive a stable independent seed for one phase.

    A phase's random stream therefore does not depend on the length of any
    preceding phase.
    """

    if isinstance(trial_seed, bool) or not isinstance(trial_seed, int):
        raise TypeError("trial seed must be an integer")
    if isinstance(phase_index, bool) or not isinstance(phase_index, int) or phase_index < 0:
        raise ValueError("phase index must be a nonnegative integer")
    return (trial_seed * 1_000_003 + phase_index * 97_409 + 0xA17E1) & ((1 << 63) - 1)


def _exact_keys(spec: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")


def _name(spec: Mapping[str, object]) -> str:
    value = spec.get("name")
    if not isinstance(value, str) or not value:
        raise ValueError("world.name must be a nonempty string")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    return out


def _nonnegative(value: object, label: str) -> float:
    out = _finite(value, label)
    if out < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return out


def _positive(value: object, label: str) -> float:
    out = _finite(value, label)
    if out <= 0.0:
        raise ValueError(f"{label} must be positive")
    return out


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _vector(value: object, dimension: int, label: str) -> list[float]:
    if dimension == 1 and isinstance(value, Number) and not isinstance(value, bool):
        return [_finite(value, label)]
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{label} must be a vector of dimension {dimension}")
    return [_finite(component, f"{label}[{index}]") for index, component in enumerate(value)]


def _vectors(value: object, dimension: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    return [_vector(item, dimension, f"{label}[{index}]") for index, item in enumerate(value)]


def _per_mode_nonnegative(value: object, count: int, label: str) -> list[float]:
    if isinstance(value, Number) and not isinstance(value, bool):
        return [_nonnegative(value, label)] * count
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} must be a scalar or contain one value per mode")
    return [_nonnegative(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _mode_weights(value: object, count: int, label: str) -> list[float]:
    if value is None:
        return [1.0 / count] * count
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} must contain one weight per mode")
    weights = [_nonnegative(item, f"{label}[{index}]") for index, item in enumerate(value)]
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError(f"{label} must contain at least one positive weight")
    return [weight / total for weight in weights]


@dataclass(slots=True)
class GaussianWorld:
    rng: random.Random
    mean: list[float]
    std: float

    def sample(self, step: int) -> Sample:
        return Sample(
            [center + self.std * self.rng.gauss(0.0, 1.0) for center in self.mean],
            {"regime": "gaussian"},
        )


@dataclass(slots=True)
class PointCycleWorld:
    rng: random.Random
    points: list[list[float]]
    noise: float

    def sample(self, step: int) -> Sample:
        mode = step % len(self.points)
        center = self.points[mode]
        return Sample(
            [value + self.noise * self.rng.gauss(0.0, 1.0) for value in center],
            {"mode": mode, "regime": "point-cycle"},
        )


@dataclass(slots=True)
class AlternatingWorld:
    rng: random.Random
    dimension: int
    axis: int
    offset: float
    noise: float

    def sample(self, step: int) -> Sample:
        branch = step & 1
        sign = -1.0 if branch == 0 else 1.0
        x = [self.noise * self.rng.gauss(0.0, 1.0) for _ in range(self.dimension)]
        x[self.axis] += sign * self.offset
        return Sample(x, {"mode": branch, "context": branch, "regime": "alternating"})


@dataclass(slots=True)
class DriftingWorld:
    rng: random.Random
    dimension: int
    amplitude: float
    period: float
    noise: float

    def sample(self, step: int) -> Sample:
        center = [0.0] * self.dimension
        angle = 2.0 * math.pi * step / self.period
        center[0] = self.amplitude * math.sin(angle)
        if self.dimension > 1:
            center[1] = self.amplitude * math.cos(angle)
        x = [value + self.noise * self.rng.gauss(0.0, 1.0) for value in center]
        return Sample(x, {"regime": "drifting", "center": center})


@dataclass(slots=True)
class SharedContrastWorld:
    rng: random.Random
    centers: list[list[float]]
    contrast: list[float]
    noise: float

    def sample(self, step: int) -> Sample:
        branch = step & 1
        mode = (step // 2) % len(self.centers)
        sign = -1.0 if branch == 0 else 1.0
        center = [
            base + sign * delta
            for base, delta in zip(self.centers[mode], self.contrast, strict=True)
        ]
        x = [value + self.noise * self.rng.gauss(0.0, 1.0) for value in center]
        return Sample(x, {"mode": mode, "context": branch, "regime": "shared"})


@dataclass(slots=True)
class ControlContrastWorld:
    rng: random.Random
    centers: list[list[float]]
    offsets: list[tuple[list[float], list[float]]]
    noise: float

    def sample(self, step: int) -> Sample:
        branch = step & 1
        mode = (step // 2) % len(self.centers)
        offset = self.offsets[mode][branch]
        center = [
            base + delta
            for base, delta in zip(self.centers[mode], offset, strict=True)
        ]
        x = [value + self.noise * self.rng.gauss(0.0, 1.0) for value in center]
        return Sample(x, {"mode": mode, "context": branch, "regime": "control"})


@dataclass(slots=True)
class GaussianMixtureWorld:
    rng: random.Random
    means: list[list[float]]
    stds: list[float]
    weights: list[float]

    def sample(self, step: int) -> Sample:
        del step
        draw = self.rng.random()
        cumulative = 0.0
        mode = len(self.weights) - 1
        for index, weight in enumerate(self.weights):
            cumulative += weight
            if draw < cumulative:
                mode = index
                break
        center = self.means[mode]
        std = self.stds[mode]
        return Sample(
            [value + std * self.rng.gauss(0.0, 1.0) for value in center],
            {"mode": mode, "regime": "gaussian-mixture", "center": center},
        )


@dataclass(slots=True)
class MovingPointCycleWorld:
    rng: random.Random
    points: list[list[float]]
    drift: list[float]
    period: float
    noise: float

    def sample(self, step: int) -> Sample:
        mode = step % len(self.points)
        angle = 2.0 * math.pi * step / self.period
        offset = math.sin(angle)
        center = [
            base + offset * delta
            for base, delta in zip(self.points[mode], self.drift, strict=True)
        ]
        return Sample(
            [value + self.noise * self.rng.gauss(0.0, 1.0) for value in center],
            {"mode": mode, "regime": "moving-point-cycle", "center": center},
        )




@dataclass(slots=True)
class MotifCycleWorld:
    rng: random.Random
    centers: list[list[float]]
    offsets: list[list[list[float]]]
    noise: float

    def sample(self, step: int) -> Sample:
        variants = len(self.offsets[0])
        variant = step % variants
        mode = (step // variants) % len(self.centers)
        center = [
            base + delta
            for base, delta in zip(
                self.centers[mode], self.offsets[mode][variant], strict=True
            )
        ]
        x = [value + self.noise * self.rng.gauss(0.0, 1.0) for value in center]
        return Sample(
            x,
            {
                "mode": mode,
                "variant": variant,
                "regime": "motif-cycle",
                "center": center,
            },
        )


@dataclass(slots=True)
class FactorMixtureWorld:
    rng: random.Random
    identities: int
    spacing: float
    amplitudes: list[float]
    noise: float

    def sample(self, step: int) -> Sample:
        del step
        levels = len(self.amplitudes)
        mode = self.rng.randrange(self.identities)
        code = self.rng.randrange(1 << levels) if levels else 0
        x = self.spacing * (mode - (self.identities - 1) / 2.0)
        for level, amplitude in enumerate(self.amplitudes):
            sign = 1.0 if ((code >> level) & 1) else -1.0
            x += sign * amplitude
        x += self.noise * self.rng.gauss(0.0, 1.0)
        truth: dict[str, object] = {
            "mode": mode,
            "context": code,
            "levels": levels,
            "regime": "factor-mixture",
        }
        for level in range(levels):
            truth[f"bit_{level}"] = (code >> level) & 1
            truth[f"active_{level}"] = int(self.amplitudes[level] > 0.0)
        return Sample([x], truth)


@dataclass(slots=True)
class OrientedFactorMixtureWorld:
    rng: random.Random
    identities: int
    spacing: float
    amplitudes: list[float]
    orientations: list[list[float]]
    noise: float

    def sample(self, step: int) -> Sample:
        del step
        levels = len(self.amplitudes)
        mode = self.rng.randrange(self.identities)
        code = self.rng.randrange(1 << levels) if levels else 0
        x = self.spacing * (mode - (self.identities - 1) / 2.0)
        for level, amplitude in enumerate(self.amplitudes):
            sign = 1.0 if ((code >> level) & 1) else -1.0
            x += sign * amplitude * self.orientations[mode][level]
        x += self.noise * self.rng.gauss(0.0, 1.0)
        truth: dict[str, object] = {
            "mode": mode,
            "context": code,
            "levels": levels,
            "regime": "oriented-factor-mixture",
        }
        for level in range(levels):
            truth[f"bit_{level}"] = (code >> level) & 1
            truth[f"active_{level}"] = int(self.amplitudes[level] > 0.0)
            truth[f"orientation_{level}"] = self.orientations[mode][level]
        return Sample([x], truth)


@dataclass(slots=True)
class FactorMotifMixtureWorld:
    """Random categorical locations carrying one or more signed local factors.

    The world is deliberately more general than the one-dimensional factor
    mixture: every mode owns a center and one vector per binary factor.  This
    lets the laboratory apply global similarities, rearrange locations, or
    rotate local frames without changing the truth-generating process.
    """

    rng: random.Random
    centers: list[list[float]]
    factors: list[list[list[float]]]
    noise: float

    def sample(self, step: int) -> Sample:
        del step
        levels = len(self.factors[0]) if self.factors else 0
        mode = self.rng.randrange(len(self.centers))
        code = self.rng.randrange(1 << levels) if levels else 0
        x = list(self.centers[mode])
        for level, vector in enumerate(self.factors[mode]):
            sign = 1.0 if ((code >> level) & 1) else -1.0
            for axis in range(len(x)):
                x[axis] += sign * vector[axis]
        if self.noise > 0.0:
            for axis in range(len(x)):
                x[axis] += self.noise * self.rng.gauss(0.0, 1.0)
        truth: dict[str, object] = {
            "mode": mode,
            "context": code,
            "levels": levels,
            "regime": "factor-motif-mixture",
        }
        for level in range(levels):
            truth[f"bit_{level}"] = (code >> level) & 1
        return Sample(x, truth)


@dataclass(slots=True)
class NestedBinaryWorld:
    identities: int
    levels: int
    spacing: float
    amplitude: float
    ratio: float

    def sample(self, step: int) -> Sample:
        patterns = 1 << self.levels
        index = step % (self.identities * patterns)
        mode = index // patterns
        code = index % patterns
        x = self.spacing * (mode - (self.identities - 1) / 2.0)
        for level in range(self.levels):
            sign = 1.0 if ((code >> level) & 1) else -1.0
            x += sign * self.amplitude * (self.ratio**level)
        truth: dict[str, object] = {
            "mode": mode,
            "context": code,
            "levels": self.levels,
            "regime": "nested-binary",
        }
        for level in range(self.levels):
            truth[f"bit_{level}"] = (code >> level) & 1
        return Sample([x], truth)


@dataclass(slots=True)
class OrderedHistoryWorld:
    """Future selected by cue order while static geometry is controlled.

    Each episode presents one of two four-cue trajectories followed by a
    common bridge, an exactly identical zero junction, and a binary outcome.
    The two cue trajectories contain the same values.  In ``kernel_balanced``
    mode their contributions to a scalar exponentially weighted W/S/Q receipt
    are exactly equal for the supplied ``memory``; only Cell-local routing can
    retain their order.  In the shuffled control, the outcome is independent
    of the trajectory while all input marginals are preserved.
    """

    rng: random.Random
    dimension: int
    memory_half_life: int
    bridge_steps: int
    predictive: bool
    kernel_balanced: bool
    cue_amplitude: float
    bridge_value: float
    outcome_amplitude: float
    enclosed: bool = False
    _episode: int = -1
    _order: int = 0
    _future: int = 0

    @property
    def motif_length(self) -> int:
        if self.kernel_balanced:
            return 6 if self.enclosed else 4
        return 2

    @property
    def episode_length(self) -> int:
        return self.motif_length + self.bridge_steps + 2

    def _motif(self, order: int) -> list[float]:
        high = self.cue_amplitude
        if self.kernel_balanced:
            chi = 2.0 ** (-1.0 / self.memory_half_life)
            low = high * chi * chi
            first = [high, -high, -low, low]
            motif = first if order == 0 else [-value for value in first]
            if self.enclosed:
                return [self.bridge_value, *motif, self.bridge_value]
            return motif
        first = [-high, high]
        return first if order == 0 else list(reversed(first))

    def sample(self, step: int) -> Sample:
        episode, phase_index = divmod(step, self.episode_length)
        if episode != self._episode:
            self._episode = episode
            self._order = self.rng.randrange(2)
            self._future = self._order if self.predictive else self.rng.randrange(2)

        motif = self._motif(self._order)
        if phase_index < len(motif):
            value = motif[phase_index]
            phase = "cue"
            cue_index = phase_index
        elif phase_index < len(motif) + self.bridge_steps:
            value = self.bridge_value
            phase = "bridge"
            cue_index = -1
        elif phase_index == len(motif) + self.bridge_steps:
            value = 0.0
            phase = "junction"
            cue_index = -1
        else:
            value = -self.outcome_amplitude if self._future == 0 else self.outcome_amplitude
            phase = "outcome"
            cue_index = -1

        x = [0.0] * self.dimension
        x[0] = value
        return Sample(
            x,
            {
                "phase": phase,
                "episode": episode,
                "cue_index": cue_index,
                "order": self._order,
                "future": self._future,
                "predictive": int(self.predictive),
                "kernel_balanced": int(self.kernel_balanced),
                "bridge_steps": self.bridge_steps,
            },
        )


@dataclass(slots=True)
class AmbiguousRegimeWorld:
    """Alternating hidden regimes sharing an exactly identical junction.

    Every pair is ``junction -> outcome``.  The junction is always the zero
    vector.  In the predictive form, the next outcome is fixed by the current
    regime; in the shuffled control it is an independent fair draw with the
    same marginal geometry.
    """

    rng: random.Random
    dimension: int
    block_junctions: int
    amplitude: float
    predictive: bool
    _pair_index: int = -1
    _future: int = 0

    def sample(self, step: int) -> Sample:
        pair_index, phase_index = divmod(step, 2)
        block, age = divmod(pair_index, self.block_junctions)
        regime = block & 1
        if pair_index != self._pair_index:
            self._pair_index = pair_index
            self._future = regime if self.predictive else self.rng.randrange(2)
        if phase_index == 0:
            x = [0.0] * self.dimension
            phase = "junction"
        else:
            sign = -1.0 if self._future == 0 else 1.0
            x = [0.0] * self.dimension
            x[0] = sign * self.amplitude
            phase = "outcome"
        return Sample(
            x,
            {
                "phase": phase,
                "regime": regime,
                "future": self._future,
                "block": block,
                "block_age": age,
                "regime_predictive": int(self.predictive),
            },
        )


def build_world(
    spec: Mapping[str, object],
    *,
    dimension: int,
    seed: int,
) -> World:
    """Validate and construct one synthetic world from a strict specification."""

    if not isinstance(spec, Mapping):
        raise ValueError("world must be an object")
    name = _name(spec)
    rng = random.Random(seed)

    if name == "gaussian":
        _exact_keys(spec, {"name", "mean", "std"}, "gaussian world")
        mean = _vector(spec.get("mean", [0.0] * dimension), dimension, "world.mean")
        std = _nonnegative(spec.get("std", 1.0), "world.std")
        return GaussianWorld(rng, mean, std)

    if name == "point-cycle":
        _exact_keys(spec, {"name", "points", "noise"}, "point-cycle world")
        points = _vectors(spec.get("points"), dimension, "world.points")
        noise = _nonnegative(spec.get("noise", 0.0), "world.noise")
        return PointCycleWorld(rng, points, noise)

    if name == "gaussian-mixture":
        _exact_keys(
            spec,
            {"name", "means", "stds", "weights"},
            "gaussian-mixture world",
        )
        means = _vectors(spec.get("means"), dimension, "world.means")
        stds = _per_mode_nonnegative(spec.get("stds", 1.0), len(means), "world.stds")
        weights = _mode_weights(spec.get("weights"), len(means), "world.weights")
        return GaussianMixtureWorld(rng, means, stds, weights)

    if name == "moving-point-cycle":
        _exact_keys(
            spec,
            {"name", "points", "drift", "period", "noise"},
            "moving-point-cycle world",
        )
        points = _vectors(spec.get("points"), dimension, "world.points")
        drift = _vector(spec.get("drift", [0.0] * dimension), dimension, "world.drift")
        period = _positive(spec.get("period", 1000.0), "world.period")
        noise = _nonnegative(spec.get("noise", 0.0), "world.noise")
        return MovingPointCycleWorld(rng, points, drift, period, noise)

    if name == "alternating":
        _exact_keys(spec, {"name", "axis", "offset", "noise"}, "alternating world")
        axis = _integer(spec.get("axis", 0), "world.axis")
        if axis >= dimension:
            raise ValueError("world.axis is outside the input dimension")
        offset = _positive(spec.get("offset", 2.0), "world.offset")
        noise = _nonnegative(spec.get("noise", 0.15), "world.noise")
        return AlternatingWorld(rng, dimension, axis, offset, noise)

    if name == "drifting":
        _exact_keys(spec, {"name", "amplitude", "period", "noise"}, "drifting world")
        amplitude = _nonnegative(spec.get("amplitude", 2.5), "world.amplitude")
        period = _positive(spec.get("period", 1000.0), "world.period")
        noise = _nonnegative(spec.get("noise", 0.25), "world.noise")
        return DriftingWorld(rng, dimension, amplitude, period, noise)

    if name == "shared-contrast":
        _exact_keys(spec, {"name", "centers", "contrast", "noise"}, "shared-contrast world")
        centers = _vectors(spec.get("centers"), dimension, "world.centers")
        contrast = _vector(spec.get("contrast"), dimension, "world.contrast")
        noise = _nonnegative(spec.get("noise", 0.0), "world.noise")
        return SharedContrastWorld(rng, centers, contrast, noise)

    if name == "control-contrast":
        _exact_keys(spec, {"name", "centers", "offsets", "noise"}, "control-contrast world")
        centers = _vectors(spec.get("centers"), dimension, "world.centers")
        raw_offsets = spec.get("offsets")
        if not isinstance(raw_offsets, list) or len(raw_offsets) != len(centers):
            raise ValueError("world.offsets must contain one pair per center")
        offsets: list[tuple[list[float], list[float]]] = []
        for index, pair in enumerate(raw_offsets):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"world.offsets[{index}] must contain two vectors")
            offsets.append(
                (
                    _vector(pair[0], dimension, f"world.offsets[{index}][0]"),
                    _vector(pair[1], dimension, f"world.offsets[{index}][1]"),
                )
            )
        noise = _nonnegative(spec.get("noise", 0.0), "world.noise")
        return ControlContrastWorld(rng, centers, offsets, noise)


    if name == "motif-cycle":
        _exact_keys(
            spec,
            {"name", "centers", "offsets", "noise"},
            "motif-cycle world",
        )
        centers = _vectors(spec.get("centers"), dimension, "world.centers")
        raw_offsets = spec.get("offsets")
        if not isinstance(raw_offsets, list) or len(raw_offsets) != len(centers):
            raise ValueError("world.offsets must contain one motif per center")
        offsets: list[list[list[float]]] = []
        variants: int | None = None
        for mode, raw_motif in enumerate(raw_offsets):
            if not isinstance(raw_motif, list) or not raw_motif:
                raise ValueError(f"world.offsets[{mode}] must be a nonempty list")
            motif = [
                _vector(item, dimension, f"world.offsets[{mode}][{variant}]")
                for variant, item in enumerate(raw_motif)
            ]
            if variants is None:
                variants = len(motif)
            elif len(motif) != variants:
                raise ValueError("all motifs must contain the same number of variants")
            offsets.append(motif)
        noise = _nonnegative(spec.get("noise", 0.0), "world.noise")
        return MotifCycleWorld(rng, centers, offsets, noise)

    if name == "factor-mixture":
        _exact_keys(
            spec,
            {"name", "identities", "spacing", "amplitudes", "noise"},
            "factor-mixture world",
        )
        if dimension != 1:
            raise ValueError("factor-mixture requires dimension 1")
        identities = _integer(spec.get("identities", 4), "world.identities", minimum=1)
        spacing = _positive(spec.get("spacing", 8.0), "world.spacing")
        raw_amplitudes = spec.get("amplitudes")
        if not isinstance(raw_amplitudes, list) or not raw_amplitudes:
            raise ValueError("world.amplitudes must be a nonempty list")
        amplitudes = [
            _nonnegative(value, f"world.amplitudes[{index}]")
            for index, value in enumerate(raw_amplitudes)
        ]
        noise = _nonnegative(spec.get("noise", 0.0), "world.noise")
        return FactorMixtureWorld(rng, identities, spacing, amplitudes, noise)


    if name == "oriented-factor-mixture":
        _exact_keys(
            spec,
            {"name", "identities", "spacing", "amplitudes", "orientations", "noise"},
            "oriented-factor-mixture world",
        )
        if dimension != 1:
            raise ValueError("oriented-factor-mixture requires dimension 1")
        identities = _integer(spec.get("identities", 4), "world.identities", minimum=1)
        spacing = _positive(spec.get("spacing", 8.0), "world.spacing")
        raw_amplitudes = spec.get("amplitudes")
        if not isinstance(raw_amplitudes, list) or not raw_amplitudes:
            raise ValueError("world.amplitudes must be a nonempty list")
        amplitudes = [
            _nonnegative(value, f"world.amplitudes[{index}]")
            for index, value in enumerate(raw_amplitudes)
        ]
        raw_orientations = spec.get("orientations")
        if not isinstance(raw_orientations, list) or len(raw_orientations) != identities:
            raise ValueError("world.orientations must contain one row per identity")
        orientations: list[list[float]] = []
        for mode, row in enumerate(raw_orientations):
            if not isinstance(row, list) or len(row) != len(amplitudes):
                raise ValueError(f"world.orientations[{mode}] must contain one value per level")
            parsed = [_finite(value, f"world.orientations[{mode}][{level}]") for level, value in enumerate(row)]
            if any(value == 0.0 for value in parsed):
                raise ValueError("world orientations must be nonzero")
            orientations.append(parsed)
        noise = _nonnegative(spec.get("noise", 0.0), "world.noise")
        return OrientedFactorMixtureWorld(rng, identities, spacing, amplitudes, orientations, noise)

    if name == "factor-motif-mixture":
        _exact_keys(
            spec,
            {"name", "centers", "factors", "noise"},
            "factor-motif-mixture world",
        )
        centers = _vectors(spec.get("centers"), dimension, "world.centers")
        raw_factors = spec.get("factors")
        if not isinstance(raw_factors, list) or len(raw_factors) != len(centers):
            raise ValueError("world.factors must contain one factor list per center")
        factors: list[list[list[float]]] = []
        levels: int | None = None
        for mode, raw_mode in enumerate(raw_factors):
            if not isinstance(raw_mode, list) or not raw_mode:
                raise ValueError(f"world.factors[{mode}] must be a nonempty list")
            parsed = [
                _vector(vector, dimension, f"world.factors[{mode}][{level}]")
                for level, vector in enumerate(raw_mode)
            ]
            if levels is None:
                levels = len(parsed)
            elif len(parsed) != levels:
                raise ValueError("all modes must contain the same number of factors")
            factors.append(parsed)
        noise = _nonnegative(spec.get("noise", 0.0), "world.noise")
        return FactorMotifMixtureWorld(rng, centers, factors, noise)

    if name == "nested-binary":
        _exact_keys(
            spec,
            {"name", "identities", "levels", "spacing", "amplitude", "ratio"},
            "nested-binary world",
        )
        if dimension != 1:
            raise ValueError("nested-binary requires dimension 1")
        identities = _integer(spec.get("identities", 4), "world.identities", minimum=1)
        levels = _integer(spec.get("levels", 1), "world.levels", minimum=0)
        spacing = _positive(spec.get("spacing", 4.0), "world.spacing")
        amplitude = _nonnegative(spec.get("amplitude", 0.7), "world.amplitude")
        ratio = _positive(spec.get("ratio", 0.25), "world.ratio")
        return NestedBinaryWorld(identities, levels, spacing, amplitude, ratio)

    if name == "ordered-history":
        _exact_keys(
            spec,
            {
                "name", "memory", "bridge_steps", "predictive",
                "kernel_balanced", "cue_amplitude", "bridge_value",
                "outcome_amplitude", "enclosed",
            },
            "ordered-history world",
        )
        if dimension != 1:
            raise ValueError("ordered-history requires dimension 1")
        memory_half_life = _integer(
            spec.get("memory", 100), "world.memory", minimum=1
        )
        bridge_steps = _integer(
            spec.get("bridge_steps", 1), "world.bridge_steps", minimum=1
        )
        predictive = spec.get("predictive", True)
        kernel_balanced = spec.get("kernel_balanced", True)
        if not isinstance(predictive, bool):
            raise ValueError("world.predictive must be boolean")
        if not isinstance(kernel_balanced, bool):
            raise ValueError("world.kernel_balanced must be boolean")
        cue_amplitude = _positive(
            spec.get("cue_amplitude", 2.0), "world.cue_amplitude"
        )
        bridge_value = _finite(
            spec.get("bridge_value", 0.25), "world.bridge_value"
        )
        outcome_amplitude = _positive(
            spec.get("outcome_amplitude", 4.0), "world.outcome_amplitude"
        )
        enclosed = spec.get("enclosed", False)
        if not isinstance(enclosed, bool):
            raise ValueError("world.enclosed must be boolean")
        return OrderedHistoryWorld(
            rng,
            dimension,
            memory_half_life,
            bridge_steps,
            predictive,
            kernel_balanced,
            cue_amplitude, bridge_value, outcome_amplitude, enclosed,
        )

    if name == "ambiguous-regime":
        _exact_keys(
            spec,
            {"name", "block_junctions", "amplitude", "predictive"},
            "ambiguous-regime world",
        )
        block_junctions = _integer(
            spec.get("block_junctions", 50), "world.block_junctions", minimum=1
        )
        amplitude = _positive(spec.get("amplitude", 1.0), "world.amplitude")
        predictive = spec.get("predictive", True)
        if not isinstance(predictive, bool):
            raise ValueError("world.predictive must be boolean")
        return AmbiguousRegimeWorld(
            rng, dimension, block_junctions, amplitude, predictive
        )

    raise ValueError(f"unknown world {name!r}")
