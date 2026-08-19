"""Auxein v0.2.0 reference implementation.

This module implements the mathematical/material canon in
``Auxein_Canon_v0.2.0.md`` using only the Python standard library.

The persistent cognitive state is deliberately small:

    NETWORK -> ordered LAYERs -> {CELL kernels, private Sigma kernels}

External vectors enter as point kernels ``(r, C, V=0)``.  Internal layers
receive at most one contextual kernel per presentation.  All contexts,
responsibilities, reports, allocation tables and readouts are ephemeral.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import math
import struct
from numbers import Real
from typing import Iterable, Mapping, Sequence


FORMAT_VERSION = 2
SCALARS = {"f32": 4, "f64": 8}

Vector = tuple[float, ...]
Recognition = tuple[str, Vector, Vector]


# ---------------------------------------------------------------------------
# Numeric and validation helpers


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    return 0.0 if out == 0.0 else out


def _positive(value: object, label: str) -> float:
    out = _finite(value, label)
    if out <= 0.0:
        raise ValueError(f"{label} must be positive")
    return out


def _eta(value: object) -> float:
    out = _finite(value, "eta")
    if not 0.0 <= out <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    return out


def _vector(value: object, dimension: int, label: str) -> Vector:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a vector")
    if len(value) != dimension:
        raise ValueError(f"{label} must have dimension {dimension}")
    return tuple(_finite(component, f"{label}[{i}]") for i, component in enumerate(value))


def _is_vector(value: object, dimension: int) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    if len(value) != dimension:
        return False
    return all(isinstance(v, Real) and not isinstance(v, bool) for v in value)


def _zero(v: Vector) -> bool:
    return all(component == 0.0 for component in v)


def _norm2(a: Vector) -> float:
    return math.fsum(x * x for x in a)


def _dist2(a: Vector, b: Vector) -> float:
    return math.fsum((x - y) * (x - y) for x, y in zip(a, b, strict=True))


def _decimal_nonnegative(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        out = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a decimal number") from exc
    if not out.is_finite() or out < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return out


def _exact_keys(mapping: Mapping[str, object], expected: set[str], label: str) -> None:
    keys = set(mapping)
    missing = expected - keys
    unknown = keys - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError(f"{label}: {'; '.join(details)}")


class _Projector:
    __slots__ = ("scalar",)

    def __init__(self, scalar: str) -> None:
        if scalar not in SCALARS:
            raise ValueError("scalar must be 'f32' or 'f64'")
        self.scalar = scalar

    def real(self, value: float) -> float:
        # Public inputs are validated before reaching the projector; all
        # internal callers already supply binary64 numbers.
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("persistent real must be finite")
        if self.scalar == "f32":
            try:
                value = struct.unpack("!f", struct.pack("!f", value))[0]
            except OverflowError as exc:
                raise ValueError("persistent f32 overflow") from exc
            if not math.isfinite(value):
                raise ValueError("persistent f32 overflow")
        return 0.0 if value == 0.0 else value

    def vector(self, value: Vector) -> Vector:
        return tuple(self.real(component) for component in value)


# ---------------------------------------------------------------------------
# Centered kernels


@dataclass(slots=True)
class Kernel:
    W: float
    C: Vector
    V: float

    def copy(self) -> "Kernel":
        return Kernel(self.W, self.C, self.V)

    @property
    def energy(self) -> float:
        return self.W * self.V

    @property
    def geometry(self) -> tuple[Vector, float]:
        return self.C, self.V

    def merged(self, other: "Kernel") -> "Kernel":
        if self.W <= 0.0 or other.W <= 0.0:
            raise ValueError("kernel weights must be positive")
        total = self.W + other.W
        delta2 = _dist2(self.C, other.C)
        ratio = other.W / total
        center = tuple(
            a + ratio * (b - a) for a, b in zip(self.C, other.C, strict=True)
        )
        variance = (
            (self.W * self.V + other.W * other.V) / total
            + (self.W * other.W / (total * total)) * delta2
        )
        return Kernel(total, center, variance)

    def ema(self, target: "Kernel", beta: float, lam: float) -> "Kernel":
        if self.W <= 0.0 or target.W <= 0.0:
            raise ValueError("EMA kernels must have positive weight")
        a = lam * self.W
        b = beta * target.W
        if b == 0.0:
            return Kernel(a, self.C, self.V)
        total = a + b
        if total <= 0.0:
            raise ValueError("EMA produced nonpositive support")
        delta2 = _dist2(self.C, target.C)
        ratio = b / total
        center = tuple(
            old + ratio * (new - old)
            for old, new in zip(self.C, target.C, strict=True)
        )
        variance = (
            (a * self.V + b * target.V) / total
            + (a * b / (total * total)) * delta2
        )
        return Kernel(total, center, variance)


class _KernelAccumulator:
    """Stable deterministic accumulator for a weighted point cloud.

    Coalesced presentation atoms are processed in lexicographic vector order,
    making the floating trajectory independent of caller iteration order.
    """

    __slots__ = ("_kernel",)

    def __init__(self) -> None:
        self._kernel: Kernel | None = None

    def add(self, center: Vector, variance: float, weight: float) -> None:
        if weight <= 0.0:
            return
        atom = Kernel(weight, center, variance)
        self._kernel = atom if self._kernel is None else self._kernel.merged(atom)

    def get(self) -> Kernel | None:
        return None if self._kernel is None else self._kernel.copy()


# ---------------------------------------------------------------------------
# Persistent layer state


@dataclass(slots=True)
class Layer:
    sigma: list[Kernel]
    cells: list[Kernel]


@dataclass(slots=True)
class _CellStart:
    kernel: Kernel
    norm2: float


@dataclass(slots=True)
class _LayerResult:
    output: list[Kernel]
    readout: set[Recognition]
    seed_requests: list[Kernel]
    transformations: list[dict[str, object]]
    report: dict[str, object]


# ---------------------------------------------------------------------------
# Auxein network


class Auxein:
    """Reference Auxein v0.2.0 network."""

    def __init__(
        self,
        *,
        dimension: int,
        memory: float,
        eta: float = 1.0,
        scalar: str = "f64",
        universe: str = "auxein",
        budget: object | None = None,
        budget_units: int | None = None,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension
        self.scalar = scalar
        self._project = _Projector(scalar)
        self.memory = self._project.real(_positive(memory, "memory"))
        self.eta = self._project.real(_eta(eta))
        if not isinstance(universe, str) or not universe:
            raise ValueError("universe must be a nonempty string")
        self.universe = universe
        self.steps_seen = 0
        self.layers: list[Layer] = [Layer([], [])]
        self._refresh_clock()
        self._set_initial_budget(budget=budget, budget_units=budget_units)

    # ----- configuration -------------------------------------------------

    def _refresh_clock(self) -> None:
        self.chi = 2.0 ** (-1.0 / self.memory)
        self.alpha = -math.expm1(-math.log(2.0) / self.memory)
        self.beta = self.eta * self.alpha
        self.lam = 1.0 - self.beta

    @property
    def scalar_bytes(self) -> int:
        return SCALARS[self.scalar]

    @property
    def kernel_units(self) -> int:
        return (self.dimension + 2) * self.scalar_bytes

    @property
    def network_units(self) -> int:
        return 33 + 2 * self.scalar_bytes

    @property
    def min_units(self) -> int:
        return self.network_units + 16

    def _budget_to_units(self, value: object) -> int:
        capacity = _decimal_nonnegative(value, "budget")
        extra = (capacity * Decimal(self.kernel_units)).to_integral_value(rounding=ROUND_FLOOR)
        return self.min_units + int(extra)

    def _set_initial_budget(self, *, budget: object | None, budget_units: int | None) -> None:
        if (budget is None) == (budget_units is None):
            raise ValueError("provide exactly one of budget or budget_units")
        if budget_units is not None:
            if isinstance(budget_units, bool) or not isinstance(budget_units, int) or budget_units < 0:
                raise ValueError("budget_units must be a nonnegative integer")
            units = budget_units
        else:
            units = self._budget_to_units(budget)
        if units < self.min_units:
            raise ValueError("budget is below the minimal executable state")
        self.budget_units = units

    @property
    def budget(self) -> Decimal:
        return Decimal(self.budget_units - self.min_units) / Decimal(self.kernel_units)

    def set_budget(self, *, budget: object | None = None, budget_units: int | None = None) -> None:
        if (budget is None) == (budget_units is None):
            raise ValueError("provide exactly one of budget or budget_units")
        units = self._budget_to_units(budget) if budget is not None else budget_units
        assert units is not None
        if isinstance(units, bool) or not isinstance(units, int) or units < self.min_units:
            raise ValueError("budget is below the minimal executable state")
        self.budget_units = units

    def set_eta(self, eta: float) -> None:
        self.eta = self._project.real(_eta(eta))
        self._refresh_clock()

    # ----- persistence ---------------------------------------------------

    def _project_kernel(self, kernel: Kernel) -> Kernel:
        W = self._project.real(kernel.W)
        C = self._project.vector(kernel.C)
        V = self._project.real(kernel.V)
        if W <= 0.0:
            # A positive support lost only to floating underflow is not a
            # cognitive deletion.  Python's f64 path effectively never hits
            # this in realistic runs; f32 preserves the least positive
            # subnormal as a representation of "still positive".
            if kernel.W > 0.0 and self.scalar == "f32":
                W = struct.unpack("!f", b"\x00\x00\x00\x01")[0]
            else:
                raise ValueError("persistent kernel support is not positive")
        if V < 0.0:
            raise ValueError("persistent kernel variance is negative")
        return Kernel(W, C, V)

    def export_state(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "dimension": self.dimension,
            "scalar": self.scalar,
            "memory": self._project.real(self.memory),
            "eta": self._project.real(self.eta),
            "steps_seen": self.steps_seen,
            "layers": [
                {
                    "sigma": [self._kernel_state(k) for k in layer.sigma],
                    "cells": [self._kernel_state(k) for k in layer.cells],
                }
                for layer in self.layers
            ],
        }

    @staticmethod
    def _kernel_state(kernel: Kernel) -> dict[str, object]:
        return {"W": kernel.W, "C": list(kernel.C), "V": kernel.V}

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, object],
        *,
        budget: object | None = None,
        budget_units: int | None = None,
        universe: str = "auxein",
    ) -> "Auxein":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        expected = {
            "format_version", "dimension", "scalar", "memory", "eta", "steps_seen", "layers",
        }
        _exact_keys(state, expected, "state")
        if state["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported format_version")
        dimension = state["dimension"]
        steps_seen = state["steps_seen"]
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("invalid state.dimension")
        if isinstance(steps_seen, bool) or not isinstance(steps_seen, int) or steps_seen < 0:
            raise ValueError("invalid state.steps_seen")
        network = cls(
            dimension=dimension,
            memory=_positive(state["memory"], "state.memory"),
            eta=_eta(state["eta"]),
            scalar=str(state["scalar"]),
            universe=universe,
            budget=budget,
            budget_units=budget_units,
        )
        raw_layers = state["layers"]
        if not isinstance(raw_layers, list) or not raw_layers:
            raise ValueError("state.layers must be a nonempty list")
        layers: list[Layer] = []
        for li, raw_layer in enumerate(raw_layers):
            if not isinstance(raw_layer, Mapping):
                raise ValueError(f"state.layers[{li}] must be an object")
            _exact_keys(raw_layer, {"sigma", "cells"}, f"state.layers[{li}]")
            sigma = network._load_kernel_list(raw_layer["sigma"], f"state.layers[{li}].sigma")
            cells = network._load_kernel_list(raw_layer["cells"], f"state.layers[{li}].cells")
            layers.append(Layer(sigma, cells))
        network.layers = layers
        network.steps_seen = steps_seen
        network._validate_state()
        return network

    def _load_kernel_list(self, value: object, label: str) -> list[Kernel]:
        if not isinstance(value, list):
            raise ValueError(f"{label} must be a list")
        kernels: list[Kernel] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(f"{label}[{index}] must be an object")
            _exact_keys(item, {"W", "C", "V"}, f"{label}[{index}]")
            W = _positive(item["W"], f"{label}[{index}].W")
            C = _vector(item["C"], self.dimension, f"{label}[{index}].C")
            V = _finite(item["V"], f"{label}[{index}].V")
            if V < 0.0:
                raise ValueError(f"{label}[{index}].V must be nonnegative")
            projected = self._project_kernel(Kernel(W, C, V))
            # Strict scalar validation: loading may canonicalize -0.0 only.
            if projected.W != W or projected.C != C or projected.V != V:
                raise ValueError(f"{label}[{index}] is not exactly representable in state.scalar")
            kernels.append(projected)
        return kernels

    def _validate_state(self) -> None:
        if not self.layers:
            raise ValueError("network must contain L0")
        for li, layer in enumerate(self.layers):
            self._assert_unique_geometry(layer.cells, f"layers[{li}].cells")
            self._assert_unique_geometry(layer.sigma, f"layers[{li}].sigma")
            cell_norms = [(cell, _norm2(cell.C)) for cell in layer.cells]
            for cell in layer.cells:
                if _zero(cell.C):
                    raise ValueError("persistent CELL center must be nonzero")
            for kernel in layer.sigma:
                if _zero(kernel.C):
                    raise ValueError("persistent Sigma center must be nonzero")
                atom = Kernel(1.0, kernel.C, kernel.V)
                c2 = _norm2(kernel.C)
                if any(self._concern(cell, norm2, atom, c2)[0] for cell, norm2 in cell_norms):
                    raise ValueError("persistent Sigma kernel is already covered by a CELL")
        # A normal state cannot keep useless empty terminal layers beyond one.
        while len(self.layers) > 1 and not self.layers[-1].cells and not self.layers[-1].sigma and not self.layers[-2].cells:
            raise ValueError("state contains redundant terminal layers")

    @staticmethod
    def _assert_unique_geometry(kernels: Sequence[Kernel], label: str) -> None:
        seen: set[tuple[Vector, float]] = set()
        for kernel in kernels:
            key = kernel.geometry
            if key in seen:
                raise ValueError(f"{label} contains uncoalesced exact clones")
            seen.add(key)

    # ----- material economy ---------------------------------------------

    def maintenance_units(self) -> int:
        payloads = sum(len(layer.sigma) + len(layer.cells) for layer in self.layers)
        return self.network_units + 16 * len(self.layers) + payloads * self.kernel_units

    def _force_solvency(self, transformations: list[dict[str, object]]) -> None:
        if self.maintenance_units() <= self.budget_units:
            return

        # Work in progress is discarded as one simultaneous wave.
        removed_sigma = sum(len(layer.sigma) for layer in self.layers)
        if removed_sigma:
            for layer in self.layers:
                layer.sigma.clear()
            transformations.append({
                "phase": "solvency", "type": "clear_sigma", "count": removed_sigma
            })

        # Terminal empty layers are material only and can be removed immediately.
        trimmed = 0
        while len(self.layers) > 1 and not self.layers[-1].cells:
            self.layers.pop()
            trimmed += 1
        if trimmed:
            transformations.append({
                "phase": "solvency", "type": "trim_layers", "count": trimmed
            })
        if self.maintenance_units() <= self.budget_units:
            return

        # Canonically, CELL destruction proceeds by increasing exact K waves.
        # Compute all waves once, then simulate payload/layer removal to find
        # the required cutoff.  This is O(C log C + L), not O(C^2) rescanning.
        valued: list[tuple[float, int, Kernel]] = []
        counts = [len(layer.cells) for layer in self.layers]
        total_cells = 0
        for li, layer in enumerate(self.layers):
            for cell in layer.cells:
                valued.append((self.cell_value(cell), li, cell))
                total_cells += 1
        if not valued:
            raise RuntimeError("minimal Auxein state exceeds the execution budget")
        valued.sort(key=lambda item: item[0])

        active_layers = len(counts)
        cutoff: float | None = None
        removed_cells = 0
        waves = 0
        position = 0
        while position < len(valued):
            k = valued[position][0]
            stop = position
            while stop < len(valued) and valued[stop][0] == k:
                _, li, _ = valued[stop]
                counts[li] -= 1
                total_cells -= 1
                removed_cells += 1
                stop += 1
            waves += 1
            while active_layers > 1 and counts[active_layers - 1] == 0:
                active_layers -= 1
            simulated_units = (
                self.network_units
                + 16 * active_layers
                + total_cells * self.kernel_units
            )
            cutoff = k
            if simulated_units <= self.budget_units:
                break
            position = stop
        else:
            raise RuntimeError("minimal Auxein state exceeds the execution budget")

        assert cutoff is not None
        value_by_identity = {id(cell): k for k, _, cell in valued}
        for layer in self.layers:
            layer.cells = [cell for cell in layer.cells if value_by_identity[id(cell)] > cutoff]
        trimmed_after = 0
        while len(self.layers) > 1 and not self.layers[-1].cells:
            self.layers.pop()
            trimmed_after += 1
        transformations.append({
            "phase": "solvency",
            "type": "destroy_cells",
            "count": removed_cells,
            "waves": waves,
            "K_through": cutoff,
        })
        if trimmed_after:
            transformations.append({
                "phase": "solvency", "type": "trim_layers", "count": trimmed_after
            })
        if self.maintenance_units() > self.budget_units:
            raise RuntimeError("internal error: forced solvency did not reach the budget")

    @staticmethod
    def cell_value(cell: Kernel) -> float:
        c2 = _norm2(cell.C)
        return c2 / (c2 + cell.V)

    # ----- presentation parsing ----------------------------------------

    def _presentation(self, value: object) -> list[Kernel]:
        """Build L0's uniform point-kernel presentation.

        The public NETWORK boundary accepts only a finite nonempty sequence
        of vectors.  Internal masses and variances are a LAYER-to-LAYER
        concern and are never accepted from the caller.
        """
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or len(value) == 0
        ):
            raise ValueError("external presentation must be a nonempty sequence of vectors")
        vectors = [
            _vector(item, self.dimension, f"presentation[{i}]")
            for i, item in enumerate(value)
        ]
        mass = 1.0 / len(vectors)
        return self._coalesce_kernels(Kernel(mass, x, 0.0) for x in vectors)

    @staticmethod
    def _coalesce_kernels(kernels: Iterable[Kernel]) -> list[Kernel]:
        groups: dict[tuple[Vector, float], list[float]] = {}
        representative: dict[tuple[Vector, float], Kernel] = {}
        for kernel in kernels:
            key = kernel.geometry
            groups.setdefault(key, []).append(kernel.W)
            representative.setdefault(key, kernel)
        out: list[Kernel] = []
        for key in sorted(groups, key=lambda k: (k[0], k[1])):
            base = representative[key]
            out.append(Kernel(math.fsum(groups[key]), base.C, base.V))
        return out

    def _coalesce_projected_kernels(self, kernels: Iterable[Kernel]) -> list[Kernel]:
        """Coalesce kernels whose C/V are already persistent values.

        Only a summed support needs a new scalar projection; unchanged
        geometry is not redundantly reprojected.
        """
        groups: dict[tuple[Vector, float], list[Kernel]] = {}
        for kernel in kernels:
            groups.setdefault(kernel.geometry, []).append(kernel)
        out: list[Kernel] = []
        for key in sorted(groups, key=lambda k: (k[0], k[1])):
            items = groups[key]
            if len(items) == 1:
                out.append(items[0])
            else:
                out.append(Kernel(self._project.real(math.fsum(k.W for k in items)), key[0], key[1]))
        return out

    # ----- cognitive primitives ----------------------------------------

    @staticmethod
    def _concern(
        kernel: Kernel, norm2_center: float, atom: Kernel, norm2_atom_center: float
    ) -> tuple[bool, float]:
        """Canonical CONCERN for an internal atom ``(r,c,v)``."""
        geometric = _dist2(atom.C, kernel.C)
        da = geometric + atom.V
        d0 = norm2_atom_center + atom.V
        return da < d0 and da < norm2_center + kernel.V, norm2_atom_center - geometric

    def _process_layer(
        self, layer_index: int, presentation: list[Kernel], *, detailed: bool
    ) -> _LayerResult:
        layer = self.layers[layer_index]
        cells_start = [_CellStart(cell.copy(), _norm2(cell.C)) for cell in layer.cells]
        sigma_start = [kernel.copy() for kernel in layer.sigma]
        sigma_norms = [_norm2(kernel.C) for kernel in sigma_start]

        cell_targets = [_KernelAccumulator() for _ in cells_start]
        cell_received = [0.0] * len(cells_start)
        context = _KernelAccumulator()
        readout: set[Recognition] = set()
        unknown: list[tuple[Kernel, float]] = []
        recognised_atoms = 0

        # CONCERN -> ALLOCATE -> RECOGNISE from the frozen CELL state.
        # Context geometry uses recognised values only; learning
        # responsibilities never weight the vertical representation.
        for atom in presentation:
            c2 = _norm2(atom.C)
            if _zero(atom.C):
                unknown.append((atom, c2))
                continue
            concerned: list[tuple[int, float]] = []  # index, gain
            for ci, snapshot in enumerate(cells_start):
                ok, gain = self._concern(snapshot.kernel, snapshot.norm2, atom, c2)
                if ok:
                    concerned.append((ci, gain))
            if not concerned:
                unknown.append((atom, c2))
                continue

            recognised_atoms += 1
            scale = max(cells_start[ci].kernel.W for ci, _ in concerned)
            scores = [
                (ci, (cells_start[ci].kernel.W / scale) * gain)
                for ci, gain in concerned
            ]
            denominator = math.fsum(score for _, score in scores)
            for ci, score in scores:
                rho = atom.W * score / denominator
                cell_targets[ci].add(atom.C, atom.V, rho)
                cell_received[ci] += rho
                center = cells_start[ci].kernel.C
                readout.add((self.universe, atom.C, center))

            # R_s is the exact quotient of recognised snapshot values.
            recognised_values = sorted({cells_start[ci].kernel.C for ci, _ in concerned})
            share = atom.W / len(recognised_values)
            for center in recognised_values:
                context.add(center, 0.0, share)

        # The context is fixed from L^- before any local learning.
        context_kernel = context.get()
        context_emitted = (
            context_kernel is not None
            and context_kernel.V > 0.0
            and not _zero(context_kernel.C)
        )
        output = [context_kernel] if context_emitted and context_kernel is not None else []

        # CELL update exactly once per preexisting cell.
        updated_cells: list[Kernel] = []
        changed_cell_geometries: set[tuple[Vector, float]] = set()
        for ci, snapshot in enumerate(cells_start):
            target = cell_targets[ci].get()
            if target is None:
                candidate = Kernel(
                    self.lam * snapshot.kernel.W,
                    snapshot.kernel.C,
                    snapshot.kernel.V,
                )
            else:
                candidate = snapshot.kernel.ema(target, self.beta, self.lam)
            candidate = self._project_kernel(candidate)
            if _zero(candidate.C):
                continue
            if candidate.geometry != snapshot.kernel.geometry:
                changed_cell_geometries.add(candidate.geometry)
            updated_cells.append(candidate)
        layer.cells = self._coalesce_projected_kernels(updated_cells)

        # DETECT is private and sees only atoms unknown to CELL.
        sigma_targets = [_KernelAccumulator() for _ in sigma_start]
        seed_requests: list[Kernel] = []
        for atom, c2 in unknown:
            if _zero(atom.C):
                continue
            concerned_sigma: list[tuple[int, float]] = []
            for si, kernel in enumerate(sigma_start):
                ok, gain = self._concern(kernel, sigma_norms[si], atom, c2)
                if ok:
                    concerned_sigma.append((si, gain))
            if concerned_sigma:
                scale = max(sigma_start[si].W for si, _ in concerned_sigma)
                scores = [
                    (si, (sigma_start[si].W / scale) * gain)
                    for si, gain in concerned_sigma
                ]
                denominator = math.fsum(score for _, score in scores)
                for si, score in scores:
                    tau = atom.W * score / denominator
                    sigma_targets[si].add(atom.C, atom.V, tau)
            elif self.beta > 0.0:
                seed_requests.append(Kernel(self.beta * atom.W, atom.C, atom.V))

        updated_sigma: list[Kernel] = []
        sigma_changed: set[tuple[Vector, float]] = set()
        for si, old in enumerate(sigma_start):
            target = sigma_targets[si].get()
            if target is None:
                candidate = Kernel(self.lam * old.W, old.C, old.V)
            else:
                candidate = old.ema(target, self.beta, self.lam)
            candidate = self._project_kernel(candidate)
            if _zero(candidate.C):
                continue
            if candidate.geometry != old.geometry:
                sigma_changed.add(candidate.geometry)
            updated_sigma.append(candidate)
        updated_sigma = self._coalesce_projected_kernels(updated_sigma)

        transformations: list[dict[str, object]] = []
        promoted: list[Kernel] = []
        remaining_sigma: list[Kernel] = []
        if self.beta > 0.0:
            for kernel in updated_sigma:
                if kernel.W > self.beta and not _zero(kernel.C):
                    promoted.append(kernel)
                else:
                    remaining_sigma.append(kernel)
        else:
            remaining_sigma = updated_sigma
        if promoted:
            layer.cells = self._coalesce_projected_kernels([*layer.cells, *promoted])
            changed_cell_geometries.update(kernel.geometry for kernel in promoted)
            transformations.append({
                "phase": "geometry",
                "type": "promote",
                "layer": layer_index,
                "count": len(promoted),
            })

        # Drop private work now covered by current knowledge.  Unchanged
        # Sigma was already uncovered at the previous causal boundary, so
        # only changed CELL geometry can newly cover it.
        current_cells = layer.cells
        current_with_norm = [(cell, _norm2(cell.C)) for cell in current_cells]
        changed_with_norm = [
            (cell, norm2)
            for cell, norm2 in current_with_norm
            if cell.geometry in changed_cell_geometries
        ]
        cleaned_sigma: list[Kernel] = []
        for kernel in remaining_sigma:
            candidates = (
                current_with_norm if kernel.geometry in sigma_changed else changed_with_norm
            )
            atom = Kernel(1.0, kernel.C, kernel.V)
            c2 = _norm2(kernel.C)
            covered = any(
                self._concern(cell, norm2, atom, c2)[0]
                for cell, norm2 in candidates
            )
            if not covered:
                cleaned_sigma.append(kernel)
        layer.sigma = cleaned_sigma

        # Seeds were unknown to the frozen CELL state.  Only cells whose
        # geometry changed (including promotions) can newly cover them.
        admissible_seeds: list[Kernel] = []
        for seed in self._coalesce_kernels(seed_requests):
            atom = Kernel(1.0, seed.C, seed.V)
            c2 = _norm2(seed.C)
            covered = any(
                self._concern(cell, norm2, atom, c2)[0]
                for cell, norm2 in changed_with_norm
            )
            if not covered:
                admissible_seeds.append(seed)

        report: dict[str, object] = {}
        if detailed:
            report = {
                "layer_index": layer_index,
                "input_atom_count": len(presentation),
                "input_mass": math.fsum(atom.W for atom in presentation),
                "unknown_atom_count": len(unknown),
                "recognised_atom_count": recognised_atoms,
                "cell_count_before": len(cells_start),
                "cell_count_after": len(layer.cells),
                "sigma_count_before": len(sigma_start),
                "sigma_count_after": len(layer.sigma),
                "promoted": len(promoted),
                "seed_requests": len(admissible_seeds),
                "context_emitted": context_emitted,
                "output_atom_count": len(output),
                "output_mass": 0.0 if not output else output[0].W,
                "context_center": None if context_kernel is None else list(context_kernel.C),
                "context_variance": None if context_kernel is None else context_kernel.V,
                "recognition_count": len(readout),
                "cell_responsibility_mass": list(cell_received),
            }
        return _LayerResult(output, readout, admissible_seeds, transformations, report)

    # ----- public step ---------------------------------------------------

    def step(self, presentation: object, *, detailed_report: bool = False) -> dict[str, object]:
        current = self._presentation(presentation)
        transformations: list[dict[str, object]] = []
        self._force_solvency(transformations)
        maintenance_open = self.maintenance_units()
        layer_count_start = len(self.layers)
        readout: set[Recognition] = set()
        all_seed_requests: list[tuple[int, Kernel]] = []
        layer_reports: list[dict[str, object]] = []
        frontier_requested = False

        for layer_index in range(layer_count_start):
            if not current:
                break
            result = self._process_layer(layer_index, current, detailed=detailed_report)
            readout.update(result.readout)
            transformations.extend(result.transformations)
            all_seed_requests.extend((layer_index, seed) for seed in result.seed_requests)
            if detailed_report:
                layer_reports.append(result.report)
            if layer_index == layer_count_start - 1 and result.output and self.beta > 0.0:
                frontier_requested = True
            current = result.output

        # Global material growth transaction: all surviving Sigma seeds plus
        # the optional frontier layer caused by an emitted terminal context.
        coalesced_requests: dict[tuple[int, Vector, float], list[float]] = {}
        for layer_index, seed in all_seed_requests:
            key = (layer_index, seed.C, seed.V)
            coalesced_requests.setdefault(key, []).append(seed.W)
        seeds: list[tuple[int, Kernel]] = [
            (key[0], Kernel(math.fsum(weights), key[1], key[2]))
            for key, weights in sorted(
                coalesced_requests.items(),
                key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]),
            )
        ]
        growth_cost = len(seeds) * self.kernel_units + (16 if frontier_requested else 0)
        if growth_cost:
            if self.maintenance_units() + growth_cost <= self.budget_units:
                seeds_by_layer: dict[int, list[Kernel]] = {}
                for layer_index, seed in seeds:
                    seeds_by_layer.setdefault(layer_index, []).append(seed)
                for layer_index, layer_seeds in seeds_by_layer.items():
                    projected_seeds = [self._project_kernel(seed) for seed in layer_seeds]
                    self.layers[layer_index].sigma = self._coalesce_projected_kernels(
                        [*self.layers[layer_index].sigma, *projected_seeds]
                    )
                if frontier_requested:
                    self.layers.append(Layer([], []))
                transformations.append({
                    "phase": "growth",
                    "type": "commit",
                    "seeds": len(seeds),
                    "layer_created": frontier_requested,
                    "units": growth_cost,
                })
            else:
                transformations.append({
                    "phase": "growth",
                    "type": "reject",
                    "seeds": len(seeds),
                    "layer_requested": frontier_requested,
                    "units": growth_cost,
                })

        self.steps_seen += 1
        maintenance_end = self.maintenance_units()
        if maintenance_end > self.budget_units:
            raise RuntimeError("internal error: post-step state exceeds budget")

        sorted_readout = sorted(readout, key=lambda item: (item[0], item[1], item[2]))
        return {
            "step_index": self.steps_seen - 1,
            "readout": [
                [universe, list(local_input), list(center)]
                for universe, local_input, center in sorted_readout
            ],
            "transformations": transformations,
            "maintenance_open_units": maintenance_open,
            "maintenance_units": maintenance_end,
            "budget_units": self.budget_units,
            "layer_reports": layer_reports,
        }

    # ----- derived views -------------------------------------------------

    def summary(self) -> dict[str, object]:
        return {
            "steps_seen": self.steps_seen,
            "dimension": self.dimension,
            "universe": self.universe,
            "scalar": self.scalar,
            "memory": self.memory,
            "eta": self.eta,
            "chi": self.chi,
            "alpha": self.alpha,
            "effective_alpha": self.beta,
            "layer_count": len(self.layers),
            "cells_per_layer": [len(layer.cells) for layer in self.layers],
            "sigma_per_layer": [len(layer.sigma) for layer in self.layers],
            "maintenance_units": self.maintenance_units(),
            "budget": str(self.budget),
            "budget_units": self.budget_units,
            "budget_margin_units": self.budget_units - self.maintenance_units(),
            "is_solvent": self.maintenance_units() <= self.budget_units,
        }


__all__ = ["Auxein", "Kernel", "Layer", "FORMAT_VERSION"]
