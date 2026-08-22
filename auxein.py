"""Auxein v0.5.0 reference implementation.

This module implements ``spec/auxein.md`` using only the Python standard
library. Geometry mode learns and emits centered geometric knowledge. Predictive
mode keeps that geometry unchanged and adds a private ``E⊕E`` succession
population used only to emit candidate future geometric presentations.

Persistent cognitive state remains deliberately small:

    NETWORK -> ordered LAYERs
                 |- geometric {CELL, Sigma}
                 `- predictive-private {CELL^T, Sigma^T, previous context}

Recognised presentations, contexts, responsibilities and readouts are ephemeral.
Sequence boundaries are explicit: ``step(P)`` is an atomic sequence, while
``sequence([...])`` processes a non-atomic causal sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import math
import struct
import sys
from numbers import Real
from typing import Iterable, Mapping, Sequence


FORMAT_VERSION = 5
SCALARS = {"f32": 4, "f64": 8}
MODES = {"geometry", "predictive"}

Vector = tuple[float, ...]
PredictionTarget = Vector


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


def _mode(value: object) -> str:
    if not isinstance(value, str) or value not in MODES:
        raise ValueError("mode must be 'geometry' or 'predictive'")
    return value


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
    try:
        return math.fsum(x * x for x in a)
    except OverflowError:
        return math.inf


def _dist2(a: Vector, b: Vector) -> float:
    try:
        return math.fsum((x - y) * (x - y) for x, y in zip(a, b, strict=True))
    except OverflowError:
        return math.inf


_MIN_F64_SUBNORMAL = float.fromhex("0x0.0000000000001p-1022")


def _decayed_support(weight: float, lam: float) -> float:
    """One homothetic forgetting step without cognitive death by underflow."""
    if weight <= 0.0:
        return weight
    out = lam * weight
    return _MIN_F64_SUBNORMAL if out <= 0.0 else out


def _orderless_sum(values: Iterable[float]) -> float:
    """High-accuracy expansion sum with no caller-order authority."""
    partials: list[float] = []
    for value in values:
        x = float(value)
        i = 0
        for y0 in partials:
            y = y0
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo != 0.0:
                if i < len(partials):
                    partials[i] = lo
                else:
                    partials.append(lo)
                i += 1
            x = hi
        del partials[i:]
        if x != 0.0:
            partials.append(x)
    return sum(partials, 0.0)

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
            + (self.W / total) * (other.W / total) * delta2
        )
        return Kernel(total, center, variance)

    def ema(self, target: "Kernel", beta: float, lam: float) -> "Kernel":
        if self.W <= 0.0 or target.W <= 0.0:
            raise ValueError("EMA kernels must have positive weight")
        a = _decayed_support(self.W, lam)
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
            + (a / total) * (b / total) * delta2
        )
        return Kernel(total, center, variance)


class _KernelAccumulator:
    """Order-independent positive accumulator for a finite kernel cloud.

    Contributions are kept only until the current presentation is closed.
    Finalisation applies the canonical centered-kernel definition directly:
    one stable pass for W/C, then one stable positive pass for V.  No merge
    order, input permutation or coordinate ordering has numerical authority.
    """

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[tuple[Vector, float, float]] = []

    def add(self, center: Vector, variance: float, weight: float) -> None:
        if weight > 0.0:
            self._items.append((center, variance, weight))

    def get(self) -> Kernel | None:
        if not self._items:
            return None
        if len(self._items) == 1:
            center, variance, weight = self._items[0]
            return Kernel(weight, center, variance)
        first_center, first_variance, _ = self._items[0]
        if all(
            center == first_center and variance == first_variance
            for center, variance, _ in self._items[1:]
        ):
            return Kernel(
                _orderless_sum(weight for _, _, weight in self._items),
                first_center,
                first_variance,
            )
        W = _orderless_sum(weight for _, _, weight in self._items)
        D = len(self._items[0][0])
        C = tuple(
            _orderless_sum(weight * center[j] for center, _, weight in self._items) / W
            for j in range(D)
        )
        V = _orderless_sum(
            weight * (variance + _dist2(center, C))
            for center, variance, weight in self._items
        ) / W
        return Kernel(W, C, V)


# ---------------------------------------------------------------------------
# Persistent layer state


@dataclass(slots=True)
class Layer:
    sigma: list[Kernel]
    cells: list[Kernel]
    temporal_sigma: list[Kernel] = field(default_factory=list)
    temporal_cells: list[Kernel] = field(default_factory=list)
    previous: Kernel | None = None


@dataclass(slots=True)
class _CellStart:
    kernel: Kernel
    norm2: float


@dataclass(slots=True)
class _PopulationResult:
    cells: list[Kernel]
    sigma: list[Kernel]
    context: Kernel | None
    knowledge: list[Kernel]
    seed_requests: list[Kernel]
    transformations: list[dict[str, object]]
    report: dict[str, object]


@dataclass(slots=True)
class _LayerResult:
    output: list[Kernel]
    context: Kernel | None
    present: list[Kernel] | None
    seed_requests: list[Kernel]
    transformations: list[dict[str, object]]
    report: dict[str, object]


# ---------------------------------------------------------------------------
# Auxein network


class Auxein:
    """Reference Auxein v0.5.0 network."""

    def __init__(
        self,
        *,
        dimension: int,
        memory: float,
        eta: float = 1.0,
        scalar: str = "f64",
        mode: str = "geometry",
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
        self.mode = _mode(mode)
        self.steps_seen = 0
        self._sequence_open = False
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
    def temporal_kernel_units(self) -> int:
        return (2 * self.dimension + 2) * self.scalar_bytes

    @property
    def network_units(self) -> int:
        return 34 + 2 * self.scalar_bytes

    @property
    def layer_units(self) -> int:
        if self.mode == "geometry":
            return 16
        return 33 + self.kernel_units

    @property
    def min_units(self) -> int:
        return self.network_units + self.layer_units

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
        layers: list[dict[str, object]] = []
        for layer in self.layers:
            payload: dict[str, object] = {
                "sigma": [self._kernel_state(k) for k in layer.sigma],
                "cells": [self._kernel_state(k) for k in layer.cells],
            }
            if self.mode != "geometry":
                payload.update({
                    "temporal_sigma": [self._kernel_state(k) for k in layer.temporal_sigma],
                    "temporal_cells": [self._kernel_state(k) for k in layer.temporal_cells],
                    "previous": None if layer.previous is None else self._kernel_state(layer.previous),
                })
            layers.append(payload)
        return {
            "format_version": FORMAT_VERSION,
            "dimension": self.dimension,
            "scalar": self.scalar,
            "memory": self._project.real(self.memory),
            "eta": self._project.real(self.eta),
            "mode": self.mode,
            "steps_seen": self.steps_seen,
            "layers": layers,
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
    ) -> "Auxein":
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        expected = {
            "format_version", "dimension", "scalar", "memory", "eta", "mode",
            "steps_seen", "layers",
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
        mode = _mode(state["mode"])
        network = cls(
            dimension=dimension,
            memory=_positive(state["memory"], "state.memory"),
            eta=_eta(state["eta"]),
            scalar=str(state["scalar"]),
            mode=mode,
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
            if mode == "geometry":
                _exact_keys(raw_layer, {"sigma", "cells"}, f"state.layers[{li}]")
            else:
                _exact_keys(
                    raw_layer,
                    {"sigma", "cells", "temporal_sigma", "temporal_cells", "previous"},
                    f"state.layers[{li}]",
                )
            sigma = network._load_kernel_list(
                raw_layer["sigma"], f"state.layers[{li}].sigma", network.dimension
            )
            cells = network._load_kernel_list(
                raw_layer["cells"], f"state.layers[{li}].cells", network.dimension
            )
            if mode != "geometry":
                temporal_sigma = network._load_kernel_list(
                    raw_layer["temporal_sigma"],
                    f"state.layers[{li}].temporal_sigma",
                    2 * network.dimension,
                )
                temporal_cells = network._load_kernel_list(
                    raw_layer["temporal_cells"],
                    f"state.layers[{li}].temporal_cells",
                    2 * network.dimension,
                )
                previous = network._load_optional_kernel(
                    raw_layer["previous"], f"state.layers[{li}].previous", network.dimension
                )
            else:
                temporal_sigma = []
                temporal_cells = []
                previous = None
            layers.append(Layer(sigma, cells, temporal_sigma, temporal_cells, previous))
        network.layers = layers
        network.steps_seen = steps_seen
        network._validate_state()
        return network

    def _load_optional_kernel(self, value: object, label: str, dimension: int) -> Kernel | None:
        if value is None:
            return None
        items = self._load_kernel_list([value], label, dimension)
        return items[0]

    def _load_kernel_list(self, value: object, label: str, dimension: int) -> list[Kernel]:
        if not isinstance(value, list):
            raise ValueError(f"{label} must be a list")
        kernels: list[Kernel] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(f"{label}[{index}] must be an object")
            _exact_keys(item, {"W", "C", "V"}, f"{label}[{index}]")
            W = _positive(item["W"], f"{label}[{index}].W")
            C = _vector(item["C"], dimension, f"{label}[{index}].C")
            V = _finite(item["V"], f"{label}[{index}].V")
            if V < 0.0:
                raise ValueError(f"{label}[{index}].V must be nonnegative")
            projected = self._project_kernel(Kernel(W, C, V))
            # Strict scalar validation: loading may canonicalize -0.0 only.
            if projected.W != W or projected.C != C or projected.V != V:
                raise ValueError(f"{label}[{index}] is not exactly representable in state.scalar")
            kernels.append(projected)
        return kernels

    def _validate_compartment(
        self, cells: Sequence[Kernel], sigma: Sequence[Kernel], label: str
    ) -> None:
        self._assert_unique_geometry(cells, f"{label}.cells")
        self._assert_unique_geometry(sigma, f"{label}.sigma")
        cell_norms = [(cell, _norm2(cell.C)) for cell in cells]
        for cell in cells:
            if _zero(cell.C):
                raise ValueError(f"{label} CELL center must be nonzero")
        for kernel in sigma:
            if _zero(kernel.C):
                raise ValueError(f"{label} Sigma center must be nonzero")
            atom = Kernel(1.0, kernel.C, kernel.V)
            c2 = _norm2(kernel.C)
            if any(self._concern(cell, norm2, atom, c2)[0] for cell, norm2 in cell_norms):
                raise ValueError(f"{label} Sigma kernel is already covered by a CELL")

    def _validate_state(self) -> None:
        if not self.layers:
            raise ValueError("network must contain L0")
        for li, layer in enumerate(self.layers):
            self._validate_compartment(layer.cells, layer.sigma, f"layers[{li}]")
            if self.mode != "geometry":
                self._validate_compartment(
                    layer.temporal_cells, layer.temporal_sigma, f"layers[{li}].temporal"
                )
                if layer.previous is not None and len(layer.previous.C) != self.dimension:
                    raise ValueError("previous context has invalid dimension")
            elif layer.temporal_sigma or layer.temporal_cells or layer.previous is not None:
                raise ValueError("geometry mode cannot contain temporal state")
        # A normal state cannot keep a redundant empty terminal layer when its
        # predecessor has no geometric knowledge capable of a vertical frontier.
        if len(self.layers) > 1:
            last = self.layers[-1]
            previous = self.layers[-2]
            if (
                not last.cells and not last.sigma
                and not last.temporal_cells and not last.temporal_sigma
                and not previous.cells
            ):
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
        if self.mode == "geometry":
            payload_units = sum(
                (len(layer.sigma) + len(layer.cells)) * self.kernel_units
                for layer in self.layers
            )
        else:
            payload_units = sum(
                (len(layer.sigma) + len(layer.cells)) * self.kernel_units
                + (len(layer.temporal_sigma) + len(layer.temporal_cells))
                * self.temporal_kernel_units
                for layer in self.layers
            )
        return self.network_units + self.layer_units * len(self.layers) + payload_units

    def _invalidate_previous(self) -> None:
        if self.mode != "geometry":
            for layer in self.layers:
                layer.previous = None

    def _layer_has_cells(self, layer: Layer) -> bool:
        return bool(layer.cells or (self.mode != "geometry" and layer.temporal_cells))

    def _force_solvency(self, transformations: list[dict[str, object]]) -> None:
        if self.maintenance_units() <= self.budget_units:
            return

        # Work in progress is discarded as one simultaneous wave in both spaces.
        removed_sigma = sum(
            len(layer.sigma)
            + (len(layer.temporal_sigma) if self.mode != "geometry" else 0)
            for layer in self.layers
        )
        if removed_sigma:
            for layer in self.layers:
                layer.sigma.clear()
                if self.mode != "geometry":
                    layer.temporal_sigma.clear()
            transformations.append({
                "phase": "solvency", "type": "clear_sigma", "count": removed_sigma
            })

        trimmed = 0
        while len(self.layers) > 1 and not self._layer_has_cells(self.layers[-1]):
            self.layers.pop()
            trimmed += 1
        if trimmed:
            transformations.append({
                "phase": "solvency", "type": "trim_layers", "count": trimmed
            })
        if self.maintenance_units() <= self.budget_units:
            self._invalidate_previous()
            return

        # CELL destruction is one common K ordering across geometric and
        # temporal knowledge. Equal K values are destroyed as whole waves.
        valued: list[tuple[float, int, str, Kernel]] = []
        g_counts = [len(layer.cells) for layer in self.layers]
        t_counts = [len(layer.temporal_cells) for layer in self.layers]
        for li, layer in enumerate(self.layers):
            valued.extend((self.cell_value(cell), li, "geometry", cell) for cell in layer.cells)
            if self.mode != "geometry":
                valued.extend(
                    (self.cell_value(cell), li, "temporal", cell)
                    for cell in layer.temporal_cells
                )
        if not valued:
            self._invalidate_previous()
            if self.maintenance_units() > self.budget_units:
                raise RuntimeError("minimal Auxein state exceeds the execution budget")
            return
        valued.sort(key=lambda item: item[0])

        active_layers = len(self.layers)
        cutoff: float | None = None
        removed_cells = 0
        waves = 0
        position = 0
        while position < len(valued):
            k = valued[position][0]
            stop = position
            while stop < len(valued) and valued[stop][0] == k:
                _, li, space, _ = valued[stop]
                if space == "geometry":
                    g_counts[li] -= 1
                else:
                    t_counts[li] -= 1
                removed_cells += 1
                stop += 1
            waves += 1
            while (
                active_layers > 1
                and g_counts[active_layers - 1] == 0
                and t_counts[active_layers - 1] == 0
            ):
                active_layers -= 1
            simulated_units = self.network_units + self.layer_units * active_layers
            simulated_units += sum(g_counts[:active_layers]) * self.kernel_units
            if self.mode != "geometry":
                simulated_units += sum(t_counts[:active_layers]) * self.temporal_kernel_units
            cutoff = k
            if simulated_units <= self.budget_units:
                break
            position = stop
        else:
            # The minimal state is guaranteed affordable by set_budget/from_state.
            cutoff = math.inf

        assert cutoff is not None
        for layer in self.layers:
            layer.cells = [cell for cell in layer.cells if self.cell_value(cell) > cutoff]
            if self.mode != "geometry":
                layer.temporal_cells = [
                    cell for cell in layer.temporal_cells if self.cell_value(cell) > cutoff
                ]
        trimmed_after = 0
        while len(self.layers) > 1 and not self._layer_has_cells(self.layers[-1]):
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
        self._invalidate_previous()
        if self.maintenance_units() > self.budget_units:
            raise RuntimeError("internal error: forced solvency did not reach the budget")

    @staticmethod
    def cell_value(cell: Kernel) -> float:
        c2 = _norm2(cell.C)
        return c2 / (c2 + cell.V)

    # ----- presentation parsing ----------------------------------------

    def _presentation(self, value: object) -> list[Kernel]:
        """Parse the canonical weighted boundary or vector-list sugar."""
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or len(value) == 0
        ):
            raise ValueError("presentation must be a nonempty sequence")

        # Sugar: a finite nonempty list of vectors is the uniform point-kernel
        # presentation. Exact duplicate vectors are coalesced canonically.
        if all(_is_vector(item, self.dimension) for item in value):
            vectors = [
                _vector(item, self.dimension, f"presentation[{i}]")
                for i, item in enumerate(value)
            ]
            counts: dict[Vector, int] = {}
            for vector in vectors:
                counts[vector] = counts.get(vector, 0) + 1
            total = len(vectors)
            return [
                Kernel(count / total, vector, 0.0)
                for vector, count in sorted(counts.items())
            ]

        kernels: list[Kernel] = []
        for i, item in enumerate(value):
            if isinstance(item, Kernel):
                W = _positive(item.W, f"presentation[{i}].W")
                C = _vector(item.C, self.dimension, f"presentation[{i}].C")
                V = _finite(item.V, f"presentation[{i}].V")
            elif (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
                and len(item) == 3
            ):
                W = _positive(item[0], f"presentation[{i}][0]")
                C = _vector(item[1], self.dimension, f"presentation[{i}][1]")
                V = _finite(item[2], f"presentation[{i}][2]")
            else:
                raise ValueError(
                    "presentation items must be all vectors or weighted [W, C, V] kernels"
                )
            if V < 0.0:
                raise ValueError(f"presentation[{i}] variance must be nonnegative")
            kernels.append(Kernel(W, C, V))

        total = math.fsum(kernel.W for kernel in kernels)
        if not 0.0 < total <= 1.0:
            raise ValueError("presentation mass must lie in (0, 1]")
        return self._coalesce_kernels(kernels)

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

    def _complete_presentation(self, kernels: Iterable[Kernel]) -> list[Kernel]:
        """Coalesce a cognitive presentation and complete its mass to one."""
        coalesced = self._coalesce_kernels(kernels)
        total = math.fsum(kernel.W for kernel in coalesced)
        remainder = math.fsum((1.0, -total))
        if remainder < 0.0:
            if abs(remainder) <= 8.0 * math.ulp(1.0):
                remainder = 0.0
            else:
                raise RuntimeError("internal error: presentation mass exceeds one")
        if remainder > 0.0:
            coalesced = self._coalesce_kernels(
                [*coalesced, Kernel(remainder, (0.0,) * self.dimension, 0.0)]
            )
        return coalesced

    @staticmethod
    def _readout_presentation(presentation: Sequence[Kernel]) -> list[list[object]]:
        return [[kernel.W, list(kernel.C), kernel.V] for kernel in presentation]

    @staticmethod
    def _presentation_key(presentation: Sequence[Kernel]) -> tuple[tuple[float, Vector, float], ...]:
        return tuple((kernel.W, kernel.C, kernel.V) for kernel in presentation)

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
        """Canonical CONCERN for an internal atom ``(r,c,v)``.

        The ordinary squared form is retained whenever ``||c||²`` is a
        positive finite binary64 value.  At the representational extremes,
        where squaring a finite nonzero center would overflow or underflow,
        the same homogeneous inequalities are evaluated in units of the
        incoming atom.  The returned gain is then in those common scaled
        units; ALLOCATE only uses gains from the same atom, so the common
        positive factor has no authority.
        """
        if (not math.isfinite(norm2_atom_center)) or (
            norm2_atom_center == 0.0 and not _zero(atom.C)
        ):
            scale = max(
                max(abs(x) for x in atom.C),
                math.sqrt(atom.V) if atom.V > 0.0 else 0.0,
            )
            if scale == 0.0:
                return False, 0.0
            x_scaled = tuple(x / scale for x in atom.C)
            c_scaled = tuple(x / scale for x in kernel.C)
            x2 = _norm2(x_scaled)
            geometric = _dist2(x_scaled, c_scaled)
            if not geometric < x2:
                return False, x2 - geometric
            vin = (math.sqrt(atom.V) / scale) ** 2 if atom.V > 0.0 else 0.0
            vmem = (math.sqrt(kernel.V) / scale) ** 2 if kernel.V > 0.0 else 0.0
            rhs = _norm2(c_scaled) + vmem
            da = geometric + vin
            return da < rhs, x2 - geometric

        geometric = _dist2(atom.C, kernel.C)
        da = geometric + atom.V
        d0 = norm2_atom_center + atom.V
        return da < d0 and da < norm2_center + kernel.V, norm2_atom_center - geometric

    def _process_population(
        self,
        cells: list[Kernel],
        sigma: list[Kernel],
        presentation: list[Kernel],
        *,
        layer_index: int,
        phase: str,
        detailed: bool,
        collect_context: bool,
    ) -> _PopulationResult:
        cells_start = [_CellStart(cell.copy(), _norm2(cell.C)) for cell in cells]
        sigma_start = [kernel.copy() for kernel in sigma]
        sigma_norms = [_norm2(kernel.C) for kernel in sigma_start]

        cell_targets = [_KernelAccumulator() for _ in cells_start]
        cell_received = [0.0] * len(cells_start)
        context = _KernelAccumulator() if collect_context else None
        knowledge_weights: dict[Vector, list[float]] = {}
        unknown: list[tuple[Kernel, float]] = []
        recognised_atoms = 0
        recognition_count = 0

        # CONCERN -> ALLOCATE -> RECOGNISE from the frozen CELL state.
        for atom in presentation:
            c2 = _norm2(atom.C)
            if _zero(atom.C):
                unknown.append((atom, c2))
                continue
            concerned: list[tuple[int, float]] = []
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

            recognised_by_center: dict[Vector, float] = {}
            for ci, gain in concerned:
                recognised_by_center.setdefault(cells_start[ci].kernel.C, gain)
            recognition_count += len(recognised_by_center)

            if context is not None:
                # R_s is the exact quotient of recognised snapshot values.
                # Knowledge mass is partitioned only by CONCERN gain; support
                # and ALLOCATE responsibilities have no authority here.
                items = sorted(recognised_by_center.items())
                if len(items) == 1:
                    weighted = [(items[0][0], atom.W)]
                else:
                    gain_scale = max(gain for _, gain in items)
                    scaled = [(center, gain / gain_scale) for center, gain in items]
                    denominator = math.fsum(gain for _, gain in scaled)
                    weighted: list[tuple[Vector, float]] = []
                    assigned: list[float] = []
                    for center, gain in scaled[:-1]:
                        omega = atom.W * gain / denominator
                        assigned.append(omega)
                        weighted.append((center, omega))
                    last_center, _ = scaled[-1]
                    last_weight = atom.W - math.fsum(assigned)
                    if last_weight <= 0.0:
                        # Extreme rounding fallback: the ratio form remains
                        # authoritative and all gains are strictly positive.
                        last_weight = atom.W * scaled[-1][1] / denominator
                    weighted.append((last_center, last_weight))
                for center, omega in weighted:
                    context.add(center, 0.0, omega)
                    knowledge_weights.setdefault(center, []).append(omega)

        context_kernel = None if context is None else context.get()
        knowledge = [
            Kernel(math.fsum(weights), center, 0.0)
            for center, weights in sorted(knowledge_weights.items())
            if math.fsum(weights) > 0.0
        ]

        # CELL update exactly once per preexisting cell.
        updated_cells: list[Kernel] = []
        changed_cell_geometries: set[tuple[Vector, float]] = set()
        for ci, snapshot in enumerate(cells_start):
            target = cell_targets[ci].get()
            if target is None:
                candidate = Kernel(
                    _decayed_support(snapshot.kernel.W, self.lam),
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
        cells = self._coalesce_projected_kernels(updated_cells)

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
                candidate = Kernel(_decayed_support(old.W, self.lam), old.C, old.V)
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
            cells = self._coalesce_projected_kernels([*cells, *promoted])
            changed_cell_geometries.update(kernel.geometry for kernel in promoted)
            transformations.append({
                "phase": phase,
                "type": "promote",
                "layer": layer_index,
                "count": len(promoted),
            })

        current_with_norm = [(cell, _norm2(cell.C)) for cell in cells]
        changed_with_norm = [
            (cell, norm2)
            for cell, norm2 in current_with_norm
            if cell.geometry in changed_cell_geometries
        ]
        cleaned_sigma: list[Kernel] = []
        for kernel in remaining_sigma:
            candidates = current_with_norm if kernel.geometry in sigma_changed else changed_with_norm
            atom = Kernel(1.0, kernel.C, kernel.V)
            c2 = _norm2(kernel.C)
            covered = any(
                self._concern(cell, norm2, atom, c2)[0]
                for cell, norm2 in candidates
            )
            if not covered:
                cleaned_sigma.append(kernel)
        sigma = cleaned_sigma

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
                "cell_count_after": len(cells),
                "sigma_count_before": len(sigma_start),
                "sigma_count_after": len(sigma),
                "promoted": len(promoted),
                "seed_requests": len(admissible_seeds),
                "recognition_count": recognition_count,
                "knowledge_mass": math.fsum(kernel.W for kernel in knowledge),
                "cell_responsibility_mass": list(cell_received),
            }
        return _PopulationResult(
            cells, sigma, context_kernel, knowledge, admissible_seeds, transformations, report
        )

    def _process_layer(
        self, layer_index: int, presentation: list[Kernel], *, detailed: bool
    ) -> _LayerResult:
        layer = self.layers[layer_index]
        result = self._process_population(
            layer.cells,
            layer.sigma,
            presentation,
            layer_index=layer_index,
            phase="geometry",
            detailed=detailed,
            collect_context=True,
        )
        layer.cells = result.cells
        layer.sigma = result.sigma
        context_kernel = result.context
        context_emitted = (
            context_kernel is not None
            and context_kernel.V > 0.0
            and not _zero(context_kernel.C)
        )
        output = [context_kernel] if context_emitted and context_kernel is not None else []
        present = None if not result.knowledge else self._complete_presentation(result.knowledge)
        report = result.report
        if detailed:
            report.update({
                "context_emitted": context_emitted,
                "output_atom_count": len(output),
                "output_mass": 0.0 if context_kernel is None else context_kernel.W,
                "context_center": None if context_kernel is None else list(context_kernel.C),
                "context_variance": None if context_kernel is None else context_kernel.V,
                "present_atom_count": 0 if present is None else len(present),
                "present_mass": 0.0 if present is None else math.fsum(k.W for k in present),
            })
        return _LayerResult(
            output,
            context_kernel,
            present,
            result.seed_requests,
            result.transformations,
            report,
        )

    def _process_temporal(
        self, layer_index: int, presentation: list[Kernel], *, detailed: bool
    ) -> _PopulationResult:
        layer = self.layers[layer_index]
        result = self._process_population(
            layer.temporal_cells,
            layer.temporal_sigma,
            presentation,
            layer_index=layer_index,
            phase="temporal",
            detailed=detailed,
            collect_context=False,
        )
        layer.temporal_cells = result.cells
        layer.temporal_sigma = result.sigma
        return result

    def _temporal_atom(self, previous: Kernel, current: Kernel) -> Kernel:
        return Kernel(
            previous.W * current.W,
            previous.C + current.C,
            previous.V + current.V,
        )

    def _split_temporal(self, center: Vector) -> tuple[Vector, Vector]:
        return center[: self.dimension], center[self.dimension :]

    @staticmethod
    def _point_relative_gain_with_norm(
        a: Vector, a2: float, b: Vector
    ) -> float | None:
        """Point/point CONCERN gain with a precomputed current squared norm."""
        b2 = _norm2(b)
        a_extreme = (not math.isfinite(a2)) or (0.0 < a2 < sys.float_info.min) or (
            a2 == 0.0 and not _zero(a)
        )
        b_extreme = (not math.isfinite(b2)) or (0.0 < b2 < sys.float_info.min) or (
            b2 == 0.0 and not _zero(b)
        )
        if a_extreme or b_extreme:
            scale = max(
                max((abs(x) for x in a), default=0.0),
                max((abs(x) for x in b), default=0.0),
            )
            if scale == 0.0:
                return None
            aa = tuple(x / scale for x in a)
            bb = tuple(x / scale for x in b)
            a2 = _norm2(aa)
            b2 = _norm2(bb)
            d2 = _dist2(aa, bb)
        else:
            d2 = _dist2(a, b)
        if not (d2 < a2 and d2 < b2):
            return None
        return (a2 - d2) / a2

    @staticmethod
    def _point_relative_gain(a: Vector, b: Vector) -> float | None:
        """Point/point CONCERN and its relative gain, with extreme scaling."""
        return Auxein._point_relative_gain_with_norm(a, _norm2(a), b)

    @staticmethod
    def _point_concern(a: Vector, b: Vector) -> bool:
        """Canonical point/point CONCERN."""
        return Auxein._point_relative_gain(a, b) is not None

    def _predict_temporal(
        self, layer: Layer, context: Kernel
    ) -> dict[PredictionTarget, float]:
        """Project frozen temporal CELLs to target -> maximal relative gain."""
        current = context.C
        current2 = _norm2(current)
        out: dict[PredictionTarget, float] = {}
        for cell in layer.temporal_cells:
            source, target = self._split_temporal(cell.C)
            gamma = self._point_relative_gain_with_norm(current, current2, source)
            if gamma is not None:
                previous = out.get(target)
                if previous is None or gamma > previous:
                    out[target] = gamma
        return out

    # ----- public presentation / sequence API ---------------------------

    def begin_sequence(self, *, resume: bool = False) -> None:
        """Open an explicit causal sequence.

        By default all previous-context registers are cleared. ``resume=True``
        is the explicit opt-in required to continue causal registers restored
        from a mid-sequence persistent state.
        """
        if self._sequence_open:
            raise RuntimeError("a sequence is already open")
        if not resume:
            self._invalidate_previous()
        self._sequence_open = True

    def end_sequence(self) -> None:
        """Close the current causal sequence and destroy causal continuity."""
        if not self._sequence_open:
            raise RuntimeError("no sequence is open")
        self._invalidate_previous()
        self._sequence_open = False

    def sequence_step(
        self, presentation: object, *, detailed_report: bool = False
    ) -> dict[str, object]:
        """Process one presentation inside an explicitly open sequence."""
        if not self._sequence_open:
            raise RuntimeError("sequence_step requires begin_sequence()")
        return self._step_in_sequence(presentation, detailed_report=detailed_report)

    def step(self, presentation: object, *, detailed_report: bool = False) -> dict[str, object]:
        """Process one atomic sequence.

        Successive calls to ``step`` never create temporal continuity. Use
        ``sequence`` or the explicit begin/sequence_step/end API for a real
        causal sequence.
        """
        if self._sequence_open:
            raise RuntimeError("step cannot be used while a sequence is open")
        self.begin_sequence()
        try:
            return self._step_in_sequence(presentation, detailed_report=detailed_report)
        finally:
            self.end_sequence()

    def sequence(
        self, presentations: object, *, detailed_report: bool = False
    ) -> list[dict[str, object]]:
        """Process a finite nonempty explicit causal sequence."""
        if self._sequence_open:
            raise RuntimeError("sequence cannot be nested")
        if (
            not isinstance(presentations, Sequence)
            or isinstance(presentations, (str, bytes, bytearray))
            or len(presentations) == 0
        ):
            raise ValueError("sequence must be a nonempty sequence of presentations")
        self.begin_sequence()
        try:
            return [
                self._step_in_sequence(presentation, detailed_report=detailed_report)
                for presentation in presentations
            ]
        finally:
            self.end_sequence()

    def consume(
        self, present_family: object, *, detailed_report: bool = False
    ) -> list[dict[str, object]]:
        """Consume an upstream present family as depth-ordered atomic sequences.

        This is the canonical direct NETWORK→NETWORK composition helper. An
        empty family still destroys any residual causal register.
        """
        if self._sequence_open:
            raise RuntimeError("consume cannot be used while a sequence is open")
        if (
            not isinstance(present_family, Sequence)
            or isinstance(present_family, (str, bytes, bytearray))
        ):
            raise ValueError("present_family must be a sequence of presentations")
        self._invalidate_previous()
        reports: list[dict[str, object]] = []
        for presentation in present_family:
            reports.append(self.step(presentation, detailed_report=detailed_report))
        self._invalidate_previous()
        return reports

    def _step_in_sequence(
        self, presentation: object, *, detailed_report: bool = False
    ) -> dict[str, object]:
        current = self._presentation(presentation)
        transformations: list[dict[str, object]] = []
        self._force_solvency(transformations)
        maintenance_open = self.maintenance_units()
        layer_count_start = len(self.layers)
        present_family: list[list[Kernel]] = []
        future_by_key: dict[tuple[tuple[float, Vector, float], ...], list[Kernel]] = {}
        all_seed_requests: list[tuple[str, int, Kernel]] = []
        layer_reports: list[dict[str, object]] = []
        temporal_reports: list[dict[str, object]] = []
        contexts: list[Kernel | None] = [None] * layer_count_start
        frontier_requested = False

        # Complete geometric recursion first. Predictive-private temporal
        # cognition can observe these contexts but never feeds back into the
        # geometry of the same presentation.
        for layer_index in range(layer_count_start):
            if not current:
                break
            result = self._process_layer(layer_index, current, detailed=detailed_report)
            contexts[layer_index] = result.context
            if result.present is not None:
                present_family.append(result.present)
            transformations.extend(result.transformations)
            all_seed_requests.extend(
                ("geometry", layer_index, seed) for seed in result.seed_requests
            )
            if detailed_report:
                layer_reports.append(result.report)
            if layer_index == layer_count_start - 1 and result.output and self.beta > 0.0:
                frontier_requested = True
            current = result.output

        if self.mode == "predictive":
            for layer_index in range(layer_count_start):
                layer = self.layers[layer_index]
                previous = layer.previous
                context = contexts[layer_index]

                # Prediction reads only the temporal snapshot that existed
                # before this presentation's temporal learning phase.
                if context is not None:
                    for target, gamma in self._predict_temporal(layer, context).items():
                        candidate = self._complete_presentation(
                            [Kernel(context.W * gamma, target, 0.0)]
                        )
                        future_by_key.setdefault(self._presentation_key(candidate), candidate)

                if previous is not None and context is not None:
                    temporal_presentation = [self._temporal_atom(previous, context)]
                    result = self._process_temporal(
                        layer_index, temporal_presentation, detailed=detailed_report
                    )
                    transformations.extend(result.transformations)
                    all_seed_requests.extend(
                        ("temporal", layer_index, seed) for seed in result.seed_requests
                    )
                    if detailed_report:
                        temporal_reports.append(result.report)

                # P_k is causal state, not learned memory: it advances even at
                # eta=0, but only inside the explicit sequence boundary.
                layer.previous = None if context is None else self._project_kernel(context)

        # One material growth transaction spans geometric and predictive-private
        # temporal seeds plus the optional frontier layer.
        projected_requests: dict[tuple[str, int], list[Kernel]] = {}
        for space, layer_index, seed in all_seed_requests:
            projected = self._project_kernel(seed)
            if _zero(projected.C):
                continue
            layer = self.layers[layer_index]
            cells = layer.cells if space == "geometry" else layer.temporal_cells
            atom = Kernel(1.0, projected.C, projected.V)
            c2 = _norm2(projected.C)
            if any(self._concern(cell, _norm2(cell.C), atom, c2)[0] for cell in cells):
                continue
            projected_requests.setdefault((space, layer_index), []).append(projected)

        future_sigma: dict[tuple[str, int], list[Kernel]] = {}
        net_new_geometry = 0
        net_new_temporal = 0
        geometric_seed_requests = 0
        temporal_seed_requests = 0
        for (space, layer_index), requests in projected_requests.items():
            layer = self.layers[layer_index]
            existing = layer.sigma if space == "geometry" else layer.temporal_sigma
            future = self._coalesce_projected_kernels([*existing, *requests])
            future_sigma[(space, layer_index)] = future
            added = max(0, len(future) - len(existing))
            if space == "geometry":
                net_new_geometry += added
                geometric_seed_requests += len(requests)
            else:
                net_new_temporal += added
                temporal_seed_requests += len(requests)

        growth_units = (
            net_new_geometry * self.kernel_units
            + net_new_temporal * self.temporal_kernel_units
            + (self.layer_units if frontier_requested else 0)
        )
        transaction_requested = bool(future_sigma) or frontier_requested
        if transaction_requested:
            if self.maintenance_units() + growth_units <= self.budget_units:
                for (space, layer_index), future in future_sigma.items():
                    layer = self.layers[layer_index]
                    if space == "geometry":
                        layer.sigma = future
                    else:
                        layer.temporal_sigma = future
                if frontier_requested:
                    self.layers.append(Layer([], []))
                transformations.append({
                    "phase": "growth",
                    "type": "commit",
                    "geometric_seeds": geometric_seed_requests,
                    "temporal_seeds": temporal_seed_requests,
                    "seeds": geometric_seed_requests + temporal_seed_requests,
                    "layer_created": frontier_requested,
                    "units": growth_units,
                })
            else:
                transformations.append({
                    "phase": "growth",
                    "type": "reject",
                    "geometric_seeds": geometric_seed_requests,
                    "temporal_seeds": temporal_seed_requests,
                    "seeds": geometric_seed_requests + temporal_seed_requests,
                    "layer_requested": frontier_requested,
                    "units": growth_units,
                })

        self.steps_seen += 1
        maintenance_end = self.maintenance_units()
        if maintenance_end > self.budget_units:
            raise RuntimeError("internal error: post-step state exceeds budget")

        readout: dict[str, object] = {
            "present": [self._readout_presentation(item) for item in present_family]
        }
        if self.mode == "predictive":
            readout["future"] = [
                self._readout_presentation(future_by_key[key])
                for key in sorted(future_by_key)
            ]

        return {
            "step_index": self.steps_seen - 1,
            "readout": readout,
            "transformations": transformations,
            "maintenance_open_units": maintenance_open,
            "maintenance_units": maintenance_end,
            "budget_units": self.budget_units,
            "layer_reports": layer_reports,
            "temporal_reports": temporal_reports,
        }

    # ----- derived views -------------------------------------------------

    def summary(self) -> dict[str, object]:
        maintenance = self.maintenance_units()
        return {
            "steps_seen": self.steps_seen,
            "dimension": self.dimension,
            "scalar": self.scalar,
            "memory": self.memory,
            "eta": self.eta,
            "mode": self.mode,
            "chi": self.chi,
            "alpha": self.alpha,
            "effective_alpha": self.beta,
            "layer_count": len(self.layers),
            "cells_per_layer": [len(layer.cells) for layer in self.layers],
            "sigma_per_layer": [len(layer.sigma) for layer in self.layers],
            "temporal_cells_per_layer": [
                len(layer.temporal_cells) if self.mode != "geometry" else 0
                for layer in self.layers
            ],
            "temporal_sigma_per_layer": [
                len(layer.temporal_sigma) if self.mode != "geometry" else 0
                for layer in self.layers
            ],
            "previous_context_per_layer": [
                layer.previous is not None if self.mode != "geometry" else False
                for layer in self.layers
            ],
            "maintenance_units": maintenance,
            "budget": str(self.budget),
            "budget_units": self.budget_units,
            "budget_margin_units": self.budget_units - maintenance,
            "is_solvent": maintenance <= self.budget_units,
        }


__all__ = ["Auxein", "Kernel", "Layer", "FORMAT_VERSION", "MODES"]
