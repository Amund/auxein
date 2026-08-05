"""Reference implementation of Auxein 0.1.0.

This module is intentionally self-contained: it owns the complete reference
engine and has no dependency on benchmark or test code. It favours literal,
deterministic execution over speed.

The implementation follows ``spec/auxein.md``. Execution-specific extensions
provide f32/f64 persistent rounding, exact integer maintenance, and strict
JSON-compatible state serialization without any file I/O.

The implementation uses only the Python standard library. Vectors are
represented by a tiny deterministic element-wise container; Auxein still
learns no matrix.

Implementation warning: the mathematics permits exact ephemeral indexes and
reusable derived aggregates. Do not recompute layer-wide quantities inside
nested candidate loops, do not route every Cell when only the winner writes,
and profile saturated regimes where conservation searches are active.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from decimal import (
    Decimal,
    InvalidOperation as DecimalInvalidOperation,
    ROUND_FLOOR,
)
import copy
import math
import struct
from typing import Any, Iterable, Iterator, Literal, Mapping, Protocol, Sequence


MODEL_VERSION = "0.1.0"
__version__ = MODEL_VERSION
STATE_SCHEMA_VERSION = 1
Scalar = Literal["f32", "f64"]
BudgetValue = Decimal | int | float | str
TransformationKind = Literal[
    "cell_death",
    "horizontal_reallocation",
    "horizontal_reallocation_death",
    "root_birth",
    "split",
    "truncate",
    "vertical_birth",
]
_TRANSFORMATION_KINDS = frozenset(
    {
        "cell_death",
        "horizontal_reallocation",
        "horizontal_reallocation_death",
        "root_birth",
        "split",
        "truncate",
        "vertical_birth",
    }
)


def _strict_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _positive_finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _budget_decimal(value: BudgetValue, label: str = "budget") -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (DecimalInvalidOperation, ValueError) as exc:
        raise TypeError(f"{label} must be a decimal number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _learning_rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("eta must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class NumericPolicy:
    """Logical persistent real format used by the reference engine.

    Python evaluates expressions in binary64, then the complete persistent
    state is rounded to this format at every causal mutation boundary. This is
    deterministic and makes f32/f64 trajectories distinct without claiming to
    emulate every hardware instruction in native binary32.
    """

    scalar_format: Scalar = "f64"

    def __post_init__(self) -> None:
        if self.scalar_format not in ("f32", "f64"):
            raise ValueError("scalar_format must be 'f32' or 'f64'")

    @property
    def byte_width(self) -> int:
        return 4 if self.scalar_format == "f32" else 8

    @property
    def epsilon(self) -> float:
        return 2.0 ** (-23 if self.scalar_format == "f32" else -52)

    def cast(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("persistent reals must be finite")
        if self.scalar_format == "f64":
            return value
        try:
            return struct.unpack("!f", struct.pack("!f", value))[0]
        except OverflowError as exc:
            raise ValueError("value is not representable in f32") from exc

    def cast_vector(self, value: Sequence[float]) -> "Vector":
        return Vector(self.cast(component) for component in value)

    def next_up(self, value: float) -> float:
        value = self.cast(value)
        if self.scalar_format == "f64":
            result = math.nextafter(value, math.inf)
        elif value == 0.0:
            result = struct.unpack("!f", struct.pack("!I", 1))[0]
        else:
            bits = struct.unpack("!I", struct.pack("!f", value))[0]
            bits = bits + 1 if value > 0.0 else bits - 1
            result = struct.unpack("!f", struct.pack("!I", bits))[0]
        if not math.isfinite(result):
            raise ValueError(f"no finite successor is representable in {self.scalar_format}")
        return result


@dataclass(frozen=True, slots=True)
class MemoryLaw:
    memory_half_life: float
    chi: float = field(init=False)
    alpha: float = field(init=False)

    def __post_init__(self) -> None:
        memory_half_life = _positive_finite_real(self.memory_half_life, "memory")
        exponent = -math.log(2.0) / memory_half_life
        object.__setattr__(self, "memory_half_life", memory_half_life)
        object.__setattr__(self, "chi", math.exp(exponent))
        object.__setattr__(self, "alpha", -math.expm1(exponent))
        if not (0.0 < self.alpha <= 1.0 and 0.0 <= self.chi <= 1.0):
            raise ValueError("memory law is not representable")


class Vector(list[float]):
    """Small mutable vector with deterministic element-wise arithmetic.

    It deliberately implements only the operations required by the reference
    engine.  No matrix algebra or external numerical package is involved.
    """

    __slots__ = ()

    def __init__(self, values: Iterable[float] = ()) -> None:
        super().__init__(map(float, values))

    @classmethod
    def _from_floats(cls, values: list[float]) -> "Vector":
        """Build from already-normalized floats without a second conversion pass."""

        out = cls.__new__(cls)
        list.__init__(out, values)
        return out

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def size(self) -> int:
        return len(self)

    @property
    def ndim(self) -> int:
        return 1

    def copy(self) -> "Vector":
        return Vector(self)

    def _coerce_vector(self, other: object) -> Sequence[float]:
        if isinstance(other, Vector):
            result: Sequence[float] = other
        elif isinstance(other, (list, tuple)):
            result = other
        else:
            raise TypeError("vector operation requires another sequence")
        if len(result) != len(self):
            raise ValueError("vector dimensions differ")
        return result

    def __add__(self, other: object) -> "Vector":
        right = self._coerce_vector(other)
        return Vector._from_floats([a + b for a, b in zip(self, right, strict=True)])

    def __radd__(self, other: object) -> "Vector":
        if other == 0:
            return self.copy()
        return self.__add__(other)

    def __iadd__(self, other: object) -> "Vector":
        right = self._coerce_vector(other)
        for index, value in enumerate(right):
            self[index] += value
        return self

    def __sub__(self, other: object) -> "Vector":
        right = self._coerce_vector(other)
        return Vector._from_floats([a - b for a, b in zip(self, right, strict=True)])

    def __rsub__(self, other: object) -> "Vector":
        left = self._coerce_vector(other)
        return Vector._from_floats([a - b for a, b in zip(left, self, strict=True)])

    def __isub__(self, other: object) -> "Vector":
        right = self._coerce_vector(other)
        for index, value in enumerate(right):
            self[index] -= value
        return self

    def __mul__(self, scalar: object) -> "Vector":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        factor = float(scalar)
        return Vector._from_floats([value * factor for value in self])

    def __rmul__(self, scalar: object) -> "Vector":
        return self.__mul__(scalar)

    def __imul__(self, scalar: object) -> "Vector":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        factor = float(scalar)
        for index in range(len(self)):
            self[index] *= factor
        return self

    def __truediv__(self, scalar: object) -> "Vector":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        divisor = float(scalar)
        return Vector._from_floats([value / divisor for value in self])

    def __itruediv__(self, scalar: object) -> "Vector":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        divisor = float(scalar)
        for index in range(len(self)):
            self[index] /= divisor
        return self


# ---------------------------------------------------------------------------
# Implementation guidance
# ---------------------------------------------------------------------------


IMPLEMENTATION_WARNINGS: tuple[str, ...] = (
    "route latent state only for the winning Cell",
    "reuse layer aggregates once per relevant state version",
    "evaluate each conservation value at most once per victim and state",
    "avoid rebuilding the complete proposal market after unaffected actions",
    "keep exhaustive validation and telemetry outside inner primitives",
    "profile saturated regimes; O(N^3 D) usually signals accidental nesting",
    "close structurally exact quadratic zeros at persistent-format resolution",
    "evaluate nonnegative scatter without subtracting nearly equal totals when possible",
    "recenter on a kernel's own mean by construction, not by subtractive recovery",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AuxeinError(RuntimeError):
    """Base error for the reference engine."""


class InvariantViolation(AuxeinError):
    """A canonical invariant was violated."""


class InvalidStateOperation(AuxeinError):
    """A requested operation is incompatible with the canonical state."""


class InsolventState(AuxeinError):
    """No admissible dry-loss operation can restore solvency."""


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _vector(value: Sequence[float] | Vector, dimension: int) -> Vector:
    out = value if isinstance(value, Vector) else Vector(value)
    if len(out) != dimension:
        raise ValueError(f"expected vector shape {(dimension,)}, got {(len(out),)}")
    if not all(math.isfinite(component) for component in out):
        raise ValueError("vectors must contain only finite values")
    return out


def _zero(dimension: int) -> Vector:
    return Vector._from_floats([0.0] * dimension)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _norm2(value: Sequence[float]) -> float:
    return _dot(value, value)


def _gamma_bound(epsilon: float, operations: int) -> float:
    """Classical ``gamma_n`` bound for a short floating operation chain."""

    count = max(1, int(operations))
    product = float(epsilon) * count
    if product >= 1.0:
        return math.inf
    return product / (1.0 - product)


def _roundoff_bound(*values: float, dimension: int = 1) -> float:
    """Binary64 evaluation bound, never a behavioural threshold."""

    scale = math.fsum(abs(float(value)) for value in values)
    if scale == 0.0:
        return 0.0
    operations = 2 * max(1, dimension) + 8
    return _gamma_bound(math.ulp(1.0), operations) * scale


def _persistent_roundoff_bound(
    scalar_format: Scalar,
    *values: float,
    dimension: int = 1,
) -> float:
    """Resolution bound induced by the chosen persistent real format."""

    scale = math.fsum(abs(float(value)) for value in values)
    if scale == 0.0:
        return 0.0
    operations = 2 * max(1, dimension) + 8
    return _gamma_bound(NumericPolicy(scalar_format).epsilon, operations) * scale


def _nonnegative(value: float, *, scale_values: Iterable[float], dimension: int) -> float:
    """Clamp only a value explainable by first-order floating roundoff."""

    if value >= 0.0:
        return float(value)
    bound = _roundoff_bound(value, *tuple(scale_values), dimension=dimension)
    if value >= -bound:
        return 0.0
    raise InvariantViolation(f"negative quadratic quantity {value!r} below roundoff {bound!r}")


def _resolved_quadratic_difference(
    total: float,
    explained: float,
    *,
    scalar_format: Scalar,
    dimension: int,
    label: str,
) -> float:
    """Close only an unresolved quadratic slack at format resolution."""

    total_f = float(total)
    explained_f = float(explained)
    value = math.fsum((total_f, -explained_f))
    bound = _persistent_roundoff_bound(
        scalar_format,
        total_f,
        explained_f,
        dimension=dimension,
    )
    if abs(value) <= bound:
        return 0.0
    if value < 0.0:
        raise InvariantViolation(
            f"negative {label} {value!r} below persistent roundoff {bound!r}"
        )
    return value


def _exact_zero_vector(value: Vector) -> bool:
    return all(component == 0.0 for component in value)


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    distance = math.dist(left, right)
    return distance * distance


def _nearest_other_sweep(
    points: Sequence[Sequence[float]],
    order: Sequence[int],
    axis_values: Sequence[float],
    axis: int,
    target: Sequence[float],
    excluded_index: int,
) -> int | None:
    """Exact nearest-other search by one-dimensional sweep and pruning.

    Euclidean distance is bounded below by the squared difference on any one
    coordinate.  Sorting on the widest coordinate therefore permits exact
    pruning without persistent neighbourhood state or recursive tree walks.
    """

    count = len(order)
    position = bisect_left(axis_values, target[axis])
    left = position - 1
    right = position
    best_index: int | None = None
    best_distance = math.inf

    while left >= 0 or right < count:
        left_bound = (target[axis] - axis_values[left]) ** 2 if left >= 0 else math.inf
        right_bound = (axis_values[right] - target[axis]) ** 2 if right < count else math.inf
        if left_bound > best_distance and right_bound > best_distance:
            break
        if left_bound <= right_bound:
            candidate = order[left]
            left -= 1
        else:
            candidate = order[right]
            right += 1
        if candidate == excluded_index:
            continue
        distance = _squared_distance(target, points[candidate])
        if distance < best_distance or (
            distance == best_distance
            and (best_index is None or candidate < best_index)
        ):
            best_distance = distance
            best_index = candidate

    return best_index


# ---------------------------------------------------------------------------
# Opaque identities and proposal tokens
# ---------------------------------------------------------------------------


class CellIdentity:
    """Opaque persistent identity: equality and hashing only."""

    __slots__ = ("_token",)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("CellIdentity values are created by Auxein")

    @classmethod
    def _from_token(cls, token: int) -> "CellIdentity":
        identity = cls.__new__(cls)
        identity._token = _strict_positive_int(token, "identity token")
        return identity

    @classmethod
    def _placeholder(cls) -> "CellIdentity":
        identity = cls.__new__(cls)
        identity._token = -1
        return identity

    def __hash__(self) -> int:
        return hash((CellIdentity, self._token))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CellIdentity) and self._token == other._token

    def __repr__(self) -> str:
        return "CellIdentity()"


class IdentityFactory:
    __slots__ = ("_next",)

    def __init__(self, start: int = 1) -> None:
        self._next = _strict_positive_int(start, "identity factory start")

    def new(self) -> CellIdentity:
        identity = CellIdentity._from_token(self._next)
        self._next += 1
        return identity

    @property
    def next_token(self) -> int:
        return self._next


class ProposalToken:
    """Opaque handle allowing a parent to authorize a child-owned action."""

    __slots__ = ("_owner", "_serial")

    def __init__(self, owner: int, serial: int) -> None:
        self._owner = _strict_positive_int(owner, "proposal owner")
        self._serial = _strict_nonnegative_int(serial, "proposal serial")

    @classmethod
    def _forced(cls, serial: int) -> "ProposalToken":
        token = cls.__new__(cls)
        token._owner = 0
        token._serial = _strict_nonnegative_int(serial, "proposal serial")
        return token

    def __hash__(self) -> int:
        return hash((ProposalToken, self._owner, self._serial))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ProposalToken)
            and self._owner == other._owner
            and self._serial == other._serial
        )

    def __repr__(self) -> str:
        return f"ProposalToken(opaque:{self._owner}:{self._serial})"


# ---------------------------------------------------------------------------
# Universal quadratic kernel
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QuadraticKernel:
    dimension: int
    W: float = 0.0
    S: Vector = field(default_factory=Vector)
    Q: float = 0.0

    def __post_init__(self) -> None:
        self.dimension = _strict_positive_int(self.dimension, "dimension")
        if len(self.S) == 0:
            self.S = _zero(self.dimension)
        else:
            self.S = _vector(self.S, self.dimension)
        self.W = float(self.W)
        self.Q = float(self.Q)
        self.validate()

    @classmethod
    def zero(cls, dimension: int) -> "QuadraticKernel":
        return cls(dimension=dimension)

    @classmethod
    def point(cls, value: Sequence[float] | Vector, weight: float) -> "QuadraticKernel":
        vector = Vector(value)
        if not vector:
            raise ValueError("point value must be non-empty")
        if not all(math.isfinite(component) for component in vector):
            raise ValueError("point value must contain only finite values")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("weight must be a real number")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and nonnegative")
        return cls(
            dimension=len(vector),
            W=weight,
            S=vector * weight,
            Q=weight * _norm2(vector),
        )

    def clone(self) -> "QuadraticKernel":
        return QuadraticKernel(self.dimension, self.W, self.S.copy(), self.Q)

    def validate(self) -> None:
        if not math.isfinite(self.W) or self.W < 0.0:
            raise InvariantViolation(f"invalid kernel mass W={self.W!r}")
        if not all(math.isfinite(component) for component in self.S):
            raise InvariantViolation("kernel first moment is not finite")
        if not math.isfinite(self.Q) or self.Q < 0.0:
            raise InvariantViolation(f"invalid kernel second moment Q={self.Q!r}")
        left = self.W * self.Q
        right = _norm2(self.S)
        slack = left - right
        bound = _roundoff_bound(left, right, dimension=self.dimension)
        if slack < -bound:
            raise InvariantViolation(
                f"quadratic invariant violated: WQ={left!r} < ||S||²={right!r}"
            )
        if self.W == 0.0 and (not _exact_zero_vector(self.S) or self.Q != 0.0):
            raise InvariantViolation("zero-mass kernel must have zero moments")

    def update(
        self,
        value: Sequence[float] | Vector,
        relevance: float,
        chi: float,
        *,
        alpha: float | None = None,
        validate_result: bool = True,
    ) -> None:
        value_v = _vector(value, self.dimension)
        relevance = float(relevance)
        chi = float(chi)
        if not math.isfinite(relevance) or relevance < 0.0:
            raise ValueError("relevance must be finite and nonnegative")
        if not (0.0 < chi <= 1.0):
            raise ValueError("chi must lie in (0, 1]")
        alpha = (1.0 - chi) if alpha is None else float(alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must lie in [0, 1]")
        # Compute the complete successor before mutating the persistent kernel.
        # A non-finite intermediate (notably relevance * ||x||²) therefore leaves
        # the previous state untouched.
        try:
            new_W = self.W + alpha * (relevance - self.W)
            new_S = Vector(
                self.S[index]
                + alpha * (relevance * value_v[index] - self.S[index])
                for index in range(self.dimension)
            )
            target_q = relevance * _norm2(value_v)
            new_Q = self.Q + alpha * (target_q - self.Q)
        except OverflowError as exc:
            raise InvariantViolation("quadratic kernel update overflowed") from exc

        if (
            not math.isfinite(new_W)
            or new_W < 0.0
            or not all(math.isfinite(component) for component in new_S)
            or not math.isfinite(new_Q)
            or new_Q < 0.0
        ):
            raise InvariantViolation("quadratic kernel update produced a non-finite state")

        if validate_result:
            # Validate a detached candidate so failure cannot partially commit.
            QuadraticKernel(self.dimension, new_W, new_S.copy(), new_Q)

        self.W = new_W
        self.S = new_S
        self.Q = new_Q

    def decay(
        self, chi: float, *, alpha: float | None = None, validate_result: bool = True
    ) -> None:
        alpha = (1.0 - chi) if alpha is None else float(alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must lie in [0, 1]")
        self.W += alpha * (0.0 - self.W)
        for index in range(self.dimension):
            self.S[index] += alpha * (0.0 - self.S[index])
        self.Q += alpha * (0.0 - self.Q)
        if validate_result:
            self.validate()

    @property
    def mean(self) -> Vector:
        if self.W <= 0.0:
            return _zero(self.dimension)
        return self.S / self.W

    @property
    def move_power(self) -> float:
        if self.W <= 0.0:
            return 0.0
        return _norm2(self.S) / self.W

    @property
    def structural_power(self) -> float:
        return self.structural_power_for("f64")

    def structural_power_for(self, scalar_format: Scalar) -> float:
        if self.W <= 0.0:
            return 0.0
        move = self.move_power
        return _resolved_quadratic_difference(
            self.Q,
            move,
            scalar_format=scalar_format,
            dimension=self.dimension,
            label="structural power",
        )

    def recenter_on_mean(
        self,
        *,
        scalar_format: Scalar = "f64",
        validate_result: bool = True,
    ) -> None:
        """Recenter on this kernel's own mean with exact first-moment closure."""

        if self.W <= 0.0:
            self.S = _zero(self.dimension)
            self.Q = 0.0
        else:
            self.Q = self.structural_power_for(scalar_format)
            self.S = _zero(self.dimension)
        if validate_result:
            self.validate()

    def recenter(
        self,
        delta: Sequence[float] | Vector,
        *,
        delta_norm: float | None = None,
        scalar_format: Scalar = "f64",
        validate_result: bool = True,
    ) -> None:
        delta_v = _vector(delta, self.dimension)
        del delta_norm  # retained for API compatibility; stable form does not need it
        if self.W <= 0.0:
            if validate_result:
                self.validate()
            return
        structural = self.structural_power_for(scalar_format)
        for index in range(self.dimension):
            self.S[index] -= self.W * delta_v[index]
        self.Q = structural + _norm2(self.S) / self.W
        if validate_result:
            self.validate()

    def scale_coordinates(
        self, alpha: float, *, validate_result: bool = True
    ) -> None:
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("coordinate scale must be finite and nonnegative")
        self.S *= alpha
        self.Q *= alpha * alpha
        if validate_result:
            self.validate()

    @staticmethod
    def add(left: "QuadraticKernel", right: "QuadraticKernel") -> "QuadraticKernel":
        if left.dimension != right.dimension:
            raise ValueError("kernel dimensions differ")
        return QuadraticKernel(
            left.dimension,
            left.W + right.W,
            left.S + right.S,
            left.Q + right.Q,
        )

    def neutral_split(self) -> tuple["QuadraticKernel", "QuadraticKernel"]:
        half = QuadraticKernel(self.dimension, self.W / 2.0, self.S / 2.0, self.Q / 2.0)
        return half, half.clone()


# ---------------------------------------------------------------------------
# Latent routing
# ---------------------------------------------------------------------------


BranchSign = Literal["+", "-"]


@dataclass(frozen=True, slots=True)
class LatentState:
    parent_mean: Vector
    plus_mean: Vector
    minus_mean: Vector
    delta_plus: Vector
    delta_minus: Vector
    defined: bool


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    r_plus: float
    r_minus: float
    emission_sign: BranchSign
    latent: LatentState


def latent_state(plus: QuadraticKernel, minus: QuadraticKernel) -> LatentState:
    if plus.dimension != minus.dimension:
        raise ValueError("latent kernel dimensions differ")
    parent_weight = plus.W + minus.W
    zero = _zero(plus.dimension)
    if parent_weight <= 0.0 or plus.W <= 0.0 or minus.W <= 0.0:
        return LatentState(zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero.copy(), False)
    parent_mean = (plus.S + minus.S) / parent_weight
    plus_mean = plus.S / plus.W
    minus_mean = minus.S / minus.W
    return LatentState(
        parent_mean=parent_mean,
        plus_mean=plus_mean,
        minus_mean=minus_mean,
        delta_plus=plus_mean - parent_mean,
        delta_minus=minus_mean - parent_mean,
        defined=True,
    )


def route_latent(
    plus: QuadraticKernel,
    minus: QuadraticKernel,
    value: Sequence[float] | Vector,
    relevance: float,
) -> RoutingDecision:
    """Apply the exact ordered +/- routing law on pre-injection state."""

    if plus.dimension != minus.dimension:
        raise ValueError("latent kernel dimensions differ")
    value_v = _vector(value, plus.dimension)
    relevance = float(relevance)
    if relevance < 0.0 or not math.isfinite(relevance):
        raise ValueError("relevance must be finite and nonnegative")

    latent = latent_state(plus, minus)
    parent_weight = plus.W + minus.W

    if relevance == 0.0:
        # The sign is still deterministic, although the zero-mass emission is inert.
        if not latent.defined:
            return RoutingDecision(0.0, 0.0, "+", latent)
    if parent_weight <= 0.0:
        return RoutingDecision(relevance / 2.0, relevance / 2.0, "+", latent)
    if plus.W <= 0.0 or minus.W <= 0.0:
        raise InvariantViolation(
            "nonempty canonical latent parent must have two positive branches"
        )

    axis = latent.plus_mean - latent.minus_mean
    residual = value_v - latent.parent_mean

    if _exact_zero_vector(axis):
        if _exact_zero_vector(residual):
            return RoutingDecision(relevance / 2.0, relevance / 2.0, "+", latent)
        return RoutingDecision(relevance, 0.0, "+", latent)

    signed = _dot(axis, residual)
    if signed > 0.0:
        return RoutingDecision(relevance, 0.0, "+", latent)
    if signed < 0.0:
        return RoutingDecision(0.0, relevance, "-", latent)
    return RoutingDecision(relevance / 2.0, relevance / 2.0, "+", latent)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellRead:
    recognition: float
    error: Vector
    relevance: float
    routing: RoutingDecision
    emitted_form: Vector


@dataclass(slots=True)
class Cell:
    identity: CellIdentity
    center: Vector
    plus: QuadraticKernel
    minus: QuadraticKernel

    def __post_init__(self) -> None:
        self.center = _vector(self.center, self.plus.dimension)
        if self.plus.dimension != self.minus.dimension:
            raise ValueError("cell latent kernel dimensions differ")
        self.validate()

    @property
    def dimension(self) -> int:
        return self.plus.dimension

    @property
    def parent(self) -> QuadraticKernel:
        return QuadraticKernel.add(self.plus, self.minus)

    @property
    def mass(self) -> float:
        return self.plus.W + self.minus.W

    @property
    def parent_mean(self) -> Vector:
        weight = self.mass
        if weight <= 0.0:
            return _zero(self.dimension)
        return (self.plus.S + self.minus.S) / weight

    @property
    def parent_second_moment(self) -> float:
        return self.plus.Q + self.minus.Q

    def validate(self) -> None:
        self.plus.validate()
        self.minus.validate()
        if len(self.center) != self.dimension or not all(math.isfinite(component) for component in self.center):
            raise InvariantViolation("invalid cell center")
        if self.mass > 0.0 and (self.plus.W <= 0.0 or self.minus.W <= 0.0):
            raise InvariantViolation("nonempty canonical cell must have two positive branches")

    def recognition_and_error(self, value: Vector, radius: float) -> tuple[float, Vector]:
        delta = value - self.center
        if radius > 0.0:
            error = delta / radius
            return math.exp(-_norm2(error)), error
        if value == self.center:
            return 1.0, _zero(self.dimension)
        return 0.0, _zero(self.dimension)

    def read_from_recognition(
        self, error: Vector, recognition: float, relevance: float
    ) -> CellRead:
        """Route only after the Layer has selected this Cell as writer."""

        write_relevance = relevance * recognition
        routing = route_latent(self.plus, self.minus, error, write_relevance)
        emitted = (
            routing.latent.delta_plus.copy()
            if routing.emission_sign == "+"
            else routing.latent.delta_minus.copy()
        )
        return CellRead(recognition, error, write_relevance, routing, emitted)

    def read(self, value: Vector, radius: float, relevance: float) -> CellRead:
        recognition, error = self.recognition_and_error(value, radius)
        return self.read_from_recognition(error, recognition, relevance)

    def inject(
        self, read: CellRead | None, chi: float, alpha: float, *, validate_state: bool = True
    ) -> None:
        if read is None:
            self.plus.decay(chi, alpha=alpha, validate_result=validate_state)
            self.minus.decay(chi, alpha=alpha, validate_result=validate_state)
        else:
            self.plus.update(
                read.error, read.routing.r_plus, chi, alpha=alpha, validate_result=validate_state
            )
            self.minus.update(
                read.error, read.routing.r_minus, chi, alpha=alpha, validate_result=validate_state
            )

    def move_and_recenter(
        self,
        radius_used_for_error: float,
        *,
        scalar_format: Scalar = "f64",
        validate_state: bool = True,
    ) -> Vector:
        if self.mass <= 0.0:
            return _zero(self.dimension)
        normalized = self.parent_mean
        self.center = self.center + radius_used_for_error * normalized
        self.plus.recenter(
            normalized,
            scalar_format=scalar_format,
            validate_result=validate_state,
        )
        self.minus.recenter(
            normalized,
            scalar_format=scalar_format,
            validate_result=validate_state,
        )
        return normalized

    @property
    def split_gain(self) -> float:
        parent_weight = self.mass
        if parent_weight <= 0.0 or self.plus.W <= 0.0 or self.minus.W <= 0.0:
            return 0.0
        axis_norm = math.fsum(
            (
                self.plus.S[index] / self.plus.W
                - self.minus.S[index] / self.minus.W
            )
            ** 2
            for index in range(self.dimension)
        )
        return _nonnegative(
            (self.plus.W * self.minus.W / parent_weight) * axis_norm,
            scale_values=(self.parent_second_moment, self.plus.Q, self.minus.Q),
            dimension=self.dimension,
        )

    def scale_internal_coordinates(
        self, alpha: float, *, validate_state: bool = True
    ) -> None:
        self.plus.scale_coordinates(alpha, validate_result=validate_state)
        self.minus.scale_coordinates(alpha, validate_result=validate_state)

    def materialize_split(
        self,
        radius: float,
        factory: IdentityFactory,
        *,
        scalar_format: Scalar = "f64",
        validate_state: bool = True,
    ) -> "Cell":
        gain = self.split_gain
        if not gain > 0.0:
            raise InvalidStateOperation("cell has no strictly positive split request")

        parent_mean = self.parent_mean
        plus_mean = self.plus.mean
        minus_mean = self.minus.mean
        delta_plus = plus_mean - parent_mean
        delta_minus = minus_mean - parent_mean

        mother_parent = self.plus.clone()
        daughter_parent = self.minus.clone()
        mother_parent.recenter(delta_plus, scalar_format=scalar_format)
        daughter_parent.recenter(delta_minus, scalar_format=scalar_format)

        old_center = self.center.copy()
        self.center = old_center + radius * delta_plus
        self.plus, self.minus = mother_parent.neutral_split()

        daughter_plus, daughter_minus = daughter_parent.neutral_split()
        daughter = Cell(
            identity=factory.new(),
            center=old_center + radius * delta_minus,
            plus=daughter_plus,
            minus=daughter_minus,
        )
        if validate_state:
            self.validate()
        return daughter


# ---------------------------------------------------------------------------
# Terminal bud and owner concordance
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OwnerMoments:
    dimension: int
    lambda_plus: float = 0.0
    f_plus: Vector = field(default_factory=Vector)
    lambda_minus: float = 0.0
    f_minus: Vector = field(default_factory=Vector)

    def __post_init__(self) -> None:
        if len(self.f_plus) == 0:
            self.f_plus = _zero(self.dimension)
        else:
            self.f_plus = _vector(self.f_plus, self.dimension)
        if len(self.f_minus) == 0:
            self.f_minus = _zero(self.dimension)
        else:
            self.f_minus = _vector(self.f_minus, self.dimension)
        self.validate()

    def validate(self) -> None:
        if self.lambda_plus < 0.0 or self.lambda_minus < 0.0:
            raise InvariantViolation("negative owner mass")
        if not math.isfinite(self.lambda_plus) or not math.isfinite(self.lambda_minus):
            raise InvariantViolation("nonfinite owner mass")
        if not all(math.isfinite(component) for component in self.f_plus) or not all(math.isfinite(component) for component in self.f_minus):
            raise InvariantViolation("nonfinite owner first moment")
        if self.lambda_plus == 0.0 and not _exact_zero_vector(self.f_plus):
            raise InvariantViolation("zero plus owner mass with nonzero moment")
        if self.lambda_minus == 0.0 and not _exact_zero_vector(self.f_minus):
            raise InvariantViolation("zero minus owner mass with nonzero moment")

    def update(
        self,
        value: Vector,
        r_plus: float,
        r_minus: float,
        chi: float,
        *,
        alpha: float | None = None,
        validate_result: bool = True,
    ) -> None:
        alpha = (1.0 - chi) if alpha is None else float(alpha)
        self.lambda_plus += alpha * (r_plus - self.lambda_plus)
        self.lambda_minus += alpha * (r_minus - self.lambda_minus)
        for index in range(self.dimension):
            component = value[index]
            self.f_plus[index] += alpha * (r_plus * component - self.f_plus[index])
            self.f_minus[index] += alpha * (r_minus * component - self.f_minus[index])
        if validate_result:
            self.validate()

    def decay(
        self, chi: float, *, alpha: float | None = None, validate_result: bool = True
    ) -> None:
        alpha = (1.0 - chi) if alpha is None else float(alpha)
        self.lambda_plus += alpha * (0.0 - self.lambda_plus)
        self.lambda_minus += alpha * (0.0 - self.lambda_minus)
        for index in range(self.dimension):
            self.f_plus[index] += alpha * (0.0 - self.f_plus[index])
            self.f_minus[index] += alpha * (0.0 - self.f_minus[index])
        if validate_result:
            self.validate()

    def scale_coordinates(
        self, alpha: float, *, validate_result: bool = True
    ) -> None:
        self.f_plus *= alpha
        self.f_minus *= alpha
        if validate_result:
            self.validate()

    @property
    def distinction(self) -> tuple[float, Vector]:
        if self.lambda_plus <= 0.0 or self.lambda_minus <= 0.0:
            return 0.0, _zero(self.dimension)
        mean_plus = self.f_plus / self.lambda_plus
        mean_minus = self.f_minus / self.lambda_minus
        weight = (
            self.lambda_plus * self.lambda_minus
            / (self.lambda_plus + self.lambda_minus)
        )
        return weight, mean_plus - mean_minus


@dataclass(frozen=True, slots=True)
class BudRead:
    value: Vector
    relevance: float
    source: CellIdentity | None
    routing: RoutingDecision


@dataclass(slots=True)
class Bud:
    dimension: int
    plus: QuadraticKernel
    minus: QuadraticKernel
    owners: dict[CellIdentity, OwnerMoments]
    scalar_format: Scalar = "f64"

    @classmethod
    def empty(
        cls,
        dimension: int,
        sources: Iterable[CellIdentity] = (),
        scalar_format: Scalar = "f64",
    ) -> "Bud":
        owners = {source: OwnerMoments(dimension) for source in sources}
        return cls(
            dimension=dimension,
            plus=QuadraticKernel.zero(dimension),
            minus=QuadraticKernel.zero(dimension),
            owners=owners,
            scalar_format=scalar_format,
        )

    @property
    def parent(self) -> QuadraticKernel:
        return QuadraticKernel.add(self.plus, self.minus)

    def validate(self) -> None:
        self.plus.validate()
        self.minus.validate()
        if self.plus.dimension != self.dimension or self.minus.dimension != self.dimension:
            raise InvariantViolation("bud dimension mismatch")
        for owner in self.owners.values():
            owner.validate()
        sum_lp = sum(owner.lambda_plus for owner in self.owners.values())
        sum_lm = sum(owner.lambda_minus for owner in self.owners.values())
        sum_fp = sum((owner.f_plus for owner in self.owners.values()), start=_zero(self.dimension))
        sum_fm = sum((owner.f_minus for owner in self.owners.values()), start=_zero(self.dimension))
        for actual, expected, name in (
            (sum_lp, self.plus.W, "plus owner mass"),
            (sum_lm, self.minus.W, "minus owner mass"),
        ):
            scale = max(1.0, abs(actual), abs(expected))
            bound = NumericPolicy(self.scalar_format).epsilon * max(1, self.dimension) * scale
            if abs(actual - expected) > bound:
                raise InvariantViolation(f"{name} does not sum to bud branch")
        for actual_v, expected_v, name in (
            (sum_fp, self.plus.S, "plus owner moment"),
            (sum_fm, self.minus.S, "minus owner moment"),
        ):
            diff_norm = math.sqrt(_norm2(actual_v - expected_v))
            scale = max(1.0, math.sqrt(_norm2(actual_v)), math.sqrt(_norm2(expected_v)))
            bound = NumericPolicy(self.scalar_format).epsilon * max(1, self.dimension) * scale
            if diff_norm > bound:
                raise InvariantViolation(f"{name} does not sum to bud branch")

    def add_source(
        self, source: CellIdentity, *, validate_aggregate: bool = True
    ) -> None:
        if source in self.owners:
            raise InvariantViolation("bud source identity already exists")
        self.owners[source] = OwnerMoments(self.dimension)
        if validate_aggregate:
            self.validate()

    def read(self, value: Vector, relevance: float, source: CellIdentity | None) -> BudRead:
        routing = route_latent(self.plus, self.minus, value, relevance)
        return BudRead(value.copy(), relevance, source, routing)

    def inject(
        self, read: BudRead, chi: float, alpha: float, *, validate_aggregate: bool = True
    ) -> None:
        self.plus.update(
            read.value, read.routing.r_plus, chi, alpha=alpha, validate_result=validate_aggregate
        )
        self.minus.update(
            read.value, read.routing.r_minus, chi, alpha=alpha, validate_result=validate_aggregate
        )

        for owner in self.owners.values():
            owner.decay(chi, alpha=alpha, validate_result=False)

        if read.relevance > 0.0:
            if read.source is None:
                raise InvariantViolation("positive bud relevance requires a source identity")
            if read.source not in self.owners:
                raise InvariantViolation(
                    "terminal source has no preallocated owner record; topology/maintenance drift"
                )
            owner = self.owners[read.source]
            plus_factor = alpha * read.routing.r_plus
            minus_factor = alpha * read.routing.r_minus
            owner.lambda_plus += plus_factor
            owner.lambda_minus += minus_factor
            for index in range(self.dimension):
                component = read.value[index]
                owner.f_plus[index] += plus_factor * component
                owner.f_minus[index] += minus_factor * component
            if validate_aggregate:
                owner.validate()

        if validate_aggregate:
            self.validate()

    @property
    def split_gain(self) -> float:
        parent_weight = self.plus.W + self.minus.W
        if parent_weight <= 0.0 or self.plus.W <= 0.0 or self.minus.W <= 0.0:
            return 0.0
        axis = self.plus.S / self.plus.W - self.minus.S / self.minus.W
        return _nonnegative(
            (self.plus.W * self.minus.W / parent_weight) * _norm2(axis),
            scale_values=(self.plus.Q + self.minus.Q, self.plus.Q, self.minus.Q),
            dimension=self.dimension,
        )

    @property
    def concordance(self) -> float:
        # Evaluate the explicit cross-owner form incrementally. This is O(ND),
        # stores no owner pair, and returns exact zero when fewer than two
        # owners carry a distinction. It avoids the catastrophic cancellation
        # of ||sum(w d)||² - sum(w² ||d||²).
        total_weight = 0.0
        vector_sum = _zero(self.dimension)
        cross_power = 0.0
        for owner in self.owners.values():
            weight, distinction = owner.distinction
            if weight <= 0.0:
                continue
            cross_power += 2.0 * weight * _dot(vector_sum, distinction)
            vector_sum += weight * distinction
            total_weight += weight
        if total_weight <= 0.0:
            return 0.0
        return cross_power / total_weight

    def scale_coordinates(
        self, alpha: float, *, validate_aggregate: bool = True
    ) -> None:
        self.plus.scale_coordinates(alpha, validate_result=validate_aggregate)
        self.minus.scale_coordinates(alpha, validate_result=validate_aggregate)
        for owner in self.owners.values():
            owner.scale_coordinates(alpha, validate_result=False)
        if validate_aggregate:
            self.validate()


@dataclass(slots=True)
class RootBud:
    """Persistent network input organ used only while no layer exists.

    It is the architectural homologue of a terminal bud, but not a vertical
    bud: it stores one universal quadratic kernel, has no owners and requires
    no cross-identity concordance.
    """

    dimension: int
    kernel: QuadraticKernel

    def __post_init__(self) -> None:
        if self.kernel.dimension != self.dimension:
            raise ValueError("root bud dimension mismatch")
        self.validate()

    @classmethod
    def empty(cls, dimension: int) -> "RootBud":
        return cls(dimension, QuadraticKernel.zero(dimension))

    def validate(self) -> None:
        self.kernel.validate()

    def inject(
        self,
        value: Sequence[float] | Vector,
        chi: float,
        alpha: float,
        *,
        validate_state: bool = True,
    ) -> None:
        self.kernel.update(value, 1.0, chi, alpha=alpha, validate_result=validate_state)
        if validate_state:
            self.validate()

    def reset(self) -> None:
        self.kernel = QuadraticKernel.zero(self.dimension)


# ---------------------------------------------------------------------------
# Maintenance model
# ---------------------------------------------------------------------------


class MaintenanceModel(Protocol):
    """Exact additive footprint plus ergonomic budget conversion."""

    def network_units(self, network: "Auxein") -> int: ...
    def root_bud_units(self, dimension: int, scalar: Scalar) -> int: ...
    def layer_units(self, layer: "Layer", scalar: Scalar) -> int: ...
    def cell_units(self, cell: Cell, scalar: Scalar) -> int: ...
    def bud_base_units(self, dimension: int, scalar: Scalar) -> int: ...
    def owner_record_units(self, dimension: int, scalar: Scalar) -> int: ...
    def budget_to_units(
        self,
        dimension: int,
        scalar: Scalar,
        budget: BudgetValue,
    ) -> int: ...
    def budget_from_units(
        self,
        dimension: int,
        scalar: Scalar,
        budget_units: int,
    ) -> Decimal: ...


@dataclass(frozen=True, slots=True)
class ScalarFootprintMaintenance:
    """Reference footprint in bytes, with exact integer arithmetic."""

    integer_bytes: int = 8

    def __post_init__(self) -> None:
        _strict_positive_int(self.integer_bytes, "integer_bytes")

    @staticmethod
    def real_bytes(scalar: Scalar) -> int:
        return NumericPolicy(scalar).byte_width

    def network_units(self, network: "Auxein") -> int:
        del network
        # next identity, step index and next layer serial
        return 3 * self.integer_bytes

    def root_bud_units(self, dimension: int, scalar: Scalar) -> int:
        return (dimension + 2) * self.real_bytes(scalar)

    def layer_units(self, layer: "Layer", scalar: Scalar) -> int:
        return (layer.dimension + 2) * self.real_bytes(scalar) + self.integer_bytes

    def cell_units(self, cell: Cell, scalar: Scalar) -> int:
        return (3 * cell.dimension + 4) * self.real_bytes(scalar) + self.integer_bytes

    def bud_base_units(self, dimension: int, scalar: Scalar) -> int:
        return (2 * dimension + 4) * self.real_bytes(scalar)

    def owner_record_units(self, dimension: int, scalar: Scalar) -> int:
        return (2 * dimension + 2) * self.real_bytes(scalar)

    def root_substrate_units(self, dimension: int, scalar: Scalar) -> int:
        class _Probe:
            pass
        return self.network_units(_Probe()) + self.root_bud_units(dimension, scalar)

    def active_shell_units(self, dimension: int, scalar: Scalar) -> int:
        class _LayerProbe:
            def __init__(self, dimension: int) -> None:
                self.dimension = dimension
        return self.layer_units(
            _LayerProbe(dimension), scalar
        ) + self.bud_base_units(dimension, scalar)

    def equivalent_cell_units(self, dimension: int, scalar: Scalar) -> int:
        class _CellProbe:
            def __init__(self, dimension: int) -> None:
                self.dimension = dimension
        return self.cell_units(
            _CellProbe(dimension), scalar
        ) + self.owner_record_units(dimension, scalar)

    def budget_to_units(
        self,
        dimension: int,
        scalar: Scalar,
        budget: BudgetValue,
    ) -> int:
        """Convert equivalent-cell capacity into exact footprint units."""

        equivalent_cells = _budget_decimal(budget)
        root = self.root_substrate_units(dimension, scalar)
        if equivalent_cells == 0:
            return root
        shell = self.active_shell_units(dimension, scalar)
        cell = self.equivalent_cell_units(dimension, scalar)
        variable = int(
            (equivalent_cells * cell).to_integral_value(rounding=ROUND_FLOOR)
        )
        return root + shell + variable

    def budget_from_units(
        self,
        dimension: int,
        scalar: Scalar,
        budget_units: int,
    ) -> Decimal:
        """Express exact footprint units as equivalent-cell capacity."""

        if isinstance(budget_units, bool) or not isinstance(budget_units, int):
            raise TypeError("budget_units must be an integer footprint")
        if budget_units <= 0:
            raise ValueError("budget_units must be positive")
        root = self.root_substrate_units(dimension, scalar)
        if budget_units <= root:
            return Decimal(0)
        shell = self.active_shell_units(dimension, scalar)
        cell = self.equivalent_cell_units(dimension, scalar)
        return Decimal(max(0, budget_units - root - shell)) / Decimal(cell)


# ---------------------------------------------------------------------------
# Layers and local proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LayerGeometry:
    mean: Vector
    radius: float


@dataclass(frozen=True, slots=True)
class Emission:
    source: CellIdentity | None
    value: Vector
    relevance: float


@dataclass(frozen=True, slots=True)
class LayerRead:
    input_value: Vector
    input_relevance: float
    geometry: LayerGeometry
    winner_slot: int | None
    winner_identity: CellIdentity | None
    winner_read: CellRead | None
    emission: Emission


ProposalKind = Literal["split", "vertical_birth", "horizontal_reallocation"]


@dataclass(frozen=True, slots=True)
class GrowthProposal:
    token: ProposalToken
    kind: ProposalKind
    layer_index: int
    geometric_value: float
    maintenance_delta_units: int
    cost_units: int


@dataclass(frozen=True, slots=True)
class ContractionOffer:
    token: ProposalToken
    layer_index: int
    geometric_loss: float
    maintenance_release_units: int
    kind: Literal["cell_death", "truncate"]


@dataclass(slots=True)
class Layer:
    dimension: int
    chi: float
    receipt: QuadraticKernel
    cells: list[Cell]
    bud: Bud | None = None
    _owner_serial: int = 0
    _proposal_serial: int = 0
    _scalar_format: Scalar = "f64"
    alpha: float = 0.0
    _pending: dict[ProposalToken, tuple[str, int | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            self.alpha = 1.0 - self.chi
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("layer alpha must lie in (0, 1]")
        if self.receipt.dimension != self.dimension:
            raise ValueError("layer receipt dimension mismatch")
        for cell in self.cells:
            if cell.dimension != self.dimension:
                raise ValueError("layer cell dimension mismatch")
        if self.bud is not None and self.bud.dimension != self.dimension:
            raise ValueError("layer bud dimension mismatch")
        self.validate()

    @classmethod
    def from_seed(
        cls,
        center: Sequence[float] | Vector,
        weight: float,
        chi: float,
        alpha: float,
        identity: CellIdentity,
        owner_serial: int,
    ) -> "Layer":
        center_v = Vector(center)
        if not center_v:
            raise ValueError("seed center must be a nonempty vector")
        if not all(math.isfinite(component) for component in center_v):
            raise ValueError("seed center must be finite")
        weight = _positive_finite_real(weight, "seed_weight")
        dimension = len(center_v)
        receipt = QuadraticKernel.point(center_v, weight)
        parent = QuadraticKernel(dimension, weight, _zero(dimension), 0.0)
        plus, minus = parent.neutral_split()
        cell = Cell(identity, center_v, plus, minus)
        bud = Bud.empty(dimension, [identity])
        return cls(dimension, chi, receipt, [cell], bud, owner_serial, 0, "f64", alpha)

    @property
    def receipt_scalar_format(self) -> Scalar:
        return getattr(self, "_scalar_format", "f64")

    def validate(self) -> None:
        if self._owner_serial <= 0 or self._proposal_serial < 0:
            raise InvariantViolation("invalid layer serial counters")
        if self._scalar_format not in ("f32", "f64"):
            raise InvariantViolation("invalid layer scalar format")
        self.receipt.validate()
        identities = [cell.identity for cell in self.cells]
        if len(set(identities)) != len(identities):
            raise InvariantViolation("duplicate identity inside layer")
        for cell in self.cells:
            cell.validate()
        if self.bud is not None:
            if set(self.bud.owners) != set(identities):
                raise InvariantViolation(
                    "terminal bud owner records must match terminal Cell identities"
                )
            self.bud.validate()

    @property
    def geometry(self) -> LayerGeometry:
        if self.receipt.W <= 0.0:
            return LayerGeometry(_zero(self.dimension), 0.0)
        mean = self.receipt.mean
        variance = self.receipt.structural_power_for(self.receipt_scalar_format) / self.receipt.W
        return LayerGeometry(mean, math.sqrt(variance))

    def prepare(self, input_value: Sequence[float] | Vector, relevance: float) -> LayerRead:
        value = _vector(input_value, self.dimension)
        relevance = float(relevance)
        if relevance < 0.0 or not math.isfinite(relevance):
            raise ValueError("layer relevance must be finite and nonnegative")
        geometry = self.geometry

        winner_slot: int | None = None
        winner_error: Vector | None = None
        best_recognition = 0.0
        for slot, cell in enumerate(self.cells):
            recognition, error = cell.recognition_and_error(value, geometry.radius)
            if recognition > best_recognition:
                best_recognition = recognition
                winner_slot = slot
                winner_error = error
            # Exact ties retain the earlier immutable list order.

        if winner_slot is None or winner_error is None or not best_recognition > 0.0:
            emission = Emission(None, _zero(self.dimension), 0.0)
            return LayerRead(value, relevance, geometry, None, None, None, emission)

        winner = self.cells[winner_slot]
        winner_read = winner.read_from_recognition(
            winner_error, best_recognition, relevance
        )
        source = winner.identity
        emission = Emission(source, winner_read.emitted_form.copy(), winner_read.relevance)
        return LayerRead(
            value, relevance, geometry, winner_slot, source, winner_read, emission
        )

    def inject(
        self,
        read: LayerRead,
        alpha: float,
        *,
        validate_aggregate: bool = True,
    ) -> None:
        self.receipt.update(
            read.input_value,
            read.input_relevance,
            self.chi,
            alpha=alpha,
            validate_result=validate_aggregate,
        )
        for slot, cell in enumerate(self.cells):
            cell.inject(
                read.winner_read if slot == read.winner_slot else None,
                self.chi,
                alpha,
                validate_state=validate_aggregate,
            )
        if validate_aggregate:
            self.validate()

    def move_and_recenter(
        self, radius_used_for_error: float, *, validate_aggregate: bool = True
    ) -> None:
        for cell in self.cells:
            cell.move_and_recenter(
                radius_used_for_error,
                scalar_format=self.receipt_scalar_format,
                validate_state=validate_aggregate,
            )
        if validate_aggregate:
            self.validate()

    def scale_internal_coordinates(
        self, alpha: float, *, validate_aggregate: bool = True
    ) -> None:
        for cell in self.cells:
            cell.scale_internal_coordinates(
                alpha, validate_state=validate_aggregate
            )
        if validate_aggregate:
            self.validate()

    def scale_physical_coordinates(
        self, alpha: float, *, validate_aggregate: bool = True
    ) -> None:
        self.receipt.scale_coordinates(
            alpha, validate_result=validate_aggregate
        )
        for cell in self.cells:
            cell.center *= alpha
        if validate_aggregate:
            self.validate()

    @property
    def capital(self) -> float:
        geometry = self.geometry
        if geometry.radius <= 0.0:
            return 0.0
        active = [cell for cell in self.cells if cell.mass > 0.0]
        total_mass = sum(cell.mass for cell in active)
        if total_mass <= 0.0:
            return 0.0
        running_mass = 0.0
        running_mean = _zero(self.dimension)
        value = 0.0
        for cell in active:
            z = (cell.center - geometry.mean) / geometry.radius
            p = z + cell.parent_mean
            weight = cell.mass
            new_mass = running_mass + weight
            if running_mass == 0.0:
                running_mean = p.copy()
            else:
                delta = p - running_mean
                value += weight * running_mass / new_mass * _norm2(delta)
                running_mean += (weight / new_mass) * delta
            running_mass = new_mass
        return _nonnegative(
            value,
            scale_values=(value,),
            dimension=self.dimension,
        )

    def conservation_value(self, victim_slot: int) -> float:
        """Scalar reference evaluation of one Cell death.

        The batch arbitration path below uses an exact ephemeral spatial index,
        while this literal implementation remains useful as an independent
        oracle for tests and isolated queries.
        """

        if victim_slot < 0 or victim_slot >= len(self.cells):
            raise IndexError("invalid victim slot")
        if len(self.cells) <= 1:
            return math.inf
        victim = self.cells[victim_slot]
        parent_weight = victim.mass
        geometry = self.geometry
        best = math.inf
        for slot, survivor in enumerate(self.cells):
            if slot == victim_slot:
                continue
            delta = survivor.center - victim.center
            if geometry.radius > 0.0:
                h = delta / geometry.radius
                candidate = parent_weight * _norm2(h)
            elif survivor.center == victim.center:
                candidate = 0.0
            else:
                candidate = math.inf
            if candidate < best:
                best = candidate
        if math.isinf(best):
            return best
        return max(best, 0.0)

    def _conservation_values(self) -> list[float]:
        """Evaluate every admissible Cell death with one ephemeral index.

        Conservation is evaluated after complete movement and recentering, so
        the canonical residual first moment is exactly zero and

            K = A min ||h||².

        The k-d tree changes only the search strategy; the selected candidate
        is evaluated again with the canonical formula.
        """

        count = len(self.cells)
        if count <= 1:
            return [math.inf for _ in self.cells]

        geometry = self.geometry
        radius = geometry.radius
        if radius <= 0.0:
            multiplicities: dict[tuple[float, ...], int] = {}
            for cell in self.cells:
                key = tuple(cell.center)
                multiplicities[key] = multiplicities.get(key, 0) + 1
            return [
                0.0 if multiplicities[tuple(cell.center)] > 1 else math.inf
                for cell in self.cells
            ]

        points: list[Sequence[float]] = [cell.center for cell in self.cells]
        lower = [min(point[axis] for point in points) for axis in range(self.dimension)]
        upper = [max(point[axis] for point in points) for axis in range(self.dimension)]
        sweep_axis = max(
            range(self.dimension),
            key=lambda axis: (upper[axis] - lower[axis], -axis),
        )
        order = sorted(
            range(count), key=lambda index: (points[index][sweep_axis], index)
        )
        axis_values = [points[index][sweep_axis] for index in order]
        values: list[float] = []
        inverse_radius = 1.0 / radius

        for victim_slot, victim in enumerate(self.cells):
            weight = victim.mass
            if weight <= 0.0:
                values.append(0.0)
                continue

            target = victim.center
            survivor_slot = _nearest_other_sweep(
                points,
                order,
                axis_values,
                sweep_axis,
                target,
                victim_slot,
            )
            if survivor_slot is None:
                values.append(math.inf)
                continue

            survivor = self.cells[survivor_slot]
            norm = 0.0
            for survivor_component, victim_component in zip(
                survivor.center, victim.center, strict=True
            ):
                h = (survivor_component - victim_component) * inverse_radius
                norm += h * h
            values.append(weight * norm)

        return values

    def _new_token(self) -> ProposalToken:
        self._proposal_serial += 1
        return ProposalToken(self._owner_serial, self._proposal_serial)

    def clear_proposals(self) -> None:
        self._pending.clear()

    def best_split_proposal(
        self,
        layer_index: int,
        maintenance_model: MaintenanceModel,
    ) -> GrowthProposal | None:
        best_slot: int | None = None
        best_value = 0.0
        for slot, cell in enumerate(self.cells):
            value = cell.split_gain
            if value > best_value:
                best_value = value
                best_slot = slot
        if best_slot is None or not best_value > 0.0:
            return None
        new_cell_units = maintenance_model.cell_units(
            self.cells[best_slot], self.receipt_scalar_format
        )
        owner_record_units = (
            maintenance_model.owner_record_units(
                self.dimension,
                self.receipt_scalar_format,
            )
            if self.bud is not None
            else 0
        )
        delta = new_cell_units + owner_record_units
        token = self._new_token()
        self._pending[token] = ("split", best_slot)
        return GrowthProposal(
            token=token,
            kind="split",
            layer_index=layer_index,
            geometric_value=best_value,
            maintenance_delta_units=delta,
            cost_units=max(delta, 0),
        )

    def execute_split(
        self,
        proposal: GrowthProposal,
        radius: float,
        factory: IdentityFactory,
        *,
        validate_aggregate: bool = True,
    ) -> CellIdentity:
        action = self._pending.pop(proposal.token, None)
        if action is None or action[0] != "split" or action[1] is None:
            raise InvalidStateOperation("stale or foreign split proposal")
        slot = action[1]
        daughter = self.cells[slot].materialize_split(
            radius,
            factory,
            scalar_format=self.receipt_scalar_format,
            validate_state=validate_aggregate,
        )
        self.cells.append(daughter)
        if self.bud is not None:
            self.bud.add_source(
                daughter.identity, validate_aggregate=validate_aggregate
            )
        self.clear_proposals()
        if validate_aggregate:
            self.validate()
        return daughter.identity

    def best_cell_contraction_offer(
        self,
        layer_index: int,
        maintenance_model: MaintenanceModel,
    ) -> ContractionOffer | None:
        if len(self.cells) <= 1:
            return None
        best_slot: int | None = None
        best_loss = math.inf
        for slot, value in enumerate(self._conservation_values()):
            if value < best_loss:
                best_loss = value
                best_slot = slot
        if best_slot is None:
            return None
        release_units = maintenance_model.cell_units(
            self.cells[best_slot], self.receipt_scalar_format
        )
        if self.bud is not None:
            release_units += maintenance_model.owner_record_units(
                self.dimension, self.receipt_scalar_format
            )
        token = self._new_token()
        self._pending[token] = ("death", best_slot)
        return ContractionOffer(
            token=token,
            layer_index=layer_index,
            geometric_loss=best_loss,
            maintenance_release_units=release_units,
            kind="cell_death",
        )

    def best_horizontal_reallocation(
        self,
        layer_index: int,
        maintenance_model: MaintenanceModel,
        network_maintenance_units: int,
        budget_units: int,
    ) -> GrowthProposal | None:
        if len(self.cells) <= 1:
            return None
        owner_record_units = (
            maintenance_model.owner_record_units(
                self.dimension,
                self.receipt_scalar_format,
            )
            if self.bud is not None
            else 0
        )

        # Reallocation is only meaningful for a strictly positive split that is
        # blocked by persistent capacity.  Building this list first avoids the
        # old split × victim × substitute cubic scan.
        blocked_splits: list[tuple[int, float, int]] = []
        for split_slot, split_cell in enumerate(self.cells):
            gain = split_cell.split_gain
            if not gain > 0.0:
                continue
            split_delta_units = (
                maintenance_model.cell_units(split_cell, self.receipt_scalar_format)
                + owner_record_units
            )
            if network_maintenance_units + split_delta_units <= budget_units:
                continue
            blocked_splits.append((split_slot, gain, split_delta_units))
        if not blocked_splits:
            return None

        best_margin = 0.0
        best_victim: int | None = None
        best_release_units = 0

        # A victim's conservation loss is independent of which blocked split
        # is being considered.  Compute it once per victim, then compare all
        # feasible split requests.  Complexity is O(n²), not O(n³).
        conservation_values = self._conservation_values()
        for victim_slot, victim in enumerate(self.cells):
            death_release_units = (
                maintenance_model.cell_units(victim, self.receipt_scalar_format)
                + owner_record_units
            )
            loss = conservation_values[victim_slot]
            for split_slot, gain, split_delta_units in blocked_splits:
                if victim_slot == split_slot:
                    continue
                future_after_death_and_split = (
                    network_maintenance_units - death_release_units + split_delta_units
                )
                if future_after_death_and_split > budget_units:
                    continue
                margin = gain - loss
                if margin > best_margin:
                    best_margin = margin
                    best_victim = victim_slot
                    best_release_units = death_release_units
        if best_victim is None:
            return None
        token = self._new_token()
        self._pending[token] = ("death", best_victim)
        return GrowthProposal(
            token=token,
            kind="horizontal_reallocation",
            layer_index=layer_index,
            geometric_value=best_margin,
            maintenance_delta_units=-best_release_units,
            cost_units=0,
        )

    def execute_death(
        self, token: ProposalToken, *, validate_aggregate: bool = True
    ) -> CellIdentity:
        action = self._pending.pop(token, None)
        if action is None or action[0] != "death" or action[1] is None:
            raise InvalidStateOperation("stale or foreign death proposal")
        slot = action[1]
        if len(self.cells) <= 1:
            raise InvalidStateOperation("layer cannot kill its last Cell")
        victim = self.cells.pop(slot)
        if self.bud is not None:
            # Information-theoretically required reset, then zero owner records
            # for the surviving terminal identities.
            self.bud = Bud.empty(self.dimension, (cell.identity for cell in self.cells))
        self.clear_proposals()
        if validate_aggregate:
            self.validate()
        return victim.identity


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    kind: TransformationKind
    layer_index: int
    geometric_value: float
    maintenance_delta_units: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _TRANSFORMATION_KINDS:
            raise ValueError(f"unknown transformation kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class LayerStepReport:
    layer_index: int
    input_relevance: float
    winner: CellIdentity | None
    recognition: float
    emitted_relevance: float
    split_value: float
    capital: float


@dataclass(frozen=True, slots=True)
class StepReport:
    step_index: int
    maintenance_charged_units: int
    maintenance_units: int
    budget_units: int
    remaining_step_budget_units: int
    layer_reports: tuple[LayerStepReport, ...]
    transformations: tuple[TransformationRecord, ...]
    vertical_gain: float | None
    vertical_concordance: float | None


# ---------------------------------------------------------------------------
# Network engine
# ---------------------------------------------------------------------------


class Auxein:
    """Deterministic, online, single-stream Auxein reference engine."""

    __slots__ = (
        "_dimension",
        "_memory_law",
        "_memory_half_life",
        "_chi",
        "_alpha",
        "_numeric_policy",
        "_scalar_format",
        "_maintenance_model",
        "_budget_units",
        "_eta",
        "_layers",
        "_identity_factory",
        "_root_bud",
        "_check_invariants",
        "_step_index",
        "_layer_serial",
        "_state_extensions",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "Auxein cannot be constructed directly; use Auxein.empty(), "
            "Auxein.from_seed(), or Auxein.from_state_dict()"
        )

    @staticmethod
    def _validated_maintenance_model(model: MaintenanceModel) -> MaintenanceModel:
        required = (
            "network_units",
            "root_bud_units",
            "layer_units",
            "cell_units",
            "bud_base_units",
            "owner_record_units",
            "budget_to_units",
            "budget_from_units",
        )
        missing = [name for name in required if not callable(getattr(model, name, None))]
        if missing:
            raise TypeError(
                "maintenance_model is missing callable methods: " + ", ".join(missing)
            )
        return model

    @classmethod
    def _from_parts(
        cls,
        *,
        dimension: int,
        memory: float,
        budget: BudgetValue | None = None,
        budget_units: int | None = None,
        eta: float = 1.0,
        maintenance_model: MaintenanceModel,
        layers: list[Layer],
        identity_factory: IdentityFactory,
        root_bud: RootBud | None = None,
        scalar: Scalar = "f64",
        step_index: int = 0,
        layer_serial: int | None = None,
        check_invariants: bool = True,
        state_extensions: Mapping[str, object] | None = None,
    ) -> "Auxein":
        self = cls.__new__(cls)
        self._dimension = _strict_positive_int(dimension, "dimension")
        self._memory_law = MemoryLaw(memory)
        self._memory_half_life = self._memory_law.memory_half_life
        self._chi = self._memory_law.chi
        self._alpha = self._memory_law.alpha
        self._numeric_policy = NumericPolicy(scalar)
        self._scalar_format: Scalar = self._numeric_policy.scalar_format
        self._maintenance_model = cls._validated_maintenance_model(maintenance_model)
        self._budget_units = self._resolve_budget_units(budget, budget_units)
        self._eta = _learning_rate(eta)
        if not isinstance(layers, list):
            raise TypeError("layers must be a list")
        self._layers = layers
        if not isinstance(identity_factory, IdentityFactory):
            raise TypeError("identity_factory must be an IdentityFactory")
        self._identity_factory = identity_factory
        if root_bud is not None and not isinstance(root_bud, RootBud):
            raise TypeError("root_bud must be a RootBud")
        self._root_bud = root_bud if root_bud is not None else RootBud.empty(self._dimension)
        self._check_invariants = _strict_bool(check_invariants, "check_invariants")
        self._step_index = _strict_nonnegative_int(step_index, "step_index")
        self._layer_serial = (
            _strict_positive_int(layer_serial, "layer_serial")
            if layer_serial is not None
            else max((layer._owner_serial for layer in layers), default=0) + 1
        )
        if state_extensions is None:
            self._state_extensions = {}
        elif not isinstance(state_extensions, Mapping):
            raise TypeError("state_extensions must be an object")
        else:
            self._state_extensions = copy.deepcopy(dict(state_extensions))
        self._quantize_state()
        self.validate()
        return self

    def _resolve_budget_units(
        self,
        budget: BudgetValue | None,
        budget_units: int | None,
    ) -> int:
        if (budget is None) == (budget_units is None):
            raise TypeError("provide exactly one of budget or budget_units")
        if budget is not None:
            return self._maintenance_model.budget_to_units(
                self._dimension,
                self._scalar_format,
                budget,
            )
        if isinstance(budget_units, bool) or not isinstance(budget_units, int):
            raise TypeError("budget_units must be an integer footprint")
        if budget_units <= 0:
            raise ValueError("budget_units must be positive")
        return budget_units

    @classmethod
    def empty(
        cls,
        dimension: int,
        *,
        memory: float,
        budget: BudgetValue | None = None,
        budget_units: int | None = None,
        eta: float = 1.0,
        maintenance_model: MaintenanceModel | None = None,
        scalar: Scalar = "f64",
        check_invariants: bool = True,
    ) -> "Auxein":
        dimension = _strict_positive_int(dimension, "dimension")
        _strict_bool(check_invariants, "check_invariants")
        model = (
            ScalarFootprintMaintenance()
            if maintenance_model is None
            else maintenance_model
        )
        return cls._from_parts(
            dimension=dimension,
            memory=memory,
            budget=budget,
            budget_units=budget_units,
            eta=eta,
            maintenance_model=model,
            layers=[],
            identity_factory=IdentityFactory(),
            root_bud=RootBud.empty(dimension),
            scalar=scalar,
            check_invariants=check_invariants,
        )

    @classmethod
    def from_seed(
        cls,
        seed: Sequence[float] | Vector,
        *,
        memory: float,
        budget: BudgetValue | None = None,
        budget_units: int | None = None,
        eta: float = 1.0,
        maintenance_model: MaintenanceModel | None = None,
        seed_weight: float = 1.0,
        scalar: Scalar = "f64",
        check_invariants: bool = True,
    ) -> "Auxein":
        """Construct an explicitly pre-materialized fixture.

        Canonical autonomous initialization uses :meth:`empty` and the root
        bud. This constructor remains useful for deterministic tests and for
        importing an externally established initial topology.
        """

        seed_v = Vector(seed)
        if not seed_v:
            raise ValueError("seed must be a nonempty vector")
        if not all(math.isfinite(component) for component in seed_v):
            raise ValueError("seed must be finite")
        memory_half_life = _positive_finite_real(memory, "memory")
        _strict_bool(check_invariants, "check_invariants")
        seed_weight = _positive_finite_real(seed_weight, "seed_weight")
        model = (
            ScalarFootprintMaintenance()
            if maintenance_model is None
            else maintenance_model
        )
        memory_law = MemoryLaw(memory_half_life)
        factory = IdentityFactory()
        root_identity = factory.new()
        root = Layer.from_seed(
            seed_v,
            seed_weight,
            memory_law.chi,
            memory_law.alpha,
            root_identity,
            owner_serial=1,
        )
        return cls._from_parts(
            dimension=len(seed_v),
            memory=memory_half_life,
            budget=budget,
            budget_units=budget_units,
            eta=eta,
            maintenance_model=model,
            layers=[root],
            identity_factory=factory,
            root_bud=RootBud.empty(len(seed_v)),
            scalar=scalar,
            check_invariants=check_invariants,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def chi(self) -> float:
        return self._chi

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def maintenance_model(self) -> MaintenanceModel:
        return self._maintenance_model

    @property
    def check_invariants(self) -> bool:
        return self._check_invariants

    @property
    def _terminal(self) -> Layer:
        if not self._layers:
            raise InvalidStateOperation("empty network has no terminal layer")
        return self._layers[-1]

    @property
    def memory(self) -> float:
        """Statistical memory half-life, measured in presentations."""

        return self._memory_half_life

    @property
    def scalar(self) -> Scalar:
        """Persistent real representation used by the engine."""

        return self._scalar_format

    @property
    def budget(self) -> Decimal:
        return self._maintenance_model.budget_from_units(
            self._dimension,
            self._scalar_format,
            self._budget_units,
        )

    @budget.setter
    def budget(self, value: BudgetValue) -> None:
        self._budget_units = self._maintenance_model.budget_to_units(
            self._dimension,
            self._scalar_format,
            value,
        )

    @property
    def budget_units(self) -> int:
        return self._budget_units

    @budget_units.setter
    def budget_units(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("budget_units must be an integer footprint")
        if value <= 0:
            raise ValueError("budget_units must be positive")
        self._budget_units = value

    @property
    def budget_margin_units(self) -> int:
        """Current exact budget margin after structural maintenance."""

        return self.budget_units - self.maintenance_units()

    @property
    def is_solvent(self) -> bool:
        """Whether the current topology fits within the exact budget."""

        return self.budget_margin_units >= 0

    @property
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, value: float) -> None:
        self._eta = _learning_rate(value)

    @property
    def effective_alpha(self) -> float:
        return self.eta * self._alpha

    def validate(self) -> None:
        self._root_bud.validate()
        if self._layers and self._root_bud.kernel.W != 0.0:
            raise InvariantViolation("root bud must be empty while layers exist")
        for index, layer in enumerate(self._layers):
            if layer.dimension != self._dimension:
                raise InvariantViolation("cross-layer dimension mismatch")
            if layer.chi != self._chi or layer.alpha != self._alpha:
                raise InvariantViolation("layer memory law differs from network")
            if index == len(self._layers) - 1:
                if layer.bud is None:
                    raise InvariantViolation("terminal layer must own a bud")
            elif layer.bud is not None:
                raise InvariantViolation("only terminal layer may own a bud")
            if len(layer.cells) == 0:
                raise InvariantViolation("an existing layer cannot be empty")
            layer.validate()
        owner_serials = [layer._owner_serial for layer in self._layers]
        if len(owner_serials) != len(set(owner_serials)):
            raise InvariantViolation("layer owner serial reused")
        if self._layer_serial <= max(owner_serials, default=0):
            raise InvariantViolation("next layer serial must exceed all live layers")
        identities: set[CellIdentity] = set()
        for layer in self._layers:
            for cell in layer.cells:
                if cell.identity in identities:
                    raise InvariantViolation("Cell identity reused across network")
                identities.add(cell.identity)
        maintenance_units = self.maintenance_units()
        if maintenance_units <= 0:
            raise InvariantViolation("maintenance must be positive")
        if self._identity_factory.next_token <= max((cell.identity._token for layer in self._layers for cell in layer.cells), default=0):
            raise InvariantViolation("next identity token must exceed all live identities")

    def maintenance_units(self) -> int:
        model = self._maintenance_model
        fmt = self._scalar_format
        total = model.network_units(self) + model.root_bud_units(self._dimension, fmt)
        for layer in self._layers:
            total += model.layer_units(layer, fmt)
            total += sum(model.cell_units(cell, fmt) for cell in layer.cells)
            if layer.bud is not None:
                total += model.bud_base_units(self._dimension, fmt)
                total += len(layer.bud.owners) * model.owner_record_units(self._dimension, fmt)
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise InvariantViolation(f"invalid maintenance total {total!r}")
        return total

    def _quantize_kernel(self, kernel: QuadraticKernel) -> None:
        policy = self._numeric_policy
        kernel.W = policy.cast(kernel.W)
        kernel.S = policy.cast_vector(kernel.S)
        kernel.Q = policy.cast(kernel.Q)
        if kernel.W == 0.0:
            kernel.S = _zero(kernel.dimension)
            kernel.Q = 0.0
            return
        required = _norm2(kernel.S) / kernel.W
        if kernel.Q < required:
            q = policy.cast(required)
            while q < required:
                q = policy.next_up(q)
            kernel.Q = q

    def _quantize_state(self) -> None:
        policy = self._numeric_policy
        self._quantize_kernel(self._root_bud.kernel)
        for layer in self._layers:
            layer._scalar_format = self._scalar_format
            self._quantize_kernel(layer.receipt)
            for cell in layer.cells:
                cell.center = policy.cast_vector(cell.center)
                self._quantize_kernel(cell.plus)
                self._quantize_kernel(cell.minus)
            if layer.bud is not None:
                layer.bud.scalar_format = self._scalar_format
                for owner in layer.bud.owners.values():
                    owner.lambda_plus = policy.cast(owner.lambda_plus)
                    owner.lambda_minus = policy.cast(owner.lambda_minus)
                    owner.f_plus = policy.cast_vector(owner.f_plus)
                    owner.f_minus = policy.cast_vector(owner.f_minus)
                layer.bud.plus.W = policy.cast(sum(owner.lambda_plus for owner in layer.bud.owners.values()))
                layer.bud.minus.W = policy.cast(sum(owner.lambda_minus for owner in layer.bud.owners.values()))
                layer.bud.plus.S = policy.cast_vector(sum((owner.f_plus for owner in layer.bud.owners.values()), start=_zero(self._dimension)))
                layer.bud.minus.S = policy.cast_vector(sum((owner.f_minus for owner in layer.bud.owners.values()), start=_zero(self._dimension)))
                self._quantize_kernel(layer.bud.plus)
                self._quantize_kernel(layer.bud.minus)

    @property
    def geometric_capital(self) -> float:
        return sum(layer.capital for layer in self._layers)

    def _truncate(self, start_layer: int) -> None:
        if start_layer < 0 or start_layer >= len(self._layers):
            raise InvalidStateOperation("truncate start must be an existing layer")
        if start_layer == 0:
            self._layers = []
            self._root_bud.reset()
            self.validate()
            return
        self._layers = self._layers[:start_layer]
        terminal = self._layers[-1]
        terminal.bud = Bud.empty(
            self._dimension,
            (cell.identity for cell in terminal.cells),
        )
        terminal.clear_proposals()
        self.validate()

    def _suffix_loss(self, start_layer: int) -> float:
        return sum(layer.capital for layer in self._layers[start_layer:])

    def _maintenance_of_prefix(self, stop: int) -> int:
        model = self._maintenance_model
        fmt = self._scalar_format
        total = model.network_units(self) + model.root_bud_units(self._dimension, fmt)
        kept = self._layers[:stop]
        for layer in kept:
            total += model.layer_units(layer, fmt)
            total += sum(model.cell_units(cell, fmt) for cell in layer.cells)
        if kept:
            new_terminal = kept[-1]
            total += model.bud_base_units(self._dimension, fmt)
            total += len(new_terminal.cells) * model.owner_record_units(self._dimension, fmt)
        return total

    def _suffix_release(self, start_layer: int) -> int:
        return self.maintenance_units() - self._maintenance_of_prefix(start_layer)

    def _forced_offers(self) -> list[ContractionOffer]:
        offers: list[ContractionOffer] = []
        for index, layer in enumerate(self._layers):
            layer.clear_proposals()
            offer = layer.best_cell_contraction_offer(index, self._maintenance_model)
            if offer is not None:
                offers.append(offer)
        for start in range(len(self._layers)):
            token = ProposalToken._forced(1_000_000 + start)
            offers.append(
                ContractionOffer(
                    token=token,
                    layer_index=start,
                    geometric_loss=self._suffix_loss(start),
                    maintenance_release_units=self._suffix_release(start),
                    kind="truncate",
                )
            )
        return [offer for offer in offers if offer.maintenance_release_units > 0]

    def _restore_solvency(self, records: list[TransformationRecord]) -> None:
        while self.maintenance_units() > self.budget_units:
            offers = self._forced_offers()
            if not offers:
                raise InsolventState(
                    "state exceeds budget and exposes no positive-release destruction"
                )
            kind_order = {"cell_death": 0, "truncate": 1}
            chosen = min(
                offers,
                key=lambda offer: (
                    offer.geometric_loss,
                    kind_order[offer.kind],
                    offer.layer_index,
                ),
            )
            before = self.maintenance_units()
            if chosen.kind == "cell_death":
                layer = self._layers[chosen.layer_index]
                layer.execute_death(chosen.token, validate_aggregate=False)
                note = "forced local dry loss"
            elif chosen.kind == "truncate":
                self._truncate(chosen.layer_index)
                note = "forced suffix truncation"
            else:
                raise InvariantViolation("unknown forced contraction kind")
            if self._check_invariants:
                self.validate()
            after = self.maintenance_units()
            if not after < before:
                raise InvariantViolation("forced destruction failed to reduce maintenance")
            records.append(
                TransformationRecord(
                    kind=chosen.kind,
                    layer_index=chosen.layer_index,
                    geometric_value=-chosen.geometric_loss,
                    maintenance_delta_units=after - before,
                    note=note,
                )
            )

    def _root_founder_preview(self) -> tuple[Layer, Cell]:
        kernel = self._root_bud.kernel
        if kernel.W <= 0.0:
            raise InvalidStateOperation("root birth requires positive accumulated mass")
        center = kernel.mean
        parent = kernel.clone()
        parent.recenter_on_mean(scalar_format=self._scalar_format)
        plus, minus = parent.neutral_split()
        founder = Cell(CellIdentity._placeholder(), center, plus, minus)
        layer = Layer(
            dimension=self._dimension,
            chi=self._chi,
            alpha=self._alpha,
            receipt=kernel.clone(),
            cells=[founder],
            bud=Bud.empty(self._dimension, [founder.identity]),
            _owner_serial=self._layer_serial,
        )
        return layer, founder

    def _root_birth_delta(self) -> int:
        layer, founder = self._root_founder_preview()
        model = self._maintenance_model
        return (
            model.layer_units(layer, self._scalar_format)
            + model.cell_units(founder, self._scalar_format)
            + model.bud_base_units(self._dimension, self._scalar_format)
            + model.owner_record_units(self._dimension, self._scalar_format)
        )

    def _try_root_birth(
        self,
        remaining_step_budget_units: int,
        records: list[TransformationRecord],
    ) -> int:
        if self._layers or self._root_bud.kernel.W <= 0.0:
            return remaining_step_budget_units
        delta = self._root_birth_delta()
        cost_units = max(delta, 0)
        if (
            cost_units > remaining_step_budget_units
            or self.maintenance_units() + delta > self.budget_units
        ):
            return remaining_step_budget_units

        kernel = self._root_bud.kernel.clone()
        center = kernel.mean
        parent = kernel.clone()
        parent.recenter_on_mean(scalar_format=self._scalar_format)
        plus, minus = parent.neutral_split()
        founder = Cell(self._identity_factory.new(), center, plus, minus)
        layer = Layer(
            dimension=self._dimension,
            chi=self._chi,
            alpha=self._alpha,
            receipt=kernel,
            cells=[founder],
            bud=Bud.empty(self._dimension, [founder.identity]),
            _owner_serial=self._layer_serial,
        )
        self._layer_serial += 1
        before = self.maintenance_units()
        self._layers.append(layer)
        self._root_bud.reset()
        self._quantize_state()
        after = self.maintenance_units()
        actual_delta = after - before
        if actual_delta != delta:
            raise InvariantViolation(
                f"predicted root birth delta {delta} != {actual_delta}"
            )
        records.append(
            TransformationRecord(
                kind="root_birth",
                layer_index=0,
                geometric_value=0.0,
                maintenance_delta_units=actual_delta,
                note="network incarnation from root bud; no geometric market value",
            )
        )
        if self._check_invariants:
            self.validate()
        return remaining_step_budget_units - cost_units

    def _prepare_reads(self, input_value: Vector) -> tuple[list[LayerRead], BudRead]:
        reads: list[LayerRead] = []
        value = input_value
        relevance = 1.0
        for layer in self._layers:
            read = layer.prepare(value, relevance)
            reads.append(read)
            value = read.emission.value
            relevance = read.emission.relevance
        bud = self._terminal.bud
        if bud is None:
            raise InvariantViolation("terminal layer has no bud")
        terminal_emission = reads[-1].emission
        bud_read = bud.read(
            terminal_emission.value,
            terminal_emission.relevance,
            terminal_emission.source,
        )
        return reads, bud_read

    def _inject_and_move(
        self,
        reads: list[LayerRead],
        bud_read: BudRead,
        alpha: float,
    ) -> None:
        pre_radii = [read.geometry.radius for read in reads]

        for layer, read in zip(self._layers, reads, strict=True):
            layer.inject(read, alpha, validate_aggregate=False)
        terminal_bud = self._terminal.bud
        if terminal_bud is None:
            raise InvariantViolation("terminal layer has no bud")
        terminal_bud.inject(
            bud_read, self._chi, alpha, validate_aggregate=False
        )

        # Persistent writes are rounded before the movement phase reads them.
        self._quantize_state()
        if self._check_invariants:
            self.validate()

        for layer, read in zip(self._layers, reads, strict=True):
            layer.move_and_recenter(
                read.geometry.radius, validate_aggregate=False
            )

        raw_post_radii = [layer.geometry.radius for layer in self._layers]
        alphas: list[float] = []
        for index, (old, new) in enumerate(zip(pre_radii, raw_post_radii, strict=True)):
            if new > 0.0:
                alpha = old / new
            elif old == 0.0:
                alpha = 1.0
            else:
                raise InvariantViolation(
                    f"layer {index} radius collapsed from {old!r} to exact zero"
                )
            alphas.append(alpha)

        # Same-layer normalized histories must enter the new radius unit.
        for layer, alpha in zip(self._layers, alphas, strict=True):
            if alpha != 1.0:
                layer.scale_internal_coordinates(
                    alpha, validate_aggregate=False
                )

        # The receiver of each layer's normalized vertical form must be carried
        # into the same new coordinate unit.
        for index, alpha in enumerate(alphas):
            if alpha == 1.0:
                continue
            if index + 1 < len(self._layers):
                self._layers[index + 1].scale_physical_coordinates(
                    alpha, validate_aggregate=False
                )
            else:
                bud = self._layers[index].bud
                if bud is None:
                    raise InvariantViolation("terminal layer lost its bud")
                bud.scale_coordinates(
                    alpha, validate_aggregate=False
                )

        if self._check_invariants:
            self.validate()

    def _vertical_proposal(self) -> GrowthProposal | None:
        terminal = self._terminal
        bud = terminal.bud
        if bud is None:
            raise InvariantViolation("terminal layer has no bud")
        gain = bud.split_gain
        concordance = bud.concordance
        if not gain > 0.0 or not concordance > 0.0:
            return None
        model = self._maintenance_model
        fmt = self._scalar_format
        old_bud_units = model.bud_base_units(
            self._dimension, fmt
        ) + len(bud.owners) * model.owner_record_units(self._dimension, fmt)
        # Founders have the same structural representation size as any Cell.
        representative = terminal.cells[0] if terminal.cells else None
        if representative is None:
            return None
        new_layer_units = model.layer_units(terminal, fmt)
        founder_units = 2 * model.cell_units(representative, fmt)
        new_bud_units = model.bud_base_units(
            self._dimension, fmt
        ) + 2 * model.owner_record_units(self._dimension, fmt)
        delta = new_layer_units + founder_units + new_bud_units - old_bud_units
        token = terminal._new_token()
        terminal._pending[token] = ("vertical_birth", None)
        return GrowthProposal(
            token=token,
            kind="vertical_birth",
            layer_index=len(self._layers) - 1,
            geometric_value=gain,
            maintenance_delta_units=delta,
            cost_units=max(delta, 0),
        )

    def _execute_vertical_birth(self, proposal: GrowthProposal) -> None:
        terminal = self._terminal
        action = terminal._pending.pop(proposal.token, None)
        if action is None or action[0] != "vertical_birth":
            raise InvalidStateOperation("stale or foreign vertical proposal")
        bud = terminal.bud
        if bud is None:
            raise InvariantViolation("terminal layer has no bud")
        if not bud.split_gain > 0.0 or not bud.concordance > 0.0:
            raise InvalidStateOperation("vertical proof ceased to be admissible")
        if bud.plus.W <= 0.0 or bud.minus.W <= 0.0:
            raise InvariantViolation("admissible bud has empty branch")

        parent_receipt = bud.parent.clone()
        founder_cells: list[Cell] = []
        for branch in (bud.plus, bud.minus):
            center = branch.mean
            branch_parent = branch.clone()
            branch_parent.recenter_on_mean(scalar_format=self._scalar_format)
            latent_plus, latent_minus = branch_parent.neutral_split()
            founder_cells.append(
                Cell(
                    identity=self._identity_factory.new(),
                    center=center,
                    plus=latent_plus,
                    minus=latent_minus,
                )
            )

        terminal.bud = None
        new_bud = Bud.empty(
            self._dimension,
            (cell.identity for cell in founder_cells),
        )
        new_layer = Layer(
            dimension=self._dimension,
            chi=self._chi,
            alpha=self._alpha,
            receipt=parent_receipt,
            cells=founder_cells,
            bud=new_bud,
            _owner_serial=self._layer_serial,
        )
        self._layer_serial += 1
        self._layers.append(new_layer)
        terminal.clear_proposals()
        self.validate()

    def _collect_growth_proposals(self) -> list[GrowthProposal]:
        proposals: list[GrowthProposal] = []
        for index, layer in enumerate(self._layers):
            layer.clear_proposals()
            proposal = layer.best_split_proposal(index, self._maintenance_model)
            if proposal is not None:
                proposals.append(proposal)
        vertical = self._vertical_proposal()
        if vertical is not None:
            proposals.append(vertical)
        return proposals

    @staticmethod
    def _proposal_priority(proposal: GrowthProposal) -> tuple[float, int, int]:
        # Descending geometry; exact vertical/horizontal equality favours vertical.
        vertical_first = 0 if proposal.kind == "vertical_birth" else 1
        return (-proposal.geometric_value, vertical_first, proposal.layer_index)

    def _execute_payable_growth(
        self,
        remaining_step_budget_units: int,
        records: list[TransformationRecord],
    ) -> int:
        while True:
            proposals = self._collect_growth_proposals()
            if not proposals:
                return remaining_step_budget_units
            current_maintenance_units = self.maintenance_units()
            payable = [
                proposal
                for proposal in proposals
                if proposal.cost_units <= remaining_step_budget_units
                and current_maintenance_units + proposal.maintenance_delta_units <= self.budget_units
            ]
            if not payable:
                return remaining_step_budget_units
            chosen = min(payable, key=self._proposal_priority)
            before = self.maintenance_units()
            if chosen.kind == "split":
                layer = self._layers[chosen.layer_index]
                radius = layer.geometry.radius
                layer.execute_split(
                    chosen,
                    radius,
                    self._identity_factory,
                    validate_aggregate=False,
                )
                note = "layer-authorized Cell mitosis"
            elif chosen.kind == "vertical_birth":
                self._execute_vertical_birth(chosen)
                note = "network-authorized layer birth"
            else:
                raise InvariantViolation("reallocation proposal entered growth executor")
            self._quantize_state()
            after = self.maintenance_units()
            actual_delta = after - before
            if actual_delta != chosen.maintenance_delta_units:
                raise InvariantViolation(
                    f"predicted maintenance delta {chosen.maintenance_delta_units} != {actual_delta}"
                )
            remaining_step_budget_units -= chosen.cost_units
            records.append(
                TransformationRecord(
                    kind=chosen.kind,
                    layer_index=chosen.layer_index,
                    geometric_value=chosen.geometric_value,
                    maintenance_delta_units=actual_delta,
                    note=note,
                )
            )
            if self._check_invariants:
                self.validate()

    def _execute_one_voluntary_reallocation(
        self,
        remaining_step_budget_units: int,
        records: list[TransformationRecord],
    ) -> None:
        del remaining_step_budget_units  # Dry loss never refunds the current presentation.
        current_maintenance_units = self.maintenance_units()
        proposals: list[GrowthProposal] = []
        for index, layer in enumerate(self._layers):
            layer.clear_proposals()
            proposal = layer.best_horizontal_reallocation(
                index,
                self._maintenance_model,
                current_maintenance_units,
                self.budget_units,
            )
            if proposal is not None and proposal.geometric_value > 0.0:
                proposals.append(proposal)

        # Canonical rule: a suffix cannot finance the vertical bud it destroys.
        if not proposals:
            return
        chosen = min(
            proposals,
            key=lambda proposal: (-proposal.geometric_value, proposal.layer_index),
        )
        before = self.maintenance_units()
        victim = self._layers[chosen.layer_index].execute_death(
            chosen.token, validate_aggregate=False
        )
        after = self.maintenance_units()
        if not after < before:
            raise InvariantViolation("voluntary death failed to reduce maintenance")
        records.append(
            TransformationRecord(
                kind="horizontal_reallocation_death",
                layer_index=chosen.layer_index,
                geometric_value=chosen.geometric_value,
                maintenance_delta_units=after - before,
                note="dry loss; victim removed; no same-step refund",
            )
        )
        if self._check_invariants:
            self.validate()

    def step(
        self,
        input_value: Sequence[float] | Vector,
        *,
        detailed_report: bool = True,
    ) -> StepReport:
        """Process exactly one online presentation.

        ``detailed_report=False`` omits diagnostic layer scans and returns
        empty ``layer_reports`` with ``None`` vertical diagnostics.  Transformations
        and all state changes are identical.
        """

        value = self._numeric_policy.cast_vector(_vector(input_value, self._dimension))
        try:
            input_norm2 = _norm2(value)
        except OverflowError as exc:
            raise ValueError("input squared norm is not representable") from exc
        if not math.isfinite(input_norm2):
            raise ValueError("input squared norm is not finite")
        records: list[TransformationRecord] = []

        # 1. Forced survival occurs before current maintenance is paid.
        self._restore_solvency(records)
        maintenance_charged_units = self.maintenance_units()
        if maintenance_charged_units > self.budget_units:
            raise InsolventState("forced survival returned an insolvent state")
        remaining_step_budget_units = self.budget_units - maintenance_charged_units

        # 2-5. The network presents the input either to L0 or, when no layer
        # exists, to the persistent root bud. The current observation is never
        # replayed through a topology created later in the same presentation.
        reads: list[LayerRead] = []
        effective_alpha = self.effective_alpha
        if self._layers:
            reads, bud_read = self._prepare_reads(value)
            if effective_alpha > 0.0:
                self._inject_and_move(reads, bud_read, effective_alpha)
                self._quantize_state()

                # 6-9. Geometric requests, parent arbitration, dry topology changes.
                remaining_step_budget_units = self._execute_payable_growth(
                    remaining_step_budget_units,
                    records,
                )
                self._execute_one_voluntary_reallocation(
                    remaining_step_budget_units,
                    records,
                )
        elif effective_alpha > 0.0:
            self._root_bud.inject(
                value,
                self._chi,
                effective_alpha,
                validate_state=False,
            )
            self._quantize_state()
            remaining_step_budget_units = self._try_root_birth(
                remaining_step_budget_units, records
            )

        for layer in self._layers:
            layer.clear_proposals()

        if self._check_invariants:
            self.validate()

        layer_reports: list[LayerStepReport] = []
        if detailed_report:
            for index, (layer, read) in enumerate(
                zip(self._layers[: len(reads)], reads, strict=False)
            ):
                winner = read.winner_identity
                recognition = (
                    read.winner_read.recognition
                    if read.winner_read is not None
                    else 0.0
                )
                best_split = max((cell.split_gain for cell in layer.cells), default=0.0)
                layer_reports.append(
                    LayerStepReport(
                        layer_index=index,
                        input_relevance=read.input_relevance,
                        winner=winner,
                        recognition=recognition,
                        emitted_relevance=read.emission.relevance,
                        split_value=best_split,
                        capital=layer.capital,
                    )
                )
            if self._layers:
                terminal_bud = self._terminal.bud
                if terminal_bud is None:
                    raise InvariantViolation("terminal layer has no bud at report time")
                vertical_gain = terminal_bud.split_gain
                vertical_concordance = terminal_bud.concordance
            else:
                vertical_gain = 0.0
                vertical_concordance = 0.0
        else:
            vertical_gain = None
            vertical_concordance = None

        report = StepReport(
            step_index=self._step_index,
            maintenance_charged_units=maintenance_charged_units,
            maintenance_units=self.maintenance_units(),
            budget_units=self.budget_units,
            remaining_step_budget_units=remaining_step_budget_units,
            layer_reports=tuple(layer_reports),
            transformations=tuple(records),
            vertical_gain=vertical_gain,
            vertical_concordance=vertical_concordance,
        )
        self._step_index += 1
        return report

    def to_state_dict(self) -> dict[str, object]:
        """Return the strict JSON-compatible causal state; perform no I/O."""
        for layer in self._layers:
            if layer._pending:
                raise InvalidStateOperation("cannot serialize during an active proposal arbitration")

        def kernel_dict(kernel: QuadraticKernel) -> dict[str, object]:
            return {"W": kernel.W, "S": list(kernel.S), "Q": kernel.Q}

        def cell_dict(cell: Cell) -> dict[str, object]:
            return {
                "identity": cell.identity._token,
                "center": list(cell.center),
                "plus": kernel_dict(cell.plus),
                "minus": kernel_dict(cell.minus),
            }

        def bud_dict(bud: Bud | None) -> object:
            if bud is None:
                return None
            owners = []
            for identity in sorted(bud.owners, key=lambda item: item._token):
                owner = bud.owners[identity]
                owners.append({
                    "identity": identity._token,
                    "lambda_plus": owner.lambda_plus,
                    "f_plus": list(owner.f_plus),
                    "lambda_minus": owner.lambda_minus,
                    "f_minus": list(owner.f_minus),
                })
            return {"plus": kernel_dict(bud.plus), "minus": kernel_dict(bud.minus), "owners": owners}

        result = copy.deepcopy(self._state_extensions)
        result.update({
            "schema_version": STATE_SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "dimension": self._dimension,
            "scalar": self.scalar,
            "memory": self.memory,
            "eta": self.eta,
            "step_index": self._step_index,
            "next_identity": self._identity_factory.next_token,
            "next_layer_serial": self._layer_serial,
            "root_bud": kernel_dict(self._root_bud.kernel),
            "layers": [
                {
                    "owner_serial": layer._owner_serial,
                    "proposal_serial": layer._proposal_serial,
                    "receipt": kernel_dict(layer.receipt),
                    "cells": [cell_dict(cell) for cell in layer.cells],
                    "bud": bud_dict(layer.bud),
                }
                for layer in self._layers
            ],
        })
        return result

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        budget: BudgetValue | None = None,
        budget_units: int | None = None,
        maintenance_model: MaintenanceModel | None = None,
        check_invariants: bool = True,
    ) -> "Auxein":
        """Rebuild a network from a strict causal state; perform no I/O."""
        def exact_keys(mapping: Mapping[str, object], expected: set[str], label: str) -> None:
            actual = set(mapping)
            if actual != expected:
                raise ValueError(
                    f"{label} keys mismatch: missing={sorted(expected-actual)}, "
                    f"unknown={sorted(actual-expected)}"
                )

        def strict_int(
            value: object,
            label: str,
            *,
            minimum: int | None = None,
        ) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
            if minimum is not None and value < minimum:
                raise ValueError(f"{label} must be at least {minimum}")
            return value

        def finite_real(value: object, label: str) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a JSON number")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{label} must be finite")
            return result

        canonical_root_keys = {
            "schema_version",
            "model_version",
            "dimension",
            "scalar",
            "memory",
            "eta",
            "step_index",
            "next_identity",
            "next_layer_serial",
            "root_bud",
            "layers",
        }
        missing_root_keys = canonical_root_keys - set(state)
        if missing_root_keys:
            raise ValueError(f"state keys mismatch: missing={sorted(missing_root_keys)}")
        state_extensions = {
            key: copy.deepcopy(value)
            for key, value in state.items()
            if key not in canonical_root_keys
        }
        if state["schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported state schema version")
        if state["model_version"] != MODEL_VERSION:
            raise ValueError("incompatible Auxein model version")
        dimension = strict_int(state["dimension"], "dimension", minimum=1)
        scalar_format = str(state["scalar"])
        if scalar_format not in ("f32", "f64"):
            raise ValueError("invalid serialized scalar format")
        policy = NumericPolicy(scalar_format)

        def persistent_real(value: object, label: str) -> float:
            result = finite_real(value, label)
            casted = policy.cast(result)
            if result != casted:
                raise ValueError(
                    f"{label} is not exactly representable in {scalar_format}; "
                    "silent format conversion is forbidden"
                )
            return result

        def persistent_vector(value: object, label: str) -> Vector:
            if not isinstance(value, list):
                raise ValueError(f"{label} must be a list")
            return Vector(
                persistent_real(component, f"{label}[{index}]")
                for index, component in enumerate(value)
            )

        def as_map(value: object, label: str) -> Mapping[str, object]:
            if not isinstance(value, Mapping):
                raise ValueError(f"{label} must be an object")
            return value

        def kernel_from(value: object, label: str) -> QuadraticKernel:
            data = as_map(value, label)
            exact_keys(data, {"W", "S", "Q"}, label)
            kernel = QuadraticKernel(
                dimension,
                persistent_real(data["W"], f"{label}.W"),
                persistent_vector(data["S"], f"{label}.S"),
                persistent_real(data["Q"], f"{label}.Q"),
            )
            if kernel.W > 0.0 and kernel.Q < _norm2(kernel.S) / kernel.W:
                raise ValueError(
                    f"{label} is not a canonical projected kernel: Q is below ||S||²/W"
                )
            return kernel

        identities: dict[int, CellIdentity] = {}

        def identity(token_value: object) -> CellIdentity:
            token = strict_int(token_value, "identity", minimum=1)
            return identities.setdefault(token, CellIdentity._from_token(token))

        root_bud = RootBud(dimension, kernel_from(state["root_bud"], "root_bud"))
        layers_value = state["layers"]
        if not isinstance(layers_value, list):
            raise ValueError("layers must be a list")
        layers: list[Layer] = []
        for layer_index, raw_layer in enumerate(layers_value):
            layer_label = f"layers[{layer_index}]"
            ld = as_map(raw_layer, layer_label)
            exact_keys(
                ld,
                {"owner_serial", "proposal_serial", "receipt", "cells", "bud"},
                layer_label,
            )
            cells_value = ld["cells"]
            if not isinstance(cells_value, list):
                raise ValueError(f"{layer_label}.cells must be a list")
            cells: list[Cell] = []
            for cell_index, raw_cell in enumerate(cells_value):
                cell_label = f"{layer_label}.cells[{cell_index}]"
                cd = as_map(raw_cell, cell_label)
                exact_keys(
                    cd,
                    {"identity", "center", "plus", "minus"},
                    cell_label,
                )
                cells.append(
                    Cell(
                        identity(cd["identity"]),
                        persistent_vector(cd["center"], f"{cell_label}.center"),
                        kernel_from(cd["plus"], f"{cell_label}.plus"),
                        kernel_from(cd["minus"], f"{cell_label}.minus"),
                    )
                )
            raw_bud = ld["bud"]
            bud: Bud | None
            if raw_bud is None:
                bud = None
            else:
                bud_label = f"{layer_label}.bud"
                bd = as_map(raw_bud, bud_label)
                exact_keys(bd, {"plus", "minus", "owners"}, bud_label)
                owners_raw = bd["owners"]
                if not isinstance(owners_raw, list):
                    raise ValueError(f"{bud_label}.owners must be a list")
                owners: dict[CellIdentity, OwnerMoments] = {}
                for owner_index, raw_owner in enumerate(owners_raw):
                    owner_label = f"{bud_label}.owners[{owner_index}]"
                    od = as_map(raw_owner, owner_label)
                    exact_keys(
                        od,
                        {
                            "identity",
                            "lambda_plus",
                            "f_plus",
                            "lambda_minus",
                            "f_minus",
                        },
                        owner_label,
                    )
                    ident = identity(od["identity"])
                    if ident in owners:
                        raise ValueError("duplicate serialized bud owner identity")
                    owners[ident] = OwnerMoments(
                        dimension,
                        persistent_real(od["lambda_plus"], f"{owner_label}.lambda_plus"),
                        persistent_vector(od["f_plus"], f"{owner_label}.f_plus"),
                        persistent_real(od["lambda_minus"], f"{owner_label}.lambda_minus"),
                        persistent_vector(od["f_minus"], f"{owner_label}.f_minus"),
                    )
                bud = Bud(
                    dimension,
                    kernel_from(bd["plus"], f"{bud_label}.plus"),
                    kernel_from(bd["minus"], f"{bud_label}.minus"),
                    owners,
                    scalar_format,
                )
            memory_half_life = finite_real(state["memory"], "memory")
            law = MemoryLaw(memory_half_life)
            layers.append(
                Layer(
                    dimension,
                    law.chi,
                    kernel_from(ld["receipt"], f"{layer_label}.receipt"),
                    cells,
                    bud,
                    strict_int(ld["owner_serial"], f"{layer_label}.owner_serial", minimum=1),
                    strict_int(ld["proposal_serial"], f"{layer_label}.proposal_serial", minimum=0),
                    scalar_format,
                    law.alpha,
                )
            )

        _strict_bool(check_invariants, "check_invariants")
        model = (
            ScalarFootprintMaintenance()
            if maintenance_model is None
            else maintenance_model
        )
        result = cls._from_parts(
            dimension=dimension,
            memory=finite_real(state["memory"], "memory"),
            budget=budget,
            budget_units=budget_units,
            eta=_learning_rate(finite_real(state["eta"], "eta")),
            maintenance_model=model,
            layers=layers,
            identity_factory=IdentityFactory(
                strict_int(state["next_identity"], "next_identity", minimum=1)
            ),
            root_bud=root_bud,
            scalar=scalar_format,
            step_index=strict_int(state["step_index"], "step_index", minimum=0),
            layer_serial=strict_int(
                state["next_layer_serial"], "next_layer_serial", minimum=1
            ),
            check_invariants=check_invariants,
            state_extensions=state_extensions,
        )
        return result

    def summary(self) -> dict[str, object]:
        return {
            "steps_seen": self._step_index,
            "dimension": self._dimension,
            "scalar": self.scalar,
            "memory": self.memory,
            "layer_count": len(self._layers),
            "cells_per_layer": [len(layer.cells) for layer in self._layers],
            "capital_per_layer": [layer.capital for layer in self._layers],
            "maintenance_units": self.maintenance_units(),
            "budget": str(self.budget),
            "budget_units": self.budget_units,
            "budget_margin_units": self.budget_margin_units,
            "is_solvent": self.is_solvent,
            "chi": self._chi,
            "alpha": self._alpha,
            "eta": self.eta,
            "effective_alpha": self.effective_alpha,
            "root_bud_mass": self._root_bud.kernel.W,
        }


__all__ = [
    "Auxein",
    "AuxeinError",
    "BudgetValue",
    "InsolventState",
    "InvalidStateOperation",
    "InvariantViolation",
    "LayerStepReport",
    "MaintenanceModel",
    "MODEL_VERSION",
    "ScalarFootprintMaintenance",
    "Scalar",
    "STATE_SCHEMA_VERSION",
    "StepReport",
    "TransformationKind",
    "TransformationRecord",
    "__version__",
]
