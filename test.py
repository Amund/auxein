"""Unit and invariant tests for the isolated Auxein reference engine."""

from __future__ import annotations

import copy
from decimal import Decimal
import json
import math
import random
import unittest
from contextlib import redirect_stderr
from io import StringIO

import auxein as auxein_module
from auxein import (
    Auxein,
    Bud,
    Cell,
    CellIdentity,
    GrowthProposal,
    IdentityFactory,
    InsolventState,
    InvalidStateOperation,
    InvariantViolation,
    Layer,
    MemoryLaw,
    NumericPolicy,
    OwnerMoments,
    ProposalToken,
    QuadraticKernel,
    RootBud,
    ScalarFootprintMaintenance,
    route_latent,
)


def budget_for(dimension: int, scalar_format: str, equivalent_cells: int) -> int:
    return ScalarFootprintMaintenance().budget_to_units(
        dimension,
        scalar_format,
        equivalent_cells,
    )


def kernel_from_mean(mean: float, weight: float = 1.0) -> QuadraticKernel:
    return QuadraticKernel.point([mean], weight)


def neutral_cell(identity_: CellIdentity, center: float, weight: float = 1.0) -> Cell:
    parent = QuadraticKernel(1, weight, [0.0], 0.0)
    plus, minus = parent.neutral_split()
    return Cell(identity_, [center], plus, minus)


def split_cell(identity_: CellIdentity, center: float, branch_mean: float = 1.0) -> Cell:
    return Cell(
        identity_,
        [center],
        kernel_from_mean(branch_mean),
        kernel_from_mean(-branch_mean),
    )


class QuadraticKernelTests(unittest.TestCase):
    def test_half_life_parameter(self) -> None:
        memory_half_life = 17.0
        chi = 2.0 ** (-1.0 / memory_half_life)
        self.assertAlmostEqual(chi**memory_half_life, 0.5)

    def test_update_and_recenter_preserve_structural_power(self) -> None:
        kernel = QuadraticKernel.zero(2)
        chi = 0.9
        for vector, relevance in (
            ([1.0, 0.0], 1.0),
            ([-1.0, 0.0], 0.5),
            ([0.0, 2.0], 0.25),
        ):
            kernel.update(vector, relevance, chi)
        before = kernel.structural_power
        kernel.recenter(kernel.mean)
        self.assertTrue(all(abs(value) <= math.ulp(1.0) for value in kernel.S))
        self.assertAlmostEqual(kernel.structural_power, before, places=13)
        kernel.validate()

    def test_coordinate_scaling_is_quadratic(self) -> None:
        kernel = QuadraticKernel.point([2.0, -1.0], 3.0)
        kernel.scale_coordinates(0.25)
        self.assertSequenceEqual(list(kernel.mean), [0.5, -0.25])
        self.assertAlmostEqual(kernel.Q, 3.0 * (0.5**2 + 0.25**2))

    def test_recenter_on_mean_closes_first_moment_exactly(self) -> None:
        kernel = QuadraticKernel.zero(3)
        for value, relevance in (
            ([7.0, -3.0, 0.25], 1.0),
            ([7.5, -2.0, 0.5], 0.4),
            ([6.5, -4.0, 0.0], 0.7),
        ):
            kernel.update(value, relevance, 0.93)
        before = kernel.structural_power
        kernel.recenter_on_mean()
        self.assertSequenceEqual(list(kernel.S), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(kernel.Q, before, places=14)

    def test_stable_recenter_preserves_structural_power_under_large_translation(self) -> None:
        kernel = QuadraticKernel.zero(2)
        for value in ([1.0e12, -1.0e12], [1.0e12 + 3.0, -1.0e12 + 4.0]):
            kernel.update(value, 1.0, 0.5)
        before = kernel.structural_power
        kernel.recenter([1.0e12, -1.0e12])
        self.assertAlmostEqual(kernel.structural_power, before, places=12)
        kernel.validate()

    def test_validation_tolerates_roundoff_but_rejects_material_violation(self) -> None:
        nearly_valid = QuadraticKernel(1, 1.0, [1.0], math.nextafter(1.0, 0.0))
        nearly_valid.validate()
        with self.assertRaises(InvariantViolation):
            QuadraticKernel(1, 1.0, [1.0], 0.99)


class NumericalDegeneracyTests(unittest.TestCase):
    def test_exact_repeated_point_keeps_zero_radius_in_both_formats(self) -> None:
        point = [-5.0, -0.16, -0.16]
        for scalar_format in ("f32", "f64"):
            with self.subTest(scalar_format=scalar_format):
                network = Auxein.empty(
                    3,
                    memory=100,
                    budget_units=budget_for(3, scalar_format, 2),
                    scalar=scalar_format,
                    check_invariants=True,
                )
                for _ in range(8):
                    network.step(point, detailed_report=False)
                    self.assertEqual(network._layers[0].geometry.radius, 0.0)

    def test_format_closure_does_not_erase_resolved_variance(self) -> None:
        for scalar_format, variance in (("f32", 1.0e-4), ("f64", 1.0e-12)):
            with self.subTest(scalar_format=scalar_format):
                receipt = QuadraticKernel(1, 1.0, [1.0], 1.0 + variance)
                cell = neutral_cell(CellIdentity._from_token(1), center=1.0)
                layer = Layer(
                    1,
                    0.9,
                    receipt,
                    [cell],
                    Bud.empty(1, [cell.identity], scalar_format=scalar_format),
                    1,
                    0,
                    scalar_format,
                    0.1,
                )
                self.assertGreater(layer.geometry.radius, 0.0)

    def test_root_founder_is_centered_by_construction(self) -> None:
        network = Auxein.empty(
            2,
            memory=50,
            budget_units=budget_for(2, "f64", 2),
            scalar="f64",
            check_invariants=True,
        )
        network.step([13.0, -7.0], detailed_report=False)
        founder = network._layers[0].cells[0]
        self.assertSequenceEqual(list(founder.plus.S), [0.0, 0.0])
        self.assertSequenceEqual(list(founder.minus.S), [0.0, 0.0])

    def test_identical_realized_representations_have_exact_zero_capital(self) -> None:
        receipt = QuadraticKernel(1, 2.0, [0.0], 2.0)
        left = Cell(
            CellIdentity._from_token(1),
            [-1.0],
            QuadraticKernel.point([1.0], 0.5),
            QuadraticKernel.point([1.0], 0.5),
        )
        right = Cell(
            CellIdentity._from_token(2),
            [1.0],
            QuadraticKernel.point([-1.0], 0.5),
            QuadraticKernel.point([-1.0], 0.5),
        )
        layer = Layer(
            1,
            0.9,
            receipt,
            [left, right],
            Bud.empty(1, [left.identity, right.identity]),
            1,
            0,
            "f64",
            0.1,
        )
        self.assertEqual(layer.capital, 0.0)

    def test_stable_capital_matches_aggregate_formula_away_from_degeneracy(self) -> None:
        receipt = QuadraticKernel(2, 3.0, [0.0, 0.0], 6.0)
        cells = [
            Cell(
                CellIdentity._from_token(1),
                [-1.0, 0.0],
                QuadraticKernel.point([0.2, 0.1], 0.5),
                QuadraticKernel.point([0.2, 0.1], 0.5),
            ),
            Cell(
                CellIdentity._from_token(2),
                [0.0, 1.0],
                QuadraticKernel.point([-0.3, 0.4], 0.75),
                QuadraticKernel.point([-0.3, 0.4], 0.75),
            ),
            Cell(
                CellIdentity._from_token(3),
                [1.0, -1.0],
                QuadraticKernel.point([0.1, -0.2], 0.25),
                QuadraticKernel.point([0.1, -0.2], 0.25),
            ),
        ]
        layer = Layer(
            2,
            0.9,
            receipt,
            cells,
            Bud.empty(2, [cell.identity for cell in cells]),
            1,
            0,
            "f64",
            0.1,
        )
        geometry = layer.geometry
        values = []
        for cell in cells:
            z = (cell.center - geometry.mean) / geometry.radius
            values.append((cell.mass, z + cell.parent_mean))
        total = math.fsum(weight for weight, _ in values)
        weighted = [
            math.fsum(weight * value[axis] for weight, value in values)
            for axis in range(2)
        ]
        aggregate = math.fsum(
            weight * sum(component * component for component in value)
            for weight, value in values
        ) - sum(component * component for component in weighted) / total
        self.assertAlmostEqual(layer.capital, aggregate, places=13)

    def test_stable_capital_keeps_small_separation_on_large_common_offset(self) -> None:
        receipt = QuadraticKernel(1, 2.0, [0.0], 2.0)
        left = Cell(
            CellIdentity._from_token(11),
            [0.0],
            QuadraticKernel.point([1.0e8], 0.5),
            QuadraticKernel.point([1.0e8], 0.5),
        )
        right = Cell(
            CellIdentity._from_token(12),
            [0.0],
            QuadraticKernel.point([1.0e8 + 1.0], 0.5),
            QuadraticKernel.point([1.0e8 + 1.0], 0.5),
        )
        layer = Layer(
            1,
            0.9,
            receipt,
            [left, right],
            Bud.empty(1, [left.identity, right.identity]),
            1,
            0,
            "f64",
            0.1,
        )
        self.assertEqual(layer.capital, 0.5)
        subtractive = (1.0e8**2 + (1.0e8 + 1.0) ** 2) - (
            (2.0e8 + 1.0) ** 2 / 2.0
        )
        self.assertEqual(subtractive, 0.0)

    def test_post_movement_conservation_ignores_rounding_residual_moment(self) -> None:
        receipt = QuadraticKernel(1, 2.0, [0.0], 2.0)
        victim = Cell(
            CellIdentity._from_token(1),
            [-1.0],
            QuadraticKernel(1, 0.5, [1.0e-18], 1.0e-30),
            QuadraticKernel(1, 0.5, [-0.5e-18], 1.0e-30),
        )
        survivor = neutral_cell(CellIdentity._from_token(2), center=1.0)
        layer = Layer(
            1,
            0.9,
            receipt,
            [victim, survivor],
            Bud.empty(1, [victim.identity, survivor.identity]),
            1,
            0,
            "f64",
            0.1,
        )
        expected = victim.mass * ((2.0 / layer.geometry.radius) ** 2)
        self.assertEqual(layer.conservation_value(0), expected)


class RoutingAndRecognitionTests(unittest.TestCase):
    def test_empty_latent_state_routes_first_proof_neutrally(self) -> None:
        plus = QuadraticKernel.zero(1)
        minus = QuadraticKernel.zero(1)
        decision = route_latent(plus, minus, [5.0], 0.8)
        self.assertEqual(decision.emission_sign, "+")
        self.assertEqual(decision.r_plus, 0.4)
        self.assertEqual(decision.r_minus, 0.4)

    def test_zero_axis_nonzero_residual_starts_plus_branch(self) -> None:
        parent = QuadraticKernel(1, 2.0, [0.0], 0.0)
        plus, minus = parent.neutral_split()
        decision = route_latent(plus, minus, [1.0], 0.75)
        self.assertEqual(decision.emission_sign, "+")
        self.assertEqual(decision.r_plus, 0.75)
        self.assertEqual(decision.r_minus, 0.0)

    def test_degenerate_radius_uses_exact_equality(self) -> None:
        cell = neutral_cell(CellIdentity._from_token(1), center=2.0)
        a_equal, e_equal = cell.recognition_and_error([2.0], 0.0)
        a_other, e_other = cell.recognition_and_error([math.nextafter(2.0, math.inf)], 0.0)
        self.assertEqual(a_equal, 1.0)
        self.assertTrue(e_equal == [0.0])
        self.assertEqual(a_other, 0.0)
        self.assertTrue(e_other == [0.0])

    def test_wta_selects_relatively_but_writes_absolutely(self) -> None:
        chi = 0.9
        receipt = QuadraticKernel(1, 1.0, [0.0], 1.0)
        cells = [
            neutral_cell(CellIdentity._from_token(1), -1.0),
            neutral_cell(CellIdentity._from_token(2), 1.0),
        ]
        layer = Layer(1, chi, receipt, cells, Bud.empty(1, [c.identity for c in cells]), 1)
        read = layer.prepare([0.0], 1.0)
        self.assertEqual(read.winner_slot, 0)
        self.assertAlmostEqual(read.winner_read.recognition, math.exp(-1.0))
        self.assertAlmostEqual(read.winner_read.relevance, math.exp(-1.0))


class SplitAndCapitalTests(unittest.TestCase):
    def test_split_gain_and_materialization(self) -> None:
        factory = IdentityFactory(2)
        mother_identity = CellIdentity._from_token(1)
        cell = split_cell(mother_identity, 0.0, branch_mean=1.0)
        self.assertAlmostEqual(cell.split_gain, 2.0)
        daughter = cell.materialize_split(1.0, factory)
        self.assertEqual(cell.identity, mother_identity)
        self.assertNotEqual(daughter.identity, mother_identity)
        self.assertAlmostEqual(cell.center[0], 1.0)
        self.assertAlmostEqual(daughter.center[0], -1.0)
        self.assertEqual(cell.split_gain, 0.0)
        self.assertEqual(daughter.split_gain, 0.0)

    def test_layer_capital_increases_by_split_gain(self) -> None:
        factory = IdentityFactory(2)
        cell = split_cell(CellIdentity._from_token(1), 0.0, branch_mean=1.0)
        layer = Layer(
            1,
            0.9,
            QuadraticKernel(1, 1.0, [0.0], 1.0),
            [cell],
            Bud.empty(1, [cell.identity]),
            1,
        )
        before = layer.capital
        gain = cell.split_gain
        proposal = layer.best_split_proposal(0, ScalarFootprintMaintenance())
        self.assertIsNotNone(proposal)
        layer.execute_split(proposal, layer.geometry.radius, factory)
        after = layer.capital
        self.assertAlmostEqual(before, 0.0)
        self.assertAlmostEqual(after - before, gain)

    def test_gamma_marginal_is_not_cell_conservation_value(self) -> None:
        identities = [CellIdentity._from_token(i) for i in range(1, 4)]
        cells = [neutral_cell(identity, center) for identity, center in zip(identities, [-1.0, 0.0, 1.0])]
        layer = Layer(
            1,
            0.9,
            QuadraticKernel(1, 1.0, [0.0], 1.0),
            cells,
            Bud.empty(1, identities),
            1,
        )
        gamma_before = layer.capital
        conservation = layer.conservation_value(1)
        survivors = [cells[0], cells[2]]
        reduced = Layer(
            1,
            0.9,
            layer.receipt.clone(),
            survivors,
            Bud.empty(1, [c.identity for c in survivors]),
            2,
        )
        self.assertAlmostEqual(gamma_before - reduced.capital, 0.0)
        self.assertAlmostEqual(conservation, 1.0)

    def test_same_layer_scale_transport_changes_split_power_quadratically(self) -> None:
        cell = split_cell(CellIdentity._from_token(1), 0.0, branch_mean=2.0)
        gain = cell.split_gain
        cell.scale_internal_coordinates(0.5)
        self.assertAlmostEqual(cell.split_gain, gain * 0.25)

    def test_batched_conservation_matches_scalar_reference(self) -> None:
        rng = random.Random(73)
        dimension = 4
        cells: list[Cell] = []
        identities: list[CellIdentity] = []
        for index in range(24):
            identity = CellIdentity._from_token(index + 1)
            identities.append(identity)
            center = [rng.uniform(-3.0, 3.0) for _ in range(dimension)]
            plus_point = [rng.uniform(-0.8, 0.8) for _ in range(dimension)]
            minus_point = [rng.uniform(-0.8, 0.8) for _ in range(dimension)]
            plus = QuadraticKernel.point(plus_point, rng.uniform(0.1, 2.0))
            minus = QuadraticKernel.point(minus_point, rng.uniform(0.1, 2.0))
            cells.append(Cell(identity, center, plus, minus))

        receipt = QuadraticKernel.zero(dimension)
        for _ in range(30):
            receipt.update(
                [rng.gauss(0.0, 1.0) for _ in range(dimension)],
                1.0,
                0.9,
            )
        layer = Layer(
            dimension,
            0.9,
            receipt,
            cells,
            Bud.empty(dimension, identities),
            1,
        )
        scalar = [layer.conservation_value(slot) for slot in range(len(cells))]
        batched = layer._conservation_values()
        for expected, actual in zip(scalar, batched, strict=True):
            self.assertAlmostEqual(actual, expected, places=11)


class BudTests(unittest.TestCase):
    def make_bud(self, opposite_second: bool = False) -> Bud:
        first = CellIdentity._from_token(1)
        second = CellIdentity._from_token(2)
        plus = QuadraticKernel.point([1.0], 2.0)
        minus = QuadraticKernel.point([-1.0], 2.0)
        second_plus = -1.0 if opposite_second else 1.0
        second_minus = 1.0 if opposite_second else -1.0
        owners = {
            first: OwnerMoments(1, 1.0, [1.0], 1.0, [-1.0]),
            second: OwnerMoments(
                1,
                1.0,
                [second_plus],
                1.0,
                [second_minus],
            ),
        }
        if opposite_second:
            plus = QuadraticKernel(1, 2.0, [0.0], 2.0)
            minus = QuadraticKernel(1, 2.0, [0.0], 2.0)
        bud = Bud(1, plus, minus, owners)
        bud.validate()
        return bud

    def test_identical_cross_identity_distinctions_are_positive(self) -> None:
        bud = self.make_bud(False)
        self.assertAlmostEqual(bud.split_gain, 4.0)
        self.assertGreater(bud.concordance, 0.0)

    def test_opposite_cross_identity_distinctions_are_negative(self) -> None:
        bud = self.make_bud(True)
        self.assertLess(bud.concordance, 0.0)

    def test_single_identity_has_zero_concordance(self) -> None:
        identity = CellIdentity._from_token(1)
        owner = OwnerMoments(1, 1.0, [1.0], 1.0, [-1.0])
        bud = Bud(
            1,
            QuadraticKernel.point([1.0], 1.0),
            QuadraticKernel.point([-1.0], 1.0),
            {identity: owner},
        )
        self.assertEqual(bud.concordance, 0.0)

    def test_single_identity_numerical_cancellation_stays_exactly_zero(self) -> None:
        identity = CellIdentity._from_token(9001)
        weight = 3.1376477997153496e-08
        distinction = -0.6444550101251245
        branch_mass = 2.0 * weight
        owner = OwnerMoments(
            1,
            branch_mass,
            [branch_mass * distinction / 2.0],
            branch_mass,
            [-branch_mass * distinction / 2.0],
        )
        bud = Bud.empty(1, [identity])
        bud.owners[identity] = owner
        self.assertEqual(bud.concordance, 0.0)

    def test_incremental_concordance_matches_explicit_cross_owner_formula(self) -> None:
        identities = [CellIdentity._from_token(9101), CellIdentity._from_token(9102), CellIdentity._from_token(9103)]
        bud = Bud.empty(1, identities)
        moments = [
            OwnerMoments(1, 2.0, [3.0], 1.0, [-0.5]),
            OwnerMoments(1, 1.5, [-1.5], 0.75, [-0.75]),
            OwnerMoments(1, 0.8, [0.4], 1.2, [-0.9]),
        ]
        for identity, owner in zip(identities, moments, strict=True):
            bud.owners[identity] = owner
        active = [owner.distinction for owner in moments]
        total = sum(weight for weight, _ in active)
        expected = 2.0 * sum(
            active[i][0] * active[j][0] * active[i][1][0] * active[j][1][0]
            for i in range(len(active))
            for j in range(i + 1, len(active))
        ) / total
        self.assertAlmostEqual(bud.concordance, expected, places=15)

    def test_terminal_death_resets_bud_and_keeps_survivor_records(self) -> None:
        bud = self.make_bud(False)
        cells = [neutral_cell(identity, center) for identity, center in zip(bud.owners, [-1.0, 1.0])]
        layer = Layer(
            1,
            0.9,
            QuadraticKernel(1, 1.0, [0.0], 1.0),
            cells,
            bud,
            1,
        )
        offer = layer.best_cell_contraction_offer(0, ScalarFootprintMaintenance())
        self.assertIsNotNone(offer)
        layer.execute_death(offer.token)
        self.assertEqual(len(layer.cells), 1)
        self.assertEqual(layer.bud.parent.W, 0.0)
        self.assertEqual(set(layer.bud.owners), {layer.cells[0].identity})


class NetworkTests(unittest.TestCase):
    def test_public_surface_is_deliberately_small(self) -> None:
        self.assertEqual(
            set(auxein_module.__all__),
            {
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
            },
        )

    def test_public_memory_scalar_and_seed_names_are_canonical(self) -> None:
        network = Auxein.from_seed(
            [0.0],
            memory=12.5,
            budget=2,
            scalar="f32",
        )
        self.assertEqual(network.memory, 12.5)
        self.assertEqual(network.scalar, "f32")
        self.assertFalse(hasattr(network, "scalar_format"))
        self.assertFalse(hasattr(Auxein, "seeded"))
        self.assertTrue(hasattr(Auxein, "from_seed"))

        state = network.to_state_dict()
        self.assertEqual(state["memory"], 12.5)
        self.assertEqual(state["scalar"], "f32")
        self.assertNotIn("t_mem", state)
        self.assertNotIn("scalar_format", state)

        with self.assertRaises(TypeError):
            Auxein.empty(1, budget=1, **{"t_mem": 10.0})
        with self.assertRaises(TypeError):
            Auxein.empty(1, memory=10.0, budget=1, **{"scalar_format": "f64"})
        with self.assertRaises(AttributeError):
            network.memory = 20.0  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            network.scalar = "f64"  # type: ignore[misc]

    def test_public_budget_and_budget_units_share_one_canonical_conversion(self) -> None:
        maintenance_model = ScalarFootprintMaintenance()
        network = Auxein.empty(
            2,
            memory=10.0,
            budget=Decimal("3.25"),
            maintenance_model=maintenance_model,
            scalar="f64",
        )
        self.assertEqual(
            network.budget_units,
            maintenance_model.budget_to_units(2, "f64", Decimal("3.25")),
        )
        self.assertEqual(network.budget, Decimal("3.25"))

        network.budget = Decimal("0")
        self.assertEqual(
            network.budget_units,
            maintenance_model.root_substrate_units(2, "f64"),
        )
        network.budget_units = maintenance_model.budget_to_units(2, "f64", 7)
        self.assertEqual(network.budget, Decimal(7))

        with self.assertRaises(TypeError):
            Auxein.empty(1, memory=10.0)
        with self.assertRaises(TypeError):
            Auxein.empty(1, memory=10.0, budget=1, budget_units=100)
        with self.assertRaises(TypeError):
            Auxein.empty(1, memory=10.0, budget="not-a-budget")
        with self.assertRaises(TypeError):
            network.budget_units = 1.5  # type: ignore[assignment]

        for obsolete_name in (
            "network_cost",
            "root_bud_cost",
            "layer_cost",
            "cell_cost",
            "bud_base_cost",
            "owner_record_cost",
        ):
            self.assertFalse(hasattr(maintenance_model, obsolete_name))

    def test_budget_margin_and_solvency_are_structural_properties(self) -> None:
        network = Auxein.empty(1, memory=10.0, budget=1)
        self.assertEqual(
            network.budget_margin_units,
            network.budget_units - network.maintenance_units(),
        )
        self.assertTrue(network.is_solvent)

        network.budget_units = network.maintenance_units() - 1
        self.assertEqual(network.budget_margin_units, -1)
        self.assertFalse(network.is_solvent)

    def test_compact_report_uses_none_for_omitted_diagnostics(self) -> None:
        network = Auxein.from_seed([0.0], memory=10.0, budget=2)
        report = network.step([0.5], detailed_report=False)
        self.assertEqual(report.layer_reports, ())
        self.assertIsNone(report.vertical_gain)
        self.assertIsNone(report.vertical_concordance)

    def test_eta_scales_learning_and_zero_freezes_voluntary_adaptation(self) -> None:
        full = Auxein.empty(1, memory=10.0, budget=0, eta=1.0)
        half = Auxein.empty(1, memory=10.0, budget=0, eta=0.5)
        stopped = Auxein.empty(1, memory=10.0, budget=0, eta=0.0)
        full.step([2.0], detailed_report=False)
        half.step([2.0], detailed_report=False)
        stopped.step([2.0], detailed_report=False)
        self.assertAlmostEqual(half._root_bud.kernel.W, 0.5 * full._root_bud.kernel.W)
        self.assertEqual(stopped._root_bud.kernel.W, 0.0)
        self.assertAlmostEqual(half.effective_alpha, 0.5 * full.alpha)

        frozen = Auxein.from_seed([0.0], memory=10.0, budget=10, eta=0.0)
        before = frozen.to_state_dict()
        report = frozen.step([10.0], detailed_report=True)
        after = frozen.to_state_dict()
        after["step_index"] = before["step_index"]
        self.assertEqual(after, before)
        self.assertEqual(report.transformations, ())
        self.assertEqual(len(report.layer_reports), 1)

    def test_eta_is_validated_and_serialized(self) -> None:
        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    Auxein.empty(1, memory=10.0, budget=1, eta=invalid)
        for invalid_type in (True, "0.5"):
            with self.subTest(invalid_type=invalid_type):
                with self.assertRaises(TypeError):
                    Auxein.empty(1, memory=10.0, budget=1, eta=invalid_type)  # type: ignore[arg-type]

        network = Auxein.empty(1, memory=10.0, budget=1, eta=0.25)
        state = network.to_state_dict()
        self.assertEqual(state["eta"], 0.25)
        restored = Auxein.from_state_dict(state, budget=network.budget)
        self.assertEqual(restored.eta, 0.25)
        self.assertEqual(restored.to_state_dict(), state)

    def test_vertical_birth_consumes_bud_and_creates_fresh_terminal_bud(self) -> None:
        net = Auxein.from_seed([0.0], memory=10.0, budget_units=budget_for(1, "f64", 100))
        root = net._terminal
        identities = [root.cells[0].identity, net._identity_factory.new()]
        # Add a second terminal Cell solely so the bud's two owner identities are canonical.
        root.cells.append(neutral_cell(identities[1], 2.0))
        root.bud = Bud.empty(1, identities)
        root.bud.plus = QuadraticKernel.point([1.0], 2.0)
        root.bud.minus = QuadraticKernel.point([-1.0], 2.0)
        root.bud.owners = {
            identities[0]: OwnerMoments(1, 1.0, [1.0], 1.0, [-1.0]),
            identities[1]: OwnerMoments(1, 1.0, [1.0], 1.0, [-1.0]),
        }
        root.validate()
        proposal = net._vertical_proposal()
        self.assertIsNotNone(proposal)
        net._execute_vertical_birth(proposal)
        self.assertEqual(len(net._layers), 2)
        self.assertIsNone(net._layers[0].bud)
        self.assertEqual(len(net._layers[1].cells), 2)
        self.assertEqual(net._layers[1].bud.parent.W, 0.0)
        self.assertEqual(set(net._layers[1].bud.owners), {c.identity for c in net._layers[1].cells})
        self.assertTrue(all(cell.split_gain == 0.0 for cell in net._layers[1].cells))

    def make_saturated_reallocation_network(self, budget: int | None = None) -> Auxein:
        chi = 0.9
        factory = IdentityFactory(3)
        q = split_cell(CellIdentity._from_token(1), 0.0, branch_mean=2.0)
        victim = neutral_cell(CellIdentity._from_token(2), 0.1, weight=1.0)
        receipt = QuadraticKernel(1, 1.0, [0.0], 1.0)
        bud = Bud.empty(1, [q.identity, victim.identity])
        layer = Layer(1, chi, receipt, [q, victim], bud, 1)
        net = Auxein._from_parts(
            dimension=1,
            memory=math.log(0.5) / math.log(chi),
            budget_units=10**9,
            maintenance_model=ScalarFootprintMaintenance(),
            layers=[layer],
            identity_factory=factory,
        )
        net.budget_units = net.maintenance_units() if budget is None else int(budget)
        return net

    def test_voluntary_reallocation_is_dry_and_delayed(self) -> None:
        net = self.make_saturated_reallocation_network()
        report = net.step([0.0])
        kinds = [record.kind for record in report.transformations]
        self.assertIn("horizontal_reallocation_death", kinds)
        self.assertNotIn("split", kinds)
        self.assertEqual(len(net._layers[0].cells), 1)
        self.assertEqual(report.remaining_step_budget_units, 0.0)
        self.assertLess(report.maintenance_units, report.maintenance_charged_units)

        next_report = net.step([0.0])
        self.assertIn("split", [record.kind for record in next_report.transformations])

    def test_forced_solvency_can_truncate_layer_zero_before_perception(self) -> None:
        one_active = budget_for(1, "f64", 1)
        net = self.make_saturated_reallocation_network(one_active - 1)
        report = net.step([0.0])
        self.assertIn("truncate", [record.kind for record in report.transformations])
        self.assertLessEqual(report.maintenance_charged_units, net.budget_units)
        self.assertEqual(len(net._layers), 0)
        self.assertGreater(net._root_bud.kernel.W, 0.0)

    def test_eta_zero_keeps_forced_solvency_active(self) -> None:
        one_active = budget_for(1, "f64", 1)
        net = self.make_saturated_reallocation_network(one_active - 1)
        net.eta = 0.0
        report = net.step([0.0])
        self.assertIn("truncate", [record.kind for record in report.transformations])
        self.assertLessEqual(report.maintenance_charged_units, net.budget_units)
        self.assertEqual(len(net._layers), 0)
        self.assertEqual(net._root_bud.kernel.W, 0.0)

    def test_root_bud_warms_up_and_reincarnates_layer_zero(self) -> None:
        one_active = budget_for(1, "f64", 1)
        net = Auxein.from_seed([0.0], memory=10.0, budget_units=one_active - 1)
        report = net.step([1.0])
        self.assertIn("truncate", [record.kind for record in report.transformations])
        self.assertEqual(len(net._layers), 0)
        self.assertGreater(net._root_bud.kernel.W, 0.0)

        # The root bud is persistent but cannot materialize topology while the
        # economic veto remains active.
        second = net.step([0.0])
        self.assertEqual(len(net._layers), 0)
        self.assertGreater(net._root_bud.kernel.W, 0.0)

        net.budget_units = one_active
        third = net.step([0.5])
        self.assertIn("root_birth", [record.kind for record in third.transformations])
        self.assertEqual(len(net._layers), 1)
        self.assertEqual(len(net._layers[0].cells), 1)
        self.assertEqual(net._root_bud.kernel.W, 0.0)

    def test_random_stream_preserves_invariants(self) -> None:
        rng = random.Random(7)
        net = Auxein.from_seed([0.0, 0.0, 0.0], memory=25.0, budget_units=budget_for(3, "f64", 100))
        for _ in range(250):
            net.step([rng.gauss(0.0, 1.0) for _ in range(3)])
        net.validate()
        self.assertGreaterEqual(len(net._layers), 1)

    def test_invariant_checks_do_not_change_the_causal_trajectory(self) -> None:
        budget = budget_for(2, "f64", 20)
        checked = Auxein.empty(
            2,
            memory=25.0,
            budget_units=budget,
            check_invariants=True,
        )
        unchecked = Auxein.empty(
            2,
            memory=25.0,
            budget_units=budget,
            check_invariants=False,
        )
        rng = random.Random(991)
        for _ in range(150):
            value = [rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)]
            report_checked = checked.step(value, detailed_report=False)
            report_unchecked = unchecked.step(value, detailed_report=False)
            self.assertEqual(report_checked.transformations, report_unchecked.transformations)
            self.assertEqual(checked.to_state_dict(), unchecked.to_state_dict())

    def test_proposals_are_opaque_to_network_level(self) -> None:
        net = self.make_saturated_reallocation_network(10**9)
        proposal = net._layers[0].best_split_proposal(0, net.maintenance_model)
        self.assertIsNotNone(proposal)
        self.assertFalse(hasattr(proposal, "cell_identity"))
        self.assertFalse(hasattr(proposal, "victim_identity"))

    def test_empty_network_has_only_root_bud_maintenance(self) -> None:
        net = Auxein.empty(3, memory=10.0, budget_units=budget_for(3, "f64", 1))
        self.assertEqual(len(net._layers), 0)
        self.assertIsInstance(net._root_bud, RootBud)
        self.assertEqual(
            net.maintenance_units(),
            ScalarFootprintMaintenance().root_substrate_units(3, "f64"),
        )
        report = net.step([1.0, 2.0, 3.0])
        self.assertIn("root_birth", [record.kind for record in report.transformations])
        self.assertEqual(len(net._layers), 1)

    def test_literal_positive_gain_can_split_after_one_informative_write(self) -> None:
        net = Auxein.from_seed([0.0], memory=100.0, budget_units=budget_for(1, "f64", 100))
        net.step([1.0])  # widens the layer receipt; radius was initially zero
        report = net.step([-1.0])
        splits = [record for record in report.transformations if record.kind == "split"]
        self.assertEqual(len(splits), 1)
        self.assertGreater(splits[0].geometric_value, 0.0)


    def test_memory_law_remains_stable_for_extreme_half_life(self) -> None:
        law = MemoryLaw(1e20)
        self.assertGreater(law.alpha, 0.0)
        self.assertEqual(law.chi, 1.0)
        kernel = QuadraticKernel.zero(1)
        kernel.update([1.0], 1.0, law.chi, alpha=law.alpha)
        self.assertGreater(kernel.W, 0.0)

    def test_f32_f64_budget_abstraction_and_rounding(self) -> None:
        maintenance = ScalarFootprintMaintenance()
        budget32 = budget_for(2, "f32", 10)
        budget64 = budget_for(2, "f64", 10)
        self.assertGreater(budget64, budget32)
        self.assertEqual(
            (budget32 - maintenance.root_substrate_units(2, "f32") - maintenance.active_shell_units(2, "f32"))
            // maintenance.equivalent_cell_units(2, "f32"),
            10,
        )
        self.assertEqual(
            (budget64 - maintenance.root_substrate_units(2, "f64") - maintenance.active_shell_units(2, "f64"))
            // maintenance.equivalent_cell_units(2, "f64"),
            10,
        )

        root_only = maintenance.root_substrate_units(1, "f32")
        net = Auxein.empty(1, memory=10.0, budget_units=root_only, scalar="f32")
        net.step([0.1], detailed_report=False)
        policy = NumericPolicy("f32")
        self.assertEqual(net._root_bud.kernel.W, policy.cast(net._root_bud.kernel.W))
        self.assertEqual(net._root_bud.kernel.S[0], policy.cast(net._root_bud.kernel.S[0]))


    def test_economy_remains_exact_above_binary64_integer_resolution(self) -> None:
        huge_budget = 10**30 + 123
        net = Auxein.empty(
            1, memory=10.0, budget_units=huge_budget, scalar="f64"
        )
        maintenance = net.maintenance_units()
        self.assertIsInstance(maintenance, int)
        self.assertEqual((huge_budget - maintenance) + maintenance, huge_budget)

    def test_strict_serialization_round_trip_and_replay(self) -> None:
        budget = budget_for(2, "f32", 20)
        first = Auxein.empty(
            2, memory=37.0, budget_units=budget, scalar="f32"
        )
        prefix = [[0.1, 0.2], [1.0, -1.0], [-1.0, 1.0], [0.3, 0.4]]
        for value in prefix:
            first.step(value, detailed_report=False)
        state = first.to_state_dict()
        self.assertNotIn("budget_units", state)
        restored = Auxein.from_state_dict(
            json.loads(json.dumps(state, allow_nan=False)),
            budget_units=budget,
        )
        self.assertEqual(state, restored.to_state_dict())

        for value in [[0.5, 0.6], [-0.25, 0.75]]:
            report_a = first.step(value, detailed_report=False)
            report_b = restored.step(value, detailed_report=False)
            self.assertEqual(first.to_state_dict(), restored.to_state_dict())
            self.assertEqual(report_a.transformations, report_b.transformations)

        malformed = copy.deepcopy(state)
        malformed["surprise"] = 1
        with self.assertRaises(ValueError):
            Auxein.from_state_dict(malformed, budget_units=budget)

    def test_benchmark_uses_canonical_public_options(self) -> None:
        from benchmark import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--scalar",
                "f32",
                "--memory",
                "50",
                "--steps",
                "1",
                "--warmup",
                "0",
            ]
        )
        self.assertEqual(args.scalar, "f32")
        self.assertEqual(args.memory, 50.0)
        for obsolete_option in ("--scalar-format", "--t-mem"):
            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([obsolete_option, "50"])


class EdgeCaseRiskTests(unittest.TestCase):
    def test_numeric_policy_rejects_nonfinite_and_f32_overflow(self) -> None:
        for scalar_format in ("f32", "f64"):
            policy = NumericPolicy(scalar_format)
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(scalar_format=scalar_format, value=value):
                    with self.assertRaises(ValueError):
                        policy.cast(value)
        with self.assertRaises(ValueError):
            NumericPolicy("f32").cast(1.0e39)

    def test_next_up_is_finite_or_rejected_at_format_maximum(self) -> None:
        maxima = {
            "f32": float.fromhex("0x1.fffffep+127"),
            "f64": float.fromhex("0x1.fffffffffffffp+1023"),
        }
        for scalar_format, maximum in maxima.items():
            with self.subTest(scalar_format=scalar_format):
                with self.assertRaises(ValueError):
                    NumericPolicy(scalar_format).next_up(maximum)

    def test_next_up_treats_signed_zero_as_the_same_boundary(self) -> None:
        policy = NumericPolicy("f32")
        positive = policy.next_up(0.0)
        negative = policy.next_up(-0.0)
        self.assertEqual(positive, negative)
        self.assertGreater(positive, 0.0)
        self.assertEqual(policy.cast(positive), positive)

    def test_f32_underflow_is_deterministic_zero(self) -> None:
        policy = NumericPolicy("f32")
        self.assertEqual(policy.cast(1.0e-50), 0.0)
        self.assertEqual(policy.cast(-1.0e-50), 0.0)

    def test_memory_law_rejects_nonpositive_or_nonfinite_half_life(self) -> None:
        for value in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MemoryLaw(value)

    def test_zero_mass_kernel_requires_exact_zero_moments(self) -> None:
        with self.assertRaises(InvariantViolation):
            QuadraticKernel(1, 0.0, [math.ulp(0.0)], 0.0)
        with self.assertRaises(InvariantViolation):
            QuadraticKernel(1, 0.0, [0.0], math.ulp(0.0))

    def test_f32_projection_raises_q_to_preserve_quadratic_invariant(self) -> None:
        network = Auxein.empty(
            1,
            memory=10.0,
            budget_units=budget_for(1, "f32", 1),
            scalar="f32",
        )
        policy = NumericPolicy("f32")
        kernel = QuadraticKernel.point([1.0], 1.0)
        kernel.S[0] = policy.next_up(1.0)
        kernel.Q = 1.0
        network._quantize_kernel(kernel)
        required = kernel.S[0] * kernel.S[0] / kernel.W
        self.assertGreaterEqual(kernel.Q, required)
        self.assertEqual(kernel.Q, policy.cast(kernel.Q))
        kernel.validate()

    def test_exact_signed_routing_boundary_splits_neutrally_and_favors_plus(self) -> None:
        plus = QuadraticKernel.point([1.0], 1.0)
        minus = QuadraticKernel.point([-1.0], 1.0)
        decision = route_latent(plus, minus, [0.0], 0.75)
        self.assertEqual(decision.r_plus, 0.375)
        self.assertEqual(decision.r_minus, 0.375)
        self.assertEqual(decision.emission_sign, "+")

    def test_exact_growth_value_tie_favors_vertical_then_lower_layer(self) -> None:
        proposals = [
            GrowthProposal(ProposalToken(1, 1), "split", 0, 2.0, 1, 1),
            GrowthProposal(ProposalToken(2, 1), "vertical_birth", 3, 2.0, 1, 1),
            GrowthProposal(ProposalToken(3, 1), "split", 1, 2.0, 1, 1),
        ]
        chosen = min(proposals, key=Auxein._proposal_priority)
        self.assertEqual(chosen.kind, "vertical_birth")
        horizontal = [proposal for proposal in proposals if proposal.kind == "split"]
        self.assertEqual(min(horizontal, key=Auxein._proposal_priority).layer_index, 0)

    def test_root_birth_is_allowed_at_exact_budget_and_denied_one_unit_below(self) -> None:
        def prepared() -> tuple[Auxein, int, int]:
            net = Auxein.empty(
                1,
                memory=10.0,
                budget_units=10**9,
                scalar="f64",
            )
            net._root_bud.inject([0.25], net.chi, net.alpha)
            net._quantize_state()
            base = net.maintenance_units()
            return net, base, net._root_birth_delta()

        exact, base, delta = prepared()
        exact.budget_units = base + delta
        exact_report = exact.step([0.25], detailed_report=False)
        self.assertIn("root_birth", [record.kind for record in exact_report.transformations])

        blocked, base, delta = prepared()
        blocked.budget_units = base + delta - 1
        blocked_report = blocked.step([0.25], detailed_report=False)
        self.assertNotIn("root_birth", [record.kind for record in blocked_report.transformations])
        self.assertEqual(len(blocked._layers), 0)

    def test_empty_state_below_irreducible_maintenance_is_inexecutable(self) -> None:
        net = Auxein.empty(
            1,
            memory=10.0,
            budget_units=budget_for(1, "f64", 0),
        )
        net.budget_units = net.maintenance_units() - 1
        with self.assertRaises(InsolventState):
            net.step([0.0], detailed_report=False)

    def test_consumed_split_proposal_cannot_be_replayed(self) -> None:
        factory = IdentityFactory(2)
        cell = split_cell(CellIdentity._from_token(1), 0.0, branch_mean=1.0)
        layer = Layer(
            1,
            0.9,
            QuadraticKernel(1, 1.0, [0.0], 1.0),
            [cell],
            Bud.empty(1, [cell.identity]),
            1,
        )
        proposal = layer.best_split_proposal(0, ScalarFootprintMaintenance())
        self.assertIsNotNone(proposal)
        layer.execute_split(proposal, layer.geometry.radius, factory)
        with self.assertRaises(InvalidStateOperation):
            layer.execute_split(proposal, layer.geometry.radius, factory)

    def test_serialization_rejects_active_proposal_arbitration(self) -> None:
        net = Auxein.from_seed(
            [0.0], memory=10.0, budget_units=budget_for(1, "f64", 10)
        )
        net._layers[0].cells[0] = split_cell(
            net._layers[0].cells[0].identity, 0.0, branch_mean=1.0
        )
        proposal = net._layers[0].best_split_proposal(0, net.maintenance_model)
        self.assertIsNotNone(proposal)
        with self.assertRaises(InvalidStateOperation):
            net.to_state_dict()

    def test_serialization_rejects_nonfinite_geometry(self) -> None:
        net = Auxein.empty(
            1, memory=10.0, budget_units=budget_for(1, "f64", 1)
        )
        state = net.to_state_dict()
        state["root_bud"]["W"] = math.nan
        with self.assertRaises(ValueError):
            Auxein.from_state_dict(state, budget_units=net.budget_units)

    def test_serialization_rejects_fractional_and_boolean_counters(self) -> None:
        net = Auxein.empty(
            1, memory=10.0, budget_units=budget_for(1, "f64", 1)
        )
        for field, value in (("step_index", 0.5), ("next_identity", True)):
            state = net.to_state_dict()
            state[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    Auxein.from_state_dict(state, budget_units=net.budget_units)

    def test_serialization_rejects_silent_f32_conversion(self) -> None:
        net = Auxein.empty(
            1,
            memory=10.0,
            budget_units=budget_for(1, "f32", 1),
            scalar="f32",
        )
        state = net.to_state_dict()
        state["root_bud"] = {"W": 1.0, "S": [0.1], "Q": 1.0}
        with self.assertRaises(ValueError):
            Auxein.from_state_dict(state, budget_units=net.budget_units)

    def test_serialization_rejects_noncanonical_quadratic_projection(self) -> None:
        net = Auxein.empty(
            1, memory=10.0, budget_units=budget_for(1, "f64", 1)
        )
        state = net.to_state_dict()
        state["root_bud"] = {
            "W": 1.0,
            "S": [1.0],
            "Q": math.nextafter(1.0, 0.0),
        }
        with self.assertRaises(ValueError):
            Auxein.from_state_dict(state, budget_units=net.budget_units)

    def test_quadratic_overflow_is_atomic_at_kernel_and_step_boundaries(self) -> None:
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        kernel = QuadraticKernel.point([1.0], 1.0)
        kernel_before = (kernel.W, list(kernel.S), kernel.Q)
        with self.assertRaises(InvariantViolation):
            kernel.update([maximum], 1.0, 0.5)
        self.assertEqual((kernel.W, list(kernel.S), kernel.Q), kernel_before)

        net = Auxein.empty(
            1, memory=10.0, budget_units=budget_for(1, "f64", 1)
        )
        state_before = copy.deepcopy(net.to_state_dict())
        with self.assertRaises(ValueError):
            net.step([maximum], detailed_report=False)
        self.assertEqual(net.to_state_dict(), state_before)
        self.assertEqual(net.step_index, 0)

    def test_new_topology_does_not_emit_or_request_during_its_birth_step(self) -> None:
        root_net = Auxein.empty(
            1, memory=10.0, budget_units=budget_for(1, "f64", 4)
        )
        root_report = root_net.step([0.25])
        self.assertIn("root_birth", [record.kind for record in root_report.transformations])
        self.assertEqual(root_report.layer_reports, ())
        self.assertEqual(root_net._terminal.bud.parent.W, 0.0)
        self.assertFalse(root_net._terminal._pending)

        vertical_net = Auxein.from_seed(
            [0.0], memory=10.0, budget_units=budget_for(1, "f64", 100)
        )
        root = vertical_net._terminal
        identities = [root.cells[0].identity, vertical_net._identity_factory.new()]
        root.cells.append(neutral_cell(identities[1], 2.0))
        root.bud = Bud.empty(1, identities)
        root.bud.plus = QuadraticKernel.point([1.0], 2.0)
        root.bud.minus = QuadraticKernel.point([-1.0], 2.0)
        root.bud.owners = {
            identities[0]: OwnerMoments(1, 1.0, [1.0], 1.0, [-1.0]),
            identities[1]: OwnerMoments(1, 1.0, [1.0], 1.0, [-1.0]),
        }
        vertical_net.validate()

        old_layer_count = len(vertical_net._layers)
        vertical_report = vertical_net.step([0.0])
        self.assertIn(
            "vertical_birth", [record.kind for record in vertical_report.transformations]
        )
        self.assertEqual(len(vertical_report.layer_reports), old_layer_count)
        self.assertEqual(vertical_net._terminal.bud.parent.W, 0.0)
        self.assertTrue(
            all(cell.split_gain == 0.0 for cell in vertical_net._terminal.cells)
        )
        self.assertTrue(all(not layer._pending for layer in vertical_net._layers))

    def test_f32_positive_branch_underflow_closes_canonically(self) -> None:
        policy = NumericPolicy("f32")
        minimum_subnormal = policy.next_up(0.0)
        net = Auxein.from_seed(
            [0.0],
            memory=10.0,
            budget_units=budget_for(1, "f32", 4),
            scalar="f32",
        )
        bud = net._terminal.bud
        self.assertIsNotNone(bud)
        owner = next(iter(bud.owners.values()))
        below_half_ulp = minimum_subnormal / 2.0
        owner.lambda_plus = below_half_ulp
        bud.plus = QuadraticKernel(1, below_half_ulp, [0.0], 0.0)

        net._quantize_state()
        self.assertEqual(owner.lambda_plus, 0.0)
        self.assertEqual(bud.plus.W, 0.0)
        self.assertSequenceEqual(list(bud.plus.S), [0.0])
        self.assertEqual(bud.plus.Q, 0.0)
        net.validate()

        survivor = QuadraticKernel(1, minimum_subnormal, [0.0], 0.0)
        net._quantize_kernel(survivor)
        self.assertEqual(survivor.W, minimum_subnormal)

    def test_serialization_rejects_duplicate_live_identities_and_stale_factory(self) -> None:
        net = Auxein.from_seed(
            [0.0], memory=10.0, budget_units=budget_for(1, "f64", 8)
        )

        duplicated = net.to_state_dict()
        first = copy.deepcopy(duplicated["layers"][0])
        second = copy.deepcopy(first)
        first["bud"] = None
        second["owner_serial"] = 2
        duplicated["layers"] = [first, second]
        duplicated["next_layer_serial"] = 3
        with self.assertRaises(InvariantViolation):
            Auxein.from_state_dict(duplicated, budget_units=net.budget_units)

        stale = net.to_state_dict()
        stale["next_identity"] = stale["layers"][0]["cells"][0]["identity"]
        with self.assertRaises(InvariantViolation):
            Auxein.from_state_dict(stale, budget_units=net.budget_units)

    def test_step_rejects_nonfinite_or_wrong_dimension_input(self) -> None:
        net = Auxein.empty(
            1, memory=10.0, budget_units=budget_for(1, "f64", 1)
        )
        state_before = copy.deepcopy(net.to_state_dict())
        for value in ([math.nan], [math.inf], [0.0, 1.0]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    net.step(value, detailed_report=False)
                self.assertEqual(net.to_state_dict(), state_before)



class PublicApiValidationTests(unittest.TestCase):
    def test_direct_constructor_is_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "Auxein.empty"):
            Auxein()

    def test_public_parameters_are_not_silently_coerced(self) -> None:
        invalid_calls = (
            lambda: Auxein.empty(2.7, memory=10.0, budget=1),
            lambda: Auxein.empty(True, memory=10.0, budget=1),
            lambda: Auxein.empty(2, memory=True, budget=1),
            lambda: Auxein.empty(2, memory="10", budget=1),
            lambda: Auxein.empty(2, memory=10.0, budget=1, check_invariants="no"),
            lambda: Auxein.from_seed([0.0], memory=10.0, budget=1, seed_weight=True),
            lambda: Auxein.from_seed([0.0], memory=10.0, budget=1, seed_weight="1"),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_scalar_footprint_integer_width_is_strict(self) -> None:
        for value in (True, 8.0, "8"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    ScalarFootprintMaintenance(value)
        with self.assertRaises(ValueError):
            ScalarFootprintMaintenance(0)

    def test_maintenance_model_surface_is_validated(self) -> None:
        with self.assertRaisesRegex(TypeError, "maintenance_model"):
            Auxein.empty(1, memory=10.0, budget=1, maintenance_model=object())

    def test_causal_configuration_is_read_only(self) -> None:
        network = Auxein.empty(2, memory=10.0, budget=2)
        for name, value in (
            ("dimension", 3),
            ("memory", 20.0),
            ("scalar", "f32"),
            ("chi", 1.0),
            ("alpha", 0.0),
            ("step_index", 4),
            ("maintenance_model", ScalarFootprintMaintenance()),
            ("check_invariants", False),
            ("layers", []),
            ("root_bud", None),
            ("identity_factory", None),
            ("terminal", None),
        ):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    setattr(network, name, value)
        network.budget = 3
        network.budget_units = network.budget_units
        network.eta = 0.5
        self.assertEqual(network.eta, 0.5)

    def test_identity_is_opaque_and_transform_kinds_are_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "created by Auxein"):
            CellIdentity(1)
        network = Auxein.from_seed([0.0], memory=10.0, budget=2)
        report = network.step([0.0])
        winner = report.layer_reports[0].winner
        self.assertIsNotNone(winner)
        self.assertEqual(repr(winner), "CellIdentity()")
        with self.assertRaises(ValueError):
            auxein_module.TransformationRecord("unknown", 0, 0.0, 0)

    def test_summary_and_version_names_are_unambiguous(self) -> None:
        network = Auxein.empty(1, memory=10.0, budget=1)
        summary = network.summary()
        self.assertIn("steps_seen", summary)
        self.assertIn("layer_count", summary)
        self.assertNotIn("step", summary)
        self.assertNotIn("layers", summary)
        self.assertEqual(auxein_module.__version__, auxein_module.MODEL_VERSION)



if __name__ == "__main__":
    unittest.main(verbosity=1)
