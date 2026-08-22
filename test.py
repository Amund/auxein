"""Canonical regression tests for the Auxein v0.5.0 reference implementation."""

from __future__ import annotations

import math
import unittest

from auxein import Auxein, FORMAT_VERSION, Kernel, MODES


class KernelTests(unittest.TestCase):
    def test_merge_matches_total_variance(self) -> None:
        a = Kernel(1.0, (0.0,), 1.0)
        b = Kernel(3.0, (4.0,), 9.0)
        m = a.merged(b)
        self.assertEqual(m.W, 4.0)
        self.assertEqual(m.C, (3.0,))
        self.assertEqual(m.V, 10.0)

    def test_merge_preserves_internal_variance(self) -> None:
        a = Kernel(1.0, (2.0,), 7.0)
        b = Kernel(1.0, (2.0,), 7.0)
        self.assertEqual(a.merged(b), Kernel(2.0, (2.0,), 7.0))

    def test_ema_without_target_is_homothetic_forgetting(self) -> None:
        k = Kernel(3.0, (2.0,), 5.0)
        self.assertEqual(k.ema(Kernel(9.0, (99.0,), 8.0), 0.0, 1.0), k)

    def test_ema_uses_target_variance(self) -> None:
        old = Kernel(2.0, (0.0,), 1.0)
        target = Kernel(1.0, (2.0,), 3.0)
        got = old.ema(target, 0.5, 0.5)
        self.assertGreater(got.V, 0.0)
        self.assertGreater(got.W, 0.0)


class CanonTests(unittest.TestCase):
    def make(self, **kwargs: object) -> Auxein:
        defaults: dict[str, object] = dict(
            dimension=1, memory=10.0, eta=1.0, scalar="f64", budget=100
        )
        defaults.update(kwargs)
        return Auxein(**defaults)

    @staticmethod
    def exact_pair_cells() -> list[Kernel]:
        return [Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (3.0,), 0.0)]

    @staticmethod
    def present(report: dict[str, object]) -> list[object]:
        readout = report["readout"]
        assert isinstance(readout, dict)
        present = readout["present"]
        assert isinstance(present, list)
        return present

    @staticmethod
    def future(report: dict[str, object]) -> list[object]:
        readout = report["readout"]
        assert isinstance(readout, dict)
        future = readout.get("future", [])
        assert isinstance(future, list)
        return future

    def test_format_and_modes(self) -> None:
        self.assertEqual(FORMAT_VERSION, 5)
        self.assertEqual(MODES, {"geometry", "predictive"})
        self.assertEqual(self.make().mode, "geometry")
        self.assertEqual(self.make(mode="predictive").mode, "predictive")
        with self.assertRaises(ValueError):
            self.make(mode="temporal")

    def test_packing(self) -> None:
        g = self.make(budget=0)
        self.assertEqual(g.kernel_units, 24)
        self.assertEqual(g.network_units, 50)
        self.assertEqual(g.layer_units, 16)
        self.assertEqual(g.min_units, 66)
        p = self.make(mode="predictive", budget=0)
        self.assertEqual(p.temporal_kernel_units, 32)
        self.assertEqual(p.layer_units, 57)
        self.assertEqual(p.min_units, 107)

    def test_vector_sugar_is_uniform_point_presentation(self) -> None:
        n = self.make()
        atoms = n._presentation([[-2.0], [2.0]])
        self.assertEqual(
            [(a.W, a.C, a.V) for a in atoms],
            [(0.5, (-2.0,), 0.0), (0.5, (2.0,), 0.0)],
        )

    def test_weighted_kernel_boundary_is_canonical(self) -> None:
        n = self.make()
        atoms = n._presentation([
            [0.25, [1.0], 0.5],
            [0.25, [1.0], 0.5],
            [0.25, [2.0], 0.0],
            [0.25, [0.0], 0.0],
        ])
        self.assertEqual(
            atoms,
            [
                Kernel(0.25, (0.0,), 0.0),
                Kernel(0.5, (1.0,), 0.5),
                Kernel(0.25, (2.0,), 0.0),
            ],
        )

    def test_weighted_boundary_rejects_invalid_mass(self) -> None:
        n = self.make()
        with self.assertRaises(ValueError):
            n.step([[0.6, [1.0], 0.0], [0.6, [2.0], 0.0]])
        with self.assertRaises(ValueError):
            n.step([])

    def test_zero_atom_is_causally_present_but_cognitively_silent(self) -> None:
        n = self.make()
        for _ in range(4):
            report = n.step([[1.0, [0.0], 0.0]])
            self.assertEqual(self.present(report), [])
        self.assertEqual(n.layers[0].cells, [])
        self.assertEqual(n.layers[0].sigma, [])

    def test_first_occurrence_seeds_and_later_recurrence_promotes(self) -> None:
        n = self.make()
        self.assertEqual(self.present(n.step([[2.0]])), [])
        self.assertEqual(len(n.layers[0].sigma), 1)
        self.assertEqual(self.present(n.step([[2.0]])), [])
        self.assertEqual(len(n.layers[0].cells), 1)
        third = n.step([[2.0]])
        self.assertEqual(self.present(third), [[[1.0, [2.0], 0.0]]])

    def test_gain_weighted_context(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(1.0, (2.0,), 10.0)]
        layer = n.step([[3.0]], detailed_report=True)["layer_reports"][0]
        self.assertAlmostEqual(layer["context_center"][0], 21.0 / 13.0)
        self.assertAlmostEqual(layer["context_variance"], 0.23668639053254437)
        self.assertEqual(layer["output_mass"], 1.0)

    def test_gain_context_ignores_learning_support(self) -> None:
        a = self.make(eta=0.0)
        b = self.make(eta=0.0)
        a.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(1.0, (2.0,), 10.0)]
        b.layers[0].cells = [Kernel(1e-200, (1.0,), 10.0), Kernel(1e200, (2.0,), 10.0)]
        ra = a.step([[3.0]], detailed_report=True)["layer_reports"][0]
        rb = b.step([[3.0]], detailed_report=True)["layer_reports"][0]
        self.assertNotEqual(ra["cell_responsibility_mass"], rb["cell_responsibility_mass"])
        self.assertEqual(ra["context_center"], rb["context_center"])
        self.assertEqual(ra["context_variance"], rb["context_variance"])

    def test_recognised_mass_is_conserved_per_atom(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(1.0, (2.0,), 10.0)]
        layer = n.step([[0.4, [3.0], 0.0]], detailed_report=True)["layer_reports"][0]
        self.assertAlmostEqual(layer["knowledge_mass"], 0.4)
        self.assertAlmostEqual(layer["output_mass"], 0.4)

    def test_output_is_completed_with_zero_remainder(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(9.0, (2.0,), 77.0)]
        report = n.step([[0.25, [2.0], 0.0], [0.75, [99.0], 0.0]])
        self.assertEqual(
            self.present(report),
            [[[0.75, [0.0], 0.0], [0.25, [2.0], 0.0]]],
        )

    def test_output_uses_cell_center_but_not_cell_variance(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(123.0, (2.0,), 999.0)]
        self.assertEqual(self.present(n.step([[2.0]])), [[[1.0, [2.0], 0.0]]])

    def test_empty_knowledge_is_empty_family_not_zero_presentation(self) -> None:
        n = self.make(eta=0.0)
        self.assertEqual(self.present(n.step([[99.0]])), [])

    def test_single_recognised_position_is_vertical_silence(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (2.0,), 0.0)]
        layer = n.step([[2.0]], detailed_report=True)["layer_reports"][0]
        self.assertEqual(layer["context_variance"], 0.0)
        self.assertFalse(layer["context_emitted"])
        self.assertEqual(layer["output_atom_count"], 0)

    def test_symmetric_multiatom_context_can_be_zero_center(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (-1.0,), 0.0), Kernel(1.0, (1.0,), 0.0)]
        layer = n.step([[-1.0], [1.0]], detailed_report=True)["layer_reports"][0]
        self.assertEqual(layer["context_center"], [0.0])
        self.assertEqual(layer["context_variance"], 1.0)
        self.assertFalse(layer["context_emitted"])

    def test_vertical_context_can_create_next_layer(self) -> None:
        n = self.make(budget=1000)
        n.layers[0].cells = self.exact_pair_cells()
        n.step([[1.0], [3.0]])
        self.assertEqual(len(n.layers), 2)

    def test_split_and_permutation_invariance(self) -> None:
        a = self.make()
        b = self.make()
        p = [[-2.0], [1.0], [4.0], [1.0]]
        q = [[1.0], [4.0], [1.0], [-2.0]]
        for _ in range(5):
            self.assertEqual(a.step(p), b.step(q))
        self.assertEqual(a.export_state(), b.export_state())

    def test_eta_zero_freezes_learned_state(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = self.exact_pair_cells()
        before = n.export_state()
        report = n.step([[1.0], [3.0]])
        self.assertTrue(self.present(report))
        after = n.export_state()
        before["steps_seen"] = after["steps_seen"]
        self.assertEqual(after, before)

    def test_step_is_atomic_and_never_learns_cross_call_transition(self) -> None:
        n = self.make(mode="predictive", budget=1000)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (5.0,), 0.0)]
        n.step([[1.0]])
        n.step([[5.0]])
        self.assertEqual(n.layers[0].temporal_sigma, [])
        self.assertEqual(n.layers[0].temporal_cells, [])
        self.assertIsNone(n.layers[0].previous)

    def test_non_atomic_sequence_learns_adjacent_transition(self) -> None:
        n = self.make(mode="predictive", budget=1000)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (5.0,), 0.0)]
        n.sequence([[[1.0]], [[5.0]]])
        self.assertEqual(len(n.layers[0].temporal_sigma), 1)
        self.assertEqual(n.layers[0].temporal_sigma[0].C, (1.0, 5.0))
        self.assertIsNone(n.layers[0].previous)

    def test_sequence_boundary_prevents_cross_sequence_transition(self) -> None:
        n = self.make(mode="predictive", budget=1000)
        n.layers[0].cells = [
            Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (10.0,), 0.0),
            Kernel(1.0, (-1.0,), 0.0), Kernel(1.0, (-10.0,), 0.0),
        ]
        n.sequence([[[1.0]], [[10.0]]])
        n.sequence([[[-1.0]], [[-10.0]]])
        centers = {k.C for k in n.layers[0].temporal_sigma}
        self.assertIn((1.0, 10.0), centers)
        self.assertIn((-1.0, -10.0), centers)
        self.assertNotIn((10.0, -1.0), centers)

    def test_explicit_multicall_sequence_api(self) -> None:
        n = self.make(mode="predictive", budget=1000)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (2.0,), 0.0)]
        n.begin_sequence()
        n.sequence_step([[1.0]])
        self.assertEqual(n.layers[0].previous.C, (1.0,))
        n.sequence_step([[2.0]])
        n.end_sequence()
        self.assertIsNone(n.layers[0].previous)
        self.assertEqual(n.layers[0].temporal_sigma[0].C, (1.0, 2.0))

    def test_singleton_can_predict_but_cannot_leave_previous(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0)]
        n.layers[0].temporal_cells = [Kernel(1.0, (1.0, 2.0), 0.0)]
        report = n.step([[1.0]])
        self.assertEqual(self.future(report), [[[1.0, [2.0], 0.0]]])
        self.assertIsNone(n.layers[0].previous)

    def test_prediction_preserves_current_context_mass(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (2.0,), 0.0)]
        n.layers[0].temporal_cells = [Kernel(1.0, (2.0, 7.0), 0.0)]
        report = n.step([[0.25, [2.0], 0.0], [0.75, [99.0], 0.0]])
        self.assertEqual(
            self.future(report),
            [[[0.75, [0.0], 0.0], [0.25, [7.0], 0.0]]],
        )

    def test_predictive_branching_emits_independent_candidates(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0)]
        n.layers[0].temporal_cells = [
            Kernel(1.0, (1.0, 2.0), 0.0),
            Kernel(999.0, (1.0, 3.0), 77.0),
        ]
        self.assertEqual(
            self.future(n.step([[1.0]])),
            [[[1.0, [2.0], 0.0]], [[1.0, [3.0], 0.0]]],
        )

    def test_predictive_relative_gain_weights_each_branch_independently(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (2.0,), 0.0)]
        n.layers[0].temporal_cells = [
            Kernel(1.0, (2.0, 7.0), 0.0),
            Kernel(999.0, (3.9, 8.0), 77.0),
        ]
        report = n.step([[0.25, [2.0], 0.0], [0.75, [99.0], 0.0]])
        self.assertEqual(
            self.future(report),
            [
                [[0.75, [0.0], 0.0], [0.25, [7.0], 0.0]],
                [[0.975625, [0.0], 0.0], [0.25 * (1.0 - (1.9 * 1.9) / 4.0), [8.0], 0.0]],
            ],
        )

    def test_predictive_same_target_uses_max_relative_gain(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (2.0,), 0.0)]
        n.layers[0].temporal_cells = [
            Kernel(1.0, (3.9, 7.0), 0.0),
            Kernel(1.0, (2.0, 7.0), 0.0),
            Kernel(1.0, (3.9, 7.0), 0.0),
        ]
        report = n.step([[0.25, [2.0], 0.0], [0.75, [99.0], 0.0]])
        self.assertEqual(
            self.future(report),
            [[[0.75, [0.0], 0.0], [0.25, [7.0], 0.0]]],
        )

    def test_predictive_relative_gain_is_scale_invariant_at_extremes(self) -> None:
        expected = 0.0975
        for scale in (1.0, 1e-158, 1e-200, 1e158, 1e200):
            gamma = Auxein._point_relative_gain((2.0 * scale,), (3.9 * scale,))
            self.assertIsNotNone(gamma)
            assert gamma is not None
            self.assertAlmostEqual(gamma, expected, places=14)

    def test_predictive_zero_target_is_explicit_candidate(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0)]
        n.layers[0].temporal_cells = [Kernel(1.0, (1.0, 0.0), 0.0)]
        self.assertEqual(self.future(n.step([[1.0]])), [[[1.0, [0.0], 0.0]]])

    def test_prediction_is_not_reinjected_recursively(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0)]
        n.layers[0].temporal_cells = [
            Kernel(1.0, (1.0, 2.0), 0.0), Kernel(1.0, (2.0, 3.0), 0.0)
        ]
        self.assertEqual(self.future(n.step([[1.0]])), [[[1.0, [2.0], 0.0]]])

    def test_temporal_product_kernel_is_direct_sum_quotient(self) -> None:
        n = self.make(mode="predictive")
        temporal = n._temporal_atom(Kernel(0.5, (2.0,), 1.0), Kernel(0.25, (7.0,), 4.0))
        self.assertEqual(temporal, Kernel(0.125, (2.0, 7.0), 5.0))

    def test_temporal_population_does_not_age_without_temporal_presentation(self) -> None:
        n = self.make(mode="predictive", eta=1.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0)]
        n.layers[0].temporal_cells = [Kernel(2.0, (1.0, 1.0), 0.5)]
        before = n.layers[0].temporal_cells[0].copy()
        n.sequence([[[99.0]], [[99.0]]])
        self.assertEqual(n.layers[0].temporal_cells[0], before)

    def test_eta_zero_advances_previous_inside_sequence_only(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (3.0,), 0.0)]
        n.begin_sequence()
        n.sequence_step([[1.0]])
        self.assertEqual(n.layers[0].previous.C, (1.0,))
        n.sequence_step([[3.0]])
        self.assertEqual(n.layers[0].previous.C, (3.0,))
        n.end_sequence()
        self.assertIsNone(n.layers[0].previous)
        self.assertEqual(n.layers[0].temporal_sigma, [])

    def test_mid_sequence_state_requires_explicit_resume(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (3.0,), 0.0)]
        n.begin_sequence()
        n.sequence_step([[1.0]])
        state = n.export_state()
        n.end_sequence()
        restored = Auxein.from_state(state, budget=100)
        self.assertIsNotNone(restored.layers[0].previous)
        restored.begin_sequence(resume=True)
        restored.sequence_step([[3.0]])
        self.assertEqual(restored.layers[0].previous.C, (3.0,))
        restored.end_sequence()
        self.assertIsNone(restored.layers[0].previous)

    def test_from_state_step_does_not_continue_saved_previous_implicitly(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (3.0,), 0.0)]
        n.layers[0].previous = Kernel(1.0, (1.0,), 0.0)
        state = n.export_state()
        restored = Auxein.from_state(state, budget=100)
        restored.step([[3.0]])
        self.assertIsNone(restored.layers[0].previous)

    def test_roundtrip_format5_predictive_state(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0)]
        n.layers[0].temporal_cells = [Kernel(1.0, (1.0, 2.0), 0.0)]
        state = n.export_state()
        self.assertEqual(state["format_version"], 5)
        restored = Auxein.from_state(state, budget=100)
        self.assertEqual(restored.export_state(), state)
        self.assertEqual(self.future(restored.step([[1.0]])), [[[1.0, [2.0], 0.0]]])

    def test_budget_is_not_serialized(self) -> None:
        n = self.make(budget=123)
        self.assertNotIn("budget", n.export_state())
        self.assertNotIn("budget_units", n.export_state())

    def test_forced_solvency_invalidates_previous(self) -> None:
        n = self.make(mode="predictive", budget=1000, eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0)]
        n.layers[0].temporal_cells = [Kernel(1.0, (1.0, 1.0), 0.0)]
        n.begin_sequence()
        n.sequence_step([[1.0]])
        self.assertIsNotNone(n.layers[0].previous)
        n.set_budget(budget_units=n.min_units)
        n.sequence_step([[1.0]])
        self.assertEqual(n.layers[0].cells, [])
        self.assertEqual(n.layers[0].temporal_cells, [])
        self.assertIsNone(n.layers[0].previous)
        n.end_sequence()

    def test_geometry_and_predictive_share_geometric_trajectory(self) -> None:
        g = self.make(mode="geometry", budget=10000)
        p = self.make(mode="predictive", budget=10000)
        sequence = [[[1.0]], [[5.0]], [[1.0]], [[5.0]], [[1.0], [5.0]]] * 4
        for presentation in sequence:
            g.step(presentation)
        p.sequence(sequence)
        self.assertEqual(len(g.layers), len(p.layers))
        for gl, pl in zip(g.layers, p.layers, strict=True):
            self.assertEqual(gl.cells, pl.cells)
            self.assertEqual(gl.sigma, pl.sigma)

    def test_scale_invariance_of_gain_weighted_geometry(self) -> None:
        a = self.make(eta=0.0)
        b = self.make(eta=0.0)
        a.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(1.0, (2.0,), 10.0)]
        b.layers[0].cells = [Kernel(1.0, (10.0,), 1000.0), Kernel(1.0, (20.0,), 1000.0)]
        ra = a.step([[3.0]], detailed_report=True)["layer_reports"][0]
        rb = b.step([[30.0]], detailed_report=True)["layer_reports"][0]
        self.assertAlmostEqual(10.0 * ra["context_center"][0], rb["context_center"][0])
        self.assertAlmostEqual(100.0 * ra["context_variance"], rb["context_variance"])

    def test_composable_output_can_be_reinjected_as_weighted_input(self) -> None:
        up = self.make(eta=0.0)
        up.layers[0].cells = self.exact_pair_cells()
        out = self.present(up.step([[1.0], [3.0]]))
        self.assertEqual(len(out), 1)
        down = self.make(eta=0.0)
        parsed = down._presentation(out[0])
        self.assertAlmostEqual(math.fsum(k.W for k in parsed), 1.0)
        self.assertTrue(all(len(k.C) == 1 for k in parsed))

    def test_direct_composition_as_atomic_singletons_has_no_temporal_authority(self) -> None:
        up = self.make(eta=0.0)
        up.layers[0].cells = self.exact_pair_cells()
        up.layers.append(type(up.layers[0])([], [Kernel(1.0, (2.0,), 1.0)]))
        family = self.present(up.step([[1.0], [3.0]]))
        down = self.make(mode="predictive", budget=1000)
        down.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(1.0, (2.0,), 10.0), Kernel(1.0, (3.0,), 10.0)]
        for presentation in family:
            down.step(presentation)
        self.assertEqual(down.layers[0].temporal_sigma, [])
        self.assertIsNone(down.layers[0].previous)

    def test_prediction_family_deduplicates_identical_candidates(self) -> None:
        n = self.make(mode="predictive", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 0.0)]
        n.layers[0].temporal_cells = [
            Kernel(1.0, (1.0, 2.0), 0.0),
            Kernel(2.0, (1.1, 2.0), 10.0),
        ]
        self.assertEqual(self.future(n.step([[1.0]])), [[[1.0, [2.0], 0.0]]])

    def test_sequence_counter_counts_presentations_not_sequences(self) -> None:
        n = self.make(mode="predictive")
        n.sequence([[[1.0]], [[2.0]], [[3.0]]])
        n.step([[4.0]])
        self.assertEqual(n.steps_seen, 4)

    def test_sequence_api_rejects_implicit_misuse(self) -> None:
        n = self.make(mode="predictive")
        with self.assertRaises(ValueError):
            n.sequence([])
        n.begin_sequence()
        with self.assertRaises(RuntimeError):
            n.step([[1.0]])
        with self.assertRaises(RuntimeError):
            n.begin_sequence()
        n.end_sequence()


if __name__ == "__main__":
    unittest.main(verbosity=2)
