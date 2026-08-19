from __future__ import annotations

from decimal import Decimal
import math
import unittest

from auxein import Auxein, Kernel, Layer


class KernelTests(unittest.TestCase):
    def test_merge_matches_total_variance(self) -> None:
        a = Kernel(2.0, (0.0,), 1.0)
        b = Kernel(3.0, (4.0,), 2.0)
        merged = a.merged(b)
        self.assertEqual(merged.W, 5.0)
        self.assertAlmostEqual(merged.C[0], 2.4)
        expected = (2 * 1 + 3 * 2) / 5 + (2 * 3 / 25) * 16
        self.assertAlmostEqual(merged.V, expected)

    def test_merge_preserves_internal_variance(self) -> None:
        a = Kernel(0.5, (2.0,), 3.0)
        b = Kernel(0.5, (2.0,), 7.0)
        merged = a.merged(b)
        self.assertEqual(merged.C, (2.0,))
        self.assertEqual(merged.V, 5.0)

    def test_ema_without_target_is_homothetic_forgetting(self) -> None:
        h = Kernel(2.0, (3.0,), 4.0)
        target = Kernel(1.0, (99.0,), 5.0)
        out = h.ema(target, 0.0, 1.0)
        self.assertEqual(out, h)


class CanonTests(unittest.TestCase):
    def make(self, **kwargs: object) -> Auxein:
        defaults = dict(dimension=1, memory=10.0, eta=1.0, scalar="f64", budget=100)
        defaults.update(kwargs)
        return Auxein(**defaults)

    @staticmethod
    def exact_pair_cells() -> list[Kernel]:
        return [Kernel(1.0, (1.0,), 0.0), Kernel(1.0, (3.0,), 0.0)]

    def test_packing(self) -> None:
        n = self.make(budget=0)
        self.assertEqual(n.kernel_units, 24)
        self.assertEqual(n.network_units, 49)
        self.assertEqual(n.min_units, 65)
        self.assertEqual(n.budget_units, 65)
        self.assertEqual(n.maintenance_units(), 65)

    def test_external_presentation_is_uniform_point_kernels(self) -> None:
        n = self.make()
        atoms = n._presentation([[-2.0], [2.0]])
        self.assertEqual([(a.W, a.C, a.V) for a in atoms], [(0.5, (-2.0,), 0.0), (0.5, (2.0,), 0.0)])

    def test_external_duplicate_coalescence(self) -> None:
        n = self.make()
        a = n._presentation([[2.0]])
        b = n._presentation([[2.0], [2.0], [2.0], [2.0]])
        self.assertEqual(a, b)

    def test_first_occurrence_seeds_but_does_not_promote(self) -> None:
        n = self.make()
        report = n.step([[2.0]])
        self.assertEqual(report["readout"], [])
        self.assertEqual(n.summary()["cells_per_layer"], [0])
        self.assertEqual(n.summary()["sigma_per_layer"], [1])
        self.assertEqual(n.layers[0].sigma[0].V, 0.0)

    def test_recurrence_promotes_only_after_later_occurrence(self) -> None:
        n = self.make()
        n.step([[2.0]])
        report = n.step([[2.0]], detailed_report=True)
        self.assertEqual(report["readout"], [])
        self.assertEqual(n.layers[0].cells[0].C, (2.0,))
        self.assertEqual(report["layer_reports"][0]["promoted"], 1)
        third = n.step([[2.0]])
        self.assertEqual(third["readout"], [["auxein", [2.0], [2.0]]])

    def test_zero_is_not_learned_or_emitted(self) -> None:
        n = self.make()
        for _ in range(5):
            report = n.step([[0.0]])
            self.assertEqual(report["readout"], [])
        self.assertEqual(n.summary()["cells_per_layer"], [0])
        self.assertEqual(n.summary()["sigma_per_layer"], [0])

    def test_internal_variance_participates_in_second_concern_bound(self) -> None:
        n = self.make(eta=0.0)
        memory = Kernel(1.0, (2.0,), 1.0)
        norm = 4.0
        close = Kernel(1.0, (2.0,), 0.5)
        broad = Kernel(1.0, (2.0,), 6.0)
        self.assertTrue(n._concern(memory, norm, close, 4.0)[0])
        self.assertFalse(n._concern(memory, norm, broad, 4.0)[0])
        self.assertEqual(n._concern(memory, norm, close, 4.0)[1], 4.0)

    def test_multi_winner_allocation_is_conservative(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(3.0, (2.0,), 10.0)]
        report = n.step([[3.0]], detailed_report=True)
        layer = report["layer_reports"][0]
        masses = layer["cell_responsibility_mass"]
        self.assertEqual(len(report["readout"]), 2)
        self.assertAlmostEqual(math.fsum(masses), 1.0)
        self.assertAlmostEqual(masses[0], 5.0 / 29.0)
        self.assertAlmostEqual(masses[1], 24.0 / 29.0)

    def test_context_geometry_ignores_learning_responsibility(self) -> None:
        a = self.make(eta=0.0)
        b = self.make(eta=0.0)
        a.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(1.0, (2.0,), 10.0)]
        b.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(100.0, (2.0,), 10.0)]
        ra = a.step([[3.0]], detailed_report=True)["layer_reports"][0]
        rb = b.step([[3.0]], detailed_report=True)["layer_reports"][0]
        self.assertNotEqual(ra["cell_responsibility_mass"], rb["cell_responsibility_mass"])
        self.assertEqual(ra["context_center"], rb["context_center"])
        self.assertEqual(ra["context_variance"], rb["context_variance"])
        self.assertEqual(ra["output_mass"], rb["output_mass"])

    def test_context_mass_is_recognised_input_mass(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = self.exact_pair_cells()
        report = n.step([[1.0], [3.0], [-10.0]], detailed_report=True)
        layer = report["layer_reports"][0]
        self.assertAlmostEqual(layer["input_mass"], 1.0)
        self.assertAlmostEqual(layer["output_mass"], 2.0 / 3.0)
        self.assertEqual(layer["context_center"], [2.0])
        self.assertEqual(layer["context_variance"], 1.0)

    def test_per_atom_multi_recognition_does_not_duplicate_mass(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 10.0), Kernel(1.0, (2.0,), 10.0)]
        report = n.step([[3.0]], detailed_report=True)
        layer = report["layer_reports"][0]
        self.assertEqual(layer["context_center"], [1.5])
        self.assertEqual(layer["context_variance"], 0.25)
        self.assertEqual(layer["output_mass"], 1.0)

    def test_single_recognised_value_is_vertical_silence(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (2.0,), 0.0)]
        report = n.step([[2.0]], detailed_report=True)
        layer = report["layer_reports"][0]
        self.assertTrue(report["readout"])
        self.assertEqual(layer["context_center"], [2.0])
        self.assertEqual(layer["context_variance"], 0.0)
        self.assertFalse(layer["context_emitted"])
        self.assertEqual(layer["output_atom_count"], 0)

    def test_zero_center_context_is_vertical_silence(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (-1.0,), 0.0), Kernel(1.0, (1.0,), 0.0)]
        report = n.step([[-1.0], [1.0]], detailed_report=True)
        layer = report["layer_reports"][0]
        self.assertEqual(layer["context_center"], [0.0])
        self.assertEqual(layer["context_variance"], 1.0)
        self.assertFalse(layer["context_emitted"])

    def test_perfect_pair_emits_context_instead_of_zero_differences(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = self.exact_pair_cells()
        report = n.step([[1.0], [3.0]], detailed_report=True)
        layer = report["layer_reports"][0]
        self.assertEqual(layer["context_center"], [2.0])
        self.assertEqual(layer["context_variance"], 1.0)
        self.assertTrue(layer["context_emitted"])
        self.assertEqual(layer["output_mass"], 1.0)

    def test_constant_input_with_two_explanations_does_not_build_deep_cascade(self) -> None:
        n = self.make(budget=1000)
        n.layers[0].cells = [Kernel(1.0, (1.0,), 1.0), Kernel(1.0, (2.0,), 1.0)]
        for _ in range(40):
            n.step([[1.5]])
        self.assertLessEqual(len(n.layers), 2)
        if len(n.layers) == 2:
            self.assertLessEqual(len(n.layers[1].cells), 1)

    def test_context_frontier_and_higher_learning_stop_after_one_context_cell(self) -> None:
        n = self.make(budget=1000)
        n.layers[0].cells = self.exact_pair_cells()
        first = n.step([[1.0], [3.0]], detailed_report=True)
        self.assertEqual(n.summary()["layer_count"], 2)
        self.assertEqual(len(first["layer_reports"]), 1)  # new L1 did not read this step
        self.assertTrue(any(t.get("layer_created") for t in first["transformations"]))

        second = n.step([[1.0], [3.0]], detailed_report=True)
        self.assertEqual(n.summary()["sigma_per_layer"], [0, 1])
        self.assertEqual(second["layer_reports"][1]["context_variance"], None)

        third = n.step([[1.0], [3.0]], detailed_report=True)
        self.assertEqual(n.summary()["cells_per_layer"], [2, 1])
        fourth = n.step([[1.0], [3.0]], detailed_report=True)
        self.assertEqual(n.summary()["layer_count"], 2)
        self.assertFalse(fourth["layer_reports"][1]["context_emitted"])

    def test_context_coalescence_split_invariance(self) -> None:
        a = self.make(eta=0.0)
        b = self.make(eta=0.0)
        a.layers[0].cells = self.exact_pair_cells()
        b.layers[0].cells = self.exact_pair_cells()
        ra = a.step([[1.0], [3.0]], detailed_report=True)["layer_reports"][0]
        rb = b.step([[1.0], [1.0], [3.0], [3.0]], detailed_report=True)["layer_reports"][0]
        for key in ("output_mass", "context_center", "context_variance", "context_emitted"):
            self.assertEqual(ra[key], rb[key])
        self.assertEqual(a.export_state(), b.export_state())

    def test_permutation_invariance(self) -> None:
        a = self.make()
        b = self.make()
        p1 = [[-2.0], [1.0], [4.0], [1.0]]
        p2 = [[1.0], [4.0], [1.0], [-2.0]]
        for _ in range(4):
            a.step(p1)
            b.step(p2)
        self.assertEqual(a.export_state(), b.export_state())

    def test_eta_zero_freezes_state_but_not_context_or_readout(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = self.exact_pair_cells()
        before = n.export_state()
        report = n.step([[1.0], [3.0]], detailed_report=True)
        after = n.export_state()
        self.assertEqual(before["layers"], after["layers"])
        self.assertEqual(after["steps_seen"], before["steps_seen"] + 1)
        self.assertEqual(len(report["readout"]), 2)
        self.assertTrue(report["layer_reports"][0]["context_emitted"])
        self.assertEqual(len(n.layers), 1)  # beta=0 forbids frontier creation

    def test_readout_contract_keeps_only_vectors(self) -> None:
        n = self.make(universe="lab", eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (2.0,), 1.0)]
        rec = n.step([[2.0]])["readout"][0]
        self.assertEqual(rec, ["lab", [2.0], [2.0]])

    def test_growth_transaction_is_all_or_nothing(self) -> None:
        n = self.make(budget=1)
        report = n.step([[-2.0], [2.0]])
        self.assertEqual(n.summary()["sigma_per_layer"], [0])
        self.assertTrue(any(t["type"] == "reject" for t in report["transformations"]))

        m = self.make(budget=2)
        m.step([[-2.0], [2.0]])
        self.assertEqual(m.summary()["sigma_per_layer"], [2])

    def test_frontier_layer_is_in_same_growth_transaction_as_seeds(self) -> None:
        # L0 already knows a context and also sees an unrelated unknown atom.
        n = self.make(budget=1000)
        n.layers[0].cells = self.exact_pair_cells()
        report = n.step([[1.0], [3.0], [10.0]])
        growth = [t for t in report["transformations"] if t["phase"] == "growth"][-1]
        self.assertEqual(growth["type"], "commit")
        self.assertEqual(growth["seeds"], 1)
        self.assertTrue(growth["layer_created"])

    def test_forced_solvency_destroys_work_then_knowledge(self) -> None:
        n = self.make(budget=20)
        for x in (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0):
            n.step([[x]])
        self.assertGreater(sum(n.summary()["cells_per_layer"]), 0)
        n.set_budget(budget=0)
        report = n.step([[0.0]])
        self.assertEqual(n.summary()["layer_count"], 1)
        self.assertEqual(n.summary()["cells_per_layer"], [0])
        self.assertEqual(n.summary()["sigma_per_layer"], [0])
        kinds = {(t["phase"], t["type"]) for t in report["transformations"]}
        self.assertIn(("solvency", "destroy_cells"), kinds)

    def test_cell_value_ignores_support(self) -> None:
        a = Kernel(1e-9, (3.0,), 1.0)
        b = Kernel(1000.0, (3.0,), 1.0)
        self.assertEqual(Auxein.cell_value(a), Auxein.cell_value(b))

    def test_roundtrip(self) -> None:
        n = self.make(universe="roundtrip")
        for x in (-2.0, 2.0, -2.0, 2.0, -2.0):
            n.step([[x]])
        state = n.export_state()
        restored = Auxein.from_state(state, budget_units=n.budget_units, universe=n.universe)
        self.assertEqual(restored.export_state(), state)
        self.assertEqual(restored.summary()["budget_units"], n.budget_units)

    def test_import_rejects_sigma_already_covered_by_cell(self) -> None:
        n = self.make()
        state = n.export_state()
        state["layers"][0]["cells"] = [{"W": 1.0, "C": [2.0], "V": 1.0}]
        state["layers"][0]["sigma"] = [{"W": 1.0, "C": [2.0], "V": 0.0}]
        with self.assertRaises(ValueError):
            Auxein.from_state(state, budget_units=n.budget_units)

    def test_f32_persistent_projection(self) -> None:
        n = self.make(scalar="f32", memory=10.1, eta=0.7)
        state = n.export_state()
        self.assertNotEqual(state["memory"], 10.1)
        restored = Auxein.from_state(state, budget_units=n.budget_units, universe=n.universe)
        self.assertEqual(restored.export_state(), state)

    def test_budget_is_not_serialized(self) -> None:
        n = self.make()
        state = n.export_state()
        self.assertNotIn("budget", state)
        self.assertNotIn("budget_units", state)

    def test_clone_coalescence_uses_geometry_not_identity(self) -> None:
        n = self.make()
        clones = n._coalesce_kernels([Kernel(1.0, (2.0,), 3.0), Kernel(4.0, (2.0,), 3.0)])
        self.assertEqual(len(clones), 1)
        self.assertEqual(clones[0].W, 5.0)
        self.assertEqual(clones[0].C, (2.0,))
        self.assertEqual(clones[0].V, 3.0)


    def test_recognised_center_quotient_ignores_cell_dispersion(self) -> None:
        n = self.make(eta=0.0)
        n.layers[0].cells = [Kernel(1.0, (2.0,), 1.0), Kernel(1.0, (2.0,), 4.0)]
        report = n.step([[2.0]], detailed_report=True)
        layer = report["layer_reports"][0]
        self.assertEqual(len(report["readout"]), 1)
        self.assertEqual(layer["context_center"], [2.0])
        self.assertEqual(layer["context_variance"], 0.0)
        self.assertFalse(layer["context_emitted"])

    def test_internal_context_variance_is_seeded_unchanged_in_next_layer(self) -> None:
        n = self.make(budget=1000)
        n.layers[0].cells = self.exact_pair_cells()
        n.step([[1.0], [3.0]])  # creates empty L1
        n.step([[1.0], [3.0]])  # L1 sees (1,2,1) once
        self.assertEqual(len(n.layers[1].sigma), 1)
        seed = n.layers[1].sigma[0]
        self.assertEqual(seed.C, (2.0,))
        self.assertEqual(seed.V, 1.0)
        self.assertAlmostEqual(seed.W, n.beta)

    def test_scale_invariance_of_learned_geometry(self) -> None:
        a = Auxein(dimension=1, memory=10.0, eta=1.0, scalar="f64", budget=100)
        b = Auxein(dimension=1, memory=10.0, eta=1.0, scalar="f64", budget=100)
        seq = [[[1.0]], [[3.0]], [[1.0]], [[3.0]], [[1.0], [3.0]]] * 4
        for presentation in seq:
            a.step(presentation)
            b.step([[10.0 * x for x in vector] for vector in presentation])
        self.assertEqual(len(a.layers), len(b.layers))
        for la, lb in zip(a.layers, b.layers, strict=True):
            self.assertEqual(len(la.cells), len(lb.cells))
            self.assertEqual(len(la.sigma), len(lb.sigma))
            for ka, kb in zip(la.cells, lb.cells, strict=True):
                self.assertAlmostEqual(ka.W, kb.W)
                self.assertEqual(tuple(10.0 * x for x in ka.C), kb.C)
                self.assertAlmostEqual(100.0 * ka.V, kb.V)
            for ka, kb in zip(la.sigma, lb.sigma, strict=True):
                self.assertAlmostEqual(ka.W, kb.W)
                self.assertEqual(tuple(10.0 * x for x in ka.C), kb.C)
                self.assertAlmostEqual(100.0 * ka.V, kb.V)

    def test_exact_rotation_invariance(self) -> None:
        a = Auxein(dimension=2, memory=10.0, eta=1.0, scalar="f64", budget=200)
        b = Auxein(dimension=2, memory=10.0, eta=1.0, scalar="f64", budget=200)
        def rot(v: list[float]) -> list[float]:
            return [-v[1], v[0]]
        seq = [
            [[2.0, 0.0]], [[0.0, 3.0]], [[2.0, 0.0]], [[0.0, 3.0]],
            [[2.0, 0.0], [0.0, 3.0]],
        ] * 4
        for presentation in seq:
            a.step(presentation)
            b.step([rot(v) for v in presentation])
        self.assertEqual(len(a.layers), len(b.layers))
        for la, lb in zip(a.layers, b.layers, strict=True):
            ga = sorted((rot(list(k.C)), k.V, k.W) for k in la.cells)
            gb = sorted((list(k.C), k.V, k.W) for k in lb.cells)
            self.assertEqual(ga, gb)

    def test_zero_padding_invariance(self) -> None:
        a = Auxein(dimension=1, memory=10.0, eta=1.0, scalar="f64", budget=200)
        b = Auxein(dimension=3, memory=10.0, eta=1.0, scalar="f64", budget=200)
        seq = [[[1.0]], [[3.0]], [[1.0]], [[3.0]], [[1.0], [3.0]]] * 4
        for presentation in seq:
            a.step(presentation)
            b.step([[v[0], 0.0, 0.0] for v in presentation])
        self.assertEqual(len(a.layers), len(b.layers))
        for la, lb in zip(a.layers, b.layers, strict=True):
            self.assertEqual(len(la.cells), len(lb.cells))
            for ka, kb in zip(la.cells, lb.cells, strict=True):
                self.assertEqual(kb.C, (ka.C[0], 0.0, 0.0))
                self.assertEqual(kb.V, ka.V)
                self.assertEqual(kb.W, ka.W)

    def test_external_input_is_strictly_vectors(self) -> None:
        n = self.make()
        with self.assertRaises((TypeError, ValueError)):
            n.step([2.0])
        with self.assertRaises((TypeError, ValueError)):
            n.step([([2.0], 1.0)])
        with self.assertRaises((TypeError, ValueError)):
            n.step([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
