import Auxein.Topology

namespace Auxein

/-- Network-visible summary of one materialized layer. The network needs only
capital and integer maintenance, never individual Cell state. -/
structure LayerSummary where
  capital : ℝ
  maintenance : ℕ

namespace LayerSummary

/-- Total realized capital exposed by a finite hierarchy. -/
def totalCapital : List LayerSummary → ℝ
  | [] => 0
  | layer :: layers => layer.capital + totalCapital layers

/-- Total layer maintenance, excluding persistent network/root overhead. -/
def totalMaintenance : List LayerSummary → ℕ
  | [] => 0
  | layer :: layers => layer.maintenance + totalMaintenance layers

@[simp] theorem totalCapital_append (xs ys : List LayerSummary) :
    totalCapital (xs ++ ys) = totalCapital xs + totalCapital ys := by
  induction xs with
  | nil => simp [totalCapital]
  | cons x xs ih => simp [totalCapital, ih, add_assoc]

@[simp] theorem totalMaintenance_append (xs ys : List LayerSummary) :
    totalMaintenance (xs ++ ys) = totalMaintenance xs + totalMaintenance ys := by
  induction xs with
  | nil => simp [totalMaintenance]
  | cons x xs ih => simp [totalMaintenance, ih, Nat.add_assoc]

/-- Capital decomposes exactly into a kept prefix and a destroyed suffix. -/
theorem totalCapital_take_add_drop (n : ℕ) (layers : List LayerSummary) :
    totalCapital (layers.take n) + totalCapital (layers.drop n) =
      totalCapital layers := by
  calc
    totalCapital (layers.take n) + totalCapital (layers.drop n) =
        totalCapital (layers.take n ++ layers.drop n) := by
          rw [totalCapital_append]
    _ = totalCapital layers := by rw [List.take_append_drop]

/-- Integer maintenance has the same exact prefix/suffix decomposition. -/
theorem totalMaintenance_take_add_drop (n : ℕ) (layers : List LayerSummary) :
    totalMaintenance (layers.take n) + totalMaintenance (layers.drop n) =
      totalMaintenance layers := by
  calc
    totalMaintenance (layers.take n) + totalMaintenance (layers.drop n) =
        totalMaintenance (layers.take n ++ layers.drop n) := by
          rw [totalMaintenance_append]
    _ = totalMaintenance layers := by rw [List.take_append_drop]

end LayerSummary

/-- Finite network state visible to the global economic jurisdiction. -/
structure NetworkSummary where
  baseMaintenance : ℕ
  layers : List LayerSummary

namespace NetworkSummary

/-- Exact persistent footprint in integer maintenance units. -/
def maintenance (state : NetworkSummary) : ℕ :=
  state.baseMaintenance + LayerSummary.totalMaintenance state.layers

/-- Additive realized geometric capital exposed by all materialized layers. -/
def capital (state : NetworkSummary) : ℝ :=
  LayerSummary.totalCapital state.layers

/-- Constitutional solvency predicate from specification §16.1. -/
def Solvent (state : NetworkSummary) (budget : ℕ) : Prop :=
  maintenance state ≤ budget

/-- Suffix truncation keeps exactly the layers strictly below `start`. -/
def truncate (state : NetworkSummary) (start : ℕ) : NetworkSummary where
  baseMaintenance := state.baseMaintenance
  layers := state.layers.take start

/-- Geometric loss exposed for suffix truncation at `start`. -/
def suffixLoss (state : NetworkSummary) (start : ℕ) : ℝ :=
  LayerSummary.totalCapital (state.layers.drop start)

/-- Integer maintenance released by suffix truncation at `start`. -/
def suffixRelease (state : NetworkSummary) (start : ℕ) : ℕ :=
  LayerSummary.totalMaintenance (state.layers.drop start)

/-- Truncation loses exactly the capital of the removed suffix. -/
theorem capital_truncate_add_suffixLoss
    (state : NetworkSummary) (start : ℕ) :
    capital (truncate state start) + suffixLoss state start = capital state := by
  exact LayerSummary.totalCapital_take_add_drop start state.layers

/-- Truncation releases exactly the maintenance of the removed suffix. -/
theorem maintenance_truncate_add_suffixRelease
    (state : NetworkSummary) (start : ℕ) :
    maintenance (truncate state start) + suffixRelease state start =
      maintenance state := by
  change
    (state.baseMaintenance +
        LayerSummary.totalMaintenance (state.layers.take start)) +
        LayerSummary.totalMaintenance (state.layers.drop start) =
      state.baseMaintenance + LayerSummary.totalMaintenance state.layers
  rw [Nat.add_assoc,
    LayerSummary.totalMaintenance_take_add_drop start state.layers]

/-- Suffix truncation can never increase persistent maintenance. -/
theorem maintenance_truncate_le
    (state : NetworkSummary) (start : ℕ) :
    maintenance (truncate state start) ≤ maintenance state := by
  have h := maintenance_truncate_add_suffixRelease state start
  omega

/-- A positive-release truncation strictly decreases maintenance. -/
theorem maintenance_truncate_lt
    (state : NetworkSummary) (start : ℕ)
    (hrelease : 0 < suffixRelease state start) :
    maintenance (truncate state start) < maintenance state := by
  have h := maintenance_truncate_add_suffixRelease state start
  omega

/-- Truncating from level zero destroys the whole hierarchy but preserves the
persistent network/root footprint. -/
@[simp] theorem maintenance_truncate_zero (state : NetworkSummary) :
    maintenance (truncate state 0) = state.baseMaintenance := by
  simp [maintenance, truncate, LayerSummary.totalMaintenance]

/-- The empty-hierarchy fallback is solvent whenever the irreducible persistent
footprint fits the budget. -/
theorem truncate_zero_solvent
    (state : NetworkSummary) {budget : ℕ}
    (hbase : state.baseMaintenance ≤ budget) :
    Solvent (truncate state 0) budget := by
  simpa [Solvent] using hbase

/-- A creation is payable exactly when its post-creation footprint fits. -/
def Payable (state : NetworkSummary) (delta budget : ℕ) : Prop :=
  maintenance state + delta ≤ budget

/-- Executing a payable creation preserves global solvency at its predicted
integer maintenance delta. -/
theorem payable_post_state_solvent
    (state post : NetworkSummary) (delta budget : ℕ)
    (hdelta : maintenance post = maintenance state + delta)
    (hpayable : Payable state delta budget) :
    Solvent post budget := by
  unfold Payable at hpayable
  unfold Solvent
  rw [hdelta]
  exact hpayable

end NetworkSummary

/-- Maintenance-only relation followed by every forced destruction. -/
def MaintenanceDrop (next current : ℕ) : Prop := next < current

/-- Forced solvency restoration terminates whenever every executed destruction
strictly lowers integer maintenance: the step relation is well founded. -/
theorem maintenanceDrop_wellFounded : WellFounded MaintenanceDrop := by
  apply WellFounded.intro
  intro current
  refine Nat.strong_induction_on current ?_
  intro n ih
  apply Acc.intro
  intro next hdrop
  exact ih next (show next < n from hdrop)

/-- Applying one positive release that does not exceed current maintenance
strictly decreases the footprint. -/
theorem sub_positive_release_lt
    {current release : ℕ} (hrelease : 0 < release)
    (hle : release ≤ current) :
    current - release < current := by
  omega

/-- Sequential dry releases, used to state a finite restoration certificate. -/
def applyReleases : ℕ → List ℕ → ℕ
  | current, [] => current
  | current, release :: releases =>
      applyReleases (current - release) releases

/-- A release schedule is exactly subtraction by its total released footprint. -/
theorem applyReleases_eq_sub_sum (current : ℕ) (releases : List ℕ) :
    applyReleases current releases = current - releases.sum := by
  induction releases generalizing current with
  | nil => simp [applyReleases]
  | cons release releases ih =>
      simp [applyReleases, ih, Nat.sub_sub]

/-- Releases never increase maintenance. -/
theorem applyReleases_le (current : ℕ) (releases : List ℕ) :
    applyReleases current releases ≤ current := by
  rw [applyReleases_eq_sub_sum]
  exact Nat.sub_le current releases.sum

/-- Any finite destruction certificate releasing at least the current budget
excess restores solvency. -/
theorem applyReleases_restores
    {current budget : ℕ} (releases : List ℕ)
    (henough : current - budget ≤ releases.sum) :
    applyReleases current releases ≤ budget := by
  rw [applyReleases_eq_sub_sum]
  omega

end Auxein
