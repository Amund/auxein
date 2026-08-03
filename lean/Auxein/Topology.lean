import Auxein.Admissibility
import Auxein.Split

namespace Auxein

noncomputable section

open scoped RealInnerProductSpace

namespace Kernel

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Exact mean of a positive-mass kernel. The zero-mass convention is handled
by callers and by `Valid`; no geometric claim is made when `W = 0`. -/
def mean (H : Kernel E) : E := H.W⁻¹ • H.S

/-- The latent split value carried by two ordered branch kernels. -/
def splitGain (Hplus Hminus : Kernel E) : ℝ :=
  structuralPower (Kernel.add Hplus Hminus) -
    (structuralPower Hplus + structuralPower Hminus)

/-- Multiplying a kernel mean by its nonzero mass recovers its first moment. -/
@[simp] theorem mass_smul_mean (H : Kernel E) (hW : H.W ≠ 0) :
    H.W • mean H = H.S := by
  simp [mean, smul_smul, hW]

/-- Kernel form of the exact split-gain identity from specification §8.1. -/
theorem splitGain_eq_weighted (Hplus Hminus : Kernel E)
    (hplus : Hplus.W ≠ 0) (hminus : Hminus.W ≠ 0)
    (hsum : Hplus.W + Hminus.W ≠ 0) :
    splitGain Hplus Hminus =
      (Hplus.W * Hminus.W / (Hplus.W + Hminus.W)) *
        sqNorm (mean Hplus - mean Hminus) := by
  have h := split_gain_identity
    Hplus.W Hminus.W Hplus.Q Hminus.Q
    (mean Hplus) (mean Hminus) hplus hminus hsum
  rw [mass_smul_mean Hplus hplus, mass_smul_mean Hminus hminus] at h
  simpa [splitGain, structuralPower, movePower, Kernel.add] using h

/-- Positive branch masses make the exact split value nonnegative. -/
theorem splitGain_nonnegative (Hplus Hminus : Kernel E)
    (hplus : 0 < Hplus.W) (hminus : 0 < Hminus.W) :
    0 ≤ splitGain Hplus Hminus := by
  have hsumPos : 0 < Hplus.W + Hminus.W := add_pos hplus hminus
  rw [splitGain_eq_weighted Hplus Hminus
    (ne_of_gt hplus) (ne_of_gt hminus) (ne_of_gt hsumPos)]
  exact weighted_two_variance_nonnegative
    (le_of_lt hplus) (le_of_lt hminus) hsumPos
    (mean Hplus) (mean Hminus)

/-- A neutral split cannot carry an inherited request for another mitosis. -/
theorem splitGain_half_self (H : Kernel E) (hW : H.W ≠ 0) :
    splitGain (Kernel.half H) (Kernel.half H) = 0 := by
  have hhalf : (Kernel.half H).W ≠ 0 := by
    dsimp [Kernel.half]
    exact div_ne_zero hW (by norm_num)
  have hsum : (Kernel.half H).W + (Kernel.half H).W ≠ 0 := by
    dsimp [Kernel.half]
    intro h
    apply hW
    linarith
  rw [splitGain_eq_weighted (Kernel.half H) (Kernel.half H)
    hhalf hhalf hsum]
  simp

/-- A branch kernel transferred to a child is recentered on its own mean. -/
def materializedBranch (H : Kernel E) : Kernel E :=
  Kernel.recenter H (mean H)

@[simp] theorem materializedBranch_W (H : Kernel E) :
    (materializedBranch H).W = H.W := rfl

/-- Materialization closes the child's first moment exactly. -/
@[simp] theorem materializedBranch_S (H : Kernel E) (hW : H.W ≠ 0) :
    (materializedBranch H).S = 0 := by
  exact recenter_on_mean_S H hW

/-- The transferred child second moment is exactly the branch structural power. -/
@[simp] theorem materializedBranch_Q (H : Kernel E) (hW : H.W ≠ 0) :
    (materializedBranch H).Q = structuralPower H := by
  exact recenter_on_mean_Q H hW

/-- Admissibility survives transfer of a branch to its materialized child. -/
theorem Valid.materializedBranch {H : Kernel E} (hH : Valid H) :
    Valid (materializedBranch H) := by
  exact hH.recenter (mean H)

/-- The child's fresh latent split reconstructs its inherited parent kernel. -/
@[simp] theorem neutral_child_reconstructs (H : Kernel E) :
    Kernel.add (Kernel.half (materializedBranch H))
      (Kernel.half (materializedBranch H)) = materializedBranch H := by
  exact add_half_self (materializedBranch H)

/-- Both fresh latent branches of a valid materialized child are valid. -/
theorem valid_neutral_child {H : Kernel E} (hH : Valid H) :
    Valid (Kernel.half (materializedBranch H)) := by
  exact Valid.half (Valid.materializedBranch hH)

/-- Materializing both branches preserves their total mass. -/
@[simp] theorem materialized_branches_mass (Hplus Hminus : Kernel E) :
    (materializedBranch Hplus).W + (materializedBranch Hminus).W =
      (Kernel.add Hplus Hminus).W := rfl

/-- After materialization, the total stored second moment is the sum of the two
branch structural powers. -/
theorem materialized_branches_second
    (Hplus Hminus : Kernel E)
    (hplus : Hplus.W ≠ 0) (hminus : Hminus.W ≠ 0) :
    (materializedBranch Hplus).Q + (materializedBranch Hminus).Q =
      structuralPower Hplus + structuralPower Hminus := by
  rw [materializedBranch_Q Hplus hplus, materializedBranch_Q Hminus hminus]

end Kernel

/-- Exact weighted mean used to materialize a parent as two children. -/
def weightedMean {E : Type*} [NormedAddCommGroup E] [Module ℝ E]
    (a b : ℝ) (x y : E) : E :=
  (a + b)⁻¹ • (a • x + b • y)

/-- The first moment of a nondegenerate weighted mean is exact. -/
@[simp] theorem weightedMean_moment
    {E : Type*} [NormedAddCommGroup E] [Module ℝ E]
    (a b : ℝ) (x y : E) (hab : a + b ≠ 0) :
    (a + b) • weightedMean a b x y = a • x + b • y := by
  simp [weightedMean, smul_smul, hab]

/-- Materialized child centers preserve the parent barycenter for every common
geometric radius. -/
theorem materialized_centers_barycenter
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (a b radius : ℝ) (parent x y : E) (hab : a + b ≠ 0) :
    let μ := weightedMean a b x y
    a • (parent + radius • (x - μ)) +
        b • (parent + radius • (y - μ)) =
      (a + b) • parent := by
  dsimp [weightedMean]
  have hbalance := child_offsets_balance a b x y hab
  dsimp at hbalance
  calc
    a • (parent + radius •
          (x - (a + b)⁻¹ • (a • x + b • y))) +
        b • (parent + radius •
          (y - (a + b)⁻¹ • (a • x + b • y))) =
      (a + b) • parent + radius •
        (a • (x - (a + b)⁻¹ • (a • x + b • y)) +
          b • (y - (a + b)⁻¹ • (a • x + b • y))) := by
            module
    _ = (a + b) • parent := by
      rw [hbalance, smul_zero, add_zero]

namespace Kernel

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Aggregated layer moment obtained by replacing one parent point with its two
materialized children. -/
def splitLayerMoment (background : Kernel E)
    (a b : ℝ) (x y : E) : Kernel E :=
  Kernel.add background (Kernel.add (point a x) (point b y))

/-- Aggregated layer moment before materialization of the same barycentric
parent. -/
def mergedLayerMoment (background : Kernel E)
    (a b : ℝ) (x y : E) : Kernel E :=
  Kernel.add background (point (a + b) (weightedMean a b x y))

/-- Equal mass and first moment make the moving terms cancel when comparing two
structural powers. -/
theorem structuralPower_sub_eq_Q_sub
    (Hnew Hold : Kernel E)
    (hW : Hnew.W = Hold.W) (hS : Hnew.S = Hold.S) :
    structuralPower Hnew - structuralPower Hold = Hnew.Q - Hold.Q := by
  unfold structuralPower movePower
  rw [hW, hS]
  ring

/-- Splitting a barycentric parent inside an arbitrary unchanged layer raises
realized layer capital by exactly `J_split` (specification §15.2). -/
theorem split_realized_capital_gain
    (background : Kernel E) (a b : ℝ) (x y : E)
    (hab : a + b ≠ 0) :
    structuralPower (splitLayerMoment background a b x y) -
        structuralPower (mergedLayerMoment background a b x y) =
      (a * b / (a + b)) * sqNorm (x - y) := by
  have hW : (splitLayerMoment background a b x y).W =
      (mergedLayerMoment background a b x y).W := by
    rfl
  have hS : (splitLayerMoment background a b x y).S =
      (mergedLayerMoment background a b x y).S := by
    dsimp [splitLayerMoment, mergedLayerMoment, Kernel.add, point]
    rw [weightedMean_moment a b x y hab]
  have hmeanQ :
      (a + b) * sqNorm (weightedMean a b x y) =
        sqNorm (a • x + b • y) / (a + b) := by
    unfold weightedMean
    rw [sqNorm_smul]
    field_simp [hab]
  rw [structuralPower_sub_eq_Q_sub
    (splitLayerMoment background a b x y)
    (mergedLayerMoment background a b x y) hW hS]
  dsimp [splitLayerMoment, mergedLayerMoment, Kernel.add, point]
  rw [hmeanQ]
  calc
    background.Q + (a * sqNorm x + b * sqNorm y) -
        (background.Q + sqNorm (a • x + b • y) / (a + b)) =
      a * sqNorm x + b * sqNorm y -
        sqNorm (a • x + b • y) / (a + b) := by ring
    _ = (a * b / (a + b)) * sqNorm (x - y) :=
      weighted_two_variance a b x y hab

end Kernel

/-- Minimal topological state needed to state the constitutional identity rule
for a Cell split. -/
structure CellState (Id E : Type*) where
  identity : Id
  center : E
  plus : Kernel E
  minus : Kernel E

namespace CellState

/-- Result of one horizontal mitosis. -/
structure MitosisResult (Id E : Type*) where
  mother : CellState Id E
  daughter : CellState Id E

variable {Id E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Parent moment derived from the two ordered latent histories. -/
def parent (cell : CellState Id E) : Kernel E :=
  Kernel.add cell.plus cell.minus

/-- Exact real-valued materialization of specification §8.2–§8.3. The caller
supplies a fresh daughter identity; freshness is an external identity-factory
obligation. -/
def materializeMitosis (cell : CellState Id E)
    (daughterIdentity : Id) (radius : ℝ) : MitosisResult Id E :=
  let parentMean := Kernel.mean cell.parent
  let plusMean := Kernel.mean cell.plus
  let minusMean := Kernel.mean cell.minus
  let motherParent := Kernel.materializedBranch cell.plus
  let daughterParent := Kernel.materializedBranch cell.minus
  {
    mother := {
      identity := cell.identity
      center := cell.center + radius • (plusMean - parentMean)
      plus := Kernel.half motherParent
      minus := Kernel.half motherParent
    }
    daughter := {
      identity := daughterIdentity
      center := cell.center + radius • (minusMean - parentMean)
      plus := Kernel.half daughterParent
      minus := Kernel.half daughterParent
    }
  }

/-- The plus branch is constitutionally the continuation of the mother. -/
@[simp] theorem materializeMitosis_mother_identity
    (cell : CellState Id E) (daughterIdentity : Id) (radius : ℝ) :
    (materializeMitosis cell daughterIdentity radius).mother.identity =
      cell.identity := rfl

/-- The minus branch receives exactly the caller-supplied new identity. -/
@[simp] theorem materializeMitosis_daughter_identity
    (cell : CellState Id E) (daughterIdentity : Id) (radius : ℝ) :
    (materializeMitosis cell daughterIdentity radius).daughter.identity =
      daughterIdentity := rfl

/-- The mother's two fresh latent histories reconstruct the transferred plus
branch and therefore carry no inherited second split. -/
@[simp] theorem materializeMitosis_mother_parent
    (cell : CellState Id E) (daughterIdentity : Id) (radius : ℝ) :
    (materializeMitosis cell daughterIdentity radius).mother.parent =
      Kernel.materializedBranch cell.plus := by
  simp [materializeMitosis, parent]

/-- The daughter's two fresh latent histories reconstruct the transferred minus
branch and therefore carry no inherited second split. -/
@[simp] theorem materializeMitosis_daughter_parent
    (cell : CellState Id E) (daughterIdentity : Id) (radius : ℝ) :
    (materializeMitosis cell daughterIdentity radius).daughter.parent =
      Kernel.materializedBranch cell.minus := by
  simp [materializeMitosis, parent]

end CellState

end

end Auxein
