import Auxein.Geometry

namespace Auxein

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Weighted two-point variance identity. This is the algebraic core of
`J_split` and `J_B`. -/
theorem weighted_two_variance
    (a b : ℝ) (x y : E) (hab : a + b ≠ 0) :
    a * sqNorm x + b * sqNorm y -
        sqNorm (a • x + b • y) / (a + b) =
      (a * b / (a + b)) * sqNorm (x - y) := by
  rw [sqNorm_add, sqNorm_smul, sqNorm_smul, sqNorm_sub]
  simp [dot_smul_left, dot_smul_right]
  field_simp [hab]
  ring

/-- Literal `P_struct - (P₊ + P₋)` form of the split gain (§8.1). -/
theorem split_gain_identity
    (a b Gplus Gminus : ℝ) (x y : E)
    (ha : a ≠ 0) (hb : b ≠ 0) (hab : a + b ≠ 0) :
    ((Gplus + Gminus) - sqNorm (a • x + b • y) / (a + b)) -
        ((Gplus - sqNorm (a • x) / a) +
          (Gminus - sqNorm (b • y) / b)) =
      (a * b / (a + b)) * sqNorm (x - y) := by
  have hxa : sqNorm (a • x) / a = a * sqNorm x := by
    rw [sqNorm_smul]
    field_simp [ha]
  have hxb : sqNorm (b • y) / b = b * sqNorm y := by
    rw [sqNorm_smul]
    field_simp [hb]
  rw [hxa, hxb]
  calc
    ((Gplus + Gminus) - sqNorm (a • x + b • y) / (a + b)) -
        ((Gplus - a * sqNorm x) + (Gminus - b * sqNorm y)) =
      a * sqNorm x + b * sqNorm y -
        sqNorm (a • x + b • y) / (a + b) := by ring
    _ = (a * b / (a + b)) * sqNorm (x - y) :=
      weighted_two_variance a b x y hab

/-- The exact split/bud gain is nonnegative for positive branch masses. -/
theorem weighted_two_variance_nonnegative
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : 0 < a + b) (x y : E) :
    0 ≤ (a * b / (a + b)) * sqNorm (x - y) := by
  exact mul_nonneg (div_nonneg (mul_nonneg ha hb) (le_of_lt hab)) (sqNorm_nonneg _)

/-- The child offsets balance exactly around the parent mean (specification §8.2). -/
theorem child_offsets_balance
    (a b : ℝ) (x y : E) (hab : a + b ≠ 0) :
    let μ := (a + b)⁻¹ • (a • x + b • y)
    a • (x - μ) + b • (y - μ) = 0 := by
  dsimp
  calc
    a • (x - (a + b)⁻¹ • (a • x + b • y)) +
          b • (y - (a + b)⁻¹ • (a • x + b • y)) =
        (a • x + b • y) -
          (a + b) • ((a + b)⁻¹ • (a • x + b • y)) := by
            module
    _ = (a • x + b • y) - (a • x + b • y) := by
          rw [smul_smul]
          simp [hab]
    _ = 0 := sub_self _

/-- A child recentered on its own branch mean has zero first moment. -/
theorem branch_recenter_first_moment
    (a : ℝ) (x : E) : a • x - a • x = 0 := by
  abel

/-- A neutral split has zero separation gain. -/
theorem neutral_split_gain_zero (a : ℝ) (x : E) :
    (a * a / (a + a)) * sqNorm (x - x) = 0 := by
  simp

end Auxein
