import Auxein.Geometry

namespace Auxein

noncomputable section

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Two-owner specialization of the one-pass concordance formula in §11.2. -/
def crossPower₂ (w₁ w₂ : ℝ) (d₁ d₂ : E) : ℝ :=
  let W := w₁ + w₂
  if W = 0 then 0
  else
    (sqNorm (w₁ • d₁ + w₂ • d₂) -
      (w₁ ^ 2 * sqNorm d₁ + w₂ ^ 2 * sqNorm d₂)) / W

/-- For two owners, only the cross dot product remains. -/
theorem crossPower₂_eq
    (w₁ w₂ : ℝ) (d₁ d₂ : E) (hW : w₁ + w₂ ≠ 0) :
    crossPower₂ w₁ w₂ d₁ d₂ =
      (2 * w₁ * w₂ / (w₁ + w₂)) * dot d₁ d₂ := by
  simp [crossPower₂, hW, sqNorm_add, sqNorm_smul, dot_smul_left,
    dot_smul_right]
  field_simp [hW]

/-- Orthogonal owner distinctions carry no cross-owner concordance. -/
theorem crossPower₂_eq_zero_of_orthogonal
    (w₁ w₂ : ℝ) (d₁ d₂ : E) (hW : w₁ + w₂ ≠ 0)
    (horth : dot d₁ d₂ = 0) :
    crossPower₂ w₁ w₂ d₁ d₂ = 0 := by
  rw [crossPower₂_eq w₁ w₂ d₁ d₂ hW, horth, mul_zero]

/-- A constant second owner contributes no cross-owner concordance. -/
theorem crossPower₂_eq_zero_of_second_constant
    (w₁ w₂ : ℝ) (d₁ : E) (hW : w₁ + w₂ ≠ 0) :
    crossPower₂ w₁ w₂ d₁ 0 = 0 := by
  apply crossPower₂_eq_zero_of_orthogonal w₁ w₂ d₁ 0 hW
  simp

/-- Global inversion of the historical plus/minus labels preserves concordance. -/
theorem crossPower₂_global_inversion
    (w₁ w₂ : ℝ) (d₁ d₂ : E) :
    crossPower₂ w₁ w₂ (-d₁) (-d₂) = crossPower₂ w₁ w₂ d₁ d₂ := by
  by_cases hW : w₁ + w₂ = 0
  · simp [crossPower₂, hW]
  · rw [crossPower₂_eq w₁ w₂ (-d₁) (-d₂) hW,
      crossPower₂_eq w₁ w₂ d₁ d₂ hW]
    simp [dot]

/-- Positive aligned distinctions yield positive concordance. -/
theorem crossPower₂_pos
    {w₁ w₂ : ℝ} (hw₁ : 0 < w₁) (hw₂ : 0 < w₂)
    {d₁ d₂ : E} (hd : 0 < dot d₁ d₂) :
    0 < crossPower₂ w₁ w₂ d₁ d₂ := by
  have hW : w₁ + w₂ ≠ 0 := ne_of_gt (add_pos hw₁ hw₂)
  rw [crossPower₂_eq w₁ w₂ d₁ d₂ hW]
  positivity

/-- Opposed distinctions yield negative concordance. -/
theorem crossPower₂_neg
    {w₁ w₂ : ℝ} (hw₁ : 0 < w₁) (hw₂ : 0 < w₂)
    {d₁ d₂ : E} (hd : dot d₁ d₂ < 0) :
    crossPower₂ w₁ w₂ d₁ d₂ < 0 := by
  have hW : w₁ + w₂ ≠ 0 := ne_of_gt (add_pos hw₁ hw₂)
  rw [crossPower₂_eq w₁ w₂ d₁ d₂ hW]
  have hc : 0 < 2 * w₁ * w₂ / (w₁ + w₂) := by positivity
  exact mul_neg_of_pos_of_neg hc hd

end

end Auxein
