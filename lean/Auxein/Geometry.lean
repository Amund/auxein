import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

namespace Auxein

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The exact squared geometry used by the real-valued model. -/
def sqNorm (x : E) : ℝ := inner ℝ x x

/-- The exact dot product used by the real-valued model. -/
def dot (x y : E) : ℝ := inner ℝ x y

@[simp] theorem dot_zero_left (x : E) : dot (0 : E) x = 0 := by
  simp [dot]

@[simp] theorem dot_zero_right (x : E) : dot x (0 : E) = 0 := by
  simp [dot]

@[simp] theorem dot_add_left (x y z : E) : dot (x + y) z = dot x z + dot y z := by
  simp [dot, inner_add_left]

@[simp] theorem dot_add_right (x y z : E) : dot x (y + z) = dot x y + dot x z := by
  simp [dot, inner_add_right]

@[simp] theorem dot_sub_left (x y z : E) : dot (x - y) z = dot x z - dot y z := by
  simp [dot, inner_sub_left]

@[simp] theorem dot_sub_right (x y z : E) : dot x (y - z) = dot x y - dot x z := by
  simp [dot, inner_sub_right]

@[simp] theorem dot_smul_left (a : ℝ) (x y : E) : dot (a • x) y = a * dot x y := by
  simp [dot, real_inner_smul_left]

@[simp] theorem dot_smul_right (a : ℝ) (x y : E) : dot x (a • y) = a * dot x y := by
  simp [dot, inner_smul_right]

@[simp] theorem dot_neg_left (x y : E) : dot (-x) y = -dot x y := by
  simp [dot]

@[simp] theorem dot_neg_right (x y : E) : dot x (-y) = -dot x y := by
  simp [dot]

theorem dot_comm (x y : E) : dot x y = dot y x := by
  change inner ℝ x y = inner ℝ y x
  exact (real_inner_comm x y).symm

@[simp] theorem sqNorm_zero : sqNorm (0 : E) = 0 := by
  simp [sqNorm]

theorem sqNorm_nonneg (x : E) : 0 ≤ sqNorm x := by
  change 0 ≤ inner ℝ x x
  exact real_inner_self_nonneg

@[simp] theorem sqNorm_neg (x : E) : sqNorm (-x) = sqNorm x := by
  simp [sqNorm]

@[simp] theorem sqNorm_smul (a : ℝ) (x : E) : sqNorm (a • x) = a ^ 2 * sqNorm x := by
  change dot (a • x) (a • x) = a ^ 2 * dot x x
  rw [dot_smul_left, dot_smul_right]
  ring

theorem sqNorm_add (x y : E) :
    sqNorm (x + y) = sqNorm x + 2 * dot x y + sqNorm y := by
  change dot (x + y) (x + y) = dot x x + 2 * dot x y + dot y y
  rw [dot_add_left, dot_add_right, dot_add_right, dot_comm y x]
  ring

theorem sqNorm_sub (x y : E) :
    sqNorm (x - y) = sqNorm x - 2 * dot x y + sqNorm y := by
  change dot (x - y) (x - y) = dot x x - 2 * dot x y + dot y y
  rw [dot_sub_left, dot_sub_right, dot_sub_right, dot_comm y x]
  ring

theorem sqNorm_add_smul (x y : E) (a : ℝ) :
    sqNorm (x + a • y) = sqNorm x + 2 * a * dot x y + a ^ 2 * sqNorm y := by
  rw [sqNorm_add, sqNorm_smul, dot_smul_right]
  ring

theorem sqNorm_sub_smul (x y : E) (a : ℝ) :
    sqNorm (x - a • y) = sqNorm x - 2 * a * dot x y + a ^ 2 * sqNorm y := by
  rw [sqNorm_sub, sqNorm_smul, dot_smul_right]
  ring

section Translation

variable {V : Type*} [AddCommGroup V]

/-- A common translation leaves an internal shape unchanged (specification §9.3). -/
theorem internal_shape_translation (μbranch μ Δ : V) :
    (μbranch - Δ) - (μ - Δ) = μbranch - μ := by
  abel

/-- A common translation leaves the latent branch axis unchanged (§7.2). -/
theorem latent_axis_translation (μplus μminus Δ : V) :
    (μplus - Δ) - (μminus - Δ) = μplus - μminus := by
  abel

end Translation

end Auxein
