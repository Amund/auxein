import Auxein.Geometry

namespace Auxein

noncomputable section

open scoped RealInnerProductSpace

/-- Exact real-valued quadratic kernel, before persistent-format projection. -/
structure Kernel (E : Type*) where
  W : ℝ
  S : E
  Q : ℝ

namespace Kernel

/-- Two kernels are equal when their three stored moments are equal. -/
@[ext] theorem ext {E : Type*} {H₁ H₂ : Kernel E}
    (hW : H₁.W = H₂.W) (hS : H₁.S = H₂.S) (hQ : H₁.Q = H₂.Q) :
    H₁ = H₂ := by
  cases H₁
  cases H₂
  cases hW
  cases hS
  cases hQ
  rfl

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The canonical invariant from specification §3. -/
def Valid (H : Kernel E) : Prop :=
  0 ≤ H.W ∧ 0 ≤ H.Q ∧ sqNorm H.S ≤ H.W * H.Q ∧
    (H.W = 0 → H.S = 0 ∧ H.Q = 0)

/-- The moving component `‖S‖²/W`. -/
def movePower (H : Kernel E) : ℝ := sqNorm H.S / H.W

/-- The structural component `Q - ‖S‖²/W`. -/
def structuralPower (H : Kernel E) : ℝ := H.Q - movePower H

/-- Exact coordinate recentering from specification §3.1. -/
def recenter (H : Kernel E) (Δ : E) : Kernel E where
  W := H.W
  S := H.S - H.W • Δ
  Q := H.Q - 2 * dot Δ H.S + H.W * sqNorm Δ

@[simp] theorem recenter_W (H : Kernel E) (Δ : E) : (recenter H Δ).W = H.W := rfl

@[simp] theorem recenter_S (H : Kernel E) (Δ : E) :
    (recenter H Δ).S = H.S - H.W • Δ := rfl

@[simp] theorem recenter_Q (H : Kernel E) (Δ : E) :
    (recenter H Δ).Q = H.Q - 2 * dot Δ H.S + H.W * sqNorm Δ := rfl

/-- Exact recentering preserves structural power (specification §3.1). -/
theorem structuralPower_recenter (H : Kernel E) (Δ : E) (hW : H.W ≠ 0) :
    structuralPower (recenter H Δ) = structuralPower H := by
  unfold structuralPower movePower recenter
  rw [sqNorm_sub_smul]
  rw [dot_comm Δ H.S]
  field_simp [hW]
  ring

/-- Recentring on the kernel mean closes the first moment exactly. -/
theorem recenter_on_mean_S (H : Kernel E) (hW : H.W ≠ 0) :
    (recenter H (H.W⁻¹ • H.S)).S = 0 := by
  simp [recenter, smul_smul, hW]

/-- Recentring on the kernel mean leaves exactly the structural component in `Q`. -/
theorem recenter_on_mean_Q (H : Kernel E) (hW : H.W ≠ 0) :
    (recenter H (H.W⁻¹ • H.S)).Q = structuralPower H := by
  have hS : (recenter H (H.W⁻¹ • H.S)).S = 0 :=
    recenter_on_mean_S H hW
  rw [← structuralPower_recenter H (H.W⁻¹ • H.S) hW]
  unfold structuralPower movePower
  simp only [hS, sqNorm_zero, zero_div, sub_zero]

/-- Exact real-valued EMA injection, written with explicit `χ` and `α`. -/
def inject (χ α r : ℝ) (x : E) (H : Kernel E) : Kernel E where
  W := χ * H.W + α * r
  S := χ • H.S + (α * r) • x
  Q := χ * H.Q + α * r * sqNorm x

/-- Componentwise sum used to derive a parent from its two histories. -/
def add (H₁ H₂ : Kernel E) : Kernel E where
  W := H₁.W + H₂.W
  S := H₁.S + H₂.S
  Q := H₁.Q + H₂.Q

/-- If routed relevance is conserved, summing the two updated histories is
exactly the parent update (specification §7.1). -/
theorem add_inject_branches
    (χ α r rplus rminus : ℝ) (x : E) (Hplus Hminus : Kernel E)
    (hr : rplus + rminus = r) :
    add (inject χ α rplus x Hplus) (inject χ α rminus x Hminus) =
      inject χ α r x (add Hplus Hminus) := by
  apply Kernel.ext
  · dsimp [add, inject]
    rw [← hr]
    ring
  · dsimp [add, inject]
    rw [← hr]
    module
  · dsimp [add, inject]
    rw [← hr]
    ring

/-- Forgetting without injection is exact scalar multiplication (specification §3.2). -/
def decay (χ : ℝ) (H : Kernel E) : Kernel E where
  W := χ * H.W
  S := χ • H.S
  Q := χ * H.Q

/-- Neutral split of a kernel into two equal histories. -/
def half (H : Kernel E) : Kernel E where
  W := H.W / 2
  S := (1 / 2 : ℝ) • H.S
  Q := H.Q / 2

@[simp] theorem half_mass_sum (H : Kernel E) : (half H).W + (half H).W = H.W := by
  simp [half]

@[simp] theorem half_first_sum (H : Kernel E) : (half H).S + (half H).S = H.S := by
  dsimp [half]
  module

@[simp] theorem half_second_sum (H : Kernel E) : (half H).Q + (half H).Q = H.Q := by
  simp [half]

end Kernel

end

end Auxein
