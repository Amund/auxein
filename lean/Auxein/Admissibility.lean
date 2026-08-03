import Auxein.Kernel

namespace Auxein

noncomputable section

open scoped RealInnerProductSpace

namespace Kernel

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Quadratic slack of a kernel. Admissibility is the assertion that this
quantity is nonnegative, together with the canonical empty-kernel condition. -/
def gap (H : Kernel E) : ℝ := H.W * H.Q - sqNorm H.S


/-- A single weighted observation is an admissible rank-one kernel. -/
def point (r : ℝ) (x : E) : Kernel E where
  W := r
  S := r • x
  Q := r * sqNorm x

/-- Rank-one kernels have zero quadratic slack. -/
@[simp] theorem gap_point (r : ℝ) (x : E) : gap (point r x) = 0 := by
  unfold gap point
  rw [sqNorm_smul]
  ring

/-- A nonnegative weighted observation is a valid initial kernel. -/
theorem valid_point {r : ℝ} (hr : 0 ≤ r) (x : E) : Valid (point r x) := by
  refine ⟨hr, ?_, ?_, ?_⟩
  · dsimp [point]
    exact mul_nonneg hr (sqNorm_nonneg x)
  · have hgap : 0 ≤ gap (point r x) := by simp
    unfold gap at hgap
    linarith
  · intro hmass
    change r = 0 at hmass
    simp [point, hmass]

/-- The zero kernel is admissible. -/
theorem valid_zero : Valid (Kernel.mk 0 0 0 : Kernel E) := by
  simp [Valid, sqNorm]

/-- `Valid` exposes nonnegative quadratic slack. -/
theorem Valid.gap_nonnegative {H : Kernel E} (hH : Valid H) : 0 ≤ gap H := by
  rcases hH with ⟨_, _, hquad, _⟩
  unfold gap
  linarith


/-- Positive-mass valid kernels have nonnegative structural power. -/
theorem Valid.structuralPower_nonnegative {H : Kernel E} (hH : Valid H)
    (hW : 0 < H.W) : 0 ≤ structuralPower H := by
  rcases hH with ⟨_, _, hquad, _⟩
  have hdiv : sqNorm H.S / H.W ≤ H.Q := by
    apply (div_le_iff₀ hW).2
    nlinarith
  unfold structuralPower movePower
  linarith

/-- Exact recentering preserves the quadratic slack, including at zero mass. -/
@[simp] theorem gap_recenter (H : Kernel E) (Δ : E) :
    gap (Kernel.recenter H Δ) = gap H := by
  unfold gap
  rw [recenter_W, recenter_Q, recenter_S, sqNorm_sub_smul, dot_comm Δ H.S]
  ring

/-- Recentring preserves the complete kernel invariant. -/
theorem Valid.recenter {H : Kernel E} (hH : Valid H) (Δ : E) :
    Valid (Kernel.recenter H Δ) := by
  rcases hH with ⟨hW, hQ, hquad, hzero⟩
  have hgap : 0 ≤ gap H := by
    unfold gap
    linarith
  have hgap' : 0 ≤ gap (Kernel.recenter H Δ) := by
    rw [gap_recenter]
    exact hgap
  have hquad' : sqNorm (Kernel.recenter H Δ).S ≤
      (Kernel.recenter H Δ).W * (Kernel.recenter H Δ).Q := by
    unfold gap at hgap'
    linarith
  have hQ' : 0 ≤ (Kernel.recenter H Δ).Q := by
    by_cases hW0 : H.W = 0
    · rcases hzero hW0 with ⟨hS0, hQ0⟩
      simp [Kernel.recenter, hW0, hS0, hQ0]
    · have hWpos : 0 < H.W := lt_of_le_of_ne hW (Ne.symm hW0)
      have hnorm : 0 ≤ sqNorm (Kernel.recenter H Δ).S := sqNorm_nonneg _
      rw [recenter_W] at hquad'
      nlinarith
  refine ⟨?_, hQ', hquad', ?_⟩
  · rw [recenter_W]
    exact hW
  · intro hW'
    have hW0 : H.W = 0 := by
      change H.W = 0 at hW'
      exact hW'
    rcases hzero hW0 with ⟨hS0, hQ0⟩
    simp [Kernel.recenter, hW0, hS0, hQ0]

/-- Algebraic decomposition of the slack after one exact EMA injection.
The mixed term is the recentered second moment of the previous kernel around
`x`, hence is nonnegative whenever the previous kernel is admissible. -/
theorem gap_inject (χ α r : ℝ) (x : E) (H : Kernel E) :
    gap (Kernel.inject χ α r x H) =
      χ ^ 2 * gap H + χ * (α * r) * (Kernel.recenter H x).Q := by
  unfold gap
  dsimp [Kernel.inject, Kernel.recenter]
  rw [sqNorm_add, sqNorm_smul, sqNorm_smul,
    dot_smul_left, dot_smul_right, dot_comm H.S x]
  ring

/-- Exact EMA injection preserves admissibility for nonnegative retention,
learning coefficient and relevance. -/
theorem Valid.inject {H : Kernel E} (hH : Valid H)
    {χ α r : ℝ} (hχ : 0 ≤ χ) (hα : 0 ≤ α) (hr : 0 ≤ r) (x : E) :
    Valid (Kernel.inject χ α r x H) := by
  rcases hH with ⟨hW, hQ, hquad, hzero⟩
  have hb : 0 ≤ α * r := mul_nonneg hα hr
  have hrec : Valid (Kernel.recenter H x) :=
    Valid.recenter ⟨hW, hQ, hquad, hzero⟩ x
  rcases hrec with ⟨_, hrecQ, _, _⟩
  have hgap : 0 ≤ gap H := by
    unfold gap
    linarith
  have hgap' : 0 ≤ gap (Kernel.inject χ α r x H) := by
    rw [gap_inject]
    exact add_nonneg
      (mul_nonneg (sq_nonneg χ) hgap)
      (mul_nonneg (mul_nonneg hχ hb) hrecQ)
  have hquad' : sqNorm (Kernel.inject χ α r x H).S ≤
      (Kernel.inject χ α r x H).W * (Kernel.inject χ α r x H).Q := by
    unfold gap at hgap'
    linarith
  refine ⟨?_, ?_, hquad', ?_⟩
  · dsimp [Kernel.inject]
    exact add_nonneg (mul_nonneg hχ hW) hb
  · dsimp [Kernel.inject]
    exact add_nonneg (mul_nonneg hχ hQ) (mul_nonneg hb (sqNorm_nonneg x))
  · intro hmass
    dsimp [Kernel.inject] at hmass ⊢
    have hχW : 0 ≤ χ * H.W := mul_nonneg hχ hW
    have hχW0 : χ * H.W = 0 := by linarith
    have hb0 : α * r = 0 := by linarith
    rcases mul_eq_zero.mp hχW0 with hχ0 | hW0
    · simp [hχ0, hb0]
    · rcases hzero hW0 with ⟨hS0, hQ0⟩
      simp [hS0, hQ0, hb0]

/-- Canonical EMA form with `α = 1 - χ`. -/
theorem Valid.ema {H : Kernel E} (hH : Valid H)
    {χ r : ℝ} (hχ0 : 0 ≤ χ) (hχ1 : χ ≤ 1) (hr : 0 ≤ r) (x : E) :
    Valid (Kernel.inject χ (1 - χ) r x H) := by
  exact hH.inject hχ0 (sub_nonneg.mpr hχ1) hr x

/-- Pure forgetting preserves admissibility for a nonnegative decay factor. -/
theorem Valid.decay {H : Kernel E} (hH : Valid H)
    {χ : ℝ} (hχ : 0 ≤ χ) : Valid (Kernel.decay χ H) := by
  rcases hH with ⟨hW, hQ, hquad, hzero⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · dsimp [Kernel.decay]
    exact mul_nonneg hχ hW
  · dsimp [Kernel.decay]
    exact mul_nonneg hχ hQ
  · dsimp [Kernel.decay]
    rw [sqNorm_smul]
    have hscaled := mul_le_mul_of_nonneg_left hquad (sq_nonneg χ)
    nlinarith
  · intro hmass
    dsimp [Kernel.decay] at hmass ⊢
    rcases mul_eq_zero.mp hmass with hχ0 | hW0
    · simp [hχ0]
    · rcases hzero hW0 with ⟨hS0, hQ0⟩
      simp [hS0, hQ0]

/-- A neutral half-history remains admissible. -/
theorem Valid.half {H : Kernel E} (hH : Valid H) : Valid (Kernel.half H) := by
  have hhalf : Kernel.half H = Kernel.decay (1 / 2 : ℝ) H := by
    apply Kernel.ext
    · dsimp [Kernel.half, Kernel.decay]
      ring
    · rfl
    · dsimp [Kernel.half, Kernel.decay]
      ring
  rw [hhalf]
  exact hH.decay (by norm_num)

/-- The two equal neutral histories reconstruct their parent exactly. -/
@[simp] theorem add_half_self (H : Kernel E) : Kernel.add (Kernel.half H) (Kernel.half H) = H := by
  apply Kernel.ext
  · exact half_mass_sum H
  · exact half_first_sum H
  · exact half_second_sum H


/-- A common recentering commutes with branch aggregation. -/
@[simp] theorem add_recenter (Hplus Hminus : Kernel E) (Δ : E) :
    Kernel.add (Kernel.recenter Hplus Δ) (Kernel.recenter Hminus Δ) =
      Kernel.recenter (Kernel.add Hplus Hminus) Δ := by
  apply Kernel.ext
  · rfl
  · dsimp [Kernel.add, Kernel.recenter]
    module
  · dsimp [Kernel.add, Kernel.recenter]
    rw [dot_add_right]
    ring

/-- Consequently, a common branch recentering preserves parent admissibility. -/
theorem valid_add_recenter (Hplus Hminus : Kernel E) (Δ : E)
    (hparent : Valid (Kernel.add Hplus Hminus)) :
    Valid (Kernel.add (Kernel.recenter Hplus Δ) (Kernel.recenter Hminus Δ)) := by
  rw [add_recenter]
  exact hparent.recenter Δ

/-- Common forgetting commutes with branch aggregation. -/
@[simp] theorem add_decay (χ : ℝ) (Hplus Hminus : Kernel E) :
    Kernel.add (Kernel.decay χ Hplus) (Kernel.decay χ Hminus) =
      Kernel.decay χ (Kernel.add Hplus Hminus) := by
  apply Kernel.ext
  · dsimp [Kernel.add, Kernel.decay]
    ring
  · dsimp [Kernel.add, Kernel.decay]
    module
  · dsimp [Kernel.add, Kernel.decay]
    ring

/-- Parent admissibility is preserved by common nonnegative forgetting. -/
theorem valid_add_decay {χ : ℝ} (hχ : 0 ≤ χ)
    (Hplus Hminus : Kernel E) (hparent : Valid (Kernel.add Hplus Hminus)) :
    Valid (Kernel.add (Kernel.decay χ Hplus) (Kernel.decay χ Hminus)) := by
  rw [add_decay]
  exact hparent.decay hχ

/-- Coordinate dilation used by specification §13.2. Mass is unchanged,
first moments scale linearly and second moments quadratically. -/
def scaleCoordinates (a : ℝ) (H : Kernel E) : Kernel E where
  W := H.W
  S := a • H.S
  Q := a ^ 2 * H.Q

/-- A common coordinate dilation preserves admissibility. -/
theorem Valid.scaleCoordinates {H : Kernel E} (hH : Valid H) (a : ℝ) :
    Valid (Kernel.scaleCoordinates a H) := by
  rcases hH with ⟨hW, hQ, hquad, hzero⟩
  refine ⟨hW, ?_, ?_, ?_⟩
  · dsimp [Kernel.scaleCoordinates]
    exact mul_nonneg (sq_nonneg a) hQ
  · dsimp [Kernel.scaleCoordinates]
    rw [sqNorm_smul]
    have hscaled := mul_le_mul_of_nonneg_left hquad (sq_nonneg a)
    nlinarith
  · intro hW0
    rcases hzero hW0 with ⟨hS0, hQ0⟩
    simp [Kernel.scaleCoordinates, hS0, hQ0]



/-- A common coordinate dilation commutes with branch aggregation. -/
@[simp] theorem add_scaleCoordinates (a : ℝ) (Hplus Hminus : Kernel E) :
    Kernel.add (Kernel.scaleCoordinates a Hplus) (Kernel.scaleCoordinates a Hminus) =
      Kernel.scaleCoordinates a (Kernel.add Hplus Hminus) := by
  apply Kernel.ext
  · rfl
  · dsimp [Kernel.add, Kernel.scaleCoordinates]
    module
  · dsimp [Kernel.add, Kernel.scaleCoordinates]
    ring

/-- Parent admissibility survives a common change of coordinate unit. -/
theorem valid_add_scaleCoordinates (a : ℝ) (Hplus Hminus : Kernel E)
    (hparent : Valid (Kernel.add Hplus Hminus)) :
    Valid (Kernel.add (Kernel.scaleCoordinates a Hplus) (Kernel.scaleCoordinates a Hminus)) := by
  rw [add_scaleCoordinates]
  exact hparent.scaleCoordinates a

/-- Both routed histories remain admissible when their shares are nonnegative. -/
theorem valid_inject_branches
    {χ α rplus rminus : ℝ} (x : E) (Hplus Hminus : Kernel E)
    (hχ : 0 ≤ χ) (hα : 0 ≤ α)
    (hrplus : 0 ≤ rplus) (hrminus : 0 ≤ rminus)
    (hplus : Valid Hplus) (hminus : Valid Hminus) :
    Valid (Kernel.inject χ α rplus x Hplus) ∧
      Valid (Kernel.inject χ α rminus x Hminus) := by
  exact ⟨hplus.inject hχ hα hrplus x, hminus.inject hχ hα hrminus x⟩

/-- The branch aggregate after one routed write is admissible whenever the
pre-write aggregate is an admissible parent. This is the induction principle
used by the implementation; no closure theorem for arbitrary unrelated sums
is required. -/
theorem valid_add_inject_branches
    {χ α r rplus rminus : ℝ} (x : E) (Hplus Hminus : Kernel E)
    (hχ : 0 ≤ χ) (hα : 0 ≤ α) (hr : 0 ≤ r)
    (hroute : rplus + rminus = r)
    (hparent : Valid (Kernel.add Hplus Hminus)) :
    Valid (Kernel.add (Kernel.inject χ α rplus x Hplus)
      (Kernel.inject χ α rminus x Hminus)) := by
  rw [Kernel.add_inject_branches χ α r rplus rminus x Hplus Hminus hroute]
  exact hparent.inject hχ hα hr x

/-- After a routed canonical EMA write, branch aggregation remains admissible. -/
theorem valid_add_ema_branches
    {χ r rplus rminus : ℝ} (x : E) (Hplus Hminus : Kernel E)
    (hχ0 : 0 ≤ χ) (hχ1 : χ ≤ 1) (hr : 0 ≤ r)
    (hroute : rplus + rminus = r)
    (hparent : Valid (Kernel.add Hplus Hminus)) :
    Valid (Kernel.add (Kernel.inject χ (1 - χ) rplus x Hplus)
      (Kernel.inject χ (1 - χ) rminus x Hminus)) := by
  exact valid_add_inject_branches x Hplus Hminus hχ0
    (sub_nonneg.mpr hχ1) hr hroute hparent

end Kernel

end

end Auxein
