import Auxein.Geometry

namespace Auxein

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [DecidableEq E]

/-- Ordered plus/minus branch used by latent routing and vertical emission. -/
inductive Branch where
  | plus
  | minus
  deriving DecidableEq, Repr

/-- Exact pre-injection routing law from specification §7.1.

The parent-empty case is represented by `parentEmpty`; otherwise `axis` and
`residual` are the already-derived vectors `b` and `s`.
-/
noncomputable def routeLatent (parentEmpty : Bool) (axis residual : E) (r : ℝ) : ℝ × ℝ × Branch :=
  if parentEmpty then
    (r / 2, r / 2, Branch.plus)
  else if axis = 0 then
    if residual = 0 then (r / 2, r / 2, Branch.plus) else (r, 0, Branch.plus)
  else
    let signed := dot axis residual
    if 0 < signed then (r, 0, Branch.plus)
    else if signed < 0 then (0, r, Branch.minus)
    else (r / 2, r / 2, Branch.plus)

/-- Routing never creates or destroys external relevance. -/
theorem routeLatent_conserves
    (parentEmpty : Bool) (axis residual : E) (r : ℝ) :
    (routeLatent parentEmpty axis residual r).1 +
      (routeLatent parentEmpty axis residual r).2.1 = r := by
  classical
  simp [routeLatent]
  split_ifs <;> ring

/-- With nonnegative external relevance, both routed shares are nonnegative. -/
theorem routeLatent_nonnegative
    (parentEmpty : Bool) (axis residual : E) {r : ℝ} (hr : 0 ≤ r) :
    0 ≤ (routeLatent parentEmpty axis residual r).1 ∧
      0 ≤ (routeLatent parentEmpty axis residual r).2.1 := by
  classical
  simp [routeLatent]
  split_ifs <;> constructor <;> positivity

end Auxein
