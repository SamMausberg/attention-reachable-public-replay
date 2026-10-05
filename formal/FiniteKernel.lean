import Mathlib

/-!
Partial formalization, kernel-checked for the public release.
These statements do not formalize the C++ numerical checker or PyTorch.
-/

open scoped BigOperators

namespace AttentionFork

def runState {S A : Type*} (step : S → A → S) : S → List A → S
  | s, [] => s
  | s, a :: as => runState step (step s a) as

theorem shared_state {S A : Type*} (step : S → A → S)
    (s t : S) (xs : List A) (h : s = t) :
    runState step s xs = runState step t xs := by
  exact congrArg (fun u => runState step u xs) h

/-- A common normalized continuation kernel preserves the first-law L1 gap. -/
theorem common_kernel_l1 {I J : Type*} [Fintype I] [Fintype J]
    (p q : I → ℝ) (K : I → J → ℝ)
    (nonneg : ∀ i j, 0 ≤ K i j) (normalized : ∀ i, ∑ j, K i j = 1) :
    (∑ i, ∑ j, |p i * K i j - q i * K i j|) = ∑ i, |p i - q i| := by
  apply Finset.sum_congr rfl
  intro i _
  calc
    (∑ j, |p i * K i j - q i * K i j|) = ∑ j, |p i - q i| * K i j := by
      apply Finset.sum_congr rfl
      intro j _
      rw [← sub_mul, abs_mul, abs_of_nonneg (nonneg i j)]
    _ = |p i - q i| * ∑ j, K i j := by rw [Finset.mul_sum]
    _ = |p i - q i| := by rw [normalized i, mul_one]

/-- The rounded decimal target leaves room for both finite-sampler deviations. -/
theorem programme_budget :
    (1024 : ℚ) / 1200000000 <
      (924 / 1000000 - 2 * 1024 * (1 / 10^12 + 49151 / 2^64))^2 := by
  norm_num

theorem programme_reserve_positive :
    (0 : ℚ) < 924 / 1000000 - 2 * 1024 * (1 / 10^12 + 49151 / 2^64) := by
  norm_num

theorem local_single_head_tolerance :
    (4 : ℚ) * (1 / 1000) / ((7 / 25) * 3) = 1 / 210 := by
  norm_num

theorem packing_charge :
    ((1 : ℚ) / 2) * (1 / 2) / (2 * (2 + 1 / 2)) = 1 / 20 := by
  norm_num

end AttentionFork
