"""Frozen decision gates for the R6 SAR-DE route (handoff v2 §4).

Pure functions so that thresholds are locally unit-testable and auditable
before any cloud run. Constants are frozen by the approved spec; changing any
of them requires a PLAN_APPROVAL amendment, never a tuning pass.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---- G1 pre-training gate (frozen) -------------------------------------------
G1_POINT_ESTIMATE = 20.0      # SAR-only val mIoU point estimate (%)
G1_CI90_LOWER = 16.0          # per-sample one-sided 90% CI lower bound (%)
G1_SOFT_BAND_FLOOR = 15.0     # below this: fail without retry
G1_RETRY_TARGET = 18.0        # soft-band retry must reach this point estimate
G1_MONOTONIC_SPEARMAN = -0.7  # SAR degradation ladder monotonicity (<=)

# ---- q-permutation detectability (frozen) -----------------------------------
QPERM_MIN_ABS_DROP_PP = 0.3   # end-to-end mIoU drop after channel swap (pp)
QPERM_CI_UPPER_BOUND = 0.0    # group-bootstrap CI upper bound must stay < 0


@dataclass(frozen=True)
class G1Decision:
    status: str  # "pass" | "retry" | "fail"
    point_estimate: float
    ci90_lower: float
    monotonic_spearman: float | None
    tries: int
    reason: str


def g1_decision(
    *,
    point_estimate: float,
    ci90_lower: float,
    monotonic_spearman: float | None = None,
    monotonic_required: bool = True,
    tries: int = 1,
) -> G1Decision:
    """Frozen G1 logic (handoff v2 §4).

    Try 1: pass iff point >= 20.0 and CI90 lower >= 16.0 and monotone;
            fail outright below 15.0; otherwise retry with a fresh seed.
    Try 2: pass iff point >= 18.0 and CI90 lower >= 16.0 and monotone.
    """
    mono_ok = (
        not monotonic_required
        or (monotonic_spearman is not None and monotonic_spearman <= G1_MONOTONIC_SPEARMAN)
    )
    if tries == 1:
        if point_estimate >= G1_POINT_ESTIMATE and ci90_lower >= G1_CI90_LOWER and mono_ok:
            return G1Decision("pass", point_estimate, ci90_lower, monotonic_spearman, tries,
                              "pre-training gate passed (point >= 20, CI90 >= 16, monotone)")
        if point_estimate < G1_SOFT_BAND_FLOOR:
            return G1Decision("fail", point_estimate, ci90_lower, monotonic_spearman, tries,
                              "below the soft-band floor (15%) on the first try")
        return G1Decision("retry", point_estimate, ci90_lower, monotonic_spearman, tries,
                          "soft band [15,20): retry with a fresh seed, target >= 18")
    if point_estimate >= G1_RETRY_TARGET and ci90_lower >= G1_CI90_LOWER and mono_ok:
        return G1Decision("pass", point_estimate, ci90_lower, monotonic_spearman, tries,
                          "soft-band retry passed (point >= 18, CI90 >= 16, monotone)")
    return G1Decision("fail", point_estimate, ci90_lower, monotonic_spearman, tries,
                      "soft-band retry below the 18% target; route moves to Plan B")


@dataclass(frozen=True)
class QPermutationDecision:
    detectable: bool
    delta_mean_pp: float
    ci_upper_pp: float
    reason: str


def q_permutation_decision(*, delta_mean_pp: float, ci_upper_pp: float) -> QPermutationDecision:
    """Frozen q-permutation detectability (handoff v2 §4).

    ``delta_mean_pp`` is the end-to-end mIoU change caused by swapping the two
    reliability channels (negative = drop). Detectable iff the model loses at
    least 0.3pp and the group-bootstrap CI upper bound stays below 0.
    """
    detectable = (
        delta_mean_pp <= -QPERM_MIN_ABS_DROP_PP and ci_upper_pp < QPERM_CI_UPPER_BOUND
    )
    reason = (
        "q-permutation detectable (end-to-end mIoU drop, CI excludes 0)"
        if detectable
        else "q-permutation not detectable under the frozen criterion"
    )
    return QPermutationDecision(detectable, delta_mean_pp, ci_upper_pp, reason)


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_replicates: int = 2000,
    seed: int = 20260902,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap on per-sample paired differences.

    ``deltas`` are one value per paired observation (candidate - reference or
    permuted - original) in percentage points. The CI is the percentile interval
    of the resampled means. Pure function: deterministic for a fixed seed,
    locally unit-testable, and reused verbatim by the cloud audit scripts.
    """
    import numpy as np

    if n_replicates < 100:
        raise ValueError("n_replicates must be at least 100")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    arr = np.asarray([float(d) for d in deltas], dtype=np.float64)
    if arr.size < 2:
        raise ValueError("paired bootstrap requires at least two paired deltas")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_replicates), dtype=np.float64)
    for i in range(int(n_replicates)):
        means[i] = arr[rng.integers(0, arr.size, arr.size)].mean()
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "mean": float(arr.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n": int(arr.size),
        "n_replicates": int(n_replicates),
        "seed": int(seed),
    }


# ---- R6 further frozen gates (handoff v2 §4) ---------------------------------
G2_CLEAN_DELTA_PP = 1.0        # rapid clean gain (either channel)
G2_AUC_FAMILIES_OK = 4         # of 6 early AUC families with CI > 0
G2_NULL_RESIDUAL_PP = 1.0      # residual-off null drop
G2_NULL_MASK_PP = 0.3          # SAR feature-mask null drop
G2_NULL_SHUFFLE_PP = 0.15      # SAR-shuffle null drop
G3_CLEAN_MEAN_PP = 2.5         # formal clean mean gain
G3_CLEAN_PER_SEED_PP = 1.5     # per-seed floor
G3_AUC_MUST_PASS = 4           # of 5 must-families with CI > 0 (blur reported separately)
G3_AUC_SEEDS = 2               # of 3 seeds with each family CI > 0
G3_PERCLASS_MEAN_PP = 2.0      # 3-class mean gain
G3_QPERM_MIN_DROP_PP = 0.3     # q-permutation end-to-end drop
G3_TRANSFER_LINEAR_PROBE = 0.0  # EuroSAT linear-probe significance (flag only)


@dataclass(frozen=True)
class G2Decision:
    status: str  # "pass" | "fail"
    clean_delta_pp: float
    auc_families_ok: int
    nulls: dict[str, float]
    reason: str


def g2_decision(
    *,
    clean_delta_pp: float,
    auc_families_ok: int,
    null_residual_pp: float,
    null_mask_pp: float,
    null_shuffle_pp: float,
) -> G2Decision:
    """Frozen G2 rapid gate (6 epochs): dual-channel pass — clean gain OR early
    AUC family evidence — plus the three calibrated null controls."""
    nulls = {
        "residual_off": null_residual_pp,
        "mask": null_mask_pp,
        "shuffle": null_shuffle_pp,
    }
    clean_ok = clean_delta_pp >= G2_CLEAN_DELTA_PP
    auc_ok = auc_families_ok >= G2_AUC_FAMILIES_OK
    null_ok = (
        null_residual_pp >= G2_NULL_RESIDUAL_PP
        and null_mask_pp >= G2_NULL_MASK_PP
        and null_shuffle_pp >= G2_NULL_SHUFFLE_PP
    )
    if (clean_ok or auc_ok) and null_ok:
        return G2Decision("pass", clean_delta_pp, auc_families_ok, nulls,
                          "rapid gate passed (clean or AUC family channel + nulls)")
    return G2Decision("fail", clean_delta_pp, auc_families_ok, nulls,
                      "rapid gate failed (neither clean nor AUC channel, or nulls below floor)")


@dataclass(frozen=True)
class G3Decision:
    status: str  # "pass" (all musts) | "conditional" (musts pass, support 1/3) | "fail"
    g3a_clean: bool
    g3b_auc: bool
    support_hits: int
    reason: str


def g3_decision(
    *,
    clean_mean_pp: float,
    per_seed_min_pp: float,
    clean_ci_lower_pp: float,
    auc_must_families_ok: int,
    auc_seeds_ok: int,
    per_class_mean_pp: float,
    per_class_ci_lower_pp: float,
    q_perm_detectable: bool,
    transfer_significant: bool,
) -> G3Decision:
    """Frozen G3 formal gate (12 epochs x 3 seeds, last-epoch judgement):
    G3a (mandatory) clean mean/per-seed/CI; G3b (mandatory) AUC must-families
    and per-seed consistency; G3c (support, 3-of-2) per-class / q-permutation /
    transfer."""
    g3a = (
        clean_mean_pp >= G3_CLEAN_MEAN_PP
        and per_seed_min_pp >= G3_CLEAN_PER_SEED_PP
        and clean_ci_lower_pp > 0.0
    )
    g3b = (
        auc_must_families_ok >= G3_AUC_MUST_PASS
        and auc_seeds_ok >= G3_AUC_SEEDS
    )
    support = 0
    if per_class_mean_pp >= G3_PERCLASS_MEAN_PP and per_class_ci_lower_pp > 0.0:
        support += 1
    if q_perm_detectable:
        support += 1
    if transfer_significant:
        support += 1
    if not g3a or not g3b:
        return G3Decision("fail", g3a, g3b, support,
                          "G3a (clean) or G3b (AUC) mandatory gate failed")
    if support >= 2:
        return G3Decision("pass", g3a, g3b, support, "G3 passed with mandatory and support evidence")
    return G3Decision("conditional", g3a, g3b, support,
                      "G3 mandatory gates passed, support evidence below 2/3")


@dataclass(frozen=True)
class DistillEpochDecision:
    epochs: int
    reason: str


def distill_epoch_decision(*, inc6_to8_pp: float) -> DistillEpochDecision:
    """Frozen A2 epoch rule: 6 -> 8 epoch increment below 0.5pp freezes at 6;
    at or above 1.5pp freezes at 8; in between repeats the 8-epoch run and
    re-evaluates at the next freeze point (cap 10 documented in the spec)."""
    if inc6_to8_pp < 0.5:
        return DistillEpochDecision(6, "increment below 0.5pp: freeze at 6 epochs")
    if inc6_to8_pp >= 1.5:
        return DistillEpochDecision(8, "increment at or above 1.5pp: freeze at 8 epochs")
    return DistillEpochDecision(10, "increment between 0.5 and 1.5pp: re-run at 10 (cap)")