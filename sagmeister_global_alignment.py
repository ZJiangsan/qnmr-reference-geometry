#!/usr/bin/env python3
"""Compare restricted fits before and after label-free full-spectrum alignment.

Run alongside sagmeister_diagnostic.py and external_nmr_validation.py.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sagmeister_diagnostic import fit_bank, write_csv


STEP = .002
GLOBAL_TICKS = 40
LOCAL_TICKS = 3
CROP = (GLOBAL_TICKS + LOCAL_TICKS) * STEP
MODES = ("unaligned_fixed", "global_fixed", "global_local")


def center(x, axis):
    return x - x.mean(axis=axis, keepdims=True)


def reference_scores(x, derivative):
    singular = np.linalg.svd(x, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-6:
        return np.inf, np.inf
    variance = np.sum(1 / singular**2)
    shift = np.linalg.norm(np.linalg.pinv(x) @ (derivative @ np.full(3, 100.)))
    return variance, shift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        ppm, refs, spectra, truth = (np.asarray(data[k], float)
                                     for k in ("ppm", "references", "spectra", "truth"))
    if (ppm.ndim != 1 or len(ppm) < 10 or refs.shape != (len(ppm), 3)
        or truth.ndim != 2 or truth.shape[1] != 3 or not len(truth)
        or spectra.shape != (len(truth), len(ppm))
        or not all(np.isfinite(a).all() for a in (ppm, refs, spectra, truth))):
        parser.error("Unexpected NPZ arrays.")
    order = np.argsort(ppm)
    ppm, refs, spectra = ppm[order], refs[order], spectra[:, order]
    if np.any(np.diff(ppm) <= 0):
        parser.error("The ppm coordinates must be distinct.")
    keep = (ppm >= ppm[0] + CROP) & (ppm <= ppm[-1] - CROP)
    grid, spectra = ppm[keep], spectra[:, keep]
    if len(grid) < 10 or np.ptp(grid) < .27:
        parser.error("Insufficient common spectral support.")
    args.out.mkdir(parents=True, exist_ok=True)
    groups, group = np.unique(truth, axis=0, return_inverse=True)
    group = group.ravel()
    counts = np.bincount(group)
    nonzero = np.any(truth != 0, axis=1)

    def equal_group(values, subset=None):
        means = np.bincount(group, weights=values, minlength=len(counts)) / counts
        return means.mean() if subset is None else means[subset].mean()

    # Negative interpolant argument means a positive shift of the template.
    # Shift only the templates to avoid smoothing the observed spectra.
    max_ticks = GLOBAL_TICKS + LOCAL_TICKS
    bank = np.array([
        np.column_stack([np.interp(grid - tick * STEP, ppm, r) for r in refs.T])
        for tick in range(-max_ticks, max_ticks + 1)
    ])
    # Place zero first for reproducible residual ties.
    ticks = np.r_[0, np.column_stack((np.arange(1, GLOBAL_TICKS + 1),
                                    -np.arange(1, GLOBAL_TICKS + 1))).ravel()]
    full_fit, index, _ = fit_bank(spectra, bank[ticks + max_ticks])
    global_shift = ticks[index]
    strata = [(int(t), np.flatnonzero(global_shift == t)) for t in np.unique(global_shift)]
    at_bound = np.abs(global_shift) == GLOBAL_TICKS
    print(f"Spectra {len(truth)}; groups {len(groups)}; points {len(grid)}; "
          f"common crop {CROP:.3f} ppm per edge")
    for label, mask in (("nonzero", nonzero), ("zero", ~nonzero)):
        if mask.any():
            q = np.quantile(global_shift[mask] * STEP, [.05, .5, .95])
            print(f"Global shift {label}: P05/median/P95 {q}; "
                  f"boundary {at_bound[mask].sum()}/{mask.sum()}")
    write_csv(args.out / "global_shifts.csv",
              ("row", "group", "nonzero", "global_shift_ppm", "global_bound"),
              ((i, group[i], int(nonzero[i]), t * STEP, int(at_bound[i]))
               for i, t in enumerate(global_shift)))
    fixed_full, _, _ = fit_bank(spectra, bank[max_ticks:max_ticks + 1])
    for label, fit in (("unaligned", fixed_full), ("globally aligned", full_fit)):
        print(f"Full support {label} equal-group MAE: "
              f"{equal_group(np.abs(fit - truth).mean(axis=1)):.4f} mM")

    def evaluate(masks):
        x = np.concatenate([center(bank[:, m], 1) for m in masks], axis=1)
        y = np.hstack([center(spectra[:, m], 1) for m in masks])
        fixed, _, _ = fit_bank(y, x[max_ticks:max_ticks + 1])
        aligned, local = np.empty_like(fixed), np.empty_like(fixed)
        deviations = np.empty(len(truth), dtype=int)
        offsets = np.r_[0, np.column_stack((np.arange(1, LOCAL_TICKS + 1),
                                           -np.arange(1, LOCAL_TICKS + 1))).ravel()]
        for tick, rows in strata:
            ix = tick + max_ticks
            fit, chosen, _ = fit_bank(y[rows], x[ix + offsets])
            # Fit the globally aligned arm at the same center as the local search.
            aligned[rows] = fit_bank(y[rows], x[ix:ix + 1])[0]
            local[rows] = fit
            deviations[rows] = offsets[chosen]
        errors = [np.abs(fit - truth).mean(axis=1) for fit in (fixed, aligned, local)]
        return [equal_group(e) for e in errors], deviations

    x_zero = bank[max_ticks]
    derivative = -np.gradient(refs, ppm, axis=0)[keep]
    edges = grid[0] + np.arange(1 + int(np.floor(np.ptp(grid) / .01 + 1e-9))) * .01
    windows = []
    for i in range(len(edges) - 9):
        mask = (grid >= edges[i]) & (grid < edges[i + 9])
        if mask.sum() >= 5:
            x = center(x_zero[mask], 0)
            dx = center(derivative[mask], 0)
            windows.append((i % 9, edges[i], mask, x, *reference_scores(x, dx)))
    nonzero_groups = np.any(groups != 0, axis=1)
    rows, rankings, paired, summaries = [], [], [], []
    for origin in range(9):
        tiles = [w for w in windows if w[0] == origin and np.isfinite(w[4:]).all()]
        losses = []
        for _, left, mask, _, ordinary, sensitivity in tiles:
            mae, deviations = evaluate([mask])
            losses.append([ordinary, sensitivity, *mae])
            boundary = np.abs(deviations) == LOCAL_TICKS
            rate = equal_group(boundary.astype(float), nonzero_groups) if nonzero.any() else np.nan
            rows.append((origin, left, ordinary, sensitivity, *mae, rate))
        a = np.asarray(losses).reshape(-1, 5)
        if len(a) >= 3:
            for i, mode in enumerate(MODES):
                error = a[:, i + 2]
                rho = [spearmanr(a[:, col], error).statistic
                       if np.ptp(a[:, col]) and np.ptp(error) else np.nan for col in (0, 1)]
                rankings.append((origin, mode, len(a), *rho, rho[1] - rho[0]))
        for _, left, mask, x, *_ in tiles:
            candidates = []
            for _, ml, mm, mx, *_ in tiles:
                if ml == left:
                    continue
                singular = np.linalg.svd(np.vstack((x, mx)), compute_uv=False)
                if singular[-1] > singular[0] * 1e-6:
                    candidates.append((np.sum(1 / singular**2), -np.sum(mx**2), ml, mm))
            if not candidates:
                continue
            greedy = min(candidates, key=lambda c: (c[0], c[2]))
            amplitude = min(candidates, key=lambda c: (c[1], c[2]))
            for name, (_, _, ml, mm) in (("greedy", greedy), ("amplitude", amplitude)):
                mae, deviations = evaluate([mask, mm])
                boundary = np.abs(deviations) == LOCAL_TICKS
                rate = equal_group(boundary.astype(float), nonzero_groups) if nonzero.any() else np.nan
                paired.append((origin, left, ml, name, *mae, rate))
        for name in ("greedy", "amplitude"):
            subset = np.asarray([r[4:] for r in paired if r[0] == origin and r[3] == name])
            if len(subset):
                summaries.append((origin, name, len(subset), *subset.mean(axis=0)))
    write_csv(args.out / "target_errors.csv",
              ("origin", "left_ppm", "ordinary_score", "derivative_score",
               *MODES, "nonzero_group_local_bound_fraction"), rows)
    write_csv(args.out / "ranking.csv",
              ("origin", "mode", "targets", "ordinary_rho", "derivative_rho",
               "derivative_minus_ordinary"), rankings)
    write_csv(args.out / "paired_one_marker.csv",
              ("origin", "target_left_ppm", "marker_left_ppm", "method", *MODES,
               "nonzero_group_local_bound_fraction"), paired)
    write_csv(args.out / "one_marker_summary.csv",
              ("origin", "method", "targets", *MODES,
               "nonzero_group_local_bound_fraction"), summaries)
    for mode in MODES:
        valid = [r for r in rankings if r[1] == mode and np.isfinite(r[-1])]
        primary = next((r for r in valid if r[0] == 0), None)
        if primary:
            print(f"Ranking {mode}: primary ordinary {primary[3]:.3f}, "
                  f"derivative {primary[4]:.3f} (n={primary[2]})")
        if valid:
            print(f"  Derivative better on {sum(r[-1] > 0 for r in valid)}/{len(valid)} origins; "
                  f"median paired rho difference {np.median([r[-1] for r in valid]):+.3f}")
    for origin, name, n, raw, aligned, local, bound in summaries:
        if origin == 0:
            print(f"One marker {name} (n={n}): unaligned {raw:.3f}, globally aligned {aligned:.3f}, "
                  f"local ±0.006 ppm {local:.3f} mM; nonzero local boundary {bound:.1%}")


if __name__ == "__main__":
    main()
