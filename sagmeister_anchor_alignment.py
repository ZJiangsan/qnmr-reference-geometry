#!/usr/bin/env python3
"""Reference-selected alignment anchors for the Sagmeister Table 1 benchmark."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from sagmeister_diagnostic import fit_bank, write_csv


STEP = .002
GLOBAL_TICKS = 40
LOCAL_TICKS = 3
CROP = (GLOBAL_TICKS + LOCAL_TICKS) * STEP


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
    parser.add_argument("--exclude-ppm", type=float, nargs=2, default=(7.42, 7.66))
    parser.add_argument("--random-paths", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.random_paths < 1:
        parser.error("Use at least one random anchor path.")
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
    _, index, _ = fit_bank(spectra, bank[ticks + max_ticks])
    global_shift = ticks[index]
    lo, hi = sorted(args.exclude_ppm)
    if not grid[0] < lo < hi < grid[-1]:
        parser.error("The excluded region must lie inside the common crop.")
    band = (grid >= lo) & (grid <= hi)
    masked_index = fit_bank(spectra[:, ~band], bank[ticks + max_ticks][:, ~band])[1]
    masked_shift = ticks[masked_index]
    group_weights = 1 / counts[group]
    if not np.any(nonzero):
        parser.error("Shift agreement requires at least one nonzero concentration group.")
    shift_weights = group_weights[nonzero]
    shift_weights /= shift_weights.sum()

    def stack(masks):
        x = np.concatenate([center(bank[:, m], 1) for m in masks], axis=1)
        y = np.hstack([center(spectra[:, m], 1) for m in masks])
        return x, y

    def estimate_shift(x, y):
        return ticks[fit_bank(y, x[ticks + max_ticks])[1]]

    def concentration_error(x, y, shift):
        error = np.empty(len(y))
        for tick in np.unique(shift):
            sel = shift == tick
            index = tick + max_ticks
            fit = fit_bank(y[sel], x[index:index + 1])[0]
            error[sel] = np.abs(fit - truth[sel]).mean(axis=1)
        return equal_group(error)

    def agreement(shift):
        delta = (shift[nonzero] - global_shift[nonzero]) * STEP
        return (shift_weights @ np.abs(delta),
                np.sqrt(shift_weights @ delta**2),
                shift_weights @ (np.abs(shift[nonzero]) == GLOBAL_TICKS))

    def shift_information(x, dx):
        signal = dx @ np.full(x.shape[1], 100.)
        return signal @ signal

    def gain(value, floor, full):
        return (value - floor) / (full - floor) if abs(full - floor) > 1e-12 else np.nan

    derivative = -np.gradient(refs, ppm, axis=0)[keep]
    edges = grid[0] + np.arange(1 + int(np.floor(np.ptp(grid) / .01 + 1e-9))) * .01
    windows = []
    for i in range(len(edges) - 9):
        mask = (grid >= edges[i]) & (grid < edges[i + 9])
        if mask.sum() >= 5:
            x = center(bank[max_ticks, mask], 0)
            dx = center(derivative[mask], 0)
            score, _ = reference_scores(x, dx)
            if np.isfinite(score):
                windows.append((i % 9, edges[i], mask, x, score, shift_information(x, dx)))
    rng = np.random.default_rng(args.seed)
    details, anchors, results = [], [], []
    for origin in range(9):
        tiles = [w for w in windows if w[0] == origin]
        masks = [w[2] for w in tiles]
        eligible = [i for i, mask in enumerate(masks) if not np.any(mask & band)]
        if len(eligible) < 3:
            parser.error(f"Fewer than three eligible anchor windows at origin {origin}.")
        anchor_orders = {
            "shift_information": [sorted(eligible, key=lambda i: (-tiles[i][5], i))],
            "amplitude": [sorted(eligible, key=lambda i: (-np.sum(tiles[i][3]**2), i))],
            "random": [rng.permutation(eligible) for _ in range(args.random_paths)],
        }
        common = {("unaligned", 0, 0): (np.zeros_like(global_shift), None),
                  ("full_alignment", 0, 0): (global_shift, None),
                  ("masked_alignment", 0, 0): (masked_shift, ~band)}
        for rule, paths in anchor_orders.items():
            for path, ordered in enumerate(paths):
                for k in (1, 2, 3):
                    subset = sorted(ordered[:k])
                    x, y = stack([masks[i] for i in subset])
                    anchor_mask = np.logical_or.reduce([masks[i] for i in subset])
                    common[(rule, k, path)] = (estimate_shift(x, y), anchor_mask)
                    anchors.append((origin, rule, k, path,
                                    " ".join(f"{tiles[i][1]:.6f}" for i in subset)))

        self_cache = {}
        origin_rows = []
        for i, tile in enumerate(tiles):
            candidates = []
            for j, marker in enumerate(tiles):
                if i == j:
                    continue
                sv = np.linalg.svd(np.vstack((tile[3], marker[3])), compute_uv=False)
                if sv[-1] > sv[0] * 1e-6:
                    candidates.append((np.sum(1 / sv**2), -np.sum(marker[3]**2), j))
            if not candidates:
                continue
            greedy = min(candidates, key=lambda c: (c[0], c[2]))[2]
            amplitude = min(candidates, key=lambda c: (c[1], c[2]))[2]
            for fit_rule, indexes in (("target_only", [i]), ("greedy_marker", [i, greedy]),
                                      ("amplitude_marker", [i, amplitude])):
                used_masks = [masks[j] for j in indexes]
                quant_mask = np.logical_or.reduce(used_masks)
                x, y = stack(used_masks)
                pair = tuple(sorted(indexes))
                if pair not in self_cache:
                    self_cache[pair] = estimate_shift(x, y)
                methods = common | {("self_alignment", 0, 0): (self_cache[pair], quant_mask)}
                for (arm, k, path), (shift, align_mask) in methods.items():
                    used = (quant_mask if arm == "unaligned" else
                            np.ones(len(grid), dtype=bool) if align_mask is None else
                            quant_mask | align_mask)
                    row = (origin, fit_rule, tile[1], tiles[indexes[-1]][1], tile[4],
                           int(not np.any(quant_mask & band)), arm, k, path, int(used.sum()),
                           concentration_error(x, y, shift), *agreement(shift))
                    origin_rows.append(row)
        details.extend(origin_rows)
        for support in ("table1", "band_free"):
            for fit_rule in ("target_only", "greedy_marker", "amplitude_marker"):
                by_method = {}
                for row in origin_rows:
                    if row[1] == fit_rule and (support == "table1" or row[5]):
                        by_method.setdefault(row[6:9], []).append(row)
                metrics = {}
                for key, selected in by_method.items():
                    a = np.asarray(
                        [[r[4], *r[9:]] for r in selected])
                    rho = (spearmanr(a[:, 0], a[:, 2]).statistic
                           if fit_rule == "target_only" and len(a) >= 3
                           and np.ptp(a[:, 0]) and np.ptp(a[:, 2]) else np.nan)
                    metrics[key] = (len(a), rho, *a[:, 1:].mean(axis=0))
                if not metrics:
                    continue
                _, rho0, _, mae0, *_ = metrics[("unaligned", 0, 0)]
                _, rho1, _, mae1, *_ = metrics[("full_alignment", 0, 0)]
                for (arm, k, path), (n, rho, points, mae, delta, rmse, hit) in metrics.items():
                    results.append((support, origin, fit_rule, arm, k, path, n, rho,
                                    mae, delta, rmse, hit, points, gain(rho, rho0, rho1),
                                    gain(mae, mae0, mae1)))
        print(f"Finished origin {origin}: {len(tiles)} targets.", flush=True)

    write_csv(args.out / "target_results.csv",
              ("origin", "fit_rule", "target_ppm", "marker_or_target_ppm", "ordinary_score",
               "quantification_band_free", "arm", "anchors", "path", "union_points",
               "mae_mM", "shift_mae_vs_full_ppm", "shift_rmse_vs_full_ppm", "shift_bound_fraction"),
              details)
    write_csv(args.out / "anchor_windows.csv",
              ("origin", "arm", "anchors", "path", "left_edges_ppm"), anchors)
    write_csv(args.out / "per_origin.csv",
              ("support", "origin", "fit_rule", "arm", "anchors", "path", "targets",
               "ordinary_rho", "mae_mM", "shift_mae_vs_full_ppm", "shift_rmse_vs_full_ppm",
               "shift_bound_fraction", "mean_union_points", "rho_gain_fraction", "mae_gain_fraction"),
              results)
    summaries = []
    def median_finite(x):
        finite = x[np.isfinite(x)]
        return np.median(finite) if len(finite) else np.nan

    for support, fit_rule, arm, k in sorted({(r[0], r[2], r[3], r[4]) for r in results}):
        grouped = {}
        for r in results:
            if (r[0], r[2], r[3], r[4]) == (support, fit_rule, arm, k):
                grouped.setdefault(r[1], []).append(r[7:])
        # Random paths have equal weight within an origin. Origins are descriptive
        # sensitivity checks because the underlying spectral windows overlap.
        a = np.stack([np.array([median_finite(c) for c in np.array(v).T])
                      for v in grouped.values()])
        medians = [median_finite(c) for c in a.T]
        summaries.append((support, fit_rule, arm, k, len(a), *medians))
    write_csv(args.out / "summary.csv",
              ("support", "fit_rule", "arm", "anchors", "origins", "median_ordinary_rho",
               "median_mae_mM", "median_shift_mae_vs_full_ppm", "median_shift_rmse_vs_full_ppm",
               "median_shift_bound_fraction", "median_mean_union_points",
               "median_rho_gain_fraction", "median_mae_gain_fraction"), summaries)

    for r in summaries:
        if r[0] == "table1" and r[1] in ("target_only", "greedy_marker"):
            print(f"{r[1]:14} {r[2]:19} k={r[3]} | rho {r[5]:.3f} | "
                  f"MAE {r[6]:.2f} mM | shift MAE {r[7]:.4f} ppm")
    print(f"Outputs: {args.out}")


if __name__ == "__main__":
    main()
