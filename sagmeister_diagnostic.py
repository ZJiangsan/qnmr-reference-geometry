#!/usr/bin/env python3
"""Audit Sagmeister reference mismatch and fixed one-marker selections.

Run beside external_nmr_validation.py. Shift estimation uses spectral data
within each fitted region only. Concentration labels are used for scoring.
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from external_nmr_validation import write_csv


def fit_bank(y, templates):
    """Profile a shared reference shift using exact active-set enumeration."""
    best = np.full(len(y), np.inf)
    estimates = np.zeros((len(y), templates[0].shape[1]))
    shifts = np.zeros(len(y), dtype=int)
    y = y - y.mean(axis=1, keepdims=True)
    for index, raw in enumerate(templates):
        x = raw - raw.mean(axis=0)
        residual = np.einsum("ij,ij->i", y, y)
        coefficients = np.zeros_like(estimates)
        for count in range(1, x.shape[1] + 1):
            for columns in combinations(range(x.shape[1]), count):
                design = x[:, columns]
                inverse = np.linalg.pinv(design)
                candidate = y @ inverse.T
                # At a feasible least-squares solution, reduction in SSE
                # equals the inner product of coefficients and X^T y.
                rss = np.einsum("ij,ij->i", candidate, y @ design)
                sse = np.einsum("ij,ij->i", y, y) - rss
                wins = (candidate >= 0).all(axis=1) & (sse < residual)
                residual[wins] = sse[wins]
                coefficients[wins] = 0
                coefficients[np.ix_(wins, columns)] = candidate[wins]
        wins = residual < best
        best[wins] = residual[wins]
        estimates[wins] = coefficients[wins]
        shifts[wins] = index
    return estimates, shifts, np.sqrt(np.maximum(best, 0) / y.shape[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/external_validation.npz"))
    parser.add_argument("--benchmark", type=Path, default=Path("results/sagmeister_validation"))
    parser.add_argument("--out", type=Path, default=Path("results/sagmeister_diagnostic"))
    parser.add_argument("--width", type=float, default=0.09)
    parser.add_argument("--stride", type=float, default=0.01)
    parser.add_argument("--max-shift", type=float, default=0.02)
    parser.add_argument("--shift-step", type=float, default=0.002)
    parser.add_argument("--exclude", nargs=2, type=float, action="append", default=[])
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.width, args.stride, args.max_shift, args.shift_step)):
        parser.error("Width, stride, maximum shift and shift step must be positive and finite.")
    with np.load(args.input, allow_pickle=False) as data:
        ppm, refs, spectra, truth = (np.asarray(data[k], dtype=float)
                                     for k in ("ppm", "references", "spectra", "truth"))
        supplied_groups = data["groups"] if "groups" in data else None
    if refs.shape != (ppm.size, 3) or spectra.shape != (len(truth), ppm.size) or truth.shape[1] != 3:
        parser.error("Expected a common ppm grid and three Sagmeister components.")
    if not all(np.isfinite(a).all() for a in (ppm, refs, spectra, truth)):
        parser.error("Input arrays must be finite.")
    order = np.argsort(ppm)
    ppm, refs, spectra = ppm[order], refs[order], spectra[:, order]
    if np.any(np.diff(ppm) <= 0):
        parser.error("The ppm grid has duplicate points.")
    groups, membership = np.unique(
        truth if supplied_groups is None else supplied_groups, axis=0, return_inverse=True)
    membership = membership.ravel()
    if membership.shape != (len(truth),):
        parser.error("Optional groups must contain one group ID per spectrum.")
    counts = np.bincount(membership)
    group_rule = "exact concentration vectors" if supplied_groups is None else "NPZ groups"
    offsets = np.arange(1, int(np.floor(args.max_shift / args.shift_step + 1e-9)) + 1)
    offsets = offsets * args.shift_step
    shifts = np.r_[0., np.column_stack((offsets, -offsets)).ravel()]
    bank = np.array([
        np.column_stack([np.interp(ppm - shift, ppm, reference) for reference in refs.T])
        for shift in shifts
    ])
    allowed = np.ones(len(ppm), dtype=bool)
    for low, high in args.exclude:
        low, high = sorted((low, high))
        allowed &= ~((ppm >= low) & (ppm <= high))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Grid {ppm[0]:.6f}–{ppm[-1]:.6f} ppm, median spacing {np.median(np.diff(ppm)):.6g} ppm")
    print(f"{len(groups)} groups by {group_rule}, {np.sum(counts > 1)} with repeated rows")
    print("Reference peak-to-peak signal per mM:", np.ptp(refs, axis=0))

    components = ("2ClBA", "3N-2ClBA", "5N-2ClBA")
    full = [fit_bank(spectra[:, allowed], candidate[:, allowed])
            for candidate in (bank[:1], bank)]
    full_rows = []
    sample_rows = []
    for mode, (prediction, shift_index, rms) in zip(("fixed", "shifted"), full):
        errors = prediction - truth
        for name, error in zip(components, errors.T):
            full_rows.append((mode, name, np.abs(error).mean(), error.mean()))
        for row, (actual, fitted, idx, residual) in enumerate(zip(truth, prediction, shift_index, rms)):
            sample_rows.append((mode, row, membership[row], shifts[idx], residual, *actual, *fitted))
        print(f"Full spectrum {mode}: MAE {np.abs(errors).mean():.4f} mM, "
              f"median residual RMS {np.median(rms):.6g}")
    write_csv(args.out / "full_components.csv", ("fit", "component", "mae_mM", "bias_mM"), full_rows)
    write_csv(args.out / "full_samples.csv",
              ["fit", "row", "group", "shift_ppm", "residual_rms"]
              + [f"true_{c}_mM" for c in components]
              + [f"fit_{c}_mM" for c in components], sample_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    fixed_rms = full[0][2]
    representatives = np.argsort(fixed_rms)[np.rint(np.quantile(
        np.arange(len(truth)), [0.1, 0.5, 0.9])).astype(int)]
    for i, (name, row) in enumerate(zip(components, representatives)):
        ax = axes[0, i]
        for mode, (fitted, _, _) in zip(("Fixed", "Shifted"), full):
            ax.scatter(truth[:, i], fitted[:, i], s=5, alpha=0.35, label=mode)
        limit = max(truth[:, i].max(), *(result[0][:, i].max() for result in full))
        ax.plot([0, limit], [0, limit], "k--", linewidth=1)
        ax.set(title=name, xlabel="Known (mM)", ylabel="Fitted (mM)")
        ax = axes[1, i]
        axis = ppm[allowed]
        observed = spectra[row, allowed]
        ax.plot(axis, observed, label="Measured", linewidth=1)
        for mode, (fitted, indices, _) in zip(("Fixed", "Shifted"), full):
            modeled = bank[indices[row], allowed] @ fitted[row]
            modeled += (observed - modeled).mean()
            ax.plot(axis, modeled, label=mode, linewidth=1)
            ax.plot(axis, observed - modeled, "--", linewidth=0.7, label=f"{mode} residual")
        ax.set(title=f"Spectrum {row}", xlabel="Chemical shift (ppm)", ylabel="Signal")
        ax.invert_xaxis()
    axes[0, 0].legend()
    axes[1, 0].legend(fontsize=7)
    fig.savefig(args.out / "full_spectrum_diagnostic.png", dpi=170)
    plt.close(fig)

    def group_mean(values):
        return np.bincount(membership, weights=values, minlength=len(counts)) / counts

    def window(center, stride):
        start = ppm[0]
        index = round((center - args.width / 2 - start) / stride)
        left = start + np.float64(index) * stride
        if not np.isclose(left + args.width / 2, center, atol=1e-10, rtol=0):
            parser.error("Saved window centers do not match the configured width and stride.")
        right = (start + np.float64(index + 1) * stride if stride == args.width
                 else left + args.width)
        mask = (ppm >= left) & (ppm < right)
        if mask.sum() < refs.shape[1] + 2 or np.any(mask & ~allowed):
            parser.error("Saved windows do not match the input grid or exclusions.")
        return mask

    def window_errors(masks):
        y_blocks = [spectra[:, mask] for mask in masks]
        y = np.hstack([b - b.mean(axis=1, keepdims=True) for b in y_blocks])
        x_blocks = [bank[:, mask, :] for mask in masks]
        x = np.concatenate([b - b.mean(axis=1, keepdims=True) for b in x_blocks], axis=1)
        return [group_mean(np.abs(fit_bank(y, candidate)[0] - truth).mean(axis=1))
                for candidate in (x[:1], x)]

    with (args.benchmark / "selected_markers.csv").open(newline="") as handle:
        selected = {(float(r["target_ppm"]), r["method"]): float(r["marker_ppm"])
                    for r in csv.DictReader(handle)
                    if r["step"] == "1" and r["method"] in ("greedy", "amplitude")}
    with (args.benchmark / "window_diagnostics.csv").open(newline="") as handle:
        targets = {float(r["target_ppm"]): float(r["variance_proxy"])
                   for r in csv.DictReader(handle)}
    window_rows = []
    differences = [[], []]
    for target, proxy in targets.items():
        target_mask = window(target, args.stride)
        losses = window_errors([target_mask])
        window_rows.append([target, proxy, *(loss.mean() for loss in losses)])
        markers = [selected.get((target, method)) for method in ("greedy", "amplitude")]
        if None in markers:
            continue
        greedy = window_errors([target_mask, window(markers[0], args.width)])
        amplitude = greedy if markers[0] == markers[1] else window_errors(
            [target_mask, window(markers[1], args.width)])
        for mode in range(2):
            differences[mode].append(np.stack([greedy[mode], amplitude[mode]], axis=1))
    write_csv(args.out / "window_scores.csv",
              ("target_ppm", "variance_proxy", "fixed_group_mae_mM", "shifted_group_mae_mM"),
              window_rows)
    finite = np.asarray(window_rows)
    finite = finite[np.isfinite(finite[:, 1])]
    paired_rows = []
    for mode_index, mode in enumerate(("fixed", "shifted")):
        if len(finite) >= 3 and np.ptp(finite[:, 1]) and np.ptp(finite[:, mode_index + 2]):
            rho = spearmanr(finite[:, 1], finite[:, mode_index + 2]).statistic
            print(f"{mode}: reference score vs group-weighted MAE rho = {rho:.4f}")
        if differences[mode_index]:
            paired = np.mean(differences[mode_index], axis=0)
            delta = paired[:, 0] - paired[:, 1]
            print(f"{mode}: {len(differences[mode_index])} paired targets, equal-group one-marker "
                  f"MAE greedy {paired[:, 0].mean():.4f}, amplitude {paired[:, 1].mean():.4f}, "
                  f"greedy minus amplitude {delta.mean():+.4f} mM")
            paired_rows.extend((mode, i, counts[i], *values, delta[i])
                               for i, values in enumerate(paired))
    write_csv(args.out / "one_marker_paired_groups.csv",
              ("fit", "group", "spectra", "greedy_mae_mM", "amplitude_mae_mM", "delta_mM"),
              paired_rows)
    print(f"Outputs: {args.out}")


if __name__ == "__main__":
    main()
