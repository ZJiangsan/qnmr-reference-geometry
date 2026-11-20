#!/usr/bin/env python3
"""Benchmark reference-only NMR window selection on aligned, measured mixtures.

Input NPZ arrays: ppm (P,), references (P,C) in signal per mM,
spectra (N,P), and truth (N,C) in mM. Use matched differential spectra
and known increments for spike-ins with unknown endogenous concentrations.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def block(refs, spectra, mask):
    x, y = refs[mask], spectra[:, mask]
    return x - x.mean(axis=0), y - y.mean(axis=1, keepdims=True)


def score(gram):
    """A-optimal concentration variance proxy per unit spectral noise."""
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        return np.inf
    return float(np.sum(1 / eigenvalues))


def absolute_error(x, y, truth):
    predictions = np.stack([nnls(x, spectrum)[0] for spectrum in y])
    return np.abs(predictions - truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=float, default=0.09)
    parser.add_argument("--stride", type=float, default=0.01)
    parser.add_argument("--max-markers", type=int, default=3)
    parser.add_argument("--random-repeats", type=int, default=16)
    parser.add_argument("--exclude", nargs=2, type=float, action="append",
                        default=[], metavar=("LOW", "HIGH"),
                        help="Exclude a ppm interval. May be repeated.")
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.width, args.stride)):
        parser.error("--width and --stride must be finite and positive.")
    if args.max_markers < 0 or args.random_repeats < 1:
        parser.error("Use nonnegative --max-markers and positive --random-repeats.")

    keys = ("ppm", "references", "spectra", "truth")
    with np.load(args.input, allow_pickle=False) as data:
        missing = [key for key in keys if key not in data]
        if missing:
            parser.error(f"Missing NPZ arrays: {', '.join(missing)}")
        ppm, refs, spectra, truth = (np.asarray(data[key], dtype=float) for key in keys)
    if ppm.ndim != 1 or any(x.ndim != 2 for x in (refs, spectra, truth)):
        parser.error("Expected a 1D ppm axis and three 2D arrays.")
    if refs.shape[0] != ppm.size or spectra.shape[1] != ppm.size:
        parser.error("References and spectra must share the ppm axis.")
    if truth.shape != (len(spectra), refs.shape[1]) or not truth.size:
        parser.error("Truth must provide one concentration per sample and component.")
    if not all(np.isfinite(x).all() for x in (ppm, refs, spectra, truth)):
        parser.error("Input arrays must be finite.")
    if np.any(truth < 0):
        parser.error("Known concentrations or spike-in increments must be nonnegative.")
    delta = np.diff(ppm)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        parser.error("The ppm axis must be strictly increasing or decreasing.")

    allowed = np.ones(ppm.size, dtype=bool)
    min_points = refs.shape[1] + 2
    for low, high in args.exclude:
        low, high = sorted((low, high))
        allowed &= ~((ppm >= low) & (ppm <= high))
    if allowed.sum() < min_points:
        parser.error("Too few points remain after exclusions.")

    half = args.width / 2

    def build_windows(stride):
        start = ppm.min()
        available_span = np.ptp(ppm) - args.width
        if available_span < 0:
            return {}
        # Shared edges keep adjacent marker windows from counting any point twice.
        count = int(np.floor(available_span / stride + 1e-10)) + 1
        edges = start + np.arange(count + 1) * stride
        windows = {}
        for i in range(count):
            left = edges[i]
            right = edges[i + 1] if stride == args.width else left + args.width
            mask = (ppm >= left) & (ppm < right)
            if mask.sum() < min_points or np.any(mask & ~allowed):
                continue
            x, y = block(refs, spectra, mask)
            gram = x.T @ x
            if np.trace(gram) > 0:
                windows[float(left + half)] = (x, y, gram)
        return windows

    targets, markers = build_windows(args.stride), build_windows(args.width)
    if not targets:
        parser.error("No usable target windows. Check the axis and window width.")
    full_x, full_y = block(refs, spectra, allowed)
    full_mae = float(absolute_error(full_x, full_y, truth).mean())
    rng = np.random.default_rng(20261119)
    diagnostics, curves, choices = [], [], []
    identity = np.eye(refs.shape[1])

    for target, (x, y, gram) in targets.items():
        loss = absolute_error(x, y, truth)
        target_mae = float(loss.mean())
        diagnostics.append([target, score(gram), target_mae, *loss.mean(axis=0)])
        candidates = [c for c in markers if abs(c - target) > args.width + 1e-9]
        if len(candidates) < args.max_markers:
            continue

        for method, repeats in (
            ("greedy", 1), ("amplitude", 1), ("random", args.random_repeats)
        ):
            for repeat in range(repeats):
                pool = candidates.copy()
                xs, ys = [x], [y]
                information = gram.copy()
                curves.append([target, method, repeat, 0, target_mae])
                for budget in range(1, args.max_markers + 1):
                    if method == "greedy":
                        def objective(center):
                            g = information + markers[center][2]
                            ridge = max(np.trace(g) * 1e-10, np.finfo(float).tiny)
                            return score(g + ridge * identity)

                        chosen = min(pool, key=objective)
                    elif method == "amplitude":
                        chosen = max(pool, key=lambda c: np.trace(markers[c][2]))
                    else:
                        chosen = pool[int(rng.integers(len(pool)))]
                    pool.remove(chosen)
                    mx, my, mg = markers[chosen]
                    xs.append(mx)
                    ys.append(my)
                    information += mg
                    error = absolute_error(np.vstack(xs), np.hstack(ys), truth)
                    curves.append([target, method, repeat, budget, error.mean()])
                    choices.append([target, method, repeat, budget, chosen])

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "window_diagnostics.csv",
              ["target_ppm", "variance_proxy", "mae_mM"] +
              [f"mae_component_{i + 1}_mM" for i in range(refs.shape[1])], diagnostics)
    write_csv(args.out / "marker_curves.csv",
              ["target_ppm", "method", "repeat", "markers", "mae_mM"], curves)
    write_csv(args.out / "selected_markers.csv",
              ["target_ppm", "method", "repeat", "step", "marker_ppm"], choices)

    finite = np.asarray([row[:3] for row in diagnostics if np.isfinite(row[1])])
    print(f"Full usable-spectrum NNLS MAE: {full_mae:.4f} mM")
    print(f"Target windows: {len(diagnostics)}")
    print(f"Rank-deficient or numerically ill-conditioned: {len(diagnostics) - len(finite)}")
    if len(finite) >= 3 and np.ptp(finite[:, 1]) and np.ptp(finite[:, 2]):
        rho = spearmanr(finite[:, 1], finite[:, 2]).statistic
        print(f"Reference-only score vs measured MAE: Spearman rho = {rho:.4f}")
    if not curves:
        print("No targets support the requested marker budget. Reduce --max-markers.")
    for budget in range(args.max_markers + 1):
        for method in ("greedy", "amplitude", "random"):
            values = [row[4] for row in curves if row[1] == method and row[3] == budget]
            if values:
                print(f"{budget} markers | {method:9s} | MAE {np.mean(values):.4f} mM")
    print(f"CSV outputs: {args.out}")


if __name__ == "__main__":
    main()
