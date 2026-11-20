#!/usr/bin/env python3
"""Evaluate marker constraints on the Copenhagen measured ternary NMR mixtures."""
import argparse
import csv
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear, nnls
from scipy.signal import find_peaks

URL = "https://sid.erda.dk/share_redirect/hbrCZIJyig/Mixture%20design%20NMR.zip"
MAT_NAME = "NMR_mix_DoE.mat"


def get_data(path):
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = path.with_suffix(".zip")
        with urlopen(URL, timeout=60) as response:
            archive.write_bytes(response.read())
        with ZipFile(archive) as zipped:
            contents = zipped.read(MAT_NAME)
        path.write_bytes(contents)
    return loadmat(path, simplify_cells=True)


def centered(values):
    return values - values.mean(axis=0)


def window(ppm, position, half_width):
    return np.flatnonzero(np.abs(ppm - position) <= half_width)


def choose_windows(ppm, refs):
    # Only pure spectra are used for window selection.
    smooth = gaussian_filter1d(refs, 3, axis=0)
    limits = np.max(np.abs(smooth), axis=0)
    candidates = set()
    for i in range(refs.shape[1]):
        peaks, _ = find_peaks(smooth[:, i], prominence=0.015 * limits[i])
        candidates.update(peaks.tolist())
    overlaps = []
    for idx in sorted(candidates):
        if not 0.95 <= ppm[idx] <= 1.85:
            continue
        points = window(ppm, ppm[idx], .045)
        segment = centered(refs[points])
        strength = np.sqrt(np.mean(segment**2, axis=0)) / limits
        for i, j in ((0, 1), (0, 2), (1, 2)):
            similarity = np.dot(segment[:, i], segment[:, j]) / max(
                np.linalg.norm(segment[:, i]) * np.linalg.norm(segment[:, j]), 1e-12)
            overlaps.append((similarity * min(strength[i], strength[j]),
                             idx, i, j, similarity))
    _, target, first, second, similarity = max(overlaps)
    target_points = window(ppm, ppm[target], .045)

    available = set(candidates)
    markers = []
    for component in range(refs.shape[1]):
        scored = []
        for idx in sorted(available):
            points = window(ppm, ppm[idx], .025)
            if np.intersect1d(points, target_points).size:
                continue
            power = np.linalg.norm(centered(refs[points]) / limits, axis=0)
            own = power[component]
            other = np.linalg.norm(np.delete(power, component))
            scored.append((own * own / max(own + 3 * other, 1e-12), idx))
        _, marker = max(scored)
        markers.append(window(ppm, ppm[marker], .025))
        available = {idx for idx in available if abs(ppm[idx] - ppm[marker]) > .05}
    return target_points, markers, (first, second, similarity)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def normalize(scales):
    total = scales.sum()
    return scales / total if total > 0 else np.full(scales.size, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=Path("data/NMR_mix_DoE.mat"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = get_data(args.mat)
    ppm = np.asarray(data["Axis"], float)
    spectra = np.asarray(data["xData"], float)
    truth = np.asarray(data["yData"], float) / 100
    samples = np.asarray(data["ObjLabels"])
    names = [str(name).strip() for name in data["yLabels"]]
    if spectra.shape != (len(truth), len(ppm)) or truth.shape != (len(samples), 3):
        parser.error("Unexpected spectral or label dimensions.")
    if not np.allclose(truth.sum(axis=1), 1):
        parser.error("Composition rows must sum to 100%.")
    endpoints = [np.flatnonzero(np.isclose(truth[:, i], 1)) for i in range(3)]
    if any(len(index) != 1 for index in endpoints):
        parser.error("Expected one pure endpoint per component.")
    pure = np.array([index[0] for index in endpoints])
    refs = spectra[pure].T
    target, markers, pair = choose_windows(ppm, refs)

    full_x = centered(refs)
    local_x = centered(refs[target])
    marker_x = np.vstack([centered(refs[region]) for region in markers])
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "windows.csv", ["role", "low_ppm", "high_ppm"],
              [(f"overlap ({names[pair[0]]}, {names[pair[1]]})",
                ppm[target].min(), ppm[target].max()),
               *[(f"marker {name}", ppm[region].min(), ppm[region].max())
                 for name, region in zip(names, markers)]])
    print(f"Pure endpoint sample IDs: {samples[pure].tolist()}")
    print(f"Overlap {ppm[target].min():.4f}–{ppm[target].max():.4f} ppm, "
          f"pure-reference correlation {pair[2]:.3f}")
    print(f"Overlap template condition number: {np.linalg.cond(local_x):.2f}")

    rows = []
    predictions = {}
    test = np.flatnonzero(~np.isin(np.arange(len(samples)), pure))
    for i in test:
        y = spectra[i]
        local_y = centered(y[target])
        marker_y = np.concatenate([centered(y[region]) for region in markers])
        prior = nnls(marker_x, marker_y)[0]
        fits = {
            "full_spectrum": nnls(full_x, centered(y))[0],
            "marker_only": prior,
            "overlap_only": nnls(local_x, local_y)[0],
        }
        # Near-zero priors get a 0.005 scale tolerance so the hard fit can
        # recover a weak component missed by the marker fit.
        margin = np.maximum(0.10 * prior, 0.005)
        fits["hard_marker"] = lsq_linear(
            local_x, local_y, bounds=(np.maximum(prior-margin, 0), prior+margin)).x
        # Fixed pilot setting, chosen independently of blend composition labels.
        # Relative width 25%, with a five-percentage-point scale floor.
        scale_sd = 0.25 * np.maximum(prior, 0.05)
        weight = 0.05 * max(np.linalg.norm(local_y), 1e-8) / scale_sd
        fits["soft_marker"] = nnls(
            np.vstack((local_x, np.diag(weight))),
            np.concatenate((local_y, weight * prior)))[0]
        category = "binary" if (truth[i] > 0).sum() == 2 else "ternary"
        for method, scales in fits.items():
            proportion = normalize(scales)
            predictions.setdefault(method, []).append(proportion)
            local_rmse = np.sqrt(np.mean((local_x @ scales - local_y)**2))
            rows.append((samples[i], category, method, *truth[i], *proportion,
                         *scales, local_rmse))

    write_csv(args.out / "predictions.csv",
              ["sample", "mixture_type", "method",
               *[f"true_fraction_{n}" for n in names],
               *[f"predicted_fraction_{n}" for n in names],
               *[f"raw_scale_{n}" for n in names], "overlap_rmse"], rows)
    actual = truth[test]
    masks = {
        "all_components": np.ones(actual.shape, bool),
        "present_5_to_15_percent": (actual >= .05) & (actual <= .15),
        "absent_components": actual == 0,
    }
    summary = []
    for method, values in predictions.items():
        estimated = np.asarray(values)
        error = 100 * (estimated - actual)
        for group, mask in masks.items():
            scored = error[mask & np.isfinite(error)]
            summary.append((method, group, mask.sum(), len(scored),
                            np.mean(np.abs(scored)) if len(scored) else np.nan,
                            np.sqrt(np.mean(scored**2)) if len(scored) else np.nan))
    columns = ["method", "subset", "expected", "scored", "mae_percentage_points",
               "rmse_percentage_points"]
    write_csv(args.out / "summary.csv", columns, summary)
    print(f"Scored {len(test)} measured blends. Outputs: {args.out}")
    for method, group, _, scored, mae, rmse in summary:
        print(f"{method:<15} {group:<25} n={scored:3} MAE={mae:6.3f} RMSE={rmse:6.3f}")


if __name__ == "__main__":
    main()
