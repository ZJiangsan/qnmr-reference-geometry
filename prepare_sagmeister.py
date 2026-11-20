#!/usr/bin/env python3
"""Convert the five raw Sagmeister NPY files into the benchmark NPZ format."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = ("2ClBA", "3N-2ClBA", "5N-2ClBA")
    ppm = np.linspace(7., 9., 1148)
    templates = []
    for i, name in enumerate(names):
        path = args.raw / f"NMR_pure_spectrum_{name}.npy"
        values = np.load(path, allow_pickle=False)
        if values.shape != (ppm.size + 3,) or not np.isfinite(values).all() or values[i] <= 0:
            parser.error(f"Unexpected reference format or concentration in {path}")
        templates.append(values[3:].astype(float) / (1000. * float(values[i])))
    references = np.column_stack(templates)
    args.out.mkdir(parents=True, exist_ok=True)
    for source, output, count in (
        ("NMR_experimental_mixtures_set.npy", "external_validation.npz", 1198),
        ("NMR_dynamic_set.npy", "sagmeister_dynamic_validation.npz", 1506),
    ):
        path = args.raw / source
        table = np.load(path, allow_pickle=False)
        if table.shape != (count, ppm.size + 3) or not np.isfinite(table).all():
            parser.error(f"Unexpected mixture array in {path}: {table.shape}")
        np.savez_compressed(args.out / output, ppm=ppm, references=references,
                            spectra=table[:, 3:], truth=1000. * table[:, :3].astype(float))
        print(args.out / output)
    print("Reference peak-to-peak signal per mM:", np.ptp(references, axis=0))


if __name__ == "__main__":
    main()
