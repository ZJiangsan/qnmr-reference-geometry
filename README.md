# Reference Geometry and Restricted Window Error in Quantitative Proton NMR after Global Alignment

This archive contains the analysis scripts for the Copenhagen experiment and the Sagmeister alignment and interference experiments. Input spectra are distributed by their original authors.

## Setup

Install NumPy, SciPy, scikit-learn, and Matplotlib in a Python environment. Run the following commands from the extracted directory.

## Copenhagen alcohol series

Obtain `NMR_mix_DoE.mat` from the University of Copenhagen Chemometrics Group dataset described by Winning et al., doi:10.1016/j.jmr.2007.10.005. Place it in `data/`. The scripts accept the original MATLAB file.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reference_window_design.py --mat data/NMR_mix_DoE.mat --out results/copenhagen
python window3_support.py --mat data/NMR_mix_DoE.mat --out results/copenhagen
python stress_shift.py --mat data/NMR_mix_DoE.mat --out results/copenhagen
python pls_baseline.py --mat data/NMR_mix_DoE.mat --out results/copenhagen
MPLBACKEND=Agg python plot_results.py results/copenhagen
```

## Sagmeister spectra

Download these files from [Sagmeister et al.](https://doi.org/10.5281/zenodo.6066166) into `data/raw_sagmeister/`:

- `NMR_pure_spectrum_2ClBA.npy`
- `NMR_pure_spectrum_3N-2ClBA.npy`
- `NMR_pure_spectrum_5N-2ClBA.npy`
- `NMR_experimental_mixtures_set.npy`
- `NMR_dynamic_set.npy`

```sh
python prepare_sagmeister.py --raw data/raw_sagmeister --out data
python sagmeister_global_alignment.py --input data/external_validation.npz --out results/global_alignment_mixtures
python sagmeister_global_alignment.py --input data/sagmeister_dynamic_validation.npz --out results/global_alignment_dynamic
python sagmeister_consensus_pilot.py --input data/sagmeister_dynamic_validation.npz --out results/consensus_pilot_dynamic
```

Each alignment directory contains `ranking.csv` with individual-origin correlations and `one_marker_summary.csv` with the primary-grid and offset-grid MAEs. The console reports the common-crop full-spectrum MAEs. The consensus pilot writes `summary.csv` and paired per-spectrum errors. It injects the interferents after deriving the clean-spectrum alignment.
