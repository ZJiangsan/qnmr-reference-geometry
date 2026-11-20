"""Apply a shared reference-axis offset after selecting windows on clean references."""
import argparse
import csv
from pathlib import Path
import numpy as np

from copenhagen_nmr_benchmark import centered, get_data, window, write_csv
from reference_window_design import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat', type=Path, default=Path('data/NMR_mix_DoE.mat'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with (args.out/'selected_markers.csv').open(newline='') as f:
        picks = {}
        for row in csv.DictReader(f):
            if row['strategy'] == 'greedy' and int(row['step']) <= 3:
                picks.setdefault(float(row['target_ppm']), []).append(float(row['marker_ppm']))
    data = get_data(args.mat)
    x = data['Axis']
    truth = data['yData']/100.
    pure = [np.flatnonzero(truth[:, i] == 1).item() for i in range(3)]
    test = np.flatnonzero(~np.isin(np.arange(len(truth)), pure))
    refs = data['xData'][pure].T
    measured = data['xData'][test].T
    valid = (x > x.min()+.02) & (x < x.max()-.02)
    normalization = np.linalg.norm(centered(refs))
    refs /= normalization
    measured /= normalization
    rows = []
    for shift in (-.02,-.01,-.005,0,.005,.01,.02):
        moved = np.column_stack([np.interp(x-shift,x[::-1],r[::-1]) for r in refs.T])
        full = centered(moved[valid])
        full_pred = predict(full.T@full, full.T@centered(measured[valid]))
        maes = []
        for target, markers in picks.items():
            rr = [window(x,c,.045) for c in [target,*markers]]
            rr = [p[valid[p]] for p in rr]
            design = np.vstack([centered(moved[p]) for p in rr])
            observed = np.vstack([centered(measured[p]) for p in rr])
            predicted = predict(design.T@design,design.T@observed)
            maes.append(np.mean(100*abs(predicted-truth[test])))
        full_mae = np.mean(100*abs(full_pred-truth[test]))
        rows.append((shift, full_mae, np.mean(maes), sum(mae<full_mae for mae in maes),len(maes)))
        print(rows[-1])
    write_csv(args.out/'shift_stress.csv',
              ['reference_shift_ppm','full_mae_pp','greedy_three_mean_mae_pp','greedy_wins','windows'],rows)


if __name__=='__main__':
    main()
