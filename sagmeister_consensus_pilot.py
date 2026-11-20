#!/usr/bin/env python3
"""Pilot: unknown interferent positions after label-free full-spectrum alignment."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from sagmeister_diagnostic import fit_bank, write_csv


def center(x, axis):
    return x - x.mean(axis=axis, keepdims=True)


def robust_fit(x, y, iterations=12):
    """Huber IRLS with an unpenalized offset and nonnegative concentrations."""
    w = np.ones(len(y))
    for _ in range(iterations):
        x0 = np.average(x, axis=0, weights=w)
        y0 = np.average(y, weights=w)
        sw = np.sqrt(w)
        fit = nnls((x - x0) * sw[:, None], (y - y0) * sw)[0]
        residual = y - y0 - (x - x0) @ fit
        if _ == 0:
            mad = np.median(np.abs(residual - np.median(residual)))
            threshold = max(1.345 * 1.4826 * mad, np.finfo(float).eps)
        w = np.minimum(1., threshold / np.maximum(np.abs(residual), threshold))
    return fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--subset-count', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-per-group', type=int, default=1)
    args = parser.parse_args()
    if args.subset_count < 3 or args.subset_count % 2 == 0 or min(args.repeats,args.max_per_group)<1:
        parser.error('Use an odd subset count of at least three and positive repeats and group cap.')
    rng = np.random.default_rng(20261120)
    with np.load(args.input, allow_pickle=False) as data:
        ppm, refs, spectra, truth = (np.asarray(data[k], float) for k in
                                     ('ppm','references','spectra','truth'))
    if (ppm.ndim != 1 or refs.shape != (len(ppm),3) or truth.ndim != 2
        or truth.shape[1] != 3 or spectra.shape != (len(truth),len(ppm))
        or not len(truth) or not all(np.isfinite(a).all() for a in (ppm,refs,spectra,truth))):
        parser.error('Unexpected NPZ arrays.')
    order = np.argsort(ppm)
    ppm, refs, spectra = ppm[order], refs[order], spectra[:,order]
    if np.any(np.diff(ppm) <= 0):
        parser.error('The ppm axis must have unique coordinates.')
    grid = ppm[(ppm >= ppm[0]+.08) & (ppm <= ppm[-1]-.08)]
    if len(grid)<10:
        parser.error('Too few cropped points.')
    keep = np.isin(ppm,grid)
    groups, ids = np.unique(truth, axis=0, return_inverse=True)
    ids = ids.ravel()
    if not np.any(groups):
        parser.error('At least one nonzero concentration group is needed.')
    sample = np.sort(np.concatenate([
        rng.choice(rows, min(args.max_per_group,len(rows)), replace=False)
        for i in range(len(groups)) if np.any(groups[i])
        for rows in [np.flatnonzero(ids == i)]
    ]))
    y = spectra[sample][:,keep]
    target = truth[sample]
    group = ids[sample]
    _, group = np.unique(group, return_inverse=True)
    group = group.ravel()
    weights = 1 / np.bincount(group)[group]
    weights /= weights.sum()
    steps = np.arange(1,41)*.002
    shifts = np.r_[0.,np.column_stack((steps,-steps)).ravel()]
    bank = np.array([np.column_stack([np.interp(grid-s,ppm,r) for r in refs.T]) for s in shifts])
    _, shift_index, _ = fit_bank(y,bank)
    # Define identical observed-coordinate windows for every method and sample.
    edges = grid[0]+np.arange(1+int(np.floor(np.ptp(grid)/.09+1e-9)))*.09
    masks = [(grid>=lo)&(grid<hi) for lo,hi in zip(edges[:-1],edges[1:])]
    if len(masks)<3 or min(m.sum() for m in masks)<4:
        parser.error('Too few usable window points.')
    blocks = [center(bank[0,m],0) for m in masks]
    full_sv = np.linalg.svd(center(bank[0],0),compute_uv=False)
    # Require numerical identifiability and at least 5% of the full-design
    # weakest singular value. Search the same smallest subset size for both rules.
    for size in (1,2,3):
        eligible = []
        for pair in combinations(range(len(masks)),size):
            sv = np.linalg.svd(np.vstack([blocks[i] for i in pair]),compute_uv=False)
            if sv[-1]>0 and sv[-1]>=.05*full_sv[-1] and sv[0]/sv[-1]<=100:
                eligible.append((pair,np.sum(1/sv**2)))
        if len(eligible)>=args.subset_count:
            break
    if len(eligible)<args.subset_count:
        raise SystemExit('Insufficient identifiable subsets under the prespecified screen.')
    penalty_scale = min(v for _,v in eligible)
    used = np.zeros(len(masks),int)
    designed = []
    available = eligible.copy()
    for _ in range(args.subset_count):
        choice = min(available,key=lambda e:(e[1]+penalty_scale*used[list(e[0])].sum(),e[0]))
        available.remove(choice)
        designed.append(choice[0])
        used[list(choice[0])]+=1
    random_sets = [eligible[k][0] for k in rng.choice(len(eligible),args.subset_count,replace=False)]
    methods = {'designed':designed,'random':random_sets}
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'subsets.csv',('method','subset','window_left_ppm'),
              ((method,j,' '.join(f'{edges[k]:.5f}' for k in subset))
               for method,subsets in methods.items() for j,subset in enumerate(subsets)))
    print(f'{len(sample)} spectra from {len(np.unique(group))} nonzero groups; '
          f'{len(eligible)} eligible subsets of {size} windows')
    for name,subsets in methods.items():
        reuse = np.bincount([k for ss in subsets for k in ss], minlength=len(masks))
        print(f'{name}: {np.count_nonzero(reuse)} distinct windows, maximum reuse {reuse.max()}')
    strata = [(i,np.flatnonzero(shift_index==i)) for i in np.unique(shift_index)]
    designs = {ss: np.concatenate([center(bank[:,masks[k]],1) for k in ss],axis=1)
               for subsets in methods.values() for ss in subsets}

    def predict(obs):
        fixed = np.empty_like(target)
        for i,rows in strata:
            fixed[rows] = fit_bank(obs[rows],bank[i:i+1])[0]
        robust = np.array([robust_fit(bank[i],row) for row,i in zip(obs,shift_index)])
        estimates = [fixed, robust]
        for subsets in methods.values():
            fits = []
            for ss in subsets:
                fit = np.empty_like(target)
                v = np.hstack([center(obs[:,masks[k]],1) for k in ss])
                for i,rows in strata:
                    fit[rows] = fit_bank(v[rows],designs[ss][i:i+1])[0]
                fits.append(fit)
            estimates.append(np.median(fits,axis=0))
        return estimates

    # Fixed relative spectral scale. Random positions and shapes are paired
    # across methods, and locations are never given to a fitting method.
    scale = 100*np.median(np.ptp(refs,axis=0))
    if scale <= 0:
        parser.error('The reference spectral scale must be positive.')
    names = ('full_nnls','full_huber','designed_consensus','random_consensus')
    out, aggregate = [], []
    for count,strength in [(0,0.)]+[(k,s) for k in (1,2,3) for s in (.25,1.,4.)]:
        trials = 1 if count==0 else args.repeats
        errors = []
        for trial in range(trials):
            obs = y.copy()
            for _ in range(count):
                location = rng.uniform(grid[0]+.02,grid[-1]-.02,(len(y),1))
                width = rng.uniform(.0015,.004,(len(y),1))
                separation = rng.uniform(.004,.012,(len(y),1))
                peak = sum(np.exp(-.5*((grid-location-s*separation)/width)**2)
                           for s in (-.5,.5))
                obs += scale*strength*peak/peak.max(axis=1,keepdims=True)
            fits = predict(obs)
            error = np.stack([np.abs(fit-target).mean(axis=1) for fit in fits],axis=1)
            errors.append(error)
            out.extend((count,strength,trial,int(sample[i]),int(group[i]),*row)
                       for i,row in enumerate(error))
        matrix = np.vstack(errors)
        avg_weights = np.tile(weights/trials,trials)
        mean = avg_weights @ matrix
        for i,name in enumerate(names):
            order = np.argsort(matrix[:,i])
            index = np.searchsorted(np.cumsum(avg_weights[order]),.9)
            aggregate.append((count,strength,name,mean[i],matrix[order[index],i],
                              mean[i]-mean[1]))
        print(f'{count} peaks at {strength:g}× | '+', '.join(f'{k} {v:.3f} mM'
                                                            for k,v in zip(names,mean)))
    write_csv(args.out/'paired_errors.csv',
              ('interferents','relative_height','trial','source_row','group',*names),out)
    write_csv(args.out/'summary.csv',
              ('interferents','relative_height','method','mae_mM','p90_mM','mae_minus_huber_mM'),
              aggregate)


if __name__ == '__main__':
    main()
