"""Reference-only selection of NMR windows and evaluation on measured blends."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from copenhagen_nmr_benchmark import centered, get_data, window, write_csv

# Jacobian of a/sum(a) at equal composition and unit total signal.
J = np.eye(3) - np.ones((3,3))/3
SUPPORTS = [list(s) for k in range(1,4) for s in combinations(range(3),k)]


def predict(g, xy):
    """Exact active-set enumeration for three-component NNLS."""
    scores = np.zeros(xy.shape[1])
    fitted = np.zeros_like(xy)
    for selected in SUPPORTS:
        s = np.array(selected)
        block = g[np.ix_(s, s)]
        w = np.linalg.pinv(block, rcond=1e-12) @ xy[s]
        value = 2*np.sum(w*xy[s],axis=0)-np.sum(w*(block @ w),axis=0)
        take = (w >= -1e-10).all(axis=0) & (value > scores)
        fitted[:,take] = 0
        fitted[np.ix_(s,np.flatnonzero(take))] = np.maximum(w[:,take],0)
        scores[take] = value[take]
    total = fitted.sum(axis=0)
    if (total <= 0).any():
        return np.divide(fitted, total, out=np.full_like(fitted,np.nan),where=total>0).T
    return (fitted / total).T


def bounds(g):
    # Eigenvalue floor stabilizes ranking of numerically singular designs.
    eig, vec = np.linalg.eigh(g)
    cov = (vec / np.maximum(eig, 1e-12)) @ vec.T
    return np.maximum(np.diag(J @ cov @ J.T),0)


def direction_diagnostics(g):
    norms = np.sqrt(np.maximum(np.diag(g), 0))
    normalized = g / np.maximum(np.outer(norms,norms), 1e-30)
    return np.linalg.cond(normalized)**.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat',type=Path,default=Path('data/NMR_mix_DoE.mat'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    data=get_data(args.mat)
    ppm,spectra,truth=data['Axis'], data['xData'], data['yData']/100.
    names=[str(n).strip() for n in data['yLabels']]
    pure=[np.flatnonzero(truth[:,i]==1).item() for i in range(3)]
    ids=np.flatnonzero(~np.isin(np.arange(len(truth)),pure))
    truth=truth[ids]
    ref=spectra[pure].T
    power=np.linalg.norm(centered(ref),axis=0)
    intensity_scale=np.linalg.norm(centered(ref))
    ref=ref/intensity_scale
    measured=spectra[ids].T/intensity_scale
    covariates={}
    def block(center):
        if center not in covariates:
            points=window(ppm,center,.045)
            design=centered(ref[points])
            covariates[center]=(design.T@design,design.T@centered(measured[points]))
        return covariates[center]
    full_x=centered(ref)
    full_pred=predict(full_x.T@full_x,full_x.T@centered(measured))
    full_mae=np.mean(100*abs(full_pred-truth))
    candidate=np.round(np.arange(.70,3.80,.09),2)
    candidate_grams={c:block(c)[0] for c in candidate}
    rng=np.random.default_rng(20261119)
    diagnostics=[]; curves=[]; choices=[]
    for t in np.round(np.arange(.70,3.80,.01),2):
        g,xy=block(t)
        strengths=np.sqrt(np.diag(g))*intensity_scale/power
        if (strengths>=.015).sum()<2:
            continue
        variance=bounds(g)
        absolute=100*abs(predict(g,xy)-truth)
        diagnostics.append([t,*strengths,direction_diagnostics(g),*np.sqrt(variance),
                            *np.nanmean(absolute,axis=0),np.nanmean(absolute),
                            int(np.isfinite(absolute).all(axis=1).sum())])
        available=[c for c in candidate if abs(c-t)>.09+1e-8]
        for strategy,repeat in [('greedy',1),('amplitude',1),('random',16)]:
            for rep in range(repeat):
                active=list(available)
                total_g,total_xy=g.copy(),xy.copy()
                for k in range(5):
                    if k:
                        if strategy=='greedy':
                            c=min(active,key=lambda v: float(bounds(total_g+candidate_grams[v]).sum()))
                        elif strategy=='amplitude':
                            c=max(active,key=lambda v: np.trace(candidate_grams[v]))
                        else:
                            c=rng.choice(active)
                        active.remove(c)
                        gg,yy=block(c)
                        total_g+=gg
                        total_xy+=yy
                        choices.append([t,strategy,rep,k,c])
                    err=100*abs(predict(total_g,total_xy)-truth)
                    curves.append([t,strategy,rep,k,np.nanmean(err),
                                   int(np.isfinite(err).all(axis=1).sum())])
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'window_diagnostics.csv',
              ['target_ppm',*[f'relative_norm_{n}' for n in names],'shape_condition',
               *[f'predicted_sd_per_noise_{n}' for n in names],*[f'mae_pp_{n}' for n in names],
               'mae_pp','valid_blends'],diagnostics)
    write_csv(args.out/'selection_curves.csv',
              ['target_ppm','strategy','repeat','markers','mae_pp','valid_blends'],curves)
    write_csv(args.out/'selected_markers.csv',['target_ppm','strategy','repeat','step','marker_ppm'],choices)
    a=np.asarray(diagnostics,float)
    supported=(a[:,1:4]>=.015).all(axis=1)
    print(f'{len(a)} eligible windows, {sum(supported)} supporting all three components; full MAE {full_mae:.4f}')
    for label,ix in [('all',np.ones(len(a),bool)),('all components supported',supported)]:
        q=a[ix]
        for label2,mask in [('all errors',np.full(len(q),True))]:
            r=spearmanr(np.sqrt(np.sum(q[:,5:8]**2,axis=1))[mask],q[mask,11]).statistic
            print(f'{label}: predicted SD vs MAE Spearman {r:.3f}')
        print('  component rho', [round(spearmanr(q[:,5+i],q[:,8+i]).statistic,3) for i in range(3)])
    print('scoring failures', sum(r[-1]!=len(ids) for r in curves))
    for k in range(5):
        print(k,[(s,round(np.mean([r[4] for r in curves if r[1]==s and r[3]==k]),4))
                 for s in ('greedy','amplitude','random')])


if __name__=='__main__':
    main()
