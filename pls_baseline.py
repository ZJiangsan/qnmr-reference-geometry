"""Supervised full-spectrum PLS baseline with held-out propanol levels."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from copenhagen_nmr_benchmark import centered, get_data, write_csv


def closed(pred):
    positive = np.maximum(pred, 0)
    totals = positive.sum(axis=1,keepdims=True)
    return np.divide(positive, totals, out=np.full_like(positive,1/3),where=totals>0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat',type=Path,default=Path('data/NMR_mix_DoE.mat'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    data = get_data(args.mat)
    mask = (data['yData'] < 100).all(axis=1)
    x = centered(data['xData'][mask].T).T
    y = data['yData'][mask]/100.
    samples = data['ObjLabels'][mask]
    group = data['yData'][mask,0]
    estimates = np.empty_like(y)
    rows = []
    for fold,(train,test) in enumerate(GroupKFold(5).split(x,y,group),1):
        losses = {}
        for n in (2,3,4,6,8):
            err = []
            for fit,val in GroupKFold(3).split(x[train], y[train],group[train]):
                model = PLSRegression(n_components=n,scale=False).fit(x[train[fit]],y[train[fit]])
                err.extend(abs(closed(model.predict(x[train[val]]))-y[train[val]]).ravel())
            losses[n] = np.mean(err)
        n = min(losses,key=losses.get)
        model = PLSRegression(n_components=n,scale=False).fit(x[train],y[train])
        estimates[test] = closed(model.predict(x[test]))
        rows.extend([samples[i],fold,n,*y[i],*estimates[i]] for i in test)
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'pls_predictions.csv',
              ['sample','fold','latent_components','true_Propanol','true_Butanol','true_Pentanol',
               'fit_Propanol','fit_Butanol','fit_Pentanol'], rows)
    print('Held-out propanol-level PLS MAE:',100*np.mean(abs(estimates-y)))


if __name__=='__main__':
    main()
