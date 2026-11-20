"""Audit the original window 3 with full and matched binary support."""
import argparse
from pathlib import Path
import numpy as np
from copenhagen_nmr_benchmark import centered, get_data, write_csv
from reference_window_design import bounds, predict

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mat',type=Path,default=Path('data/NMR_mix_DoE.mat'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    d=get_data(args.mat)
    truth=d['yData']/100.
    full=d['xData'][[np.flatnonzero(truth[:,j]==1).item() for j in range(3)]].T
    axis=d['Axis']
    mask=(axis>=1.3327643750103961)&(axis<=1.422386154263303)
    scale=np.linalg.norm(centered(full))
    ref=centered(full[mask])/scale
    gram=ref.T@ref
    sd=np.sqrt(bounds(gram))
    rows=[]
    for label,test,cols in (
        ('all_nonpure_three_component',(truth<1).all(axis=1),[0,1,2]),
        ('binary_three_component',(truth[:,0]==0)&(truth[:,1:]>0).all(axis=1),[0,1,2]),
        ('binary_matched_support',(truth[:,0]==0)&(truth[:,1:]>0).all(axis=1),[1,2])):
        design=ref.copy()
        design[:,[i for i in range(3) if i not in cols]]=0
        y=centered(d['xData'][test][:,mask].T)/scale
        fitted=predict(design.T@design,design.T@y)
        errors=100*np.nanmean(abs(fitted-truth[test]),axis=0)
        print(label,errors,np.mean(errors[cols]))
        for j,name in enumerate(d['yLabels']):
            rows.append([label,str(name).strip(),test.sum(),j in cols,
                         np.linalg.norm(ref[:,j])*scale/np.linalg.norm(centered(full[:,j])),
                         sd[j],errors[j],np.linalg.cond(design[:,cols])])
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'window3_support.csv',
              ['case','component','blends','in_fit','relative_reference_norm',
               'three_component_predicted_sd_per_noise','mae_pp','condition'],rows)

if __name__=='__main__':
    main()
