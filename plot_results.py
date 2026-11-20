"""Plot reference-only error ranking and matched-window selection curves."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    directory=p.parse_args().results
    def data(name):
        with (directory/name).open(newline='') as f:
            return list(csv.DictReader(f))
    rows=data('window_diagnostics.csv')
    labels=('Propanol','Butanol','Pentanol')
    pred=np.array([np.linalg.norm([float(r[f'predicted_sd_per_noise_{n}']) for n in labels]) for r in rows])
    observed=np.array([float(r['mae_pp']) for r in rows])
    good=np.array([all(float(r[f'relative_norm_{n}'])>=.015 for n in labels) for r in rows])
    fig,(a,b)=plt.subplots(1,2,figsize=(8.8,3.5),layout='constrained')
    for mask,name,color in [(good,'All components supported','#236A97'),
                            (~good,'At least one weak reference','#B45726')]:
        a.scatter(pred[mask],observed[mask],s=23,label=name,color=color,alpha=.85)
    a.set(xscale='log',yscale='log',xlabel='Predicted fraction SD per unit noise',
          ylabel='Measured composition MAE (percentage points)',title='A   Identifiability by target window')
    a.legend(frameon=False,fontsize=8)
    curves=data('selection_curves.csv')
    for strategy,name,color,ls in [
        ('random','Random','#747474','-'),
        ('greedy','Greedy A-optimal','#236A97','-'),
        ('amplitude','Amplitude ranked','#B45726','--')]:
        values=[np.mean([float(r['mae_pp']) for r in curves if r['strategy']==strategy
                        and int(r['markers'])==k]) for k in range(5)]
        b.plot(range(5),values,label=name,marker='o',ms=4,color=color,ls=ls)
    b.axhline(.97123,ls=':',color='black',label='Full-spectrum NNLS')
    b.set(xticks=range(5),yscale='log',xlabel='Number of additional windows',ylabel='Mean composition MAE (percentage points)',
          title='B   Error by marker budget')
    b.legend(frameon=False,fontsize=8)
    fig.savefig(directory/'identifiability_and_selection.png',dpi=220)

if __name__=='__main__':
    main()
