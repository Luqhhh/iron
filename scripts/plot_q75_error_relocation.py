"""Export descriptive concentration and paired OOF gains as private PNG/PDF."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((source/'report.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    colors = {'iron':'#2463a6','time':'#d47b28','combined':'#38917c'}
    for seed in (42,3407):
        for target in colors:
            curve = pd.read_csv(source/f'concentration-{seed}-{target}.csv')
            x = 100*curve['rank']/len(curve)
            axes[0].plot(x,100*curve.cumulative_share,color=colors[target],linestyle='-' if seed==42 else '--',
                label=f'{target}, seed {seed}')
    axes[0].plot([0,10],[0,10],color='#bbbbbb',linewidth=1)
    axes[0].set(xlim=(0,10),ylim=(0,32),xlabel='Largest-error rows (% of all rows)',ylabel='Share of absolute-error loss (%)',
                title='Q75 OOF error concentration')
    axes[0].legend(fontsize=8,loc='upper left',frameon=False)
    names=list(report['seeds']['42']['candidates'])
    indices=np.arange(len(names))
    for j,seed in enumerate((42,3407)):
        gains=[report['seeds'][str(seed)]['candidates'][n]['gain_global_denominator'] for n in names]
        axes[1].barh(indices+(j-.5)*.34,gains,height=.34,label=f'seed {seed}',color=['#2463a6','#d47b28'][j])
    axes[1].axvline(0,color='#666666',linewidth=.8)
    axes[1].set_yticks(indices,names,fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Local package score-point gain vs Q75',title='Fixed candidates, complete 5-fold OOF')
    axes[1].legend(fontsize=8,frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('2754 rows per split seed; descriptive, not platform forecasts',fontsize=11)
    for extension in ('png','pdf'):
        fig.savefig(output/f'q75-error-relocation.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
