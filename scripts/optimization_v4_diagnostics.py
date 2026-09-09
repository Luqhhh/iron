"""Read-only post-run diagnostics, preserving origin multiplicity and week pairing."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bf_tap.artifacts import atomic_write_json, file_sha256
from bf_tap.optimization.v3_evidence import paired_week_intervals
from bf_tap.optimization.refresh_factorial import decomposition


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for p in sorted((args.run/'units').glob('O*/**/errors.csv')):
        frame=pd.read_csv(p,dtype={'sample_id':str})
        unit=p.parent.parent.name
        frame['candidate']=p.parent.name
        frame['origin']=unit.split('_')[0]
        frame['horizon']=int(unit.split('_H')[1])
        rows.append(frame)
    errors=pd.concat(rows,ignore_index=True)
    reports={c:paired_week_intervals(errors.loc[errors.candidate.isin([c,'E12-raw'])],c,'E12-raw')
             for c in ('R1','R2','R3')}
    atomic_write_json(args.output/'paired_week_intervals.json',reports)
    monthly=[]
    for (candidate,origin,horizon),f in errors.groupby(['candidate','origin','horizon']):
        row=dict(candidate=candidate,origin=origin,horizon=int(horizon),
                 eval_month=pd.to_datetime(f.reference_time).dt.strftime('%Y-%m').iloc[0],samples=len(f))
        for t in ('tap_iron','tap_time_len'):
            row[t+'_wmape']=float(f['abs_error_'+t].sum()/f[t].sum())
        row['E']=.5*(row['tap_iron_wmape']+row['tap_time_len_wmape'])
        monthly.append(row)
    table=pd.DataFrame(monthly)
    table.to_csv(args.output/'all_cells.csv',index=False)
    table.loc[table.horizon==1].to_csv(args.output/'h1_six_origins.csv',index=False)
    table.loc[table.eval_month.isin(['2024-09','2024-10','2024-11'])].to_csv(args.output/'recent_three_months.csv',index=False)
    # Draw one set of calendar-week weights for every arm and component each replicate.
    f=pd.read_csv(args.run/'factorial_predictions.csv',dtype={'sample_id':str})
    f['arm']=f.model_state.map({'old':'0','new':'1'})+f.history_state.map({'old':'0','new':'1'})
    f['week']=pd.to_datetime(f.reference_time).dt.strftime('%G-W%V')
    rng=np.random.default_rng(2026)
    weeks=sorted(f.week.unique())
    cols=['tap_iron','tap_time_len','abs_error_tap_iron','abs_error_tap_time_len']
    draws={}
    for _ in range(1000):
        counts=pd.Series(rng.choice(weeks,len(weeks),replace=True)).value_counts()
        weighted=f.copy()
        weighted[cols]=weighted[cols].mul(weighted.week.map(counts).fillna(0),axis=0)
        summed=weighted.groupby(['component','eval_month','arm'])[cols].sum()
        if (summed[['tap_iron','tap_time_len']]<=0).any().any():
            continue
        for (component,month),part in summed.groupby(level=[0,1]):
            part=part.droplevel([0,1])
            losses={t:part['abs_error_'+t]/part[t] for t in ('tap_iron','tap_time_len')}
            losses['E']=.5*(losses['tap_iron']+losses['tap_time_len'])
            for target,loss in losses.items():
                d=decomposition(loss.to_dict())
                for effect,value in d.items():
                    draws.setdefault((component,month,target,effect),[]).append(value)
    intervals=[dict(component=c,eval_month=m,target=t,effect=e,valid_draws=len(values),
                    **dict(zip(('p025','median','p975'),np.quantile(values,[.025,.5,.975]).tolist())))
               for (c,m,t,e),values in draws.items()]
    atomic_write_json(args.output/'factorial_week_intervals.json',dict(
        scope='RETROSPECTIVE_STABILITY_NOT_SELECTION_ADJUSTED_NOT_PHYSICAL_CAUSALITY',
        seed=2026,repetitions=1000,source_sha256=file_sha256(args.run/'factorial_predictions.csv'),intervals=intervals))

if __name__=='__main__': main()
