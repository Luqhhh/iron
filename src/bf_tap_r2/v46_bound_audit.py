"""Independent Fraction-only verification: no optimizer or generator imports."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import yaml


def certificate(design, anchor, query, saved):
    if saved['anchor'] != anchor:
        raise ValueError('Cell must use one common anchor')
    origin=[F(str(v)) for v in design[anchor][0]]
    upper=F(str(design[anchor][1]))+F(str(design[anchor][2]))
    displacement=[x-y for x,y in zip(query,origin)]
    actual=[F(0)]*len(origin);value=upper
    for name,weight in saved['dual_weights'].items():
        w=F(weight)
        if name not in design or w<0:
            raise ValueError('Invalid dual name or negative weight')
        point,score,error=design[name]
        for j,x in enumerate(point):actual[j]+=w*(origin[j]-F(str(x)))
        value+=w*(upper-(F(str(score))-F(str(error))))
    if actual!=displacement or F(saved['exact_upper'])!=value:
        raise ValueError('Exact dual equation/value mismatch')
    return value


def verify_partition(design, saved, denominator, dimension):
    n=denominator
    if saved['denominator']!=n or saved['dimension']!=dimension:
        raise ValueError('Grid identity differs')
    expected={}
    if dimension==1:
        for k in range(n):expected[f'I:{k}']=[[k],[k+1]]
    else:
        # Enumerate squares on the integer lattice, clipping its upper diagonal.
        for i in range(n):
            for j in range(n):
                if i+j<n:expected[f'L:{i}:{j}']=[[i,j],[i+1,j],[i,j+1]]
                if i+j+1<n:expected[f'U:{i}:{j}']=[[i+1,j],[i+1,j+1],[i,j+1]]
    if len(saved['cells'])!=len(expected) or {c['id'] for c in saved['cells']}!=set(expected):
        raise ValueError('Incomplete or duplicate cell coverage')
    total_measure=F(0);cell_values=[];vertices=set();cert_count=0
    for cell in saved['cells']:
        if cell['vertices']!=expected[cell['id']]:
            raise ValueError('Grid vertex identity differs')
        if len(cell['vertex_certificates'])!=dimension+1:
            raise ValueError('Missing cell vertex certificate')
        points=[[F(x,n) for x in v] for v in cell['vertices']]
        if dimension==1:
            measure=points[1][0]-points[0][0]
        else:
            a,b,c=points
            measure=((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
        if measure<=0:raise ValueError('Invalid cell orientation or size')
        total_measure+=measure;vertices.update(tuple(v) for v in cell['vertices'])
        values=[certificate(design,cell['anchor'],q,c) for q,c in zip(points,cell['vertex_certificates'])]
        cert_count+=len(values)
        bound=max(values)
        if bound!=F(cell['exact_upper']):raise ValueError('Cell maximum differs')
        cell_values.append(bound)
    if total_measure!=(F(1) if dimension==1 else F(1,2)):
        raise ValueError('Domain measure not covered')
    bound=max(cell_values)
    if bound!=F(saved['exact_upper']) or saved['vertex_count']!=len(vertices):
        raise ValueError('Global maximum or vertex count differs')
    if F.from_float(saved['upper'])<bound:raise ValueError('Floating presentation rounds inward')
    return bound,cert_count


def verify_coarse(design, record, expected_vertices):
    if record['vertices']!=expected_vertices:
        raise ValueError('Coarse-domain vertices differ')
    if len(record['vertex_certificates'])!=len(expected_vertices):
        raise ValueError('Missing coarse certificate')
    bound=max(certificate(design,record['anchor'],[F(str(v)) for v in q],c)
              for q,c in zip(expected_vertices,record['vertex_certificates']))
    if bound!=F(record['exact_upper']):raise ValueError('Coarse maximum differs')
    return bound


def verify(spec, base, report):
    if any(report[k]!=0 for k in ['new_fits','label_reads','test_prediction_reads','packages','desktop_writes','uploads']):
        raise ValueError('Zero-access/fit/release contract differs')
    time,count_t=verify_partition(spec['time_observations'],report['time'],spec['grid_denominator'],2)
    iron,count_i=verify_partition(spec['iron_observations'],report['iron'],spec['grid_denominator'],1)
    coarse=report['coarse_controls']
    old=verify_coarse(base['time_observations'],coarse['old_time'],base['time_domain_vertices'])
    new=verify_coarse(spec['time_observations'],coarse['new_time'],spec['time_domain_vertices'])
    old_iron=verify_coarse(spec['iron_observations'],coarse['iron'],spec['iron_domain_vertices'])
    if not time<=new<=old or iron>old_iron:raise ValueError('Bound became weaker')
    _,score,error=spec['iron_observations']['W05']
    ref=F(str(score))-F(str(error));gain=iron-ref;total=max(F(0),time+gain)
    expected={'iron_reference_exact_lower':ref,'iron_gain_exact_upper':gain,'family_exact_upper':total}
    if any(F(report[k])!=v for k,v in expected.items()):raise ValueError('Family arithmetic differs')
    if F(coarse['old_family_exact_upper'])!=old+old_iron-ref or F(coarse['new_family_exact_upper'])!=new+old_iron-ref:
        raise ValueError('Coarse family arithmetic differs')
    if len(report['milestones'])!=len(spec['platform_milestones']):raise ValueError('Missing milestones')
    for row,target in zip(report['milestones'],spec['platform_milestones']):
        t=F(str(target))
        if (F(str(row['target']))!=t or row['excluded_under_assumptions']!=(total<t)
                or F(row['exact_margin_upper_minus_target'])!=total-t):
            raise ValueError('Milestone interpretation differs')
    if F.from_float(report['family_upper'])<total:raise ValueError('Family presentation rounds inward')
    return {'status':'passed','time_cells':len(report['time']['cells']),
            'iron_cells':len(report['iron']['cells']),'cell_vertex_certificates':count_t+count_i,
            'coarse_vertex_certificates':8,'family_exact_upper':str(total),
            'optimizer_calls':0,'new_fits':0,'label_reads':0,'packages':0,'uploads':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='local/runs/round2-v46/bound-r1.json')
    parser.add_argument('--output',default='local/runs/round2-v46/audit-r1.json')
    args=parser.parse_args();root=Path.cwd().resolve()
    inp=(root/args.input).resolve();out=(root/args.output).resolve()
    if any(not p.is_relative_to(root/'local/runs/round2-v46') for p in (inp,out)):
        raise ValueError('Private V46 evidence directory required')
    report=json.loads(inp.read_text())
    for name,sha in report['source_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Source identity differs: {name}')
    spec=yaml.safe_load((root/'configs/round2_v46/SPEC.yaml').read_text())
    base=yaml.safe_load((root/spec['base_bound_spec']).read_text())
    audit=verify(spec,base,report)
    audit.update(report_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                 auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with out.open('x') as f:json.dump(audit,f,indent=2);f.write('\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
