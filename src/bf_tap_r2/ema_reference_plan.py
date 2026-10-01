"""Constructor-only catalogue of the original matching-reference native calls.

This is a budget proposal, not a runner or scientific execution admission.
Eligible shrink spouts must be verified against every frozen outer partition.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import yaml

from .ema_reference_ledger import KINDS
from .v4_1_reference import V36FixedRecipeFactory
from .v3_6_sampler import sample_v36


V31_NAMES = {
    'v31_iron_0018':'v31-s1-expr-iron-0018',
    'v31_iron_0012':'v31-s1-expr-iron-0012',
    'v31_time_0050':'v31-s1-time-0021-0050',
    'v31_time_0031':'v31-s1-time-0021-0031',
    'v31_time_0039':'v31-s1-time-0021-0039',
    'v31_time_0002':'v31-s1-time-0021-0002',
}
V3_NAMES = {'v3_0107':'v3-catboost-tap_iron-0107',
            'v3_0043':'v3-catboost-tap_iron-0043',
            'v3_0021':'v3-catboost-tap_time_len-0021'}


def _tree(trial):
    if trial.get('base_family', trial.get('family')) != 'catboost':
        raise ValueError('Matching reference tree family changed')
    params=trial['parameters']
    if params.get('thread_count') != 1 or params.get('task_type', 'CPU') != 'CPU':
        raise ValueError('Matching reference tree CPU/thread contract changed')


def _ebm(trial):
    params=trial['parameters']
    if (params.get('objective') != 'rmse' or params.get('outer_bags') != 4
            or params.get('inner_bags') != 0 or params.get('n_jobs') != 1
            or not isinstance(params.get('interactions'), int) or params['interactions'] <= 0):
        raise ValueError('Matching EBM main/interaction bag budget changed')
    # Current InterpretML fits four main bags and four interaction bags; RMSE
    # final intercept correction is analytic, without another boost() call.
    return dict(ebm_fit=1, ebm_boost=8)


def _network(trial):
    if trial.get('kind') not in {'numeric_mlp', 'numeric_tabm'}:
        raise ValueError('Matching numeric-network family changed')
    if trial['parameters'].get('optimizer') not in {'adam', 'adamw'}:
        raise ValueError('Matching numeric-network optimizer changed')
    return dict(torch_optimizer=1)


def reference_plan(root, spec, *, eligible_spouts):
    """Read existing constructor recipes; never call any estimator fit method."""
    root=Path(root); spouts=tuple(eligible_spouts)
    if spouts != tuple(sorted(set(spouts))) or not spouts:
        raise ValueError('Explicit sorted eligible shrink spouts required')
    if spec['budget']['reference_workers'] != 1:
        raise ValueError('Confirmation reference requires one original worker')
    factory=V36FixedRecipeFactory(root, workers=1); recovery=factory.a_factory._recovery
    roles=[]
    def add(name, family, recipe, calls, **metadata):
        roles.append(dict(name=name, family=family, recipe=deepcopy(recipe),
            expected_native_calls={k:calls.get(k,0) for k in KINDS}, **metadata))
    for name in recovery.BASE_NAMES:
        if name.startswith('c2_'):
            _,target,iters,seed=name.split('_')
            trial=recovery.c2_trial('tap_iron' if target=='iron' else 'tap_time_len',int(iters),int(seed))
            _tree(trial); add(name,'legacy_tree',trial,dict(catboost_fit=1))
        elif name=='ord_time':
            trial=recovery.ord_trial(); _tree(trial)
            add(name,'legacy_tree',trial,dict(catboost_fit=1))
        elif name in V3_NAMES:
            trial=recovery.V3_TRIALS[V3_NAMES[name]]; _tree(trial)
            add(name,'legacy_tree',trial,dict(catboost_fit=1))
        elif name in V31_NAMES:
            trial=recovery.V31_TRIALS[V31_NAMES[name]]; _tree(trial)
            add(name,'legacy_inner_select_refit_tree',trial,dict(catboost_fit=2),
                inner_seed=int(recovery.INNER_SEED))
        elif name.startswith('j1_'):
            params={**recovery.CONFIG_V23['common'],'loss_function':'MultiRMSE',
                    'random_seed':int(name.split('_')[1]),'thread_count':1}
            add(name,'legacy_joint_tree',params,dict(catboost_fit=1))
        elif name=='jm1':
            recipe=recovery.CONFIG_V27['models']['JM1']
            if recipe['solver']!='lbfgs':raise ValueError('Legacy joint MLP solver changed')
            add(name,'legacy_joint_mlp_lbfgs',recipe,dict(sklearn_mlp_fit=1))
        else:
            raise ValueError('Undeclared matching legacy pipeline: '+name)
    for target, names in factory.a_factory._expert_ids.items():
        for name in names:
            trial=factory.a_factory._trials[name]
            if trial['kind']=='ebm_boundary':
                add(name,'A_ebm',trial,_ebm(trial))
            elif trial['kind']=='global_spout_shrink':
                params=trial['parameters'];_tree(params['parent_trial'])
                add(name,'A_global_spout_shrink',trial,dict(catboost_fit=1+len(spouts)),
                    eligible_spouts=list(spouts), minimum_spout_samples=params['min_spout_samples'])
            else:raise ValueError('Undeclared matching A expert kind')
    for target, names in factory.selected_experts.items():
        for name in names:
            trial=factory.trials[name]
            calls=_network(trial) if trial['kind'] in {'numeric_mlp','numeric_tabm'} else _ebm(trial)
            add(name,'B36_expert',trial,calls)
    trial=next((t for t in sample_v36(root) if t['trial_id']==spec['reference']['n_trial']),None)
    if trial is None:raise ValueError('Missing matching N trial')
    add('N0048','matching_time_network',trial,_network(trial))
    for name, spec_key, recipe_key in [('V12_joint','joint_spec','joint_recipe'),
                                      ('V7_periodic','periodic_spec','periodic_recipe')]:
        frozen=yaml.safe_load((root/spec['reference'][spec_key]).read_text())
        recipe=frozen['recipes'][spec['reference'][recipe_key]]
        if recipe['backbone']!='tabm':raise ValueError('Original matching TabM backbone changed')
        add(name,'matching_selector_fresh_refit_network',
            dict(recipe=recipe,training=frozen['training']),dict(torch_optimizer=2))
    if len(roles)!=32 or len({r['name'] for r in roles})!=32:
        raise ValueError('Matching reference pipeline catalogue changed')
    total=Counter({k:0 for k in KINDS})
    for role in roles:total.update(role['expected_native_calls'])
    return dict(status='constructor_only_native_budget_proposal_not_execution_admission',
        roles=roles, pipelines=32, expected_native_calls_per_factory=dict(total),
        ebm_stage_calls_explanation='four main plus four interaction boosts per EBM; RMSE intercept correction is analytic',
        shrink_partition_requirement='every outer fold must have these exact eligible spouts at its frozen min_samples',
        eligible_spouts=list(spouts),
        fixed_reference_weights=deepcopy({k:spec['reference'][k] for k in ('iron_weights','time_weights')}),
        scope='no estimator fitting or prediction, no seed qualification, no release')
