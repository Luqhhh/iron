"""One prospective native MAE change; this module grants no fitting admission."""
from copy import deepcopy


def mae_recipe(original):
    if original['class']!='RealMLP_TD_Regressor' or original['constructor'].get('train_metric_name') is not None or original['resolved'].get('train_metric_name') is not None:
        raise ValueError('Expected the immutable native TD default-MSE recipe')
    result=deepcopy(original)
    for key in ['constructor','resolved']:result[key]['train_metric_name']='mae'
    return result
