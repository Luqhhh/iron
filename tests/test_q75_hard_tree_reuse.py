import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('hard_tree_reuse',Path(__file__).parents[1]/'scripts/q75_hard_tree_reuse.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def test_complete_fold_matches_only_its_ordered_query():
    ids=np.array([f'id{i}' for i in range(10)]);folds=np.tile(np.arange(5),2)
    assert review.aligned_unit(ids,folds,2,ids[[2,7]],[1.,2.]).tolist()==(folds==2).tolist()
    with pytest.raises(ValueError,match='Query identity'):
        review.aligned_unit(ids,folds,2,ids[[7,2]],[1.,2.])
    with pytest.raises(ValueError,match='Query identity'):
        review.aligned_unit(ids,folds,2,ids[[2]],[1.])


def test_partial_fold_pool_or_nonfinite_prediction_cannot_be_scored():
    ids=np.array([f'id{i}' for i in range(10)]);folds=np.tile(np.arange(5),2)
    with pytest.raises(ValueError,match='Complete five-fold'):
        review.aligned_unit(ids,np.zeros(10),0,ids,np.ones(10))
    with pytest.raises(ValueError,match='predictions differ'):
        review.aligned_unit(ids,folds,0,ids[[0,5]],[1,float('nan')])
