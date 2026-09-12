"""Keep synthetic fits separate from all competition-run budgets."""
import json
from pathlib import Path
from preprocessing import Preprocessor
from qrf_model import QRF

COUNTS = {'synthetic_forest_attempted':0,'synthetic_forest_completed':0,
          'synthetic_preprocessor_attempted':0,'synthetic_preprocessor_completed':0}


def pytest_sessionstart(session):
    for cls, prefix in [(QRF,'synthetic_forest'),(Preprocessor,'synthetic_preprocessor')]:
        original=cls.fit
        def tracked(self,*args,_original=original,_prefix=prefix,**kwargs):
            COUNTS[_prefix+'_attempted']+=1
            result=_original(self,*args,**kwargs)
            COUNTS[_prefix+'_completed']+=1
            return result
        cls.fit=tracked


def pytest_sessionfinish(session,exitstatus):
    path=session.config.option.xmlpath
    if path:
        Path(str(path)+'.synthetic_fit_counts.json').write_text(json.dumps(dict(**COUNTS,
            competition_fits=0,exitstatus=int(exitstatus)),sort_keys=True,indent=2))
