import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from features import build_features
from validate_release import replay

def test_feature_prefix_never_uses_future_observations():
    df=pd.DataFrame({'unit':[1]*9,'cycle':range(1,10),'sensor_2':[1,3,8,12,7,5,9,11,99]})
    full=build_features(df,sensors=['sensor_2'])
    for n in range(1,len(df)+1):
        prefix=build_features(df.iloc[:n],sensors=['sensor_2'])
        np.testing.assert_allclose(prefix,full.iloc[:n])

@pytest.mark.parametrize('remaining,expected',[(10,'late'),(30,'good'),(80,'early')])
def test_warning_classes(remaining,expected):
    out=replay(pd.DataFrame({'unit':[1],'cycle':[1],'rul':[remaining],'pred_rul':[10]}),.35)
    assert out[expected]==1 and out['warned']==1

def test_censored_engine_is_not_counted_as_successfully_covered():
    out=replay(pd.DataFrame({'unit':[1],'cycle':[1],'rul':[80],'pred_rul':[80]}),.35)
    assert out['censored_before_danger']==1 and out['eligible_observed']==0
    assert out['miss_rate_observed_eligible'] is None

def test_eligible_unwarned_engine_counts_as_miss():
    out=replay(pd.DataFrame({'unit':[1],'cycle':[1],'rul':[30],'pred_rul':[80]}),.35)
    assert out['missed_observed']==1 and out['miss_rate_observed_eligible']==1
