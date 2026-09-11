import pandas as pd
from bf_tap.optimization.trajectory_complete import canonical_history_time


def test_csv_timezone_canonicalization_preserves_exact_instants_and_detects_changes():
    csv=pd.Series(['2024-05-31 23:59:00+08:00','2024-06-01 00:00:00+08:00'])
    bundle=pd.to_datetime(csv,utc=True).dt.tz_convert('Asia/Shanghai')
    restored=pd.to_datetime(csv)
    assert (restored==bundle).all() and not restored.equals(bundle)
    assert canonical_history_time(restored).equals(bundle)
    changed=restored+pd.Timedelta(seconds=1)
    assert not canonical_history_time(changed).equals(bundle)
