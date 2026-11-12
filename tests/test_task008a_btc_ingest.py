"""Focused causal, checksum, boundary, immutability tests; never call network."""
from __future__ import annotations
from datetime import date, timedelta
from io import BytesIO
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from cita2027.btc import ingest as mod


def fake_zip(month='2017-08', data=None):
    if data is None:
        start=date(2017,8,17)
        data=[]
        for i in range(15):
            d=start+timedelta(days=i)
            op=int(pd.Timestamp(d,tz='UTC').timestamp()*1000)
            data.append([str(op),'100','110','90','105','20',str(op+86400000-1),
                         '2100','3','10','1000','0'])
    content='\n'.join(','.join(r) for r in data)+'\n'
    buf=BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(f'BTCUSDT-1d-{month}.csv',content)
    return buf.getvalue()


def test_schedule_and_final_test_prohibition():
    months=list(mod.months())
    assert months[0]=='2017-08' and months[-1]=='2024-12' and len(months)==89
    with pytest.raises(ValueError,match='OUTSIDE'):
        mod.source_name('2025-01')


def test_checksum_exact_filename_and_digest():
    z=fake_zip()
    sha=hashlib.sha256(z).hexdigest()
    name=mod.source_name('2017-08')
    assert mod.expected_checksum(f'{sha}  {name}\n',name)==sha
    assert mod.expected_checksum(f'{sha} *{name}\n',name)==sha
    for bad in (f'{sha} other.zip', f'{sha} ../{name}', 'deadbeef', f'{sha} {name}\n{sha} {name}'):
        with pytest.raises(ValueError): mod.expected_checksum(bad,name)


def test_parse_rows_and_ohlc_fail_fast():
    rows=mod.parse_month(fake_zip(),'2017-08')
    assert len(rows)==15 and rows[0]['date']=='2017-08-17'
    broken=[['1502928000000','100','99','90','105','2','1503014399999','0','1','0','0','0']]
    with pytest.raises(ValueError,match='OHLC_INCONSISTENT'):
        mod.parse_month(fake_zip(data=broken),'2017-08')


def test_timestamp_units_close_and_duplicate():
    first=mod.parse_month(fake_zip(),'2017-08')[0]
    row=[str(first['open_time_ms']),'100','110','90','105','20',str(first['close_time_ms']),
         '2100','3','10','1000','0']
    micro=row.copy();micro[0]+='000';micro[6]+='000'
    with pytest.raises(ValueError,match='timestamp unit'):
        mod.parse_month(fake_zip(data=[micro]),'2017-08')
    bad_close=row.copy();bad_close[6]=str(int(bad_close[6])-1)
    with pytest.raises(ValueError,match='Daily UTC'):
        mod.parse_month(fake_zip(data=[bad_close]),'2017-08')
    with pytest.raises(ValueError,match='DUPLICATE'):
        mod.parse_month(fake_zip(data=[row,row]),'2017-08')


def test_calendar_gap_fails():
    rows=mod.parse_month(fake_zip(),'2017-08')
    with pytest.raises(ValueError,match='CALENDAR_GAP'):
        mod.validate_calendar(rows)


def synthetic_complete_history_for_testing_only():
    """Unit-test fixture, never published or used as scientific market data."""
    n=(mod.END-mod.START).days+1
    ds=[mod.START+timedelta(days=i) for i in range(n)]
    close=np.linspace(100,200,n)
    ret=np.r_[np.nan,np.log(close[1:]/close[:-1])]
    return pd.DataFrame({'date':[d.isoformat() for d in ds],
                         'split':['warmup' if d<mod.DEV_START else
                                  'development' if d<=mod.DEV_END else 'validation' for d in ds],
                         'log_return':ret})


def test_context_ignores_future_target():
    df=synthetic_complete_history_for_testing_only()
    first=int(np.flatnonzero(df['split']=='development')[0])-1
    original=mod.forecast_context(df,first)
    assert len(original)==256 and not original.flags.writeable
    df.loc[first+1,'log_return']=999999.0
    assert np.array_equal(original,mod.forecast_context(df,first))
    df.loc[first,'log_return']=7.0
    assert not np.array_equal(original,mod.forecast_context(df,first))


def test_causal_origin_plans_boundaries_and_fixed_seed():
    df=synthetic_complete_history_for_testing_only()
    dev=mod.origin_plan(df,'a'*64,'fixture','development')
    val=mod.origin_plan(df,'a'*64,'fixture','validation')
    assert dev['count']==1461 and val['count']==731
    assert dev['development_scored_origins']==1209
    assert dev['first']['origin_date']=='2018-12-31'
    assert dev['first']['target_date']=='2019-01-01'
    assert dev['last']['target_date']=='2022-12-31'
    assert val['first']['origin_date']=='2022-12-31'
    assert val['first']['target_date']=='2023-01-01'
    assert val['last']['target_date']=='2024-12-31'
    assert val['first']['origin_index']==dev['last']['target_index']
    assert val['first']['seed']==2027+val['first']['origin_index']
    assert dev['first']['role']=='boundary_bridge'
    assert val['first']['role']=='boundary_bridge'
    assert {v['target_date'] for v in dev['origins']}.isdisjoint({v['target_date'] for v in val['origins']})
    assert max(x['target_date'] for x in val['origins']) < '2025-01-01'


def test_write_once_and_revision_protection(tmp_path):
    path=tmp_path/'test.zip'
    mod.put_once(path,b'hello')
    mod.put_once(path,b'hello')
    assert path.read_bytes()==b'hello'
    with pytest.raises(ValueError,match='IMMUTABLE_FILE_DRIFT'):
        mod.put_once(path,b'changed')


def test_fetch_month_cached_checksum_revision(tmp_path, monkeypatch):
    month='2017-08'; zip_bytes=fake_zip();name=mod.source_name(month)
    good=(hashlib.sha256(zip_bytes).hexdigest()+'  '+name+'\n').encode()
    requests=[]
    def fake_download(url,**kwargs):
        requests.append(url)
        return good if url.endswith('.CHECKSUM') else zip_bytes
    monkeypatch.setattr(mod,'download_bytes',fake_download)
    mod.fetch_month(tmp_path,month)
    mod.fetch_month(tmp_path,month)
    assert len([s for s in requests if s.endswith('.zip')])==1
    monkeypatch.setattr(mod,'download_bytes',lambda url,**kwargs: (b'f'*64+b'  '+name.encode()+b'\n'))
    with pytest.raises(ValueError,match='UPSTREAM_CHECKSUM_REVISION'):
        mod.fetch_month(tmp_path,month)


def test_snapshot_not_written_before_full_calendar(tmp_path, monkeypatch):
    # No synthetic completeness bypass allowed: archive with only 15 dates must abort.
    z=fake_zip(); month='2017-08'
    monkeypatch.setattr(mod,'months',lambda:iter([month]))
    def supply(cache,m):
        cache.mkdir(parents=True,exist_ok=True)
        (cache/mod.source_name(m)).write_bytes(z)
        return {'month':m,'zip_name':mod.source_name(m),'zip_sha256':hashlib.sha256(z).hexdigest()}
    monkeypatch.setattr(mod,'fetch_month',supply)
    with pytest.raises(ValueError,match='CALENDAR_GAP'):
        mod.build(tmp_path)
    assert not (tmp_path/'artifacts'/'btcusdt_data').exists()
