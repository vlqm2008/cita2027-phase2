"""Focused CPU tests for TASK-011. No network, GPU, Test data, or old audits."""
from __future__ import annotations
import io
import json
import zipfile
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cita2027.finaltest import lock as L
from cita2027.finaltest import execute as E
from cita2027.calibration.online import Parameters,calibrate,interval_metrics
from cita2027.calibration.development_analysis import circular_block_indices,paired_ci


def micro_zip(month='2025-01'):
    base=datetime(2025,1,1,tzinfo=timezone.utc)
    dates=[base+timedelta(days=k) for k in range(31)]
    lines=[]
    for when in dates:
        op=int(when.timestamp())*1_000_000
        cl=op+86_400_000_000-1
        lines.append(f'{op},4.0,4.5,3.5,4.2,1.0,{cl},1.0,10,0.5,0.5,0\n')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as f:f.writestr('BTCUSDT-1d-'+month+'.csv',''.join(lines))
    return stream.getvalue()


def test_spot_2025_microseconds_only():
    dates=L.date_labels_only(micro_zip(),'2025-01')
    assert len(dates)==31
    assert dates[0]=='2025-01-01' and dates[-1]=='2025-01-31'


def test_spot_2025_legacy_millisecond_timestamp_rejected():
    blob=micro_zip()
    src=io.BytesIO(blob)
    with zipfile.ZipFile(src) as z:
        content=z.read(z.namelist()[0]).decode()
    one=content.replace('1735689600000000','1735689600000',1)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr('BTCUSDT-1d-2025-01.csv',one)
    with pytest.raises(ValueError,match='MICROSECOND'):L.date_labels_only(stream.getvalue(),'2025-01')


def test_archive_expected_22_months_2025_2026_october():
    months=list(L.archive_months())
    assert len(months)==22 and months[0]=='2025-01' and months[-1]=='2026-10'
    assert (L.END-L.START).days+1==668


def test_2025_first_origin_locked_and_2026_last():
    rules=L.locked_rules()
    assert rules['target_start']=='2025-01-01' and rules['target_end_inclusive']=='2026-10-30'
    assert len(rules['primary_hypotheses'])==8
    assert len({(r['market'],r['model'],r['scale']) for r in rules['primary_hypotheses']})==8
    assert rules['window']==252 and rules['correction']=='expansion_only'
    assert rules['primary_family']=='8_comparisons_Holm_Bonferroni_alpha_0.05_for_B20'


def test_immutable_write_once_and_conflict(tmp_path):
    f=tmp_path/'locked'/'p.json'
    one=b'abc\n'
    assert L.write_once(f,one)==L.sha_bytes(one)
    assert L.write_once(f,one)==L.sha_bytes(one)
    with pytest.raises(ValueError,match='IMMUTABLE_FILE_COLLISION'):L.write_once(f,b'changed')
    assert f.read_bytes()==one


def test_no_test_target_loaded_before_lock_failure(tmp_path,monkeypatch):
    # The post-lock reader must reject absence of pretest manifest before pandas reads any file.
    def forbid(*a,**k):raise AssertionError('FINAL_DATA_PARSED_BEFORE_LOCK')
    monkeypatch.setattr(pd,'read_csv',forbid)
    with pytest.raises(ValueError,match='PRETEST_LOCK_NOT_DURABLE'):
        E.load_market_after_lock(tmp_path,'btcusdt','0'*64)


def test_future_target_does_not_affect_current_forecast_context():
    df=pd.DataFrame({'log_return':np.arange(300,dtype=float)/1000})
    a=E.context_for_origin(df,256)
    df.at[257,'log_return']=999.
    b=E.context_for_origin(df,256)
    assert np.array_equal(a,b)
    assert len(a)==256 and a.flags.writeable is False
    assert a[-1]==.256


def test_holm_prespecified_family_monotone():
    raw={str(i):p for i,p in enumerate([.003,.01,.02,.07,.1,.12,.3,.6])}
    adj=E.holm_adjust(raw)
    assert adj['0']==pytest.approx(.024)
    assert adj['1']==pytest.approx(.07)
    assert all(adj[str(i)]<=adj[str(i+1)] for i in range(7))
    assert all(adj[str(i)]>=raw[str(i)] for i in range(8))


def test_paired_statistic_rejects_zero_and_retains_sign():
    d=np.full(300,-.001)
    indices=circular_block_indices(len(d),100,20,2027)
    p=E.centered_one_sided_p(d,indices)
    assert p==pytest.approx(1/101)
    ci=paired_ci(d,indices)
    assert ci['mean_difference']==pytest.approx(-.001)
    assert ci['ci_95_percentile'][1]<0


def test_continuous_replay_with_no_reset_matches_single_run():
    p=Parameters()
    first=np.linspace(-.03,.03,256)
    rec=[]
    base_date=date(2022,1,1)
    for k in range(300):
        y=.015*np.sin(k/11)
        rec.append({'origin_index':1000+k,'target_index':1001+k,'seed':3027+k,
                    'origin_date':(base_date+timedelta(days=k)).isoformat(),
                    'target_date':(base_date+timedelta(days=k+1)).isoformat(),
                    'q10':-.01,'q50':0.,'q90':.01,'target_log_return':float(y)})
    for scale in ('none','residual_ewma','return_ewma'):
        for ctrl in ('rolling_cqr','fixed_gamma_aci','variable_gamma_aci'):
            all_=calibrate(rec,first,p,scale,ctrl)
            before=calibrate(rec[:279],first,p,scale,ctrl)
            assert all_[:len(before)]==before
            assert len(all_)==48
    m=interval_metrics(all_[-10:],.2)
    assert m['coverage']+m['lower_miss_rate']+m['upper_miss_rate']==pytest.approx(1.)


def test_scores_winkler_decomposition_and_sign():
    row={'lower':-.02,'upper':.01,'target_log_return':-.04}
    assert E.score_from_trace(row)==pytest.approx(.03+.2)
    assert E.score_from_trace({'lower':-.02,'upper':.01,'target_log_return':0.})==pytest.approx(.03)


def test_checkpoint_probes_must_match_before_lock(monkeypatch,tmp_path):
    # A cannot proceed to source or Final Test if identities are not present.
    monkeypatch.setattr(L,'accepted_anchors',lambda root:[])
    monkeypatch.setattr(L,'pinned_code',lambda root:[])
    monkeypatch.setattr(L,'spy_source_dates',lambda root:(_ for _ in ()).throw(AssertionError('SOURCE_ACCESS_BEFORE_PROBE')))
    with pytest.raises(ValueError,match='LOCAL_CHECKPOINT_SHA_PROBE_NOT_VERIFIED'):
        L.preflight_and_lock(tmp_path,{})


def test_btc_after_lock_2025_microsecond_numeric_ingest_synthetic(tmp_path,monkeypatch):
    # Synthetic 2017-2024 prefix, never a real downloaded Final Test snapshot.
    p=tmp_path/'artifacts'/'btcusdt_data'/'snapshots'/f'{L.DATA_SHA}.csv'
    p.parent.mkdir(parents=True)
    dates=pd.date_range('2017-08-17',periods=2694,freq='D').strftime('%Y-%m-%d')
    assert dates[-1]=='2024-12-31'
    old=pd.DataFrame({'row_index':range(2694),'date':dates,'split':['validation']*2694,
                      'close':np.ones(2694),'log_return':np.r_[np.nan,np.zeros(2693)]})
    old.to_csv(p,index=False)
    z=tmp_path/'artifacts'/'final_test'/'blind_sources'/'BTCUSDT-1d-2025-01.zip'
    z.parent.mkdir(parents=True)
    blob=micro_zip();z.write_bytes(blob)
    c=z.with_suffix('.zip.CHECKSUM');c.write_bytes(b'synthetic test checksum')
    actual=E.sha256_file
    monkeypatch.setattr(E,'sha256_file',lambda f:L.DATA_SHA if Path(f)==p else actual(f))
    targets=[(date(2025,1,1)+timedelta(days=j)).isoformat() for j in range(31)]
    plan={'count':31,'origins':[{'target_date':x} for x in targets],
          'source':{'archives':[{'month':'2025-01','path':z.relative_to(tmp_path).as_posix(),
                    'checksum_path':c.relative_to(tmp_path).as_posix(),
                    'sha256':actual(z),'checksum_sha256':actual(c)}]}}
    df=E.btc_after_lock(tmp_path,plan)
    assert len(df)==2725
    assert df.iloc[2693]['date']=='2024-12-31'
    assert df.iloc[2694]['date']=='2025-01-01'
    assert df.iloc[2724]['date']=='2025-01-31'
    assert df.iloc[2694]['previous_row_index']==2693
    assert float(df.iloc[2694]['log_return'])==pytest.approx(np.log(4.2))
