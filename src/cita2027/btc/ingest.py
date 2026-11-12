"""TASK-008A: Binance Spot BTCUSDT 1d, strictly bounded through 2024-12-31.

Data from archive files are retrospective and cannot establish historical PIT availability.
The only side effects are inside data/btcusdt and artifacts/btcusdt_data.
No GPU dependencies; uses existing repository provenance, cache and forecast contracts.
"""
from __future__ import annotations

import calendar
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd

from cita2027.forecasting.cache import digest, store_or_verify
from cita2027.utils.provenance import canonical_json_bytes, sha256_file

START = date(2017, 8, 17)
END = date(2024, 12, 31)
DEV_START = date(2019, 1, 1)
DEV_END = date(2022, 12, 31)
VAL_START = date(2023, 1, 1)
VAL_END = END
BASE = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/'
COLS = ('open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore')
CONTEXT = 256
BURNIN = 252
CHUNK = 1024 * 1024
SCHEMA = 'task008a-btc-spot-1d-utc-v1'


def months():
    y, m = START.year, START.month
    while (y, m) <= (END.year, END.month):
        yield f'{y:04d}-{m:02d}'
        m += 1
        if m == 13:
            m, y = 1, y + 1


def source_name(month: str) -> str:
    if month not in set(months()):
        raise ValueError('OUTSIDE_LOCKED_ARCHIVE_BOUNDARY')
    return f'BTCUSDT-1d-{month}.zip'


def expected_checksum(text: str, name: str) -> str:
    """Accept standard sha256sum checksum line for precisely this archive."""
    matches = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(matches) != 1:
        raise ValueError('Invalid CHECKSUM: expected one nonempty line')
    m = re.fullmatch(r'([a-fA-F0-9]{64})\s+\*?([^\\/\s]+)', matches[0])
    if not m or m.group(2) != name:
        raise ValueError('CHECKSUM filename or digest invalid')
    return m.group(1).lower()


def download_bytes(url: str, attempts: int = 3, max_bytes: int = 256 * 1024 * 1024) -> bytes:
    if not url.startswith(BASE) or not (url.endswith('.zip') or url.endswith('.zip.CHECKSUM')):
        raise ValueError('Unauthorized download URL')
    for attempt in range(attempts):
        try:
            req = Request(url, headers={'User-Agent': 'CITA2027-Research-BinanceArchiveAudit/1.0'})
            with urlopen(req, timeout=45) as response:
                buf = bytearray()
                while True:
                    block = response.read(CHUNK)
                    if not block: break
                    buf.extend(block)
                    if len(buf) > max_bytes:
                        raise ValueError('Archive exceeds safety size')
                return bytes(buf)
        except (HTTPError, URLError, TimeoutError) as exc:
            if isinstance(exc, HTTPError) and exc.code == 404:
                raise ValueError(f'MISSING_OFFICIAL_ARCHIVE: {url}') from exc
            if attempt + 1 == attempts:
                raise RuntimeError(f'BINANCE_DOWNLOAD_FAILED: {url}: {exc}') from exc
            time.sleep(1 + attempt * 2)
    raise AssertionError('unreachable')


def put_once(path: Path, data: bytes) -> None:
    """Write-once with temp + hard link (no overwrites); retry verifies exact bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'IMMUTABLE_FILE_DRIFT: {path}')
        return
    import tempfile
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-', delete=False) as f:
            temp_name = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temp_name, path)  # atomic no-replace, supported on local NTFS
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f'IMMUTABLE_FILE_RACE: {path}')
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def fetch_month(cache_dir: Path, month: str) -> dict:
    """Re-fetch official checksum on each run to detect archive revisions.

    Once an archive is cached, only its checksum is downloaded; no archive overwrite.
    """
    name = source_name(month)
    checksum_url = BASE + name + '.CHECKSUM'
    zip_url = BASE + name
    remote_checksum = download_bytes(checksum_url, max_bytes=8192)
    expected = expected_checksum(remote_checksum.decode('utf-8-sig'), name)
    zip_path = cache_dir / name
    checksum_path = cache_dir / (name + '.CHECKSUM')
    if checksum_path.exists() and checksum_path.read_bytes() != remote_checksum:
        raise ValueError(f'UPSTREAM_CHECKSUM_REVISION: {month}; do not replace previously frozen sources')
    if zip_path.exists():
        observed = sha256_file(zip_path)
        if observed != expected:
            raise ValueError(f'CACHED_ARCHIVE_CHECKSUM_MISMATCH_OR_REVISION: {month}')
    else:
        raw = download_bytes(zip_url)
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected:
            raise ValueError(f'OFFICIAL_ZIP_CHECKSUM_MISMATCH: {month}')
        put_once(zip_path, raw)
    put_once(checksum_path, remote_checksum)
    return {'month':month, 'zip_name':name, 'zip_sha256':observed,
            'checksum_sha256':hashlib.sha256(remote_checksum).hexdigest(),
            'official_sha256':expected, 'url':zip_url, 'checksum_url':checksum_url,
            'zip_size_bytes':zip_path.stat().st_size}


def _number(raw: str, label: str, *, positive: bool = False, integer: bool = False):
    try:
        v = Decimal(raw)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f'Invalid {label}: {raw!r}') from exc
    if not v.is_finite() or (positive and v <= 0) or (not positive and v < 0):
        raise ValueError(f'Invalid nonfinite/nonpositive {label}: {raw!r}')
    if integer and v != v.to_integral_value():
        raise ValueError(f'Invalid non-integer {label}: {raw!r}')
    return v


def parse_month(blob: bytes, month: str) -> list[dict]:
    name = source_name(month)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [info for info in zf.infolist() if not info.is_dir()]
        if len(names) != 1 or names[0].filename != name.removesuffix('.zip') + '.csv':
            raise ValueError(f'Unexpected ZIP contents: {month}')
        if names[0].file_size > 256 * 1024 * 1024:
            raise ValueError('Unsafe uncompressed ZIP size')
        with zf.open(names[0]) as stream:
            reader = csv.reader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline=''))
            rows = []
            for line_no, fields in enumerate(reader, 1):
                if not fields: raise ValueError('Blank kline row')
                if line_no == 1 and fields[0].strip().lower() in ('open_time','open time'):
                    continue
                if len(fields) != 12:
                    raise ValueError(f'Malformed archive {month} line {line_no}: {len(fields)} columns')
                if not re.fullmatch(r'\d{13}', fields[0]) or not re.fullmatch(r'\d{13}', fields[6]):
                    raise ValueError(f'Incorrect timestamp unit (expected milliseconds) in {month}')
                op_ms, close_ms = int(fields[0]), int(fields[6])
                if op_ms % 86400000 != 0 or close_ms != op_ms + 86400000 - 1:
                    raise ValueError(f'Daily UTC candle start/end invalid {month} line {line_no}')
                day = datetime.fromtimestamp(op_ms / 1000, tz=timezone.utc).date()
                if day.strftime('%Y-%m') != month:
                    raise ValueError('Candle date outside its monthly archive')
                o,h,l,c = (_number(fields[i], COLS[i], positive=True) for i in (1,2,3,4))
                if not (l <= min(o,c) <= max(o,c) <= h):
                    raise ValueError(f'OHLC_INCONSISTENT: {day}')
                for i in (5,7,8,9,10):
                    _number(fields[i], COLS[i], integer=(i==8))
                if not fields[11].strip():
                    raise ValueError('Missing ignore field')
                rows.append({'date': day.isoformat(), 'open':str(o),'high':str(h),
                             'low':str(l),'close':str(c),'volume':str(_number(fields[5], 'volume')),
                             'open_time_ms':op_ms,'close_time_ms':close_ms,
                             'trades':int(Decimal(fields[8]))})
            if not rows: raise ValueError(f'EMPTY_MONTH: {month}')
    if [r['date'] for r in rows] != sorted(set(r['date'] for r in rows)):
        raise ValueError(f'DUPLICATE_OR_UNSORTED_DAYS: {month}')
    return rows


def validate_calendar(rows: list[dict]) -> dict:
    dates = [date.fromisoformat(r['date']) for r in rows]
    expected = [START + timedelta(days=i) for i in range((END-START).days+1)]
    if len(dates) != len(expected) or dates != expected:
        ds, es = set(dates), set(expected)
        raise ValueError(f'CALENDAR_GAP_OR_EXTRA: missing={sorted(es-ds)[:8]}, extra={sorted(ds-es)[:8]}')
    return {'first_day':START.isoformat(),'last_day':END.isoformat(),
            'daily_utc_days':len(expected),'duplicate_days':0,'missing_days':0,
            'timestamps_unit':'milliseconds','utc_close_time_rule':'next_midnight_minus_1ms'}


def make_dataset(rows: list[dict]) -> pd.DataFrame:
    validate_calendar(rows)
    df = pd.DataFrame(rows)
    dates = pd.to_datetime(df['date'], utc=True)
    close = np.array([float(Decimal(x)) for x in df['close']],dtype='float64')
    ret = np.full(len(close), np.nan)
    # No data-dependent scale fitted over future; return is computed at date t from C[t-1],C[t].
    ret[1:] = np.log(close[1:] / close[:-1])
    if not np.isfinite(close).all() or not np.isfinite(ret[1:]).all():
        raise ValueError('Nonfinite closing price/return')
    df['row_index'] = np.arange(len(df),dtype=int)
    df['previous_date'] = [''] + list(df['date'].iloc[:-1])
    df['previous_row_index'] = [-1] + list(range(len(df)-1))
    df['log_return'] = ret
    df['split'] = ['warmup' if day < DEV_START else
                   'development' if day <= DEV_END else 'validation'
                   for day in (d.date() for d in dates)]
    return df[['row_index','date','split','open_time_ms','close_time_ms','open','high','low','close',
               'volume','trades','previous_date','previous_row_index','log_return']]


def forecast_context(df: pd.DataFrame, origin_index: int) -> np.ndarray:
    """Strict information set: returns [origin-255, origin]; excludes target row."""
    if not CONTEXT <= origin_index < len(df) - 1:
        raise ValueError('Insufficient context or no next target')
    context = df.iloc[origin_index-CONTEXT+1:origin_index+1]['log_return'].to_numpy(dtype=np.float64,copy=True)
    if len(context) != CONTEXT or not np.isfinite(context).all():
        raise ValueError('Context contains unavailable/invalid return')
    context.setflags(write=False)
    return context


def origin_plan(df: pd.DataFrame, dataset_sha: str, snapshot_id: str, split: str) -> dict:
    if split not in ('development','validation'):
        raise ValueError('Invalid target split')
    targets = [int(i) for i in np.flatnonzero(df['split'].to_numpy() == split)]
    if not targets or targets != list(range(targets[0], targets[-1]+1)):
        raise ValueError('Split has missing/noncontiguous targets')
    first_target = df.iloc[targets[0]]['date']
    if (split == 'development' and first_target != DEV_START.isoformat()) or (split == 'validation' and first_target != VAL_START.isoformat()):
        raise ValueError('Wrong split first target')
    origins = []
    for j, target_index in enumerate(targets):
        i = target_index - 1
        ctx = forecast_context(df, i)
        if df.iloc[i]['date'] != (date.fromisoformat(df.iloc[target_index]['date']) - timedelta(days=1)).isoformat():
            raise ValueError('Nonconsecutive forecast origin and target')
        role = 'boundary_bridge' if j==0 else f'{split}_origin'
        origins.append({'origin_index':i,'target_index':target_index,
                        'origin_date':df.iloc[i]['date'],'target_date':df.iloc[target_index]['date'],
                        'context_first_index':i-CONTEXT+1,'context_last_index':i,
                        'context_sha256':hashlib.sha256(np.asarray(ctx, dtype='<f8').tobytes()).hexdigest(),
                        'seed':2027+i, 'role':role})
    if split == 'development' and df.iloc[origins[0]['origin_index']]['split'] != 'warmup':
        raise ValueError('Missing warmup-to-Development bridge')
    if split == 'validation' and df.iloc[origins[0]['origin_index']]['split'] != 'development':
        raise ValueError('Missing Development-to-Validation bridge')
    if split=='validation' and origins[-1]['target_date'] != END.isoformat():
        raise ValueError('Validation last target drift')
    return {'schema_version':'btc-target-origin-plan-v1','source':'binance_spot_BTCUSDT_1d_UTC',
            'target_split':split,'dataset_sha256':dataset_sha,'snapshot_id':snapshot_id,
            'count':len(origins),'boundary_bridge_count':1,'context_length':CONTEXT,
            'horizon':1,'samples':100,'quantile_levels':[0.1,0.5,0.9],
            'seed_policy':'2027_plus_origin_index', 'seed':2027,
            'development_burnin_origins':BURNIN,'development_scored_origins':len(origins)-BURNIN,
            'calibration_state':'btc_independent_causal_start_dev_then_validation_carry_forward_no_reset',
            'scale':['none','residual_ewma','return_ewma'],
            'controllers':['rolling_cqr','fixed_gamma_aci','variable_gamma_aci'],
            'alpha':0.20,'W':252,'gamma0':0.01,'ewma_decay':0.94,
            'shared_gamma_driver':'return_ewma','gamma_driver_bounds':[0.5,2.0],
            'correction':'expansion_only','regime_rule':'prior_252_origin_sigma_q75',
            'bootstrap_lengths':[10,20,40],'bootstrap_replicates':1000,'bootstrap_seed':2027,
            'first':origins[0], 'last':origins[-1], 'origins':origins,
            'contamination_free_verified':False,'btc_historical_point_in_time_verified':False,
            'scientific_clearance':False}


def snapshot_bytes(df: pd.DataFrame) -> bytes:
    # Float serialization explicit, deterministic within pinned pandas runtime.
    return df.to_csv(index=False,float_format='%.17g',lineterminator='\n',na_rep='').encode('utf-8')


def build(root: Path) -> dict:
    """Network+CPU only. Fail on missing data, checksum drift, or any plan inconsistency."""
    root = Path(root).resolve()
    cache_dir = root / 'data' / 'btcusdt' / 'binance_spot_monthly_1d'
    month_items=[]; rows=[]
    for month in months():
        item=fetch_month(cache_dir,month)
        month_items.append(item)
        parsed=parse_month((cache_dir/item['zip_name']).read_bytes(),month)
        rows.extend(parsed)
        print(json.dumps({'progress':'BTC_ARCHIVE_VERIFIED','month':month,'rows':len(parsed)},sort_keys=True),flush=True)
    quality=validate_calendar(rows)
    df=make_dataset(rows)
    # Never inspect or download rows from 2025 onwards.
    if (df['date'] >= '2025-01-01').any():
        raise ValueError('FINAL_TEST_LEAKAGE')
    snap_bytes=snapshot_bytes(df)
    snap_sha=hashlib.sha256(snap_bytes).hexdigest()
    snapshot_id='BTCUSDT_SPOT_1d_UTC_'+snap_sha[:16]
    dev=origin_plan(df,snap_sha,snapshot_id,'development')
    val=origin_plan(df,snap_sha,snapshot_id,'validation')
    if dev['last']['target_date'] != DEV_END.isoformat() or val['first']['origin_index'] != dev['last']['target_index']:
        raise ValueError('CALIBRATION_BOUNDARY_DISCONTINUITY')
    if (dev['count'] != (DEV_END-DEV_START).days+1 or val['count'] != (VAL_END-VAL_START).days+1):
        raise ValueError('LOCKED_SPLIT_COUNT_DRIFT')
    # Lock the input-month inventory and derivation provenance; ZIPs are byte checked above.
    quality.update({'source':'Binance Spot monthly 1d official archives',
                    'num_source_archives':len(month_items),
                    'first_development_origin':dev['first']['origin_date'],
                    'first_development_target':dev['first']['target_date'],
                    'first_validation_origin':val['first']['origin_date'],
                    'first_validation_target':val['first']['target_date'],
                    'raw_checksum_verified':True,'all_archives_complete':True,
                    'zero_or_negative_prices':0,'ohlc_inconsistencies':0,
                    'context_days':CONTEXT,'development_burnin_origins':BURNIN,
                    'btc_historical_point_in_time_verified':False,
                    'contamination_free_verified':False,
                    'scientific_clearance':False,'final_test_read':False})
    # all outputs derive from immutable source and deterministic payload, write-once.
    folder = root / 'artifacts' / 'btcusdt_data'
    dataset_path=folder / 'snapshots' / f'{snap_sha}.csv'
    dev_path=folder / 'plans' / f'{digest(dev)}.json'
    val_path=folder / 'plans' / f'{digest(val)}.json'
    quality_obj={'schema_version':'btc-quality-v1','snapshot_sha256':snap_sha,'quality':quality}
    quality_path=folder / 'quality' / f'{digest(quality_obj)}.json'
    manifest={'schema_version':SCHEMA,'scope':'BTCUSDT_2017-08-17_through_2024-12-31_no_final_test',
              'archive_source':'https://data.binance.vision/',
              'archive_revision_policy':'freeze_exact_bytes_and_upstream_checksum_fail_on_revision',
              'archives':month_items,'processed_sha256':snap_sha,'snapshot_id':snapshot_id,
              'development_plan_sha256':digest(dev),'validation_plan_sha256':digest(val),
              'quality_sha256':digest(quality_obj),
              'date_start':START.isoformat(),'date_end':END.isoformat(),
              'target_def':'log(C_t/C_t-1)_decimal_based_on_UTC_completed_close',
              'protocol':'TASK-007 BTC pre-data protocol v1, TASK-008A ingestion',
              'contamination_free_verified':False,
              'btc_historical_point_in_time_verified':False,
              'scientific_clearance':False,'final_test_read':False}
    manifest_sha=digest(manifest)
    manifest_path=folder/'manifests'/f'{manifest_sha}.json'
    put_once(dataset_path,snap_bytes)
    store_or_verify(dev_path,dev)
    store_or_verify(val_path,val)
    store_or_verify(quality_path,quality_obj)
    store_or_verify(manifest_path,manifest)
    for p, expected in ((dataset_path,snap_sha),(dev_path,digest(dev)),(val_path,digest(val)),
                        (quality_path,digest(quality_obj)),(manifest_path,manifest_sha)):
        if sha256_file(p)!=expected: raise ValueError(f'PUBLISHED_ARTIFACT_DRIFT: {p}')
    return {'status':'BTC_DATA_INTEGRITY_PASS_NOT_SCIENTIFIC_CLEARANCE',
            'dataset_path':str(dataset_path),'dataset_sha256':snap_sha,
            'manifest_path':str(manifest_path),'manifest_sha256':manifest_sha,
            'quality_report_path':str(quality_path),'quality_report_sha256':digest(quality_obj),
            'development_plan_path':str(dev_path),'development_plan_sha256':digest(dev),
            'validation_plan_path':str(val_path),'validation_plan_sha256':digest(val),
            'source_archives_verified':len(month_items),'daily_bars':len(df),
            'development_forecasts_per_model':dev['count'],
            'development_evaluated_per_model_after_burnin':dev['count']-BURNIN,
            'validation_forecasts_per_model':val['count'],
            'btc_historical_point_in_time_verified':False,
            'contamination_free_verified':False,'scientific_clearance':False,
            'final_test_read':False,
            'btc_inference_gate':'GO_TECHNICAL_AFTER_MANUAL_LOG_REVIEW_ONLY'}
