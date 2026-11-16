"""Read-only data resolution, exact UTC interpolation and task-count audits."""
from functools import lru_cache
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from .config import resolve


def workload_source(configured):
    path = resolve(configured)
    candidates = [path, path.parent / path.stem / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, None
    archive = path.with_suffix('.zip')
    if archive.is_file():
        with zipfile.ZipFile(archive) as stream:
            names = [n for n in stream.namelist() if Path(n).name == path.name]
        if len(names) != 1:
            raise ValueError('Workload ZIP must contain exactly one matching pickle')
        return archive, names[0]
    raise FileNotFoundError(f'Workload missing: {path} (including read-only folder/ZIP fallback)')


@lru_cache(maxsize=2)
def read_workload(configured):
    path, member = workload_source(configured)
    if member:
        with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
            df = pd.read_pickle(stream)
    else:
        df = pd.read_pickle(path)
    if not {'interval_15m', 'tasks_matrix'} <= set(df.columns):
        raise ValueError('Invalid workload schema')
    df = df.copy()
    times = pd.DatetimeIndex(df['interval_15m'])
    if times.tz is None:
        raise ValueError('Workload timestamps must contain an explicit timezone')
    df['interval_15m'] = times.tz_convert('UTC')
    if df.interval_15m.duplicated().any():
        raise ValueError('Duplicate workload intervals would silently select the first row')
    if (df.interval_15m != df.interval_15m.dt.floor('15min')).any():
        raise ValueError('Workload timestamps must align to UTC quarter hours')
    return df


def window(sim):
    start = pd.Timestamp(year=sim['year'], month=sim['month'], day=sim['init_day'],
                         hour=sim['init_hour'], tz='UTC')
    return start, start + pd.Timedelta(days=sim['duration_days'])


def required_window(sim):
    start, end = window(sim)
    if sim.get('reset_mode') == 'randomized_aligned':
        year = pd.Timestamp(year=sim['year'], month=1, day=1, tz='UTC')
        day = start.dayofyear - 1
        start = year + pd.Timedelta(days=max(0, day - 7))
        end = year + pd.Timedelta(days=min(364, day + 7), hours=23) + pd.Timedelta(days=sim['duration_days'])
    return start, end


def workload_counts(sim, dcs):
    """Use the real extractor (scale=1, task_scale=5, group_size=1).

    task_scale changes CPU/GPU/memory, NOT the number of tasks. Zero-arrival
    slots are retained in quantiles. Also compute the all-Defer SLA upper bound
    to make clear that arrival <=400 does not certify future policy queues.
    """
    import random
    from utils.workload_utils import extract_tasks_from_row
    df = read_workload(sim['workload_path']).set_index('interval_15m')
    start, end = required_window(sim)
    times = pd.date_range(start, end, freq='15min', inclusive='left')
    source_times = times.map(lambda t: t.replace(year=2020))
    if source_times.min() < df.index.min() or source_times.max() > df.index.max():
        raise ValueError('Workload does not span required calendar window')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    counts, active, expirations = [], [], []
    import heapq
    try:
        for now, source in zip(times, source_times):
            while expirations and expirations[0] < now.value:
                heapq.heappop(expirations)
            if source in df.index:
                tasks = extract_tasks_from_row(df.loc[source], scale=1, task_scale=5,
                                               group_size=1, datacenter_configs=dcs,
                                               current_time_utc=now)
            else:
                tasks = []
            counts.append(len(tasks))
            # Tasks strictly overdue are forced home but still appear in one
            # decision observation before the forced return; include that step.
            for task in tasks:
                heapq.heappush(expirations, (task.sla_deadline + pd.Timedelta(minutes=15)).value)
            active.append(len(expirations))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return dict(start=str(start), end_exclusive=str(end), steps=len(counts),
                maximum=int(max(counts)), p95=float(np.percentile(counts, 95)),
                p99=float(np.percentile(counts, 99)), mean=float(np.mean(counts)),
                all_defer_queue_upper_bound=int(max(active)), task_scale=5,
                missing_arrival_intervals=int(sum(t not in df.index for t in source_times)))


def external_paths(location, year):
    return {
        'price': resolve(f'data/electricity_prices/standardized/{location}/{year}/{location}_electricity_prices_{year}.csv'),
        'ci': resolve(f'data/carbon_intensity/{location}/{year}/{location}_{year}_hourly.csv'),
        'weather': resolve(f'data/weather/{location}/{year}.json'),
    }


def _series(index, values, label):
    index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    data = pd.Series(np.asarray(values, dtype=float), index=index).sort_index()
    if data.index.has_duplicates or not np.isfinite(data.values).all():
        raise ValueError(f'Duplicate timestamps or non-finite data: {label}')
    expected = pd.date_range(data.index[0], data.index[-1], freq='h')
    if not data.index.equals(expected):
        raise ValueError(f'Missing/non-hourly timestamps: {label}')
    return data


@lru_cache(maxsize=10)
def external_data(location, year):
    paths = external_paths(location, year)
    price = pd.read_csv(paths['price'])
    ci = pd.read_csv(paths['ci'])
    raw = json.loads(paths['weather'].read_text())
    weather_times = pd.to_datetime(raw['hourly']['time'], utc=True) - pd.Timedelta(seconds=raw.get('utc_offset_seconds', 0))
    series = {
        'price': _series(price['Datetime (UTC)'], price['Price (USD/MWh)'], 'price'),
        'ci': _series(ci['Datetime (UTC)'], ci['Carbon Intensity gCO₂eq/kWh (direct)'], 'ci'),
        'temperature': _series(weather_times, raw['hourly']['temperature_2m'], 'temperature'),
        'humidity': _series(weather_times, raw['hourly']['relative_humidity_2m'], 'humidity'),
    }
    start = pd.Timestamp(year=year, month=1, day=1, tz='UTC')
    end = pd.Timestamp(year=year+1, month=1, day=1, tz='UTC')
    index = pd.date_range(start, end, freq='15min', inclusive='left')
    arrays = {}
    for name, values in series.items():
        if name == 'price':
            # Preserve the existing price manager's outlier clipping.
            q1, q3 = np.percentile(values, [15, 85])
            values = values.clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        arrays[name] = values.reindex(index.union(values.index)).interpolate(method='time').reindex(index).to_numpy()
    return dict(series=series, arrays=arrays, index=index, paths=paths)


def validate_coverage(data, start, end):
    # DC weather uses a next-step lookahead; CI forecasts four more steps.
    required_end = end + pd.Timedelta(hours=1)
    for name, values in data['series'].items():
        if start < values.index[0] or required_end > values.index[-1]:
            raise ValueError(f'{name} lacks coverage [{start}, {required_end}]')


def time_index(now):
    beginning = pd.Timestamp(year=now.year, month=1, day=1, tz='UTC')
    return int((now - beginning).total_seconds() // 900)
