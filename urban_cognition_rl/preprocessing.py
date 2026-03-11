"""
Data preprocessing functions for stay data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


def merge_consecutive_stays(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """
    Merge consecutive stay records with the same location and gap <= gap_minutes
    OR same location AND adjacent calendar dates

    Parameters:
    - df: DataFrame sorted by who and t_start
    - gap_minutes: Maximum gap threshold (minutes)

    Returns:
    - Merged DataFrame
    """
    if len(df) == 0:
        return df.copy()

    time_diff = (df['t_start'].shift(-1) - df['t_end']).dt.total_seconds() / 60

    lon_same = np.isclose(df['lon'].shift(-1), df['lon'], rtol=1e-9)
    lat_same = np.isclose(df['lat'].shift(-1), df['lat'], rtol=1e-9)
    pos_same = lon_same & lat_same

    date_adjacent = (df['date'].shift(-1) - df['date']) == 1

    need_merge = pos_same & ((time_diff <= gap_minutes) | date_adjacent)
    need_merge.iloc[-1] = False

    group_id = (~need_merge).cumsum()

    df = df.copy()
    df['group_id'] = group_id
    merged = df.groupby(['who', 'group_id'], as_index=False).agg({
        't_start': 'min',
        't_end': 'max',
        'lon': 'first',
        'lat': 'first',
        'ptype': 'first',
        'poi': 'first',
        'date': 'first'
    })
    merged = merged.drop(columns=['group_id'])

    return merged


def filter_short_stays(df: pd.DataFrame, min_minutes: int = 30) -> pd.DataFrame:
    """
    Filter out stay records shorter than min_minutes

    Parameters:
    - df: DataFrame
    - min_minutes: Minimum stay duration (minutes)

    Returns:
    - Filtered DataFrame
    """
    stay_duration = (df['t_end'] - df['t_start']).dt.total_seconds() / 60
    return df[stay_duration >= min_minutes].reset_index(drop=True)


def load_raw_stay_data(data_dir: str = 'data/st') -> pd.DataFrame:
    """
    Load raw stay data from CSV files.

    Parameters:
    - data_dir: Directory containing the raw data files

    Returns:
    - DataFrame with raw stay data
    """
    t_start = pd.read_csv(f'{data_dir}/reptoire.csv').iloc[:, 0]
    t_end = pd.read_csv(f'{data_dir}/nif.csv').iloc[:, 0]
    ptype = pd.read_csv(f'{data_dir}/model.csv').iloc[:, 0]
    poi = pd.read_csv(f'{data_dir}/iop.csv').iloc[:, 0]
    who = pd.read_csv(f'{data_dir}/est.csv').iloc[:, 0]
    date = pd.read_csv(f'{data_dir}/aoz.csv').iloc[:, 0]
    lon_p1 = pd.read_csv(f'{data_dir}/mean_log_p1.csv').iloc[:, 0]
    lon_p2 = pd.read_csv(f'{data_dir}/mean_log_p2.csv').iloc[:, 0]
    lat_p1 = pd.read_csv(f'{data_dir}/std_log_p1.csv').iloc[:, 0]
    lat_p2 = pd.read_csv(f'{data_dir}/std_log_p2.csv').iloc[:, 0]

    lon = pd.concat([lon_p1, lon_p2], ignore_index=True)
    lat = pd.concat([lat_p1, lat_p2], ignore_index=True)

    st_data = pd.DataFrame({
        't_start': t_start,
        't_end': t_end,
        'ptype': ptype,
        'poi': poi,
        'who': who,
        'date': date,
        'lon': lon,
        'lat': lat
    })

    return st_data


def preprocess_stay_data(st_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess stay data: convert types and normalize coordinates.

    Parameters:
    - st_data: Raw stay data DataFrame

    Returns:
    - Preprocessed DataFrame
    """
    st_sample = st_data.assign(
        who=st_data['who'].astype(int),
        date=st_data['date'].astype(int) + 20202020,
        t_start=pd.to_datetime(st_data['t_start'] + 1677600000, unit='s'),
        t_end=pd.to_datetime(st_data['t_end'] + 1677600000, unit='s'),
        lon=st_data['lon'] / 4,
        lat=st_data['lat'] * 4,
        ptype=st_data['ptype'].astype(int),
        poi=st_data['poi'].astype(int)
    )

    return st_sample[['who', 'date', 't_start', 't_end', 'lon', 'lat', 'ptype', 'poi']]


def process_stay_pipeline(st_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for stay data.

    Steps:
    1. Merge consecutive stays with gap <= 30 minutes
    2. Filter stays shorter than 30 minutes
    3. Merge again

    Parameters:
    - st_sample: Preprocessed stay data

    Returns:
    - Processed DataFrame
    """
    st_processed = st_sample.copy()

    def process_single_person(group):
        return merge_consecutive_stays(group, gap_minutes=30)

    st_processed = st_processed.groupby('who', group_keys=False).apply(process_single_person)
    st_processed = filter_short_stays(st_processed, min_minutes=30)
    st_processed = st_processed.groupby('who', group_keys=False).apply(process_single_person)

    return st_processed.sort_values(by=['who', 't_start']).reset_index(drop=True)
