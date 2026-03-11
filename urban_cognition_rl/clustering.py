"""
Spatial clustering functions using HDBSCAN.
"""

import numpy as np
import pandas as pd
from pyproj import Transformer
import hdbscan


def jitter_within_radius(xy: np.ndarray, max_radius_m: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply uniform random jitter within a circle of specified radius.

    Uses sqrt(r) for radius to ensure uniform area distribution.
    """
    if xy.shape[1] != 2:
        raise ValueError("Input array should have exactly 2 columns (x, y)")

    n = xy.shape[0]
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radii = max_radius_m * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    offsets = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    return xy + offsets


def add_jitter_and_cluster(
    df: pd.DataFrame,
    jitter_radius_m: float = 300,
    min_cluster_size: int = 6,
    min_samples: int = 6,
    seed: int = 114514
) -> pd.DataFrame:
    """
    Apply 300m random jitter to coordinates, then perform HDBSCAN clustering.

    This function:
    1. Converts lon/lat to UTM coordinates (EPSG:32650)
    2. Applies random jitter within specified radius
    3. Converts back to WGS84 lon/lat
    4. Performs HDBSCAN clustering on the jittered coordinates

    Cluster labels:
    - 0: Missing coordinates (outside study area)
    - -1: HDBSCAN noise points
    - 1+: HDBSCAN clusters

    Parameters:
    - df: DataFrame with 'who', 'lon', 'lat' columns
    - jitter_radius_m: Radius for random jitter in meters
    - min_cluster_size: HDBSCAN parameter
    - min_samples: HDBSCAN parameter
    - seed: Random seed

    Returns:
    - DataFrame with additional 'cluster_id' column
    """
    RNG = np.random.default_rng(seed)
    df = df.copy()

    def cluster_one_person(person_df: pd.DataFrame) -> pd.DataFrame:
        person_df = person_df.copy()

        has_coords = ~person_df[['lon', 'lat']].isna().any(axis=1)
        valid_mask = has_coords.values
        valid_count = valid_mask.sum()
        total_count = len(person_df)

        person_df['cluster_id'] = 0

        if valid_count == 0:
            return person_df

        valid_indices = person_df[has_coords].index
        valid_lon = person_df.loc[valid_indices, 'lon'].values.copy()
        valid_lat = person_df.loc[valid_indices, 'lat'].values.copy()

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
        inverse_transformer = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)

        utm_x, utm_y = transformer.transform(valid_lon, valid_lat)

        xy_valid = np.column_stack([utm_x, utm_y])
        xy_jittered = jitter_within_radius(xy_valid, max_radius_m=jitter_radius_m, rng=RNG)

        jittered_lon, jittered_lat = inverse_transformer.transform(xy_jittered[:, 0], xy_jittered[:, 1])

        person_df.loc[valid_indices, 'lon'] = jittered_lon
        person_df.loc[valid_indices, 'lat'] = jittered_lat

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        labels_valid = clusterer.fit_predict(xy_jittered)

        labels_shifted = np.where(labels_valid == -1, -1, labels_valid + 1)

        person_df.loc[valid_indices, 'cluster_id'] = labels_shifted

        return person_df

    return df.groupby('who', group_keys=False).apply(cluster_one_person)
