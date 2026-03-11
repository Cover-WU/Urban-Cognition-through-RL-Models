"""
Visualization functions for user trajectories and clusters.
"""

import plotly.express as px
import pandas as pd


def visualize_user_clusters(
    df: pd.DataFrame,
    user_id: int,
    zoom: int = 12,
    height: int = 600
) -> px.scatter_mapbox:
    """
    Create an interactive map showing cluster locations for a specific user.

    Parameters:
    - df: DataFrame with 'who', 'lon', 'lat', 'cluster_id' columns
    - user_id: User ID to visualize
    - zoom: Initial zoom level
    - height: Map height in pixels

    Returns:
    - Plotly Express scatter_mapbox figure
    """
    user_data = df[df['who'] == user_id].copy()

    if user_data.empty:
        raise ValueError(f"No data found for user {user_id}")

    user_data['cluster_label'] = user_data['cluster_id'].apply(
        lambda x: f"Missing (Outside Study Area)" if x == 0
                  else (f"HDBSCAN Noise" if x == -1
                  else f"Cluster {x}")
    )

    color_discrete_map = {
        'Missing (Outside Study Area)': 'red',
        'HDBSCAN Noise': 'gray',
    }

    cluster_labels = [f'Cluster {i}' for i in sorted(user_data[user_data['cluster_id'] >= 1]['cluster_id'].unique())]

    fig = px.scatter_mapbox(
        user_data,
        lat='lat',
        lon='lon',
        color='cluster_label',
        color_discrete_map=color_discrete_map,
        category_orders={'cluster_label': ['Missing (Outside Study Area)', 'HDBSCAN Noise'] + cluster_labels},
        hover_data={
            'who': True,
            't_start': True,
            't_end': True,
            'cluster_id': True,
            'lat': ':.5f',
            'lon': ':.5f'
        },
        title=f"Stay locations for User {user_id}",
        zoom=zoom,
        height=height,
        size_max=15
    )

    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    return fig


def compute_user_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cluster statistics per user.

    Parameters:
    - df: DataFrame with 'who' and 'cluster_id' columns

    Returns:
    - DataFrame with per-user statistics
    """
    user_stats = df.groupby('who').agg({
        'cluster_id': [
            'count',
            lambda x: (x == 0).sum(),
            lambda x: (x != 0).sum(),
            lambda x: (x == -1).sum(),
            lambda x: (x >= 1).sum(),
            lambda x: (x[x >= 1].nunique() if (x >= 1).any() else 0)
        ]
    }).reset_index()

    user_stats.columns = ['who', 'total_stays', 'missing_stays', 'valid_stays', 'noise_stays', 'clustered_stays', 'num_clusters']

    return user_stats
