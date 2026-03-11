"""
Urban Cognition through RL Models

This package provides implementations of various reinforcement learning models
for analyzing urban human mobility patterns.

Models include:
- Model-Free (MF) RL with TD learning
- Model-Free with Episodic Memory (MFE)
- Model-Based SR-Dyna with Successor Representation
"""

from .data_types import Visit, Trajectory, User

from .mf_model import MFConfig, fit_mf_model, fit_mf_for_all_users
from .mfe_model import MFEpiConfig, EpisodicMemory, fit_mfe_model, fit_mfe_for_all_users
from .srdyna_model import (
    SRDynaConfig,
    SRRecord,
    WorldModel,
    SRMemory,
    compute_Q,
    fit_sr_dyna_model,
    fit_sr_dyna_for_all_users
)

from .preprocessing import (
    load_raw_stay_data,
    preprocess_stay_data,
    process_stay_pipeline,
    merge_consecutive_stays,
    filter_short_stays
)

from .clustering import add_jitter_and_cluster, jitter_within_radius

from .visualization import visualize_user_clusters, compute_user_cluster_stats


__version__ = '1.0.0'

__all__ = [
    # Data types
    'Visit',
    'Trajectory',
    'User',

    # MF model
    'MFConfig',
    'fit_mf_model',
    'fit_mf_for_all_users',

    # MFE model
    'MFEpiConfig',
    'EpisodicMemory',
    'fit_mfe_model',
    'fit_mfe_for_all_users',

    # SR-Dyna model
    'SRDynaConfig',
    'SRRecord',
    'WorldModel',
    'SRMemory',
    'compute_Q',
    'fit_sr_dyna_model',
    'fit_sr_dyna_for_all_users',

    # Preprocessing
    'load_raw_stay_data',
    'preprocess_stay_data',
    'process_stay_pipeline',
    'merge_consecutive_stays',
    'filter_short_stays',

    # Clustering
    'add_jitter_and_cluster',
    'jitter_within_radius',

    # Visualization
    'visualize_user_clusters',
    'compute_user_cluster_stats',
]
