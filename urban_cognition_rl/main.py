"""
Main entry point for running all models.

Optimized with parallel processing support using joblib.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Optional
import time

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .data_types import User
from .preprocessing import load_raw_stay_data, preprocess_stay_data, process_stay_pipeline
from .clustering import add_jitter_and_cluster

from .mf_model import MFConfig, fit_mf_model, fit_mf_for_all_users
from .mfe_model import MFEpiConfig, fit_mfe_model, fit_mfe_for_all_users
from .srdyna_model import SRDynaConfig, fit_sr_dyna_model, fit_sr_dyna_for_all_users


# ============================================================================
# Parallel processing wrapper functions
# ============================================================================

def _fit_single_mf(args):
    """Helper function for parallel MF fitting."""
    user_id, user, config = args
    try:
        user_df = user.to_dataframe()
        result = fit_mf_model(user_df, config, verbose=False)
        result['user_id'] = user_id
        return result
    except Exception as e:
        return {
            'user_id': user_id,
            'fit_time_seconds': np.nan,
            'n_records': 0,
            'log_likelihood': np.nan,
            'AIC': np.nan,
            'BIC': np.nan,
            'alpha': np.nan,
            'beta': np.nan,
            'epsilon': np.nan,
            'phi': np.nan,
            'converged': False,
            'n_iterations': None,
            'optimization_message': str(e),
        }


def _fit_single_mfe(args):
    """Helper function for parallel MFE fitting."""
    user_id, user, config, high_performance = args
    try:
        user_df = user.to_dataframe()
        result = fit_mfe_model(user_df, config, verbose=False, high_performance=high_performance)
        result['user_id'] = user_id
        return result
    except Exception as e:
        return {
            'user_id': user_id,
            'fit_time_seconds': np.nan,
            'n_records': 0,
            'log_likelihood': np.nan,
            'AIC': np.nan,
            'BIC': np.nan,
            'alpha_td': np.nan,
            'beta': np.nan,
            'epsilon_explore': np.nan,
            'phi_forget': np.nan,
            'converged': False,
            'n_iterations': None,
            'optimization_message': str(e),
        }


def _fit_single_srdyna(args):
    """Helper function for parallel SR-Dyna fitting."""
    user_id, user, config = args
    try:
        user_df = user.to_dataframe()
        result = fit_sr_dyna_model(user_df, config, verbose=False)
        result['user_id'] = user_id
        return result
    except Exception as e:
        return {
            'user_id': user_id,
            'fit_time_seconds': np.nan,
            'n_records': 0,
            'log_likelihood': np.nan,
            'AIC': np.nan,
            'BIC': np.nan,
            'alpha': np.nan,
            'beta': np.nan,
            'epsilon': np.nan,
            'phi': np.nan,
            'converged': False,
            'n_iterations': None,
            'optimization_message': str(e),
        }


def fit_mf_for_all_users_parallel(users_dict: dict,
                                  config: MFConfig = None,
                                  sample_size: int = None,
                                  n_jobs: int = -1,
                                  verbose: bool = True) -> pd.DataFrame:
    """Fit MF model for all users in parallel.

    Parameters:
    - users_dict: Dictionary of User objects
    - config: MF model configuration
    - sample_size: Number of users to process
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    - verbose: Print progress

    Returns:
    - DataFrame with results
    """
    if config is None:
        config = MFConfig()

    if not JOBLIB_AVAILABLE:
        if verbose:
            print("joblib not available, falling back to sequential processing")
        return fit_mf_for_all_users(users_dict, config, sample_size, verbose)

    user_ids = list(users_dict.keys())
    if sample_size is not None:
        user_ids = user_ids[:sample_size]

    args_list = [(uid, users_dict[uid], config) for uid in user_ids]

    if verbose:
        print(f"Running parallel MF fitting for {len(user_ids)} users using {n_jobs if n_jobs > 0 else 'all'} CPUs...")

    t_start = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_single_mf)(args) for args in args_list
    )
    elapsed = time.time() - t_start

    if verbose:
        print(f"Parallel MF fitting completed in {elapsed:.2f}s")

    return pd.DataFrame(results)


def fit_mfe_for_all_users_parallel(users_dict: dict,
                                    config: MFEpiConfig = None,
                                    sample_size: int = None,
                                    high_performance: bool = False,
                                    n_jobs: int = -1,
                                    verbose: bool = True) -> pd.DataFrame:
    """Fit MFE model for all users in parallel.

    Parameters:
    - users_dict: Dictionary of User objects
    - config: MFE model configuration
    - sample_size: Number of users to process
    - high_performance: Use high performance mode
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    - verbose: Print progress

    Returns:
    - DataFrame with results
    """
    if config is None:
        config = MFEpiConfig()

    if not JOBLIB_AVAILABLE:
        if verbose:
            print("joblib not available, falling back to sequential processing")
        return fit_mfe_for_all_users(users_dict, config, sample_size, verbose, high_performance)

    user_ids = list(users_dict.keys())
    if sample_size is not None:
        user_ids = user_ids[:sample_size]

    args_list = [(uid, users_dict[uid], config, high_performance) for uid in user_ids]

    if verbose:
        print(f"Running parallel MFE fitting for {len(user_ids)} users using {n_jobs if n_jobs > 0 else 'all'} CPUs...")

    t_start = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_single_mfe)(args) for args in args_list
    )
    elapsed = time.time() - t_start

    if verbose:
        print(f"Parallel MFE fitting completed in {elapsed:.2f}s")

    return pd.DataFrame(results)


def fit_sr_dyna_for_all_users_parallel(users_dict: dict,
                                        config: SRDynaConfig = None,
                                        sample_size: int = None,
                                        n_jobs: int = -1,
                                        verbose: bool = True) -> pd.DataFrame:
    """Fit SR-Dyna model for all users in parallel.

    Parameters:
    - users_dict: Dictionary of User objects
    - config: SR-Dyna model configuration
    - sample_size: Number of users to process
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    - verbose: Print progress

    Returns:
    - DataFrame with results
    """
    if config is None:
        config = SRDynaConfig()

    if not JOBLIB_AVAILABLE:
        if verbose:
            print("joblib not available, falling back to sequential processing")
        return fit_sr_dyna_for_all_users(users_dict, config, sample_size, verbose)

    user_ids = list(users_dict.keys())
    if sample_size is not None:
        user_ids = user_ids[:sample_size]

    args_list = [(uid, users_dict[uid], config) for uid in user_ids]

    if verbose:
        print(f"Running parallel SR-Dyna fitting for {len(user_ids)} users using {n_jobs if n_jobs > 0 else 'all'} CPUs...")

    t_start = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_fit_single_srdyna)(args) for args in args_list
    )
    elapsed = time.time() - t_start

    if verbose:
        print(f"Parallel SR-Dyna fitting completed in {elapsed:.2f}s")

    return pd.DataFrame(results)


def load_and_preprocess_data(data_dir: str = 'data/st') -> pd.DataFrame:
    """Load and preprocess raw stay data."""
    print("Loading raw data...")
    st_data = load_raw_stay_data(data_dir)

    print("Preprocessing...")
    st_sample = preprocess_stay_data(st_data)

    print("Processing pipeline...")
    st_processed = process_stay_pipeline(st_sample)

    print("Clustering...")
    st_clustered = add_jitter_and_cluster(st_processed)

    return st_clustered


def convert_to_users(df: pd.DataFrame) -> dict:
    """Convert DataFrame to User objects."""
    print("Converting to User objects...")
    return User.from_dataframe(df)


def run_mf_model(users_dict: dict, sample_size: int = None, config: MFConfig = None,
                parallel: bool = True, n_jobs: int = -1) -> pd.DataFrame:
    """Run MF model for all users.

    Parameters:
    - users_dict: Dictionary of User objects
    - sample_size: Number of users to process
    - config: MF model configuration
    - parallel: Use parallel processing (requires joblib)
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    """
    print("\n" + "=" * 60)
    print("Running Model-Free (MF) Estimation")
    print("=" * 60)

    if config is None:
        config = MFConfig()

    if parallel and JOBLIB_AVAILABLE:
        return fit_mf_for_all_users_parallel(users_dict, config=config, sample_size=sample_size, n_jobs=n_jobs)
    return fit_mf_for_all_users(users_dict, config=config, sample_size=sample_size)


def run_mfe_model(users_dict: dict, sample_size: int = None, config: MFEpiConfig = None,
                 high_performance: bool = False, parallel: bool = True, n_jobs: int = -1) -> pd.DataFrame:
    """Run MFE model for all users.

    Parameters:
    - users_dict: Dictionary of User objects
    - sample_size: Number of users to process
    - config: MFE model configuration
    - high_performance: Use high performance mode
    - parallel: Use parallel processing (requires joblib)
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    """
    print("\n" + "=" * 60)
    print("Running Model-Free with Episodic Memory (MFE) Estimation")
    print("=" * 60)

    if config is None:
        config = MFEpiConfig()

    if parallel and JOBLIB_AVAILABLE:
        return fit_mfe_for_all_users_parallel(users_dict, config=config, sample_size=sample_size, high_performance=high_performance, n_jobs=n_jobs)
    return fit_mfe_for_all_users(users_dict, config=config, sample_size=sample_size, high_performance=high_performance)


def run_sr_dyna_model(users_dict: dict, sample_size: int = None, config: SRDynaConfig = None,
                     parallel: bool = True, n_jobs: int = -1) -> pd.DataFrame:
    """Run SR-Dyna model for all users.

    Parameters:
    - users_dict: Dictionary of User objects
    - sample_size: Number of users to process
    - config: SR-Dyna model configuration
    - parallel: Use parallel processing (requires joblib)
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    """
    print("\n" + "=" * 60)
    print("Running Model-Based SR-Dyna Estimation")
    print("=" * 60)

    if config is None:
        config = SRDynaConfig()

    if parallel and JOBLIB_AVAILABLE:
        return fit_sr_dyna_for_all_users_parallel(users_dict, config=config, sample_size=sample_size, n_jobs=n_jobs)
    return fit_sr_dyna_for_all_users(users_dict, config=config, sample_size=sample_size)


def main():
    """Main entry point for running all models."""
    print("=" * 60)
    print("Urban Cognition through RL Models")
    print("=" * 60)

    df = load_and_preprocess_data()
    users_dict = convert_to_users(df)

    # mf_results = run_mf_model(users_dict)
    mfe_results = run_mfe_model(users_dict, sample_size=10, high_performance=True, parallel=False)
    # srdyna_results = run_sr_dyna_model(users_dict, sample_size=10)

    # return mf_results, mfe_results, srdyna_results
    return mfe_results


if __name__ == "__main__":
    main()
