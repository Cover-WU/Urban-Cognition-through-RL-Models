"""
Main entry point for running all models.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from .data_types import User
from .preprocessing import load_raw_stay_data, preprocess_stay_data, process_stay_pipeline
from .clustering import add_jitter_and_cluster

from .mf_model import MFConfig, fit_mf_model, fit_mf_for_all_users
from .mfe_model import MFEpiConfig, fit_mfe_model, fit_mfe_for_all_users
from .srdyna_model import SRDynaConfig, fit_sr_dyna_model, fit_sr_dyna_for_all_users


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


def run_mf_model(users_dict: dict, sample_size: int = None, config: MFConfig = None) -> pd.DataFrame:
    """Run MF model for all users."""
    print("\n" + "=" * 60)
    print("Running Model-Free (MF) Estimation")
    print("=" * 60)

    if config is None:
        config = MFConfig()

    return fit_mf_for_all_users(users_dict, config=config, sample_size=sample_size)


def run_mfe_model(users_dict: dict, sample_size: int = None, config: MFEpiConfig = None) -> pd.DataFrame:
    """Run MFE model for all users."""
    print("\n" + "=" * 60)
    print("Running Model-Free with Episodic Memory (MFE) Estimation")
    print("=" * 60)

    if config is None:
        config = MFEpiConfig()

    return fit_mfe_for_all_users(users_dict, config=config, sample_size=sample_size)


def run_sr_dyna_model(users_dict: dict, sample_size: int = None, config: SRDynaConfig = None) -> pd.DataFrame:
    """Run SR-Dyna model for all users."""
    print("\n" + "=" * 60)
    print("Running Model-Based SR-Dyna Estimation")
    print("=" * 60)

    if config is None:
        config = SRDynaConfig()

    return fit_sr_dyna_for_all_users(users_dict, config=config, sample_size=sample_size)


def main():
    """Main entry point for running all models."""
    print("=" * 60)
    print("Urban Cognition through RL Models")
    print("=" * 60)

    df = load_and_preprocess_data()
    users_dict = convert_to_users(df)

    # mf_results = run_mf_model(users_dict)
    # mfe_results = run_mfe_model(users_dict, sample_size=10)
    srdyna_results = run_sr_dyna_model(users_dict, sample_size=10)

    # return mf_results, mfe_results, srdyna_results
    return srdyna_results


if __name__ == "__main__":
    main()
