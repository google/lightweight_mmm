"""Shared pytest fixtures and configuration for lightweight_mmm tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_data():
    """Create mock data for testing MMM models."""
    np.random.seed(42)
    n_time_periods = 52
    n_media_channels = 3
    n_geos = 2
    
    # Generate synthetic data
    data = {
        'date': pd.date_range('2023-01-01', periods=n_time_periods, freq='W'),
        'sales': np.random.poisson(1000, n_time_periods) + np.random.normal(0, 50, n_time_periods),
    }
    
    # Add media spend data
    for i in range(n_media_channels):
        data[f'media_{i}'] = np.random.exponential(1000, n_time_periods)
    
    # Add geo data
    for i in range(n_geos):
        data[f'geo_{i}_sales'] = np.random.poisson(500, n_time_periods)
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        'n_media_channels': 3,
        'n_geos': 2,
        'model_type': 'adstock',
        'priors': {
            'intercept': {'mean': 0, 'std': 1},
            'coef_media': {'mean': 0, 'std': 0.1},
        },
        'hyperparameters': {
            'learning_rate': 0.001,
            'n_iterations': 1000,
            'batch_size': 32,
        }
    }


@pytest.fixture
def sample_media_data():
    """Generate sample media spend data."""
    np.random.seed(123)
    return np.random.rand(52, 3) * 10000  # 52 weeks, 3 channels


@pytest.fixture
def sample_target_data():
    """Generate sample target (sales) data."""
    np.random.seed(123)
    base_sales = 10000
    trend = np.linspace(0, 1000, 52)
    seasonality = 500 * np.sin(np.linspace(0, 4 * np.pi, 52))
    noise = np.random.normal(0, 200, 52)
    return base_sales + trend + seasonality + noise


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    import random
    random.seed(42)
    
    # Reset JAX random seed if JAX is available
    try:
        import jax
        jax.random.PRNGKey(42)
    except ImportError:
        pass


@pytest.fixture
def mock_model_params():
    """Create mock model parameters."""
    return {
        'intercept': np.array([1000.0]),
        'coef_media': np.array([0.1, 0.2, 0.15]),
        'coef_trend': np.array([10.0]),
        'saturation_parameters': {
            'alphas': np.array([2.0, 1.5, 2.5]),
            'betas': np.array([0.5, 0.6, 0.4])
        },
        'adstock_parameters': {
            'convolve_window': 3,
            'decay_rates': np.array([0.3, 0.4, 0.35])
        }
    }


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and assert log messages."""
    with caplog.at_level('DEBUG'):
        yield caplog