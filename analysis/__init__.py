"""Janus V5 Data Analysis Pipeline.

A comprehensive analysis toolkit to validate hypothesis before deploying
complex deep learning models.
"""

__version__ = "1.0.0"
__author__ = "Janus Team"

from .step1_data_health import run_data_health_check
from .step2_target_analysis import run_target_analysis
from .step3_correlation_analysis import run_correlation_analysis
from .step4_lag_analysis import run_lag_analysis
from .step5_baseline_models import run_baseline_modeling

__all__ = [
    'run_data_health_check',
    'run_target_analysis',
    'run_correlation_analysis',
    'run_lag_analysis',
    'run_baseline_modeling',
]
