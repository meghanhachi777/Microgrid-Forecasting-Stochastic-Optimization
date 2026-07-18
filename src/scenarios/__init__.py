"""Scenario generation, power-curve conversion, and reduction."""

from .generation import (
    center_residuals,
    generate_residual_bootstrap_scenarios,
    asymmetric_scaling_factors,
    scale_residual_trajectories_asymmetric,
    generate_calibrated_scenarios,
    correlation_preservation_error,
)
from .power_curve import (
    RATED_POWER_KW,
    CUT_IN_SPEED,
    RATED_SPEED,
    CUT_OUT_SPEED,
    adjust_wind_speed_to_hub_height,
    wind_speed_to_power_v90,
    net_power,
)
from .reduction import (
    standardize_hourwise,
    reduce_scenarios_kmeans,
    weighted_mean,
    weighted_variance,
    weighted_quantile_1d,
    weighted_quantiles_by_hour,
    evaluate_reduced_set,
)
