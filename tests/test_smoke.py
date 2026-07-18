"""Smoke tests: modules import, inputs validate, and the pipeline
reproduces published reference values where data allows.

Run with:  python -m pytest tests/
Tests that need unavailable data or a Gurobi license are skipped.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

def test_src_imports():
    from src.utils import get_paths                      # noqa: F401
    from src.forecasting import persistence              # noqa: F401
    from src.evaluation import calculate_metrics         # noqa: F401
    from src.statistics import diebold_mariano_test      # noqa: F401
    from src.scenarios import wind_speed_to_power_v90    # noqa: F401
    from src.visualization import plot_vss_waterfall     # noqa: F401


def test_optimization_package_imports():
    from optimization import parameters                  # noqa: F401
    from optimization import scenario_generator          # noqa: F401
    from optimization import deterministic_model         # noqa: F401
    from optimization import stochastic_model            # noqa: F401


# ------------------------------------------------------------------
# Power curve (no data required)
# ------------------------------------------------------------------

def test_power_curve_reference_points():
    from src.scenarios import wind_speed_to_power_v90

    speeds = np.array([0.0, 3.9, 5.0, 8.0, 12.0, 15.0, 24.9, 25.0])
    expected = [0.0, 0.0, 55.2703, 405.9197, 1507.7016,
                3000.0, 3000.0, 0.0]
    assert np.allclose(wind_speed_to_power_v90(speeds), expected,
                       atol=1e-3)


# ------------------------------------------------------------------
# Statistics (no data required)
# ------------------------------------------------------------------

def test_dm_test_detects_dominated_model():
    from src.statistics import diebold_mariano_test

    rng = np.random.default_rng(0)
    actual = rng.normal(size=500)
    good = actual + rng.normal(scale=0.1, size=500)
    bad = actual + rng.normal(scale=1.0, size=500)
    result = diebold_mariano_test(actual, good, bad,
                                  forecast_horizon=1)
    assert result["p_value"] < 1e-6
    assert result["better_position"] == "Model A"


def test_holm_adjustment_monotone():
    from src.statistics import holm_adjust_p_values

    adjusted = holm_adjust_p_values([0.01, 0.04, 0.03])
    assert np.all(adjusted >= [0.01, 0.04, 0.03])
    assert np.all(adjusted <= 1.0)


# ------------------------------------------------------------------
# Data-dependent tests (skipped when files are unavailable)
# ------------------------------------------------------------------

def _has_inputs() -> bool:
    try:
        from src.utils import get_paths
        paths = get_paths()
        return (paths["OPT_INPUT_DIR"]
                / "pgu_parameters.csv").exists()
    except Exception:
        return False


needs_inputs = pytest.mark.skipif(
    not _has_inputs(), reason="optimization inputs not available")


@needs_inputs
def test_parameters_load_and_validate():
    from optimization.parameters import (
        load_parameters, load_reduced_scenarios)

    params = load_parameters()
    assert params.demand.shape == (24,)
    assert len(params.pgu) == 4
    wind, probabilities = load_reduced_scenarios(100)
    assert wind.shape == (100, 24)
    assert np.isclose(probabilities.sum(), 1.0)


@needs_inputs
def test_gurobi_model_dimensions_match_manuscript():
    gurobipy = pytest.importorskip("gurobipy")  # noqa: F841
    from optimization.parameters import (
        load_parameters, load_reduced_scenarios)
    from optimization import gurobi_model

    params = load_parameters()
    wind, probabilities = load_reduced_scenarios(100)
    model, *_ = gurobi_model.build_model(params, wind, probabilities)
    model.update()
    assert model.NumVars == 36288
    assert model.NumBinVars == 7488
    assert model.NumConstrs == 64292


@needs_inputs
def test_ev_deterministic_objective_matches_manuscript():
    pytest.importorskip("gurobipy")
    from optimization.parameters import (
        load_parameters, load_reduced_scenarios)
    from optimization.deterministic_model import (
        expected_wind_profile, solve_deterministic)

    params = load_parameters()
    wind, probabilities = load_reduced_scenarios(100)
    profile = expected_wind_profile(wind, probabilities)
    ev = solve_deterministic(params, profile)
    assert abs(ev["objective"] - 125.049545) < 1e-3
    assert ev["committed_unit_hours"] == 0
