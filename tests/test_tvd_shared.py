"""Shared TVD helpers in quick_pp.utils and the callers that delegate to them."""

import numpy as np
import pandas as pd
import pytest

from quick_pp.handover import _interpolate_tvd
from quick_pp.utils import interpolate_tvd_on_md, survey_tvd_stations

SURVEY = pd.DataFrame(
    {
        "md": [500.0, 1000.0, 1500.0],
        "inc": [0.0, 0.0, 30.0],
        "azim": [0.0, 0.0, 90.0],
    }
)


def test_survey_tvd_stations_tie_in_sort_and_dedupe():
    md, tvd, inc = survey_tvd_stations(
        [1500.0, 500.0, 1000.0, 1000.0], [30.0, 0.0, 5.0, 0.0], [90.0, 0.0, 0.0, 0.0]
    )
    assert md.tolist() == [500.0, 1000.0, 1500.0]
    # Last duplicate wins (inc=0 at 1000 m), so the hole is vertical to 1000 m
    assert inc.tolist() == [0.0, 0.0, 30.0]
    assert tvd[0] == pytest.approx(500.0)
    assert tvd[1] == pytest.approx(1000.0)
    assert 1000.0 < tvd[2] < 1500.0


def test_survey_tvd_stations_handles_empty_and_nan():
    md, tvd, inc = survey_tvd_stations([np.nan], [np.nan], [np.nan])
    assert md.size == tvd.size == inc.size == 0


def test_interpolate_tvd_on_md_rules():
    md, tvd, inc = survey_tvd_stations(SURVEY["md"], SURVEY["inc"], SURVEY["azim"])
    grid = np.array([0.0, 250.0, 1500.0, 1700.0, np.nan])
    out = interpolate_tvd_on_md(grid, md, tvd, inc)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(250.0)
    assert out[2] == pytest.approx(tvd[-1])
    slope = (tvd[-1] - tvd[-2]) / (md[-1] - md[-2])
    assert out[3] == pytest.approx(tvd[-1] + 200.0 * slope)
    assert np.isnan(out[4])
    # No stations -> all NaN; no inclination for a single station -> vertical
    assert np.isnan(interpolate_tvd_on_md([100.0], [], [])).all()
    assert interpolate_tvd_on_md([1100.0], [1000.0], [1000.0])[0] == pytest.approx(
        1100.0
    )


def test_handover_extrapolates_below_last_station():
    md = np.array([1500.0, 1700.0])
    out = _interpolate_tvd(md, None, SURVEY)
    assert out[1] > out[0]  # previously clamped flat at the last station
    assert out[1] - out[0] < 200.0  # deviated, so less than the MD step


def test_handover_tvd_curve_path_extrapolates():
    well_df = pd.DataFrame({"DEPTH": [1000.0, 1100.0], "TVD": [990.0, 1080.0]})
    out = _interpolate_tvd(np.array([1200.0]), well_df, None)
    assert out[0] == pytest.approx(1170.0)
