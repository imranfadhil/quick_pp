import numpy as np
import pytest

from quick_pp.saturation import (
    estimate_rw_at_depth,
    estimate_rw_temperature_salinity,
    formation_temperature_c,
)


def test_formation_temperature_metric_and_imperial():
    # 25 degC + 30 degC/km * 2 km = 85 degC
    assert formation_temperature_c(2000.0, 25.0, 30.0, "metric") == pytest.approx(85.0)
    # 75 degF + 15 degF/kft * (2000 m = 6.56168 kft) -> converted to degC
    t_f = 75.0 + 15.0 * (2000.0 * 3.28084 / 1000.0)
    assert formation_temperature_c(2000.0, 75.0, 15.0, "imperial") == pytest.approx(
        (t_f - 32.0) * 5.0 / 9.0
    )


def test_formation_temperature_clamps_and_validates():
    out = formation_temperature_c(np.array([0.0, 20000.0]), 25.0, 30.0)
    assert out[0] == pytest.approx(25.0)
    assert out[1] == 250.0
    assert formation_temperature_c(-5000.0, 25.0, 30.0) == 5.0
    with pytest.raises(ValueError):
        formation_temperature_c(1000.0, meas_system="bad")
    # Non-finite inputs fall back to defaults
    assert formation_temperature_c(1000.0, np.nan, np.nan) == pytest.approx(55.0)


def test_rw_at_depth_matches_bateman_konen_and_floors_salinity():
    tvd = np.array([1000.0, 2000.0, 3000.0])
    rw = estimate_rw_at_depth(tvd, 35000.0, 25.0, 30.0)
    temp = formation_temperature_c(tvd, 25.0, 30.0)
    assert np.allclose(rw, estimate_rw_temperature_salinity(temp, 35000.0))
    # Rw falls with depth (hotter water is more conductive)
    assert np.all(np.diff(rw) < 0)
    # Salinity floor: 10 ppm behaves like 100 ppm
    assert estimate_rw_at_depth(1500.0, 10.0) == pytest.approx(
        estimate_rw_at_depth(1500.0, 100.0)
    )
    # Scalar in, scalar-like out
    assert np.ndim(estimate_rw_at_depth(1500.0)) == 0
