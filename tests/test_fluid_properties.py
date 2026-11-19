import numpy as np
import pytest

from quick_pp.fluid_type import (
    brine_density,
    estimate_fluid_properties,
    obm_filtrate_density,
)


def test_normal_reservoir_gas_properties_and_methane_hydrogen_index():
    normal = estimate_fluid_properties(
        tvd=0.0,
        surf_temp=70.0,
        temp_grad=0.0,
        surf_pres=15.0,
        pres_grad=0.0,
        gas_gravity=0.60,
        gor=0.0,
    )
    assert 0.10 <= float(normal["RHO_G"]) <= 0.14
    assert 0.22 <= float(normal["HI_G"]) <= 0.30

    methane = estimate_fluid_properties(
        tvd=0.0,
        surf_temp=70.0,
        temp_grad=0.0,
        surf_pres=15.0,
        pres_grad=0.0,
        gas_gravity=0.554,
        gor=0.0,
    )
    assert float(methane["HI_G"] / methane["RHO_G"]) == pytest.approx(2.25)


def test_hpht_gas_density_and_compressibility_are_finite():
    fluids = estimate_fluid_properties(
        tvd=0.0,
        surf_temp=185.0,
        temp_grad=0.0,
        surf_pres=105.0,
        pres_grad=0.0,
        gas_gravity=0.70,
        gor=0.0,
    )

    assert float(fluids["Z_FACTOR"]) > 1.2
    assert 0.30 <= float(fluids["RHO_G"]) <= 0.48
    assert float(fluids["HI_G"]) < 1.0
    assert all(np.isfinite(value).all() for value in fluids.values())


def test_brine_density_accounts_for_thermal_expansion_and_salinity():
    pure_water = estimate_fluid_properties(
        tvd=0.0,
        surf_temp=180.0,
        temp_grad=0.0,
        surf_pres=35.0,
        pres_grad=0.0,
        salinity_ppm=0.0,
    )
    brine = estimate_fluid_properties(
        tvd=0.0,
        surf_temp=180.0,
        temp_grad=0.0,
        surf_pres=35.0,
        pres_grad=0.0,
        salinity_ppm=200000.0,
    )

    assert float(pure_water["RHO_W"]) < 0.93
    assert float(brine["RHO_W"]) > 1.05
    assert float(brine["HI_W"]) == pytest.approx(
        float(brine["RHO_W"]) * 0.8
    )


def test_depth_profiles_and_explicit_pressure_temperature_curves():
    fluids = estimate_fluid_properties(
        tvd=np.array([0.0, 1000.0]),
        temp_grad=30.0,
        surf_temp=25.0,
        pres_grad=10.2,
        surf_pres=0.1013,
        temp_curve=np.array([60.0, 120.0]),
        pres_curve=np.array([20.0, 80.0]),
    )

    np.testing.assert_allclose(fluids["TEMP_FL"], [60.0, 120.0])
    np.testing.assert_allclose(fluids["PRES_FL"], [20.0, 80.0])
    assert fluids["RHO_G"].shape == (2,)

    partial_logs = estimate_fluid_properties(
        tvd=np.array([0.0, 1000.0]),
        temp_grad=30.0,
        surf_temp=25.0,
        pres_grad=10.2,
        surf_pres=0.1013,
        temp_curve=np.array([60.0, np.nan]),
        pres_curve=np.array([20.0, np.nan]),
    )
    np.testing.assert_allclose(partial_logs["TEMP_FL"], [60.0, 55.0])
    np.testing.assert_allclose(partial_logs["PRES_FL"], [20.0, 10.3013])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gas_gravity": 0.0}, "gas_gravity must be positive"),
        ({"gor": -1.0}, "gor must be non-negative"),
        ({"salinity_ppm": 1_000_001.0}, "salinity_ppm must be between"),
        ({"temp_curve": np.array([50.0, 60.0, 70.0])}, "temp_curve must match"),
    ],
)
def test_invalid_fluid_model_inputs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        estimate_fluid_properties(tvd=np.array([100.0, 200.0]), **kwargs)


def test_brine_density_matches_batzle_wang_reference_points():
    # Pure water at 20 degC / 0.1 MPa is ~0.998 g/cc
    assert float(brine_density(20.0, 0.1, 0.0)) == pytest.approx(0.998, abs=0.003)
    # 200,000 ppm brine at 180 degC / 35 MPa is ~1.05 g/cc
    assert float(brine_density(180.0, 35.0, 200000.0)) == pytest.approx(1.05, abs=0.01)
    # Salinity monotonically increases density at fixed P-T
    assert float(brine_density(100.0, 25.0, 150000.0)) > float(
        brine_density(100.0, 25.0, 5000.0)
    )


def test_brine_density_is_consistent_with_estimate_fluid_properties():
    fluids = estimate_fluid_properties(
        tvd=np.array([0.0, 1000.0]),
        temp_grad=30.0,
        surf_temp=25.0,
        pres_grad=10.2,
        surf_pres=0.1013,
        salinity_ppm=120000.0,
    )
    expected = brine_density(fluids["TEMP_FL"], fluids["PRES_FL"], 120000.0)
    np.testing.assert_allclose(fluids["RHO_W"], expected)


def test_obm_filtrate_density_thermal_expansion_and_pressure():
    surface = float(obm_filtrate_density(15.0, 0.1, 0.82))
    downhole = float(obm_filtrate_density(120.0, 35.0, 0.82))
    # Downhole OBM filtrate sits in the 0.78-0.83 g/cc band
    assert 0.76 <= downhole <= 0.84
    assert downhole < surface
    with pytest.raises(ValueError, match="base_oil_density"):
        obm_filtrate_density(100.0, 20.0, 0.0)
