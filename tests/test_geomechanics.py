import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quick_pp.app.backend.router import api_router
from quick_pp.rock_physics.geomechanics import (
    _integrate_overburden_mpa,
    estimate_log_stress_profile,
    estimate_pore_pressure_res,
)


def test_overburden_pressure_integrates_density_over_tvd_in_original_order():
    pressure = _integrate_overburden_mpa(
        rhob=[2.0, 2.0, 2.0],
        tvd_m=[1000.0, 0.0, 500.0],
    )

    expected = np.array([19.6133, 0.0, 9.80665])
    np.testing.assert_allclose(pressure, expected, rtol=1e-5)


def test_log_stress_profile_uses_sonic_poisson_ratio_and_equivalent_stress():
    result = estimate_log_stress_profile(
        md=[0.0, 1000.0],
        tvd=[0.0, 1000.0],
        rhob=[2.0, 2.0],
        pore_pressure_gradient=0.433,
        biot_coefficient=1.0,
        dtc=[80.0, 80.0],
        dts=[140.0, 140.0],
        depth_unit="ft",
    )

    assert result["sv_psi"][0] == pytest.approx(0.0)
    assert result["pp_psi"][1] == pytest.approx(433.0)
    assert result["nu"][1] == pytest.approx(0.257, abs=0.001)
    assert result["ncs_equivalent_psi"][1] < result["sv_psi"][1]


def test_log_stress_profile_rejects_mismatched_sonic_logs():
    with pytest.raises(ValueError, match="match the shape"):
        estimate_log_stress_profile(
            md=[0.0, 1000.0],
            tvd=[0.0, 1000.0],
            rhob=[2.0, 2.0],
            dtc=[80.0],
            dts=[140.0, 140.0],
            depth_unit="ft",
        )


def test_resistivity_pore_pressure_requires_explicit_shale_trend():
    with pytest.raises(ValueError, match="res_shale is required"):
        estimate_pore_pressure_res(
            s=np.array([20.0]),
            p_hyd=np.array([10.0]),
            res=np.array([5.0]),
        )


def test_stress_profile_endpoint_returns_depth_aligned_curves():
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)
    response = client.post(
        "/quick_pp/scal/stress_profile",
        json={
            "md": [100.0, 200.0],
            "tvd": [100.0, 200.0],
            "rhob": [2.0, 2.1],
            "dtc": [80.0, 80.0],
            "dts": [140.0, 140.0],
            "depth_unit": "m",
            "pore_pressure_gradient": 1.42,
        },
    )

    assert response.status_code == 200
    rows = response.json()["data"]
    assert [row["DEPTH"] for row in rows] == [100.0, 200.0]
    assert rows[-1]["SV_EST"] > rows[0]["SV_EST"]
    assert rows[-1]["NCS_EST"] > 0
    assert rows[-1]["NU_DYN"] == pytest.approx(0.257, abs=0.001)


def test_core_corrections_accept_depth_varying_stress_column():
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)
    response = client.post(
        "/quick_pp/scal/core_corrections",
        json={
            "net_confining_stress": "NCS_EST",
            "mean_pore_pressure": None,
            "ambient_stress": 0.0,
            "compaction_coeff": 0.0001,
            "perm_compaction_coeff": 0.0,
            "data": [
                {"DEPTH": 100.0, "CPORE": 0.2, "CPERM": 100.0, "NCS_EST": 500.0},
                {"DEPTH": 200.0, "CPORE": 0.2, "CPERM": 100.0, "NCS_EST": 1500.0},
            ],
        },
    )

    assert response.status_code == 200
    corrected = response.json()["data"]
    assert corrected[0]["CPORE"] > corrected[1]["CPORE"]
