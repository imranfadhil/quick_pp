"""Integration tests for package-level exports, SCAL Rel-Perm, Sphinx docs, and FastAPI router endpoints."""

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

import quick_pp
from quick_pp.app.backend.router import api_router


def test_quick_pp_init_exports():
    """Verify quick_pp/__init__.py exports uncertainty, upscaling, scal, optimize_fwl, and stratigraphic_normalization."""
    expected_exports = [
        "uncertainty",
        "upscaling",
        "scal",
        "optimize_fwl",
        "ContactOptimizer",
        "stratigraphic_normalization",
        "normalize_by_zone",
        "mc_volumetric_uncertainty",
        "run_log_uncertainty",
        "calc_stochastic_reservoir_summary",
        "upscale_logs",
        "block_logs",
        "expand_blocked_logs",
        "cardwell_parsons_bounds",
        "correct_core_data",
        "corey_relperm",
        "brooks_corey_relperm",
        "let_relperm",
        "fit_relperm",
        "generate_swof_table",
    ]
    for name in expected_exports:
        assert hasattr(quick_pp, name), f"Missing export {name} in quick_pp"
        assert name in quick_pp.__all__, f"{name} not listed in quick_pp.__all__"


def test_sphinx_docs_list_new_modules():
    """Verify docs/source/quick_pp.rst lists quick_pp.uncertainty, quick_pp.upscaling, and quick_pp.scal."""
    rst_path = Path(__file__).resolve().parents[1] / "docs" / "source" / "quick_pp.rst"
    content = rst_path.read_text(encoding="utf-8")
    assert ".. automodule:: quick_pp.uncertainty" in content
    assert ".. automodule:: quick_pp.upscaling" in content
    assert ".. automodule:: quick_pp.scal" in content


def test_scal_relperm_and_swof():
    """Test Corey, Brooks-Corey, LET relative permeability, curve fitting, and SWOF table generation."""
    sw = np.linspace(0.15, 0.80, 15)
    corey = quick_pp.corey_relperm(
        sw, swirr=0.15, sor=0.20, krw_max=0.35, kro_max=0.85, nw=2.5, no=3.0
    )
    assert np.isclose(corey["krw"][0], 0.0)
    assert np.isclose(corey["kro"][0], 0.85)
    assert np.isclose(corey["krw"][-1], 0.35)
    assert np.isclose(corey["kro"][-1], 0.0)

    bc = quick_pp.brooks_corey_relperm(
        sw, swirr=0.15, sor=0.20, krw_max=0.35, kro_max=0.85, lam=2.0
    )
    assert np.all((bc["krw"] >= 0.0) & (bc["krw"] <= 0.35))
    assert np.all((bc["kro"] >= 0.0) & (bc["kro"] <= 0.85))

    let = quick_pp.let_relperm(
        sw,
        swirr=0.15,
        sor=0.20,
        krw_max=0.35,
        kro_max=0.85,
        lw=2.2,
        ew=1.4,
        tw=2.1,
        lo=2.5,
        eo=1.6,
        to=1.9,
    )
    fit_res = quick_pp.fit_relperm(
        sw, let["krw"], let["kro"], model="let", swirr=0.15, sor=0.20
    )
    assert fit_res["rmse_krw"] < 1e-2
    assert fit_res["rmse_kro"] < 1e-2

    swof_res = quick_pp.generate_swof_table(model="corey", n_points=10, sor=0.20)
    swof_df = swof_res["table"]
    assert list(swof_df.columns) == ["SW", "KRW", "KROW", "PCOW"]
    assert len(swof_df) == 11  # 10 mobile points + Sw=1.0 endpoint
    assert swof_res["swof_keyword"].startswith("SWOF\n")
    assert swof_res["swof_keyword"].strip().endswith("/")


def test_backend_router_endpoints():
    """Verify FastAPI router endpoints for Uncertainty, Upscaling, SCAL Rel-Perm, FWL Optimization, and Normalization."""
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)

    # 1. Uncertainty endpoints
    vol_resp = client.post(
        "/quick_pp/uncertainty/volumetric",
        json={"n_try": 500, "seed": 42},
    )
    assert vol_resp.status_code == 200
    vol_data = vol_resp.json()
    assert "P10" in vol_data["percentiles"]
    assert "P50" in vol_data["percentiles"]
    assert "P90" in vol_data["percentiles"]

    sample_logs = [
        {
            "WELL_NAME": "W1",
            "DEPTH": float(1000 + i),
            "ZONES": "Z1" if i < 10 else "Z2",
            "GR": 35.0 + i,
            "NPHI": 0.22 - 0.002 * i,
            "RHOB": 2.32 + 0.005 * i,
            "RT": 25.0 - 0.8 * i,
            "VSHALE": 0.15,
            "PHIT": 0.20 - 0.002 * i,
            "PERM": 80.0 - 2.0 * i,
            "SWT": 0.25 + 0.03 * i,
        }
        for i in range(20)
    ]

    log_unc_resp = client.post(
        "/quick_pp/uncertainty/log",
        json={"model": "ssc", "n_samples": 15, "data": sample_logs},
    )
    assert log_unc_resp.status_code == 200
    assert "SW_P50" in log_unc_resp.json()["data"][0]

    # 2. Log Blocking / Upscaling endpoints
    upscale_resp = client.post(
        "/quick_pp/upscaling/block_logs",
        json={"cell_size": 5.0, "by_zone": True, "data": sample_logs},
    )
    assert upscale_resp.status_code == 200
    assert len(upscale_resp.json()["blocked"]) > 0
    assert "KV_KH_RATIO" in upscale_resp.json()["blocked"][0]

    bounds_resp = client.post(
        "/quick_pp/upscaling/permeability_bounds",
        json={"perm": [10.0, 50.0, 200.0], "omega": 0.33},
    )
    assert bounds_resp.status_code == 200
    b_json = bounds_resp.json()
    assert b_json["k_harm"] <= b_json["k_geom"] <= b_json["k_arith"]

    # 3. SCAL Rel-Perm & SWOF endpoints
    relperm_resp = client.post(
        "/quick_pp/scal/relperm",
        json={"model": "let", "swirr": 0.15, "sor": 0.20},
    )
    assert relperm_resp.status_code == 200
    rp_rows = relperm_resp.json()["data"]
    assert len(rp_rows) == 25

    fit_resp = client.post(
        "/quick_pp/scal/fit_relperm",
        json={
            "model": "corey",
            "sw": [r["sw"] for r in rp_rows],
            "krw": [r["krw"] for r in rp_rows],
            "kro": [r["kro"] for r in rp_rows],
        },
    )
    assert fit_resp.status_code == 200
    assert "nw" in fit_resp.json()

    swof_resp = client.post(
        "/quick_pp/scal/swof",
        json={"model": "corey", "n_points": 12},
    )
    assert swof_resp.status_code == 200
    assert "SWOF" in swof_resp.json()["swof_keyword"]

    # 4. FWL Optimization endpoint
    fwl_resp = client.post(
        "/quick_pp/saturation/optimize_fwl",
        json={
            "model": "leverett_j",
            "shm_params": {"a": 0.15, "b": 1.5, "swirr": 0.10},
            "data": sample_logs,
        },
    )
    assert fwl_resp.status_code == 200
    fwl_json = fwl_resp.json()
    assert "fwl" in fwl_json
    assert "owc" in fwl_json
    assert "ttz" in fwl_json

    # 5. Stratigraphic Normalization endpoint
    norm_resp = client.post(
        "/quick_pp/qaqc/stratigraphic_normalization",
        json={"logs": ["GR", "NPHI"], "data": sample_logs},
    )
    assert norm_resp.status_code == 200
    assert "GR_NORM" in norm_resp.json()["data"][0]

