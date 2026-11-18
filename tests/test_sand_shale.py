import numpy as np

from quick_pp.lithology.sand_shale import SandShale


def test_sand_shale_fractions_bounded_and_normalized():
    model = SandShale(
        dry_sand_point=(-0.02, 2.65),
        dry_clay_point=(0.45, 2.71),
        fluid_point=(1.0, 1.0),
    )
    # 1. Clean sand past sand endpoint (negative NPHI, lower RHOB)
    # 2. Pure sand point
    # 3. Midpoint
    # 4. Pure clay point
    # 5. Heavy shale past clay endpoint
    # 6. NaN entry
    nphi = np.array([-0.05, -0.02, 0.215, 0.45, 0.55, np.nan])
    rhob = np.array([2.60, 2.65, 2.68, 2.71, 2.78, np.nan])

    vsand, vcld = model.lithology_fraction(nphi, rhob)

    # Valid points check
    valid = ~np.isnan(nphi)
    assert np.all(vsand[valid] >= 0.0)
    assert np.all(vsand[valid] <= 1.0)
    assert np.all(vcld[valid] >= 0.0)
    assert np.all(vcld[valid] <= 1.0)
    assert np.allclose(vsand[valid] + vcld[valid], 1.0)

    # Beyond sand point clamps to 1.0 sand, 0.0 clay
    assert np.isclose(vsand[0], 1.0)
    assert np.isclose(vcld[0], 0.0)

    # At sand point
    assert np.isclose(vsand[1], 1.0)
    assert np.isclose(vcld[1], 0.0)

    # At clay point
    assert np.isclose(vsand[3], 0.0)
    assert np.isclose(vcld[3], 1.0)

    # Beyond clay point clamps to 0.0 sand, 1.0 clay
    assert np.isclose(vsand[4], 0.0)
    assert np.isclose(vcld[4], 1.0)

    # NaN is preserved
    assert np.isnan(vsand[5])
    assert np.isnan(vcld[5])


def test_sand_shale_zero_coordinates_preserved():
    model = SandShale(
        dry_sand_point=(0.0, 2.65),
        dry_clay_point=(0.33, 2.71),
        wet_clay_point=(0.40, 2.40),
        fluid_point=(1.0, 1.0),
    )
    vsand, vcld, _ = model.estimate_lithology(np.array([0.1, 0.2]), np.array([2.4, 2.5]))
    assert model.dry_sand_point == (0.0, 2.65)
    assert model.dry_clay_point == (0.33, 2.71)
    assert model.wet_clay_point == (0.40, 2.40)


def test_workspace_sand_shale_lithology_and_porosity(monkeypatch):
    import pandas as pd

    from quick_pp.app.backend.services import workspace
    from quick_pp.app.backend.services.workspace import (
        WorkspaceLithologyRequest,
        WorkspacePorosityRequest,
        workspace_lithology,
        workspace_porosity,
    )

    df = pd.DataFrame(
        {
            "DEPTH": [1000.0, 1001.0, 1002.0],
            "NPHI": [0.0, 0.215, 0.45],
            "RHOB": [2.60, 2.68, 2.71],
            "GR": [20.0, 70.0, 120.0],
            "BADHOLE": [0.0, 0.0, 0.0],
        }
    )

    monkeypatch.setattr(workspace, "_get_filtered_well_df", lambda *args, **kwargs: df.copy())
    monkeypatch.setattr(workspace, "_update_in_memory_well_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(workspace, "_record_well_model_config", lambda *args, **kwargs: None)

    req = WorkspaceLithologyRequest(
        model="sand_shale",
        dry_sand_point=(-0.02, 2.65),
        dry_clay_point=(0.45, 2.71),
        fluid_point=(1.0, 1.0),
    )
    res = workspace_lithology(1, "well", req)
    data = res["data"]

    assert "VSAND" in data
    assert "VCLAY" in data
    assert "VSHALE" in data
    assert "VSILT" in data
    assert np.all(np.asarray(data["VSILT"]) == 0.0)
    assert np.allclose(np.asarray(data["VSAND"]) + np.asarray(data["VCLAY"]), 1.0)
    assert data["VCLAY"][2] > data["VCLAY"][0]

    # Test porosity execution with sand_shale lithology model
    df_with_litho = df.copy()
    for col in ("VSAND", "VCLAY", "VSILT", "VSHALE"):
        df_with_litho[col] = data[col]
    monkeypatch.setattr(workspace, "_get_filtered_well_df", lambda *args, **kwargs: df_with_litho.copy())

    poro_req = WorkspacePorosityRequest(
        method="den",
        lithology_model="sand_shale",
        dry_sand_point=(-0.02, 2.65),
        dry_clay_point=(0.45, 2.71),
        fluid_point=(1.0, 1.0),
    )
    poro_res = workspace_porosity(1, "well", poro_req)
    p_data = poro_res["data"]
    assert "PHIT" in p_data
    assert "PHIE" in p_data

