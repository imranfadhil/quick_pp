import numpy as np
import pandas as pd

from quick_pp.fluid_type import classify_fluid_types
from quick_pp.saturation import extract_fluid_contacts_and_limits


def test_classify_fluid_types_neutron_density():
    """Verify fluid typing from neutron-density porosity crossover."""
    depths = np.linspace(2000, 2100, 21)  # 5m steps: 2000 to 2100
    # 2000-2030: Gas (low RHOB, low NPHI -> large crossover Phi_D - Phi_N > 0.04)
    # 2035-2070: Oil (normal RHOB, NPHI tracking Phi_D)
    # 2075-2100: Water (Sw = 1.0)
    sw = np.ones(len(depths))
    phit = np.full(len(depths), 0.25)
    vsh = np.full(len(depths), 0.05)
    rhob = np.full(len(depths), 2.30)
    nphi = np.full(len(depths), 0.25)

    for i, d in enumerate(depths):
        if d <= 2030:  # Gas
            sw[i] = 0.15
            rhob[i] = 2.15  # Phi_D = (2.65 - 2.15) / 1.65 = 0.303
            nphi[i] = 0.12  # Crossover = 0.303 - 0.12 = 0.183 >> 0.04
        elif d <= 2070:  # Oil
            sw[i] = 0.20
            rhob[i] = 2.30  # Phi_D = (2.65 - 2.30) / 1.65 = 0.212
            nphi[i] = 0.22  # Crossover = -0.008 < 0.04
        else:  # Water leg
            sw[i] = 1.0
            rhob[i] = 2.30
            nphi[i] = 0.25

    df = pd.DataFrame({
        "WELL_NAME": ["Well-Test"] * len(depths),
        "DEPTH": depths,
        "SWT": sw,
        "PHIT": phit,
        "VSHALE": vsh,
        "RHOB": rhob,
        "NPHI": nphi,
    })

    res = classify_fluid_types(
        df,
        method="neutron_density",
        sw_cutoff=0.65,
        phit_cutoff=0.05,
        vsh_cutoff=0.50,
        crossover_cutoff=0.04,
        enforce_segregation=True,
    )

    # Check columns exist
    for col in ["GAS_FLAG", "OIL_FLAG", "HC_FLAG", "VOIL", "VGAS", "BVHC", "FLUID_TYPE", "RHO_HC"]:
        assert col in res.columns

    # Verify gas pay flags
    gas_rows = res[res["DEPTH"] <= 2030]
    assert (gas_rows["GAS_FLAG"] == 1).all()
    assert (gas_rows["OIL_FLAG"] == 0).all()
    assert (gas_rows["HC_FLAG"] == 1).all()
    assert (gas_rows["VGAS"] > 0).all()
    assert (gas_rows["VOIL"] == 0).all()
    assert (gas_rows["FLUID_TYPE"] == "GAS").all()

    # Verify oil pay flags
    oil_rows = res[(res["DEPTH"] > 2030) & (res["DEPTH"] <= 2070)]
    assert (oil_rows["GAS_FLAG"] == 0).all()
    assert (oil_rows["OIL_FLAG"] == 1).all()
    assert (oil_rows["HC_FLAG"] == 1).all()
    assert (oil_rows["VOIL"] > 0).all()
    assert (oil_rows["VGAS"] == 0).all()
    assert (oil_rows["FLUID_TYPE"] == "OIL").all()

    # Verify water leg
    water_rows = res[res["DEPTH"] > 2070]
    assert (water_rows["GAS_FLAG"] == 0).all()
    assert (water_rows["OIL_FLAG"] == 0).all()
    assert (water_rows["HC_FLAG"] == 0).all()
    assert (water_rows["FLUID_TYPE"] == "WATER").all()


def test_classify_fluid_types_rho_hc():
    """Verify fluid typing using inverted apparent hydrocarbon density."""
    depths = np.array([2000.0, 2010.0, 2020.0, 2030.0])
    # Case 1: Gas (rhob very low -> rho_hc ~ 0.20 g/cc)
    # Case 2: Oil (rhob higher -> rho_hc ~ 0.70 g/cc)
    # Case 3: Water (Sw = 1.0)
    sw = np.array([0.15, 0.20, 1.0, 1.0])
    phit = np.array([0.25, 0.25, 0.25, 0.02])
    vsh = np.array([0.05, 0.05, 0.05, 0.80])
    rhob = np.array([2.05, 2.22, 2.30, 2.55])

    df = pd.DataFrame({
        "WELL_NAME": "Well-RhoHC",
        "DEPTH": depths,
        "SWT": sw,
        "PHIT": phit,
        "VSHALE": vsh,
        "RHOB": rhob,
    })

    res = classify_fluid_types(
        df,
        method="rho_hc",
        rho_hc_cutoff=0.40,
        enforce_segregation=False,
    )

    assert res.loc[0, "GAS_FLAG"] == 1
    assert res.loc[0, "OIL_FLAG"] == 0
    assert res.loc[0, "RHO_HC"] < 0.40

    assert res.loc[1, "GAS_FLAG"] == 0
    assert res.loc[1, "OIL_FLAG"] == 1
    assert res.loc[1, "RHO_HC"] > 0.40

    assert res.loc[2, "FLUID_TYPE"] == "WATER"
    assert res.loc[3, "FLUID_TYPE"] == "NON_RESERVOIR"


def test_classify_fluid_types_sonic_vp_vs():
    """Verify fluid typing using compressional-shear velocity ratio."""
    df = pd.DataFrame({
        "WELL_NAME": "Well-Sonic",
        "DEPTH": [2100.0, 2110.0],
        "SWT": [0.20, 0.20],
        "PHIT": [0.25, 0.25],
        "VSHALE": [0.05, 0.05],
        "DTC": [100.0, 90.0],   # us/ft
        "DTS": [150.0, 165.0],   # us/ft -> Vp/Vs = DTS/DTC = 1.50 (Gas) and 1.83 (Oil)
    })

    res = classify_fluid_types(
        df,
        method="sonic_vp_vs",
        vp_vs_cutoff=1.60,
    )

    assert res.loc[0, "GAS_FLAG"] == 1
    assert res.loc[0, "OIL_FLAG"] == 0
    assert res.loc[1, "GAS_FLAG"] == 0
    assert res.loc[1, "OIL_FLAG"] == 1


def test_classify_fluid_types_gravity_segregation_correction():
    """Verify that an inverted oil-above-gas profile gets segregated properly."""
    depths = np.array([2000.0, 2005.0, 2010.0, 2015.0])
    # Say sample at 2000 had weak crossover and was flagged oil,
    # but sample at 2010 was strong gas crossover in the same sand body.
    df = pd.DataFrame({
        "WELL_NAME": "Well-Seg",
        "DEPTH": depths,
        "SWT": [0.2, 0.2, 0.2, 0.2],
        "PHIT": [0.25, 0.25, 0.25, 0.25],
        "VSHALE": [0.05, 0.05, 0.05, 0.05],
        "RHOB": [2.25, 2.25, 2.15, 2.15],
        "NPHI": [0.25, 0.25, 0.12, 0.12],  # 2000-2005 no crossover (oil), 2010-2015 crossover (gas)
    })

    # Without segregation
    raw = classify_fluid_types(df, method="neutron_density", crossover_cutoff=0.04, enforce_segregation=False)
    assert raw.loc[0, "OIL_FLAG"] == 1
    assert raw.loc[2, "GAS_FLAG"] == 1

    # With segregation enforced: oil above gas is corrected to gas
    seg = classify_fluid_types(df, method="neutron_density", crossover_cutoff=0.04, enforce_segregation=True)
    assert seg.loc[0, "GAS_FLAG"] == 1
    assert seg.loc[0, "OIL_FLAG"] == 0
    assert seg.loc[2, "GAS_FLAG"] == 1
    assert seg.loc[2, "OIL_FLAG"] == 0


def test_contact_extraction_from_classified_fluids():
    """Verify that extract_fluid_contacts_and_limits identifies GUT, GDT, OUT, ODT, WUT from classified logs."""
    depths = np.linspace(2500, 2600, 21)
    df = pd.DataFrame({
        "WELL_NAME": ["Well-Contacts"] * 21,
        "TVDSS": depths,
        "SWT": [0.15 if d < 2570 else 1.0 for d in depths],
        "PHIT": [0.25] * 21,
        "VSHALE": [0.05] * 21,
        # Gas 2500-2530, Oil 2535-2565, Water 2570+
        "RHOB": [2.15 if d <= 2530 else 2.25 for d in depths],
        "NPHI": [0.12 if d <= 2530 else 0.24 for d in depths],
    })

    classified = classify_fluid_types(df, method="neutron_density", crossover_cutoff=0.04)
    contacts = extract_fluid_contacts_and_limits(
        classified,
        depth_col="TVDSS",
        sw_col="SWT",
        phit_col="PHIT",
        vsh_col="VSHALE",
        sw_pay_cutoff=0.65,
    )

    types = contacts["contact_type"].tolist()
    assert "GUT" in types
    assert "GDT" in types
    assert "OUT" in types
    assert "ODT" in types
    assert "WUT" in types


def test_classify_fluid_types_missing_sonic_raises():
    """Verify that selecting Dipole Sonic without DTC/DTS raises a clear ValueError."""
    df = pd.DataFrame({
        "WELL_NAME": "Well-NoSonic",
        "DEPTH": [2000.0, 2010.0],
        "SWT": [0.2, 0.2],
        "PHIT": [0.2, 0.2],
        "RHOB": [2.3, 2.3],
        "NPHI": [0.2, 0.2],
    })

    try:
        classify_fluid_types(df, method="sonic_vp_vs")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Dipole sonic fluid typing requires valid compressional (DTC) and shear (DTS)" in str(e)


def test_classify_fluid_types_missing_mud_gas_raises():
    """Verify that selecting Mud Gas without gas chromatography logs raises a clear ValueError."""
    df = pd.DataFrame({
        "WELL_NAME": "Well-NoGas",
        "DEPTH": [2000.0, 2010.0],
        "SWT": [0.2, 0.2],
        "PHIT": [0.2, 0.2],
        "RHOB": [2.3, 2.3],
        "NPHI": [0.2, 0.2],
    })

    try:
        classify_fluid_types(df, method="mud_gas")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Mud gas fluid typing requires valid gas chromatography logs" in str(e)


def test_classify_fluid_types_unconfirmed_buffer():
    """Verify that ambiguous crossover intervals are flagged as UNCONFIRMED when allow_unconfirmed=True."""
    # Sand interval with 3 samples:
    # Point 0: strong crossover (Delta_Phi = 0.15) -> GAS
    # Point 1: ambiguous transition (Delta_Phi = 0.025, between 0.01 and 0.04) -> UNCONFIRMED
    # Point 2: negative crossover (Delta_Phi = -0.05, <= 0.01) -> OIL
    df = pd.DataFrame({
        "WELL_NAME": "Well-Unc",
        "DEPTH": [2000.0, 2005.0, 2010.0],
        "SWT": [0.2, 0.2, 0.2],
        "PHIT": [0.25, 0.25, 0.25],
        "VSHALE": [0.05, 0.05, 0.05],
        "RHOB": [2.15, 2.27, 2.35],
        # Phi_D: 2.15 -> 0.303, 2.27 -> 0.230, 2.35 -> 0.182
        # NPHI:  0.15 -> Delta=0.153 (Gas), 0.205 -> Delta=0.025 (Unc), 0.25 -> Delta=-0.068 (Oil)
        "NPHI": [0.15, 0.205, 0.25],
    })

    res = classify_fluid_types(
        df,
        method="neutron_density",
        crossover_cutoff=0.04,
        crossover_oil_cutoff=0.01,
        allow_unconfirmed=True,
        enforce_segregation=False,
    )

    assert res.loc[0, "GAS_FLAG"] == 1
    assert res.loc[0, "OIL_FLAG"] == 0
    assert res.loc[0, "UNC_FLAG"] == 0
    assert res.loc[0, "FLUID_TYPE"] == "GAS"

    assert res.loc[1, "GAS_FLAG"] == 0
    assert res.loc[1, "OIL_FLAG"] == 0
    assert res.loc[1, "UNC_FLAG"] == 1
    assert res.loc[1, "VUNC"] > 0
    assert res.loc[1, "FLUID_TYPE"] == "UNCONFIRMED"

    assert res.loc[2, "GAS_FLAG"] == 0
    assert res.loc[2, "OIL_FLAG"] == 1
    assert res.loc[2, "UNC_FLAG"] == 0
    assert res.loc[2, "FLUID_TYPE"] == "OIL"
