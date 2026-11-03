import numpy as np
import pandas as pd
import pytest

from quick_pp.database import objects as db_objects
from quick_pp.database.db_connector import DBConnector
from quick_pp.saturation import extract_fluid_contacts_and_limits


def test_extract_fluid_contacts_oil_water_column():
    """Verify that ODT, OUT, OWC, WUT, and FWL are correctly extracted and WDT is excluded."""
    depths = np.linspace(2900, 3050, 31)  # 5m steps: 2900, 2905, ..., 2990, 2995, 3000, 3005, 3010, ...
    phit = np.full_like(depths, 0.22)
    vsh = np.full_like(depths, 0.05)
    sw = np.ones_like(depths)
    for i, d in enumerate(depths):
        if d <= 2990:
            sw[i] = 0.20  # clean oil pay
        elif d < 3020:
            sw[i] = 0.20 + (d - 2990) * 0.025  # transition zone: 2995->0.325, 3000->0.45, 3005->0.575 (<0.65), 3010->0.70 (>0.65)
        else:
            sw[i] = 1.00  # 100% water leg (d >= 3020)

    df = pd.DataFrame({
        "WELL_NAME": ["Well-30/7a-8"] * len(depths),
        "ZONES": ["Balmoral Sandstone"] * len(depths),
        "FAULT_BLOCK": ["Main"] * len(depths),
        "TVDSS": depths,
        "SWT": sw,
        "PHIT": phit,
        "VSH": vsh,
    })

    contacts = extract_fluid_contacts_and_limits(
        df=df,
        fwl_dict={"Balmoral Sandstone": 3025.0},
        sw_pay_cutoff=0.65,
        sw_water_cutoff=0.90,
    )

    assert not contacts.empty
    types = contacts["contact_type"].tolist()

    # Verify WDT is strictly NOT present
    assert "WDT" not in types

    # Verify oil limits
    assert "OUT" in types
    assert "ODT" in types
    out_row = contacts[contacts["contact_type"] == "OUT"].iloc[0]
    odt_row = contacts[contacts["contact_type"] == "ODT"].iloc[0]
    assert out_row["depth"] == 2900.0
    # Lowest pay point with Sw < 0.65 is at 3005m
    assert odt_row["depth"] == 3005.0
    assert "Oil column top" in out_row["source"]
    assert "Lowest proved oil pay" in odt_row["source"]
    assert "Well-30/7a-8" in odt_row["remarks"]

    # Verify water limit: first 100% water (Sw >= 0.90) is at 3020m
    assert "WUT" in types
    wut_row = contacts[contacts["contact_type"] == "WUT"].iloc[0]
    assert wut_row["depth"] == 3020.0
    assert "100% Sw water leg" in wut_row["source"]

    # Verify OWC bracket: midpoint between ODT (3005) and WUT (3020) = 3012.5m
    assert "OWC" in types
    owc_row = contacts[contacts["contact_type"] == "OWC"].iloc[0]
    assert owc_row["depth"] == pytest.approx(3012.5, abs=0.5)
    assert "ODT-WUT midpoint" in owc_row["source"]

    # Verify FWL
    assert "FWL" in types
    fwl_row = contacts[contacts["contact_type"] == "FWL"].iloc[0]
    assert fwl_row["depth"] == 3025.0
    assert "FWL Optimizer" in fwl_row["source"]

    # Verify fault block and zone are captured
    assert all(contacts["fault_block"] == "Main")
    assert all(contacts["zone"] == "Balmoral Sandstone")


def test_extract_fluid_contacts_gas_cap():
    """Verify GUT, GDT, GOC, and OUT are extracted when a gas cap exists above oil."""
    depths = np.linspace(2500, 2600, 21)
    df = pd.DataFrame({
        "WELL_NAME": ["Well-GasCap"] * len(depths),
        "ZONES": ["GasZone"] * len(depths),
        "TVDSS": depths,
        "SWT": [0.15] * 21,
        "PHIT": [0.25] * 21,
        "VSH": [0.02] * 21,
        "GAS_FLAG": [1 if d <= 2540 else 0 for d in depths],
        "OIL_FLAG": [1 if d > 2540 else 0 for d in depths],
    })

    contacts = extract_fluid_contacts_and_limits(df=df)
    types = contacts["contact_type"].tolist()

    assert "GUT" in types
    assert "GDT" in types
    assert "OUT" in types
    assert "ODT" in types
    assert "GOC" in types
    assert "WDT" not in types

    gdt_row = contacts[contacts["contact_type"] == "GDT"].iloc[0]
    out_row = contacts[contacts["contact_type"] == "OUT"].iloc[0]
    goc_row = contacts[contacts["contact_type"] == "GOC"].iloc[0]

    assert gdt_row["depth"] == 2540.0
    assert out_row["depth"] == 2545.0
    assert goc_row["depth"] == pytest.approx(2542.5, abs=0.5)
    assert "GDT-OUT midpoint" in goc_row["source"]


def test_database_contacts_persistence():
    """Test saving and retrieving fluid contacts with zone, fault_block, source, and remarks."""
    db_conn = DBConnector(db_url="sqlite:///:memory:")
    db_conn.setup_db()
    with db_conn.get_session() as session:
        proj = db_objects.Project(session, name="TestContactsProj")
        proj.save()
        well = db_objects.Well(session, project_id=proj.project_id, name="TestWell", uwi="TW-01")
        well.save()

        test_contacts = [
            {
                "contact_type": "ODT",
                "depth": 2995.0,
                "zone": "Balmoral Sandstone",
                "fault_block": "FB-West",
                "source": "Log Interpretation (Lowest Sw pay)",
                "remarks": "Deepest proved oil pay at 2995.0 m TVDSS",
            },
            {
                "contact_type": "WUT",
                "depth": 3030.0,
                "zone": "Balmoral Sandstone",
                "fault_block": "FB-West",
                "source": "Log Interpretation (100% Sw)",
                "remarks": "Shallowest 100% water at 3030.0 m TVDSS",
            },
            {
                "contact_type": "FWL",
                "depth": 3025.0,
                "zone": "Balmoral Sandstone",
                "fault_block": "FB-West",
                "source": "Multi-Well SHF FWL Optimizer",
                "remarks": "Zero capillary pressure level (Pc=0)",
            },
        ]

        well.add_fluid_contacts(test_contacts, replace=True)
        retrieved_df = well.get_fluid_contacts()

        assert len(retrieved_df) == 3
        assert set(retrieved_df["contact_type"]) == {"ODT", "WUT", "FWL"}
        assert set(retrieved_df["zone"]) == {"Balmoral Sandstone"}
        assert set(retrieved_df["fault_block"]) == {"FB-West"}
        assert all(retrieved_df["source"].notna())
        assert all(retrieved_df["remarks"].notna())


def test_fdp_report_saturation_contacts_rendering():
    """Verify that 06_saturation_contacts.md.j2 renders contacts table and proved column narrative without error."""
    import jinja2
    template_dir = r"quick_pp/app/backend/template/reports"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    tmpl = env.get_template("sections/06_saturation_contacts.md.j2")

    sample_context = {
        "report_type": "FIELD_FDP",
        "sw_params": {
            "model_name": "Waxman-Smits Shaly-Sand Model",
            "fwl": 3025.0,
            "salinity": 35000,
            "rw": 0.08,
            "temp": 75.0,
            "a": 1.0,
            "m": 2.0,
            "n": 2.0,
        },
        "fluid_contacts_rows": [
            {
                "well_name": "30/7a-8",
                "zone": "Balmoral Sandstone",
                "fault_block": "Main",
                "contact_type": "OUT",
                "depth": 2950.0,
                "source": "Log Interpretation (Oil column top)",
                "remarks": "Shallowest proved oil pay",
            },
            {
                "well_name": "30/7a-8",
                "zone": "Balmoral Sandstone",
                "fault_block": "Main",
                "contact_type": "ODT",
                "depth": 2995.0,
                "source": "Log Interpretation (Lowest proved oil pay)",
                "remarks": "Deepest proved oil pay at 2995.0 m TVDSS",
            },
            {
                "well_name": "30/7a-8",
                "zone": "Balmoral Sandstone",
                "fault_block": "Main",
                "contact_type": "OWC",
                "depth": 3005.0,
                "source": "Log Bracket (ODT-WUT midpoint)",
                "remarks": "Interpreted OWC bracketed between proved ODT and WUT",
            },
            {
                "well_name": "30/7a-8",
                "zone": "Balmoral Sandstone",
                "fault_block": "Main",
                "contact_type": "WUT",
                "depth": 3020.0,
                "source": "Log Interpretation (100% Sw water leg)",
                "remarks": "Shallowest 100% water-bearing sand",
            },
            {
                "well_name": "30/7a-8",
                "zone": "Balmoral Sandstone",
                "fault_block": "Main",
                "contact_type": "FWL",
                "depth": 3025.0,
                "source": "Multi-Well SHF FWL Optimizer",
                "remarks": "Zero capillary pressure level (Pc=0)",
            },
        ],
        "fluid_contacts_summary": [
            {
                "zone": "Balmoral Sandstone",
                "fault_block": "Main",
                "gut": None,
                "gdt": None,
                "out": 2950.0,
                "odt": 2995.0,
                "wut": 3020.0,
                "goc": None,
                "gwc": None,
                "owc": 3005.0,
                "fwl": 3025.0,
            }
        ],
        "assets": {},
        "include_ai_commentary": False,
    }

    rendered = tmpl.render(**sample_context)
    assert "Multi-Well & Zonal Fluid Contacts" in rendered
    assert "30/7a-8" in rendered
    assert "Balmoral Sandstone" in rendered
    assert "ODT" in rendered
    assert "Proved Oil Column:" in rendered
    assert "Free Water Level (FWL):" in rendered
    assert "WDT" not in rendered


def test_extract_fluid_contacts_wut_never_shallower_than_out():
    """Verify that when a wet interval exists at the top of a zone (shallower than OUT),
    WUT is picked strictly from the basal water leg beneath the hydrocarbon accumulation."""
    depths = np.linspace(2800, 2900, 21)  # 2800, 2805, ..., 2900
    sw = np.ones_like(depths)
    # 2800 - 2820: wet sand (Sw = 1.0)
    # 2825 - 2860: clean oil pay (Sw = 0.20) -> OUT = 2825, ODT = 2860
    # 2865 - 2900: basal water leg (Sw = 1.0) -> WUT = 2865
    for i, d in enumerate(depths):
        if d < 2825:
            sw[i] = 1.00
        elif d <= 2860:
            sw[i] = 0.20
        else:
            sw[i] = 1.00

    df = pd.DataFrame({
        "WELL_NAME": ["Well-TopWet"] * len(depths),
        "ZONES": ["Balmoral Sandstone"] * len(depths),
        "TVDSS": depths,
        "SWT": sw,
        "PHIT": [0.22] * len(depths),
        "VSH": [0.05] * len(depths),
    })

    contacts = extract_fluid_contacts_and_limits(df=df)
    types = contacts["contact_type"].tolist()

    assert "OUT" in types
    assert "ODT" in types
    assert "WUT" in types
    out_row = contacts[contacts["contact_type"] == "OUT"].iloc[0]
    odt_row = contacts[contacts["contact_type"] == "ODT"].iloc[0]
    wut_row = contacts[contacts["contact_type"] == "WUT"].iloc[0]

    assert out_row["depth"] == 2825.0
    assert odt_row["depth"] == 2860.0
    # WUT MUST be at 2865.0 (below ODT), NOT at 2800.0 or 2820.0 (above OUT)!
    assert wut_row["depth"] == 2865.0
    assert wut_row["depth"] >= odt_row["depth"]
    assert wut_row["depth"] > out_row["depth"]


def test_extract_fluid_contacts_no_sand_and_no_pay_skips_zone():
    """Verify that:
    1. A shale zone with NO reservoir sand (e.g. Sele / Balder Tuff) registers ZERO contacts.
    2. A clean sand zone with NO hydrocarbon pay (100% wet sand) registers ZERO contacts by default,
       avoiding phantom WUT lines at the top of the formation.
    3. A pay zone in the same well registers valid contacts."""
    # Build 3 zones in one well:
    # Zone 1: 'Sele' (Pure Shale: VSH = 0.85, PHIT = 0.02)
    # Zone 2: 'Andrew Sandstone' (Clean Sand, 100% Wet: VSH = 0.08, PHIT = 0.22, SWT = 1.00)
    # Zone 3: 'Balmoral Sandstone' (Oil Reservoir: VSH = 0.05, PHIT = 0.25, SWT = 0.20 in pay, 1.00 in water leg)
    z1_depths = np.linspace(2700, 2750, 11)  # Sele shale
    z2_depths = np.linspace(2760, 2800, 9)   # Andrew wet sand
    z3_depths = np.linspace(2810, 2880, 15)  # Balmoral oil reservoir

    d_all = np.concatenate([z1_depths, z2_depths, z3_depths])
    zones_all = (
        ["Sele"] * len(z1_depths)
        + ["Andrew Sandstone"] * len(z2_depths)
        + ["Balmoral Sandstone"] * len(z3_depths)
    )

    vsh_all = (
        [0.85] * len(z1_depths)
        + [0.08] * len(z2_depths)
        + [0.05] * len(z3_depths)
    )
    phit_all = (
        [0.02] * len(z1_depths)
        + [0.22] * len(z2_depths)
        + [0.25] * len(z3_depths)
    )
    sw_all = (
        [1.0] * len(z1_depths)
        + [1.0] * len(z2_depths)
        + [0.20 if d <= 2840 else 1.0 for d in z3_depths]
    )

    df = pd.DataFrame({
        "WELL_NAME": ["Well-MultiZone"] * len(d_all),
        "ZONES": zones_all,
        "TVDSS": d_all,
        "SWT": sw_all,
        "PHIT": phit_all,
        "VSH": vsh_all,
    })

    contacts = extract_fluid_contacts_and_limits(df=df, require_pay=True)

    # 1. Sele (no sand) must have ZERO contacts
    sele_contacts = contacts[contacts["zone"] == "Sele"]
    assert len(sele_contacts) == 0, f"Expected 0 contacts in shale zone Sele, found: {sele_contacts}"

    # 2. Andrew Sandstone (clean sand, but 100% wet / no pay) must have ZERO contacts
    andrew_contacts = contacts[contacts["zone"] == "Andrew Sandstone"]
    assert len(andrew_contacts) == 0, f"Expected 0 contacts in 100% wet sand Andrew, found: {andrew_contacts}"

    # 3. Balmoral Sandstone (oil reservoir) must have OUT, ODT, WUT, OWC
    balm_contacts = contacts[contacts["zone"] == "Balmoral Sandstone"]
    assert len(balm_contacts) > 0
    balm_types = balm_contacts["contact_type"].tolist()
    assert "OUT" in balm_types
    assert "ODT" in balm_types
    assert "WUT" in balm_types
    assert "OWC" in balm_types
    assert "WDT" not in balm_types


def test_clean_slate_for_zone_under_purview():
    """Verify that identifying and saving new contacts starts a clean slate for the zones
    under purview, clearing old obsolete/stale contacts for that zone without altering other zones."""
    db_conn = DBConnector(db_url="sqlite:///:memory:")
    db_conn.setup_db()
    with db_conn.get_session() as session:
        proj = db_objects.Project(session, name="CleanSlateProj")
        proj.save()
        well = db_objects.Well(session, project_id=proj.project_id, name="Well-CS", uwi="CS-01")
        well.save()

        # Step 1: Initial state has contacts across 3 zones:
        # - 'Balmoral Sandstone': OUT, ODT, and an old obsolete WUT
        # - 'Forties': OUT, ODT (unrelated zone)
        # - 'Sele': a bogus WUT
        initial_contacts = [
            {"zone": "Balmoral Sandstone", "contact_type": "OUT", "depth": 2830.0},
            {"zone": "Balmoral Sandstone", "contact_type": "ODT", "depth": 2840.0},
            {"zone": "Balmoral Sandstone", "contact_type": "WUT", "depth": 2820.0},  # Old shallow WUT
            {"zone": "Forties", "contact_type": "OUT", "depth": 2500.0},
            {"zone": "Forties", "contact_type": "ODT", "depth": 2550.0},
            {"zone": "Sele", "contact_type": "WUT", "depth": 2740.0},  # Bogus shale contact
        ]
        well.add_fluid_contacts(initial_contacts, replace=True)
        assert len(well.get_fluid_contacts()) == 6

        # Step 2: Re-evaluating Balmoral Sandstone with fresh consistent contacts:
        # [OUT at 2833.0, ODT at 2839.0, FWL at 2895.0] (notice: NO WUT, and new OUT/ODT depths)
        fresh_balm_contacts = [
            {"zone": "Balmoral Sandstone", "contact_type": "OUT", "depth": 2833.0},
            {"zone": "Balmoral Sandstone", "contact_type": "ODT", "depth": 2839.0},
            {"zone": "Balmoral Sandstone", "contact_type": "FWL", "depth": 2895.0},
        ]
        well.add_fluid_contacts(fresh_balm_contacts)

        df_after_balm = well.get_fluid_contacts()
        balm_df = df_after_balm[df_after_balm["zone"] == "Balmoral Sandstone"]

        # Balmoral Sandstone MUST have only the 3 new contacts.
        # The old obsolete WUT (2820.0) and old OUT/ODT must be completely gone!
        assert len(balm_df) == 3
        assert set(balm_df["contact_type"]) == {"OUT", "ODT", "FWL"}
        assert set(balm_df["depth"]) == {2833.0, 2839.0, 2895.0}

        # Forties and Sele must still be intact at this point
        assert len(df_after_balm[df_after_balm["zone"] == "Forties"]) == 2
        assert len(df_after_balm[df_after_balm["zone"] == "Sele"]) == 1

        # Step 3: Cleaning the slate for Sele (which is under purview and determined to have no sand/pay)
        deleted = well.delete_fluid_contacts(zones=["Sele"])
        assert deleted == 1

        df_final = well.get_fluid_contacts()
        assert len(df_final[df_final["zone"] == "Sele"]) == 0
        assert len(df_final[df_final["zone"] == "Forties"]) == 2
        assert len(df_final[df_final["zone"] == "Balmoral Sandstone"]) == 3



