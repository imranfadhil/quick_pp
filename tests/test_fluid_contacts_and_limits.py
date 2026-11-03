import numpy as np
import pandas as pd
import pytest

from quick_pp.saturation import extract_fluid_contacts_and_limits
from quick_pp.database.db_connector import DBConnector
from quick_pp.database import objects as db_objects


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
