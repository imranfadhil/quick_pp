"""Unit tests for optional permeability cutoff in interval flagging and reservoir summary."""

import numpy as np
import pandas as pd

from quick_pp.ressum import calc_reservoir_summary, flag_interval
from quick_pp.upscaling import block_logs


def test_flag_interval_standard_cutoffs():
    """Verify that flag_interval works as expected with standard 3 cutoffs."""
    vsh = np.array([0.1, 0.5, 0.1, 0.1])
    phi = np.array([0.2, 0.2, 0.02, 0.2])
    sw = np.array([0.3, 0.3, 0.3, 0.9])

    cutoffs = {"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.8}
    rock, res, pay = flag_interval(vsh, phi, sw, cutoffs)

    np.testing.assert_array_equal(rock, [1, 0, 1, 1])
    np.testing.assert_array_equal(res, [1, 0, 0, 1])
    np.testing.assert_array_equal(pay, [1, 0, 0, 0])


def test_flag_interval_with_permeability_cutoff():
    """Verify that optional PERM cutoff filters reservoir_flag and pay_flag."""
    vsh = np.array([0.1, 0.1, 0.1, 0.1])
    phi = np.array([0.2, 0.2, 0.2, 0.2])
    sw = np.array([0.3, 0.3, 0.3, 0.3])
    # Four samples: high perm, border perm, low perm, nan perm
    perm = np.array([50.0, 1.0, 0.05, np.nan])

    # 1. Without PERM cutoff -> all 4 are pay
    cutoffs_no_perm = {"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.8}
    r0, res0, pay0 = flag_interval(vsh, phi, sw, cutoffs_no_perm, perm=perm)
    np.testing.assert_array_equal(res0, [1, 1, 1, 1])
    np.testing.assert_array_equal(pay0, [1, 1, 1, 1])

    # 2. With PERM cutoff = 1.0 mD -> only index 0 and 1 qualify
    cutoffs_with_perm = {"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.8, "PERM": 1.0}
    r1, res1, pay1 = flag_interval(vsh, phi, sw, cutoffs_with_perm, perm=perm)
    np.testing.assert_array_equal(r1, [1, 1, 1, 1])
    np.testing.assert_array_equal(res1, [1, 1, 0, 0])
    np.testing.assert_array_equal(pay1, [1, 1, 0, 0])


def test_calc_reservoir_summary_with_perm_cutoff():
    """Verify calc_reservoir_summary adjusts net pay thickness when PERM cutoff is applied."""
    depth = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    vshale = np.array([0.10, 0.10, 0.10, 0.10])
    phit = np.array([0.20, 0.20, 0.20, 0.20])
    swt = np.array([0.30, 0.30, 0.30, 0.30])
    # Points 0 & 1 have 100 mD; points 2 & 3 have 0.05 mD
    perm = np.array([100.0, 100.0, 0.05, 0.05])
    zones = np.array(["ZONE_A", "ZONE_A", "ZONE_A", "ZONE_A"])

    # 1. Without PERM cutoff: all 4 points qualify as pay (net pay = 3.0 m)
    summary_no_perm = calc_reservoir_summary(
        depth=depth,
        vshale=vshale,
        phit=phit,
        swt=swt,
        perm=perm,
        zones=zones,
        cutoffs={"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.8},
    )
    pay_no_perm = summary_no_perm[
        (summary_no_perm["zones"] == "ZONE_A") & (summary_no_perm["flag"] == "pay")
    ].iloc[0]
    assert pay_no_perm["net"] == 3.0

    # 2. With PERM cutoff = 1.0 mD: only points 0 & 1 qualify (net pay = 2.0 m)
    summary_with_perm = calc_reservoir_summary(
        depth=depth,
        vshale=vshale,
        phit=phit,
        swt=swt,
        perm=perm,
        zones=zones,
        cutoffs={"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.8, "PERM": 1.0},
    )
    pay_with_perm = summary_with_perm[
        (summary_with_perm["zones"] == "ZONE_A") & (summary_with_perm["flag"] == "pay")
    ].iloc[0]
    assert pay_with_perm["net"] == 2.0
    # Average perm on pay should now reflect only the passing layers (100 mD)
    np.testing.assert_allclose(pay_with_perm["av_perm_arith_pay"], 100.0)


def test_upscaling_with_perm_cutoff():
    """Verify log blocking and NTG calculations honor the PERM cutoff."""
    fine_df = pd.DataFrame(
        {
            "DEPTH": [1000.0, 1000.5, 1001.0, 1001.5],
            "VSHALE": [0.10, 0.10, 0.10, 0.10],
            "PHIT": [0.20, 0.20, 0.20, 0.20],
            "SWT": [0.30, 0.30, 0.30, 0.30],
            "PERM": [50.0, 50.0, 0.02, 0.02],
            "ZONES": ["ZONE_A", "ZONE_A", "ZONE_A", "ZONE_A"],
        }
    )

    blocked = block_logs(
        fine_df,
        cell_size=1.0,
        net_only=True,
        net_flag_type="pay",
        cutoffs={"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.8, "PERM": 1.0},
    )

    # Cell 1 (1000.0 - 1000.5) has 2 samples with perm=50 mD -> NTG = 1.0
    # Cell 2 (1001.0 - 1001.5) has 2 samples with perm=0.02 mD -> NTG = 0.0
    assert blocked.iloc[0]["NTG"] == 1.0
    assert blocked.iloc[1]["NTG"] == 0.0
