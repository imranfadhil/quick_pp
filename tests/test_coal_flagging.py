import numpy as np
import pandas as pd

from quick_pp.app.backend.services.workspace import _compute_coal_mask
from quick_pp.machine_learning.feature_engineering import (
    refine_coal_mask,
)


def test_refine_coal_mask_gap_bridging():
    """Verify Option 2: narrow non-coal gaps bounded by coal are bridged."""
    # Two coal intervals separated by 2 non-coal samples (gap=2) and 4 non-coal samples (gap=4)
    # Refinement with bridge_gap=2 should bridge the 2-sample gap, but not the 4-sample gap.
    raw_mask = np.array(
        [
            False,
            False,
            True,
            True,  # Coal seam 1 (indices 2, 3)
            False,
            False,  # 2-sample gap (indices 4, 5)
            True,
            True,  # Coal seam 2 (indices 6, 7)
            False,
            False,
            False,
            False,  # 4-sample gap (indices 8, 9, 10, 11)
            True,
            True,  # Coal seam 3 (indices 12, 13)
            False,
            False,  # Trailing non-coal (indices 14, 15)
        ],
        dtype=bool,
    )

    refined = refine_coal_mask(
        raw_mask,
        buffer_pts=0,
        bridge_gap=2,
        adaptive_shoulder=False,
    )

    # Indices 4, 5 should now be True (bridged)
    assert np.all(refined[2:8])  # indices 2..7 all True
    # Indices 8..11 should remain False (gap=4 > 2)
    assert not np.any(refined[8:12])
    # Leading and trailing False remain False
    assert not refined[0] and not refined[1]
    assert not refined[14] and not refined[15]


def test_refine_coal_mask_boundary_buffer():
    """Verify Option 1: boundary buffer dilates top and bottom symmetrically."""
    raw_mask = np.array(
        [False, False, False, True, True, False, False, False], dtype=bool
    )

    # 1-point buffer
    buf1 = refine_coal_mask(
        raw_mask,
        buffer_pts=1,
        bridge_gap=0,
        adaptive_shoulder=False,
    )
    # Originally 3, 4 -> now 2, 3, 4, 5
    assert not buf1[1]
    assert np.all(buf1[2:6])
    assert not buf1[6]

    # 2-point buffer
    buf2 = refine_coal_mask(
        raw_mask,
        buffer_pts=2,
        bridge_gap=0,
        adaptive_shoulder=False,
    )
    # Originally 3, 4 -> now 1, 2, 3, 4, 5, 6
    assert not buf2[0]
    assert np.all(buf2[1:7])
    assert not buf2[7]


def test_refine_coal_mask_adaptive_shoulder():
    """Verify Option 3: adaptive gradient shoulder recovery captures transition ramp."""
    # Synthetic RHOB profile across coal:
    # Index: 0     1     2     3     4     5     6     7     8
    # Depth: top -> -> -> -> -> -> -> -> bottom
    # RHOB:  2.55, 2.30, 2.10, 1.40, 1.45, 2.05, 2.25, 2.32, 2.58
    # Baseline shale is ~2.55 - 2.58 g/cc.
    # Indices 3, 4 are pure coal.
    # Indices 1, 2 (top shoulder) and 5, 6, 7 (bottom shoulder) are in transition.
    rhob = np.array([2.55, 2.30, 2.10, 1.40, 1.45, 2.05, 2.25, 2.32, 2.58])
    nphi = np.array([0.28, 0.32, 0.38, 0.55, 0.52, 0.40, 0.33, 0.30, 0.27])
    raw_mask = np.zeros(len(rhob), dtype=bool)
    raw_mask[3:5] = True  # initial raw detection

    # Test adaptive shoulder tracing without fixed buffer
    refined = refine_coal_mask(
        raw_mask,
        rhob=rhob,
        nphi=nphi,
        buffer_pts=0,
        bridge_gap=0,
        adaptive_shoulder=True,
        max_adaptive_pts=3,
        rhob_shoulder_max=2.35,
    )

    # Indices 2 and 1 are <= 2.35 and ramping towards coal -> should be absorbed
    assert refined[2]
    assert refined[1]
    # Index 0 is 2.55 > 2.35 -> should NOT be absorbed
    assert not refined[0]

    # Indices 5, 6, 7 are <= 2.35 and ramping away from coal -> should be absorbed
    assert refined[5]
    assert refined[6]
    assert refined[7]
    # Index 8 is 2.58 > 2.35 -> should NOT be absorbed
    assert not refined[8]


def test_refine_coal_mask_guards_against_gas_sand():
    """Verify adaptive shoulder does not encroach into gas sand with low NPHI."""
    # Coal adjacent to a clean gas sand (RHOB=2.20, NPHI=0.08)
    rhob = np.array([2.20, 1.45, 1.45, 2.55])
    nphi = np.array([0.08, 0.55, 0.55, 0.30])  # gas crossover at index 0
    raw_mask = np.array([False, True, True, False])

    refined = refine_coal_mask(
        raw_mask,
        rhob=rhob,
        nphi=nphi,
        buffer_pts=0,
        bridge_gap=0,
        adaptive_shoulder=True,
        max_adaptive_pts=2,
    )
    # Gas sand at index 0 should be protected and NOT flagged
    assert not refined[0]
    assert refined[1] and refined[2]


def test_compute_coal_mask_integration():
    """Verify _compute_coal_mask in workspace service applies all three refinements."""
    n = 20
    dept = np.linspace(1000, 1010, n)
    rhob = np.full(n, 2.55)
    nphi = np.full(n, 0.25)

    # Embed two thin coals at indices 6..7 and 10..11 (gap of 2 between them)
    # with shoulder ramping at indices 5 and 12
    rhob[6:8] = 1.40
    nphi[6:8] = 0.50
    rhob[10:12] = 1.45
    nphi[10:12] = 0.48

    # Shoulders
    rhob[5] = 2.20
    nphi[5] = 0.35
    rhob[12] = 2.25
    nphi[12] = 0.33

    df = pd.DataFrame({"DEPTH": dept, "RHOB": rhob, "NPHI": nphi})

    # When enable_coal_flagging is True with buffer_pts=1, bridge_gap=2, adaptive=True
    mask = _compute_coal_mask(
        df,
        enable_coal_flagging=True,
        coal_rhob_max=2.0,
        coal_nphi_min=0.30,
        coal_window_size=15,
        coal_trend_factor=0.05,
        coal_buffer_pts=1,
        coal_bridge_gap=2,
        coal_adaptive_shoulder=True,
    )

    # Gap between 7 and 10 (indices 8, 9) should be bridged
    assert mask.iloc[8] and mask.iloc[9]
    # Shoulders at 5 and 12 should be covered
    assert mask.iloc[5] and mask.iloc[12]
    # Background shale at indices 0..4 and 14..19 should not be flagged
    assert not mask.iloc[0] and not mask.iloc[4]
    assert not mask.iloc[14] and not mask.iloc[19]
