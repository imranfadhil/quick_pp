"""Tests for the NPHI/RHOB light-hydrocarbon crossplot correction."""

import math

import numpy as np
import pytest

from quick_pp.qaqc import (
    den_correction,
    den_correction_params_from_endpoints,
    neu_den_xplot_hc_correction,
    neu_den_xplot_hc_correction_angle,
)
from quick_pp.utils import nd_triangle_coords

SAND = (-0.02, 2.65)
CLAY = (0.33, 2.70)
WATER = (1.0, 1.0)
RHO_GAS, HI_GAS = 0.3, 0.66
GAS_ANGLE = math.degrees(math.atan2(1.0 - RHO_GAS, 1.0 - HI_GAS))


def forward(phi, vcl, shc, sand=SAND, clay=CLAY, rho_hc=RHO_GAS, hi_hc=HI_GAS):
    """Linear bulk-volume mixing forward model (matrix, clay, water, hydrocarbon)."""
    vsd = 1.0 - phi - vcl
    nphi = vsd * sand[0] + vcl * clay[0] + phi * ((1.0 - shc) + shc * hi_hc)
    rhob = vsd * sand[1] + vcl * clay[1] + phi * ((1.0 - shc) + shc * rho_hc)
    return np.asarray(nphi, dtype=float), np.asarray(rhob, dtype=float)


def correct(nphi, rhob, vsh, **kwargs):
    kwargs.setdefault("dry_min1_point", SAND)
    kwargs.setdefault("dry_clay_point", CLAY)
    kwargs.setdefault("water_point", WATER)
    kwargs.setdefault("corr_angle", GAS_ANGLE)
    return neu_den_xplot_hc_correction(nphi, rhob, vsh, **kwargs)


# --------------------------------------------------------------------------- angle
def test_angle_measured_from_horizontal():
    assert neu_den_xplot_hc_correction_angle(1.0, 0.3, 0.66) == pytest.approx(64.09, abs=0.01)


def test_angle_liquid_defaults_and_hi_above_one():
    # Default HI for liquids is capped at 1.0 -> pure density correction
    assert neu_den_xplot_hc_correction_angle(1.0, 0.8) == pytest.approx(90.0)
    # HI > 1 means NPHI reads high in HC: correction angle exceeds 90 degrees
    assert neu_den_xplot_hc_correction_angle(1.0, 0.8, 1.05) > 90.0


# --------------------------------------------------------------------------- core
@pytest.mark.parametrize("phi,sg", [(0.25, 0.5), (0.30, 0.8), (0.10, 0.6), (0.05, 0.9)])
def test_clean_gas_sand_exact_recovery(phi, sg):
    nphi, rhob = forward(phi, 0.0, sg)
    nphi_wet, rhob_wet = forward(phi, 0.0, 0.0)
    nc, rc, flag = correct(np.array([nphi]), np.array([rhob]), np.array([0.0]))
    assert flag[0] == 1.0
    assert nc[0] == pytest.approx(float(nphi_wet), abs=1e-10)
    assert rc[0] == pytest.approx(float(rhob_wet), abs=1e-10)
    _, vcl = nd_triangle_coords(nc, rc, SAND, CLAY, WATER)
    assert vcl[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("rho_clay", [2.75, 2.55])
def test_clay_density_ordering_invariance(rho_clay):
    clay = (0.33, rho_clay)
    nphi, rhob = forward(0.25, 0.0, 0.5)
    nphi_wet, rhob_wet = forward(0.25, 0.0, 0.0)
    nc, rc, flag = correct(
        np.array([nphi]), np.array([rhob]), np.array([0.0]), dry_clay_point=clay
    )
    assert flag[0] == 1.0
    assert nc[0] == pytest.approx(float(nphi_wet), abs=1e-10)
    assert rc[0] == pytest.approx(float(rhob_wet), abs=1e-10)


def test_buffer_only_widens_flagging():
    # Gas point beyond the sand line plus a water-bearing point just inside it
    gas = forward(0.25, 0.0, 0.5)
    wet_inside = forward(0.25, 0.01, 0.0)
    nphi = np.array([gas[0], wet_inside[0]])
    rhob = np.array([gas[1], wet_inside[1]])
    vsh = np.array([0.0, 0.0])

    nc0, rc0, f0 = correct(nphi, rhob, vsh, buffer=0.0)
    nc1, rc1, f1 = correct(nphi, rhob, vsh, buffer=0.03)

    np.testing.assert_allclose(nc0, nc1)
    np.testing.assert_allclose(rc0, rc1)
    assert f0.tolist() == [1.0, 0.0]
    assert f1.tolist() == [1.0, 1.0]
    # Buffer-flagged point already on the shaly side of the target is not moved
    assert nc1[1] == pytest.approx(nphi[1])
    assert rc1[1] == pytest.approx(rhob[1])


def test_shaly_gas_sand_strict_vs_vsh_gr_mode():
    nphi, rhob = forward(0.22, 0.30, 0.6)
    nphi_wet, rhob_wet = forward(0.22, 0.30, 0.0)
    args = (np.array([nphi]), np.array([rhob]), np.array([0.30]))

    nc, rc, flag = correct(*args, flag_mode="strict")
    assert flag[0] == 0.0
    assert nc[0] == pytest.approx(float(nphi))

    nc, rc, flag = correct(*args, flag_mode="vsh_gr", vsh_tol=0.05)
    assert flag[0] == 1.0
    assert nc[0] == pytest.approx(float(nphi_wet), abs=1e-10)
    assert rc[0] == pytest.approx(float(rhob_wet), abs=1e-10)


def test_vsh_gr_mode_does_not_touch_water_bearing_points():
    nphi, rhob = forward(0.22, 0.30, 0.0)
    nc, rc, flag = correct(
        np.array([nphi]), np.array([rhob]), np.array([0.30]), flag_mode="vsh_gr"
    )
    assert flag[0] == 0.0
    assert rc[0] == pytest.approx(float(rhob))


def test_bulk_target_recovers_shaly_gas_and_solid_target_matches_projection():
    nphi, rhob = forward(0.22, 0.20, 0.6)
    nphi_wet, rhob_wet = forward(0.22, 0.20, 0.0)
    args = (np.array([nphi]), np.array([rhob]), np.array([0.20]))

    nc, rc, _ = correct(*args, target_mode="bulk")
    assert nc[0] == pytest.approx(float(nphi_wet), abs=1e-10)
    assert rc[0] == pytest.approx(float(rhob_wet), abs=1e-10)

    nc, rc, _ = correct(*args, target_mode="solid")
    phi, vcl = nd_triangle_coords(nc, rc, SAND, CLAY, WATER)
    assert (vcl / (1.0 - phi))[0] == pytest.approx(0.20, abs=1e-10)


def test_nan_vsh_gr_is_skipped():
    nphi, rhob = forward(0.25, 0.0, 0.5)
    nc, rc, flag = correct(np.array([nphi]), np.array([rhob]), np.array([np.nan]))
    assert flag[0] == 0.0
    assert nc[0] == pytest.approx(float(nphi))
    assert rc[0] == pytest.approx(float(rhob))


def test_corrected_points_never_cross_zero_porosity():
    nphi = np.array([-0.06, -0.04])
    rhob = np.array([2.55, 2.60])
    vsh = np.array([0.6, 0.8])
    nc, rc, flag = correct(nphi, rhob, vsh, flag_mode="vsh_gr")
    phi, _ = nd_triangle_coords(nc, rc, SAND, CLAY, WATER)
    assert np.all(flag == 1.0)
    assert np.all(phi >= -1e-12)


def test_scalar_inputs_and_none_vsh():
    nphi, rhob = forward(0.25, 0.0, 0.5)
    nc, rc, flag = correct(float(nphi), float(rhob), None)
    assert np.ndim(nc) == 0
    assert float(flag) == 1.0


def test_invalid_modes_raise():
    with pytest.raises(ValueError):
        correct(np.array([0.1]), np.array([2.2]), None, flag_mode="bogus")
    with pytest.raises(ValueError):
        correct(np.array([0.1]), np.array([2.2]), None, target_mode="bogus")


# --------------------------------------------------------------------------- den_correction
def test_den_correction_params_reproduce_triangle_model():
    rho_ma, phin_sh, phid_sh = den_correction_params_from_endpoints(SAND, CLAY, 1.0)
    for phi, vcl in [(0.25, 0.2), (0.15, 0.3), (0.0, 1.0), (0.3, 0.0)]:
        nphi, rhob = forward(phi, vcl, 0.0)
        rr = den_correction(
            nphi, vsh_gr=vcl, phin_sh=phin_sh, phid_sh=phid_sh, rho_ma=rho_ma, rho_water=1.0
        )
        assert float(rr) == pytest.approx(float(rhob), abs=1e-10)


def test_den_correction_requires_gr_or_vsh():
    with pytest.raises(ValueError):
        den_correction(0.2)

