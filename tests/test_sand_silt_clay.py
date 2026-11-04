import math

import numpy as np

from quick_pp.lithology.sand_silt_clay import SandSiltClay
from quick_pp.utils import length_a_b, line_intersection


def test_missing_dry_silt_nphi_is_calculated_at_configured_rhob():
    model = SandSiltClay(
        dry_silt_point=(None, 2.68),
        dry_clay_point=(0.33, 2.70),
        wet_clay_point=(0.45, 1.80),
    )

    model.estimate_lithology(np.array([0.1, 0.2]), np.array([2.4, 2.5]))

    expected_nphi = model.fluid_point[0] + (
        (2.68 - model.fluid_point[1]) / math.tan(math.radians(model.silt_line_angle))
    )
    assert model.dry_silt_point[1] == 2.68
    assert np.isclose(model.dry_silt_point[0], expected_nphi)


def test_missing_dry_silt_nphi_respects_custom_rhob():
    model = SandSiltClay(
        dry_silt_point=(None, 2.60),
        dry_clay_point=(0.33, 2.70),
        wet_clay_point=(0.45, 1.80),
    )

    model.estimate_lithology(np.array([0.1, 0.2]), np.array([2.4, 2.5]))

    assert model.dry_silt_point[1] == 2.60


def test_complete_dry_silt_point_with_zero_nphi_is_preserved():
    model = SandSiltClay(
        dry_silt_point=(0.0, 2.68),
        dry_clay_point=(0.33, 2.70),
        wet_clay_point=(0.45, 1.80),
    )

    model.estimate_lithology(np.array([0.1, 0.2]), np.array([2.4, 2.5]))

    assert model.dry_silt_point == (0.0, 2.68)


def test_reservoir_length_uses_fluid_silt_intersection_with_rock_line(monkeypatch):
    model = SandSiltClay(
        dry_sand_point=(-0.02, 2.65),
        dry_silt_point=(0.25, 2.68),
        dry_clay_point=(0.33, 2.70),
        fluid_point=(1.0, 1.0),
    )
    captured = {}

    def capture_lithology_chart(proj_len, rock_len, res_ratio):
        captured["rock_len"] = rock_len
        captured["res_ratio"] = res_ratio
        return np.zeros(2), np.zeros(2), np.zeros(2)

    monkeypatch.setattr(model, "lithology_chart", capture_lithology_chart)
    model.lithology_fraction_kuttan_modified(
        np.array([0.1, 0.2]), np.array([2.4, 2.5])
    )

    intersection = line_intersection(
        (model.dry_sand_point, model.dry_clay_point),
        (model.fluid_point, model.dry_silt_point),
    )
    expected_res_len = length_a_b(model.dry_sand_point, intersection)
    expected_rock_len = length_a_b(model.dry_sand_point, model.dry_clay_point)
    assert np.isclose(captured["rock_len"], expected_rock_len)
    assert np.isclose(captured["res_ratio"], expected_res_len / expected_rock_len)
    assert not np.isclose(
        captured["res_ratio"],
        length_a_b(model.dry_sand_point, model.dry_silt_point) / expected_rock_len,
    )
