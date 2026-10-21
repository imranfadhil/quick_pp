#!/usr/bin/env python

"""Tests for `quick_pp` package."""

import unittest

import numpy as np
from click.testing import CliRunner

from quick_pp import cli, lithology, porosity


class TestPetrophysicsCalculations(unittest.TestCase):
    """Level 1: Tests for core petrophysical calculations."""

    def test_gr_to_vsh(self):
        """Test GR index calculation from Gamma Ray array."""
        gr = np.array([20.0, 120.0, 20.0])
        igr = lithology.gr_index(gr)
        np.testing.assert_array_almost_equal(
            np.asarray(igr).ravel(), np.array([0.0, 1.0, 0.0])
        )

    def test_density_porosity(self):
        """Test porosity calculation from bulk density."""
        phi = porosity.density_porosity(rhob=2.15, rho_matrix=2.65, rho_fluid=1.0)
        self.assertAlmostEqual(float(phi), 0.303, places=3)

    def test_gr_to_vsh_edge_case(self):
        """Level 3: Test GR index normalization bounds [0, 1]."""
        gr = np.array([-10.0, 50.0, 200.0])
        igr = lithology.gr_index(gr)
        self.assertAlmostEqual(float(np.min(igr)), 0.0)
        self.assertAlmostEqual(float(np.max(igr)), 1.0)


class TestCli(unittest.TestCase):
    """Tests for the CLI interface."""

    def setUp(self):
        self.runner = CliRunner()

    def test_command_line_interface(self):
        """Test the CLI help and version messages."""
        result = self.runner.invoke(cli.cli, ["--help"])
        assert result.exit_code == 0
        assert "Command-line interface for the quick_pp library." in result.output

        ver_result = self.runner.invoke(cli.cli, ["--version"])
        assert ver_result.exit_code == 0
        assert "quick_pp version:" in ver_result.output
