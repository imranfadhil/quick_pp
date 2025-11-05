#!/usr/bin/env python

"""Tests for `quick_pp` package."""

import unittest
from click.testing import CliRunner
import os
import tempfile
import numpy as np

# Assuming your core calculations are in a module like this
from quick_pp import porosity, lithology
from quick_pp import cli


class TestPetrophysicsCalculations(unittest.TestCase):
    """Level 1: Tests for core petrophysical calculations."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_gr_to_vsh(self):
        """Test VSHALE calculation from Gamma Ray."""
        # Test a simple case
        vsh = lithology.gr_index(gr=75)
        self.assertAlmostEqual(vsh, 0.5)

        # Test with numpy array
        gr = np.array([0, 75, 150])
        vsh = lithology.gr_index(gr)
        np.testing.assert_array_almost_equal(vsh, np.array([0, 0.5, 1.0]))

    def test_density_porosity(self):
        """Test porosity calculation from bulk density."""
        # Test a simple case
        phi = porosity.density_porosity(rhob=2.15, rho_matrix=2.65, rho_fluid=1.0)
        self.assertAlmostEqual(phi, 0.303, places=3)

    def test_gr_to_vsh_edge_case(self):
        """Level 3: Test VSHALE calculation for edge cases."""
        # What if GR is above max? Should be capped at 1.0
        vsh = lithology.gr_index(gr=200)
        self.assertAlmostEqual(vsh, 1.0)

        # What if GR is below min? Should be capped at 0.0
        vsh = lithology.gr_index(gr=-10)
        self.assertAlmostEqual(vsh, 0.0)


class TestCli(unittest.TestCase):
    """Tests for the CLI interface."""

    def setUp(self):
        self.runner = CliRunner()

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_command_line_interface(self):
        """Test the CLI help messages."""
        result = self.runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'quick_pp.cli.main' in result.output
        help_result = self.runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

    def test_training_cli_run(self):
        """Level 2: Test training ML model path workflow from the CLI."""
        # Create a dummy LAS file for testing
        las_content = ""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".las") as tmp:
            tmp.write(las_content)
            tmp_path = tmp.name

        try:
            result = self.runner.invoke(cli.main, ['start', tmp_path])
            assert result.exit_code == 0, result.output
            assert f"Successfully processed {tmp_path}" in result.output
        finally:
            # Clean up the dummy file
            os.remove(tmp_path)
