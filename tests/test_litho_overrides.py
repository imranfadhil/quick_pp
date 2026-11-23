import unittest

import numpy as np
import pandas as pd

from quick_pp.app.backend.services.workspace import (
    WorkspaceLithologyRequest,
    WorkspacePorosityRequest,
    _isolated_evaluation,
    _IsolatedEvaluation,
    workspace_lithology,
    workspace_porosity,
)


class TestLithoPorosityOverrides(unittest.TestCase):
    def setUp(self):
        depths = np.arange(2000.0, 2010.0, 0.5)
        n = len(depths)
        self.project_id = 9999
        self.well_name = "TEST_OVERRIDE_WELL"

        # Baseline clean sand-shale data
        self.df = pd.DataFrame(
            {
                "DEPTH": depths,
                "DEPT": depths,
                "NPHI": np.full(n, 0.20),
                "RHOB": np.full(n, 2.45),
                "GR": np.full(n, 60.0),
                "CALI": np.full(n, 8.5),
                "BS": np.full(n, 8.5),
                "BADHOLE": np.zeros(n),
                "COAL_FLAG": np.zeros(n),
            }
        )
        self.isolated = _IsolatedEvaluation(
            project_id=self.project_id,
            well_name=self.well_name,
            data=self.df.copy(),
        )
        self.token = _isolated_evaluation.set(self.isolated)

    def tearDown(self):
        _isolated_evaluation.reset(self.token)

    def test_manual_override_flag_coal(self):
        """Test forcing coal across interval 2002.0 - 2004.0."""
        req = WorkspaceLithologyRequest(
            model="ssc",
            wet_clay_point=(0.45, 1.80),
            enable_coal_flagging=False,  # auto detection disabled
            manual_overrides=[
                {
                    "id": "ov-1",
                    "top": 2002.0,
                    "bottom": 2004.0,
                    "coalMode": "flag",
                    "badholeMode": "keep",
                    "enabled": True,
                }
            ],
            save=False,
        )
        res = workspace_lithology(self.project_id, self.well_name, req)
        df_res = pd.DataFrame(res["data"])

        mask = (df_res["DEPTH"] >= 2002.0) & (df_res["DEPTH"] <= 2004.0)
        outside = ~mask

        # Within override: COAL_FLAG == 1.0, VCOAL == 1.0, other minerals == 0.0
        self.assertTrue((df_res.loc[mask, "COAL_FLAG"] == 1.0).all())
        self.assertTrue((df_res.loc[mask, "VCOAL"] == 1.0).all())
        self.assertTrue((df_res.loc[mask, "VSAND"] == 0.0).all())
        self.assertTrue((df_res.loc[mask, "VCLAY"] == 0.0).all())

        # Outside override: COAL_FLAG == 0.0, VCOAL == 0.0, sum of regular minerals == 1.0
        self.assertTrue((df_res.loc[outside, "COAL_FLAG"] == 0.0).all())
        self.assertTrue((df_res.loc[outside, "VCOAL"] == 0.0).all())
        total_minerals = df_res.loc[outside, "VSAND"] + df_res.loc[outside, "VSILT"] + df_res.loc[outside, "VCLAY"]
        self.assertTrue((total_minerals > 0.9).all())

    def test_manual_override_clear_coal(self):
        """Test clearing auto-flagged coal across interval 2002.0 - 2004.0."""
        # Artificially set coal properties in the base well for that interval
        df = self.df.copy()
        mask_seam = (df["DEPTH"] >= 2002.0) & (df["DEPTH"] <= 2006.0)
        df.loc[mask_seam, "RHOB"] = 1.4
        df.loc[mask_seam, "NPHI"] = 0.45
        self.isolated.data = df

        req = WorkspaceLithologyRequest(
            model="ssc",
            wet_clay_point=(0.45, 1.80),
            enable_coal_flagging=True,
            manual_overrides=[
                {
                    "id": "ov-clear",
                    "top": 2002.0,
                    "bottom": 2004.0,
                    "coalMode": "clear",
                    "badholeMode": "keep",
                    "enabled": True,
                }
            ],
            save=False,
        )
        res = workspace_lithology(self.project_id, self.well_name, req)
        df_res = pd.DataFrame(res["data"])

        cleared_mask = (df_res["DEPTH"] >= 2002.0) & (df_res["DEPTH"] <= 2004.0)
        uncleared_seam = (df_res["DEPTH"] > 2004.0) & (df_res["DEPTH"] <= 2006.0)

        # Cleared interval should have COAL_FLAG == 0
        self.assertTrue((df_res.loc[cleared_mask, "COAL_FLAG"] == 0.0).all())
        # Uncleared part of coal seam should still be coal
        self.assertTrue((df_res.loc[uncleared_seam, "COAL_FLAG"] == 1.0).all())

    def test_manual_override_flag_and_clear_badhole(self):
        """Test forcing and clearing badhole flag."""
        req = WorkspaceLithologyRequest(
            model="ssc",
            wet_clay_point=(0.45, 1.80),
            badhole_treatment="vsh_gr",
            manual_overrides=[
                {
                    "id": "ov-bh",
                    "top": 2005.0,
                    "bottom": 2007.0,
                    "coalMode": "keep",
                    "badholeMode": "flag",
                    "enabled": True,
                }
            ],
            save=False,
        )
        res = workspace_lithology(self.project_id, self.well_name, req)
        df_res = pd.DataFrame(res["data"])

        bh_mask = (df_res["DEPTH"] >= 2005.0) & (df_res["DEPTH"] <= 2007.0)
        outside = ~bh_mask

        self.assertTrue((df_res.loc[bh_mask, "BADHOLE"] == 1.0).all())
        self.assertTrue((df_res.loc[outside, "BADHOLE"] == 0.0).all())

    def test_porosity_with_coal_and_badhole_overrides(self):
        """Test that porosity zeros out in flagged coal and applies badhole treatment."""
        req = WorkspacePorosityRequest(
            method="den",
            lithology_model="ssc",
            badhole_treatment="ignore",
            manual_overrides=[
                {
                    "id": "ov-coal",
                    "top": 2001.0,
                    "bottom": 2002.5,
                    "coalMode": "flag",
                    "badholeMode": "keep",
                    "enabled": True,
                },
                {
                    "id": "ov-bh",
                    "top": 2005.0,
                    "bottom": 2006.5,
                    "coalMode": "keep",
                    "badholeMode": "flag",
                    "enabled": True,
                },
            ],
            save=False,
        )
        res = workspace_porosity(self.project_id, self.well_name, req)
        df_res = pd.DataFrame(res["data"])

        coal_mask = (df_res["DEPTH"] >= 2001.0) & (df_res["DEPTH"] <= 2002.5)
        bh_mask = (df_res["DEPTH"] >= 2005.0) & (df_res["DEPTH"] <= 2006.5)

        # Flagged coal should have PHIT == 0.0
        self.assertTrue((df_res.loc[coal_mask, "PHIT"] == 0.0).all())

        # Flagged badhole with 'ignore' mode should have NaN for PHIT
        self.assertTrue(df_res.loc[bh_mask, "PHIT"].isna().all())


if __name__ == "__main__":
    unittest.main()
