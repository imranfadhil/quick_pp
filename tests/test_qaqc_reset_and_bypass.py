import numpy as np
import pandas as pd

from quick_pp.app.backend.services import workspace


class _DummyReq:
    def __init__(
        self,
        *,
        use_env_corrected: bool = True,
        confidence_weighting: bool = True,
        use_raw_ori_logs: bool = False,
        use_normalized_logs: bool = False,
        qc_good_conf: float = 0.8,
        qc_min_conf: float = 0.2,
    ):
        self.use_env_corrected = use_env_corrected
        self.confidence_weighting = confidence_weighting
        self.use_raw_ori_logs = use_raw_ori_logs
        self.use_normalized_logs = use_normalized_logs
        self.qc_good_conf = qc_good_conf
        self.qc_min_conf = qc_min_conf


def test_apply_qc_weighted_inputs_turn_off_and_bypass_modes():
    df = pd.DataFrame(
        {
            "DEPTH": [1000.0, 1001.0, 1002.0],
            "GR": [55.0, 65.0, 75.0],
            "GR_ORI": [40.0, 50.0, 60.0],
            "GR_NORM": [48.0, 58.0, 68.0],
            "GR_ENV_CORR": [52.0, 62.0, 72.0],
            "GR_QC_CONF": [1.0, 1.0, 1.0],
            "NPHI": [0.20, 0.22, 0.24],
            "NPHI_ENV_CORR": [0.18, 0.19, 0.21],
            "NPHI_QC_CONF": [1.0, 1.0, 1.0],
            "CPORE": [0.15, np.nan, 0.19],
            "CPORE_ORI": [0.17, np.nan, 0.21],
        }
    )

    # 1. Default: uses *_ENV_CORR
    out_env = workspace._apply_qc_weighted_inputs(df, _DummyReq())
    np.testing.assert_allclose(out_env["GR"], [52.0, 62.0, 72.0])
    np.testing.assert_allclose(out_env["NPHI"], [0.18, 0.19, 0.21])

    # 2. Turn off env corrections & confidence weighting -> uses edited GR / NPHI
    out_off = workspace._apply_qc_weighted_inputs(
        df, _DummyReq(use_env_corrected=False, confidence_weighting=False)
    )
    np.testing.assert_allclose(out_off["GR"], [55.0, 65.0, 75.0])
    np.testing.assert_allclose(out_off["NPHI"], [0.20, 0.22, 0.24])

    # 3. Turn on normalized logs (*_NORM) with env off -> uses GR_NORM
    out_norm = workspace._apply_qc_weighted_inputs(
        df,
        _DummyReq(
            use_normalized_logs=True,
            use_env_corrected=False,
            confidence_weighting=False,
        ),
    )
    np.testing.assert_allclose(out_norm["GR"], [48.0, 58.0, 68.0])

    # 4. Bypass raw-log edits (use_raw_ori_logs=True) -> restores GR_ORI, protects CPORE,
    #    and keeps NPHI_ENV_CORR on un-edited NPHI when use_env_corrected=True
    out_ori = workspace._apply_qc_weighted_inputs(
        df, _DummyReq(use_raw_ori_logs=True, use_env_corrected=True)
    )
    np.testing.assert_allclose(out_ori["GR"], [40.0, 50.0, 60.0])
    np.testing.assert_allclose(out_ori["NPHI"], [0.18, 0.19, 0.21])
    np.testing.assert_allclose(out_ori["CPORE"], [0.15, np.nan, 0.19], equal_nan=True)


def test_batch_conditioning_preserves_ori_and_qaqc_reset_restores_and_clears(
    monkeypatch,
):
    store = {
        "W-1": pd.DataFrame(
            {
                "WELL_NAME": ["W-1", "W-1", "W-1"],
                "DEPTH": [100.0, 101.0, 102.0],
                "NPHI": [20.0, 25.0, 30.0],
                "GR_NORM": [45.0, 55.0, 65.0],
                "GR_IMP": [44.0, 54.0, 64.0],
                "PERM_TRANS": [12.0, 25.0, 80.0],
                "CPORE": [0.15, np.nan, 0.18],
                "CPORE_ORI": [0.16, np.nan, 0.19],
            }
        )
    }

    def fake_load_multi(_pid, well_names, min_depth, max_depth, zones):
        names = well_names or list(store.keys())
        return [store[n].copy() for n in names if n in store]

    def fake_merged(_pid, name):
        return store[name].copy() if name in store else pd.DataFrame()

    def fake_save(_pid, name, upd_df):
        cur = store[name].set_index("DEPTH")
        upd = upd_df.set_index("DEPTH")
        for col in upd.columns:
            cur[col] = upd[col]
        store[name] = cur.reset_index()
        return len(upd_df)

    def fake_drop(_pid, name, cols, save=True):
        cur = store[name]
        present = [c for c in cols if c in cur.columns]
        store[name] = cur.drop(columns=present, errors="ignore")
        return present

    monkeypatch.setattr(workspace, "_load_multi_well_dfs", fake_load_multi)
    monkeypatch.setattr(workspace, "_get_merged_dataframe_sync", fake_merged)
    monkeypatch.setattr(workspace, "_save_curves_to_well", fake_save)
    monkeypatch.setattr(workspace, "_drop_columns_from_well", fake_drop)
    monkeypatch.setattr(
        workspace,
        "get_project_summary",
        lambda _pid: {"wells": [{"name": k} for k in store]},
    )

    # 1. Batch unit_scale backs up NPHI_ORI before scaling NPHI by 0.01
    workspace.workspace_multi_well_conditioning(
        1,
        workspace.WorkspaceMultiWellConditioningRequest(
            action="unit_scale",
            source_curve="NPHI",
            target_curve="NPHI",
            scale=0.01,
            offset=0.0,
            save=True,
        ),
    )
    assert "NPHI_ORI" in store["W-1"].columns
    np.testing.assert_allclose(store["W-1"]["NPHI_ORI"], [20.0, 25.0, 30.0])
    np.testing.assert_allclose(store["W-1"]["NPHI"], [0.20, 0.25, 0.30])

    # 2. Reset restore_ori restores NPHI, drops NPHI_ORI, and protects CPORE_ORI
    res_ori = workspace.workspace_qaqc_reset(
        1,
        workspace.WorkspaceQaqcResetRequest(
            action="restore_ori",
            drop_ori=True,
            save=True,
        ),
    )
    assert res_ori["restored_curves"] == ["NPHI"]
    assert "NPHI_ORI" not in store["W-1"].columns
    assert "CPORE_ORI" in store["W-1"].columns
    np.testing.assert_allclose(store["W-1"]["NPHI"], [20.0, 25.0, 30.0])

    # 3. Reset clear_curves clears *_NORM, *_IMP, and explicit PERM_TRANS
    res_clear = workspace.workspace_qaqc_reset(
        1,
        workspace.WorkspaceQaqcResetRequest(
            action="clear_curves",
            curves=["PERM_TRANS"],
            suffixes=["_NORM", "_IMP"],
            save=True,
        ),
    )
    assert set(res_clear["cleared_curves"]) == {"GR_NORM", "GR_IMP", "PERM_TRANS"}
    for dropped_col in ("GR_NORM", "GR_IMP", "PERM_TRANS"):
        assert dropped_col not in store["W-1"].columns


def test_reservoir_summary_perm_source_bypass(monkeypatch):
    df = pd.DataFrame(
        {
            "DEPTH": [1000.0, 1001.0, 1002.0, 1003.0],
            "ZONES": ["Z1", "Z1", "Z1", "Z1"],
            "VCLAY": [0.1, 0.1, 0.1, 0.1],
            "PHIT": [0.20, 0.20, 0.20, 0.20],
            "SWT": [0.30, 0.30, 0.30, 0.30],
            "PERM": [0.2, 0.2, 0.2, 0.2],
            "PERM_TRANS": [50.0, 50.0, 50.0, 50.0],
        }
    )
    monkeypatch.setattr(
        workspace,
        "_get_filtered_well_df",
        lambda _pid, _wn, _min_d, _max_d, _zones: df.copy(),
    )
    monkeypatch.setattr(
        workspace, "_update_in_memory_well_cache", lambda *_a, **_kw: None
    )

    # Default ('auto') uses PERM_TRANS (50 mD >= 5 mD cutoff -> net pay > 0)
    res_auto = workspace.workspace_reservoir_summary(
        1,
        "W-1",
        workspace.WorkspaceReservoirSummaryRequest(
            cut_offs={"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.7, "PERM": 5.0},
            perm_source="auto",
            save=False,
        ),
    )
    pay_auto = [
        r
        for r in res_auto["summary"]
        if str(r.get("flag", r.get("FLAG", ""))).lower() == "pay"
    ]
    assert pay_auto and float(pay_auto[0].get("net", pay_auto[0].get("NET", 0))) > 0
    assert res_auto["flags"]["data"]["PAY_FLAG"] == [1, 1, 1, 1]

    # Bypass ('perm') uses log PERM (0.2 mD < 5 mD cutoff -> net pay == 0)
    res_perm = workspace.workspace_reservoir_summary(
        1,
        "W-1",
        workspace.WorkspaceReservoirSummaryRequest(
            cut_offs={"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.7, "PERM": 5.0},
            perm_source="perm",
            save=False,
        ),
    )
    pay_perm = [
        r
        for r in res_perm["summary"]
        if str(r.get("flag", r.get("FLAG", ""))).lower() == "pay"
    ]
    assert not pay_perm or float(pay_perm[0].get("net", pay_perm[0].get("NET", 0))) == 0
    assert res_perm["flags"]["data"]["PAY_FLAG"] == [0, 0, 0, 0]


if __name__ == "__main__":

    class _MP:
        def __init__(self):
            self._orig = []

        def setattr(self, obj, attr, val):
            self._orig.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, val)

        def restore(self):
            for o, a, v in reversed(self._orig):
                setattr(o, a, v)

    test_apply_qc_weighted_inputs_turn_off_and_bypass_modes()
    mp1 = _MP()
    try:
        test_batch_conditioning_preserves_ori_and_qaqc_reset_restores_and_clears(mp1)
    finally:
        mp1.restore()
    mp2 = _MP()
    try:
        test_reservoir_summary_perm_source_bypass(mp2)
    finally:
        mp2.restore()
    print("All test_qaqc_reset_and_bypass checks passed!")
