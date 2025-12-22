import math

import numpy as np
import pandas as pd


# Sanitize lists for JSON serialization: convert numpy/pandas types
# to native Python types and replace NaN/Inf with None.
def _sanitize_list(lst):
    out = []
    for v in lst:
        # pandas NA / NaN
        try:
            if pd.isna(v):
                out.append(None)
                continue
        except Exception:
            pass

        # numeric types (numpy, python)
        try:
            if isinstance(v, (int, float, np.floating, np.integer)):
                fv = float(v)
                if math.isfinite(fv):
                    out.append(fv)
                else:
                    out.append(None)
                continue
        except Exception:
            pass

        # fallback: keep as-is (strings, etc.)
        out.append(v)
    return out
