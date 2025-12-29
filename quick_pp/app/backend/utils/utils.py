import math
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


# Sanitize lists for JSON serialization: convert numpy/pandas types
# to native Python types and replace NaN/Inf with None.
def sanitize_list(lst):
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


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename derived from the provided name.

    Keeps only alphanumeric characters, dots, dashes and underscores and
    replaces spaces with underscores. Falls back to a UUID if resulting
    name is empty.
    """
    # Strip any path components
    name = Path(name).name
    # Replace spaces with underscore
    name = name.replace(" ", "_")
    # Remove characters other than alnum, dot, dash, underscore
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    # Limit length to avoid extremely long filenames
    if len(name) > 200:
        name = name[:200]
    if not name:
        name = uuid4().hex
    return name
