"""As-of-`t` feature builder (scaffold for the feature-layer milestone).

MS0 ships the leakage-safe Elo (see ``elo.py``). The remaining as-of features —
exponential time-decayed form (half-life ~180d), rest-days, comp_weight (already
emitted by Elo), squad-value/age diffs (when Transfermarkt is wired), and the
neutral / is_host_home flags — are built here. Implemented in the feature-layer
milestone; stubbed now so the module/imports exist and the pipeline shape is
fixed.

Hard rule: every feature for a match at time `t` uses only data with date < `t`.
"""

from __future__ import annotations

import pandas as pd

FORM_HALFLIFE_DAYS = 180.0


def add_form(matches: pd.DataFrame, halflife_days: float = FORM_HALFLIFE_DAYS) -> pd.DataFrame:
    """Exponential time-decayed recent-result strength, as-of each match.

    TODO(feature-layer): implement per-team decayed result accumulation in date
    order, snapshotting the pre-match value (same leakage discipline as Elo).
    """
    raise NotImplementedError("form features land in the feature-layer milestone")


def add_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous match (NaN for first appearance)."""
    raise NotImplementedError("rest-days land in the feature-layer milestone")
