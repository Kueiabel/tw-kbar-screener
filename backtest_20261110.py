"""Reproducible 2026-11-10 signal-date acceptance backtest.

This module consumes OFFICIAL daily OHLCV bars already loaded by twse_source.
It never inserts post-11/10 candles into signal discovery or trading-volume
ranking. Outcomes are compared against completed 11/11--11/13 daily bars.

Historical company-master membership is NOT reconstructed (survivorship
bias may remain); prices are TWSE unadjusted and exclude trading costs.
"""
from __future__ import annotations

import pandas as pd

from patterns import screen

SIGNAL_DAY = "2026-11-10"
OUTCOME_DAYS = ("2026-11-11", "2026-11-12", "2026-11-13")
V_CATS = ("V底槌頭", "V底長下影", "V底三K反轉")
SAKATA_CATS = ("酒田三川晨星", "酒田紅三兵", "酒田上升三法")


def _metrics(frame: pd.DataFrame) -> dict:
    """All rates use only 4 observed daily closes; missing bars excluded."""
    if frame.empty:
        return {
            "candidates": 0, "valid": 0, "missing": 0,
            "all_three_daily_up": 0, "all_three_vs_tuesday": 0,
            "ever_above_tuesday": 0, "friday_above_tuesday": 0,
            "all_three_daily_up_rate": None, "all_three_vs_tuesday_rate": None,
            "ever_above_tuesday_rate": None, "friday_above_tuesday_rate": None,
        }
    v = frame[frame["complete"]].copy()
    n = len(v)
    result = {
        "candidates": int(len(frame)), "valid": int(n),
        "missing": int(len(frame)-n),
    }
    mapping = {
        "all_three_daily_up": "連續三天收漲",
        "all_three_vs_tuesday": "三天收盤均高於週二",
        "ever_above_tuesday": "三天內曾高於週二",
        "friday_above_tuesday": "週五收盤高於週二",
    }
    for key, col in mapping.items():
        count = int(v[col].sum()) if n else 0
        result[key] = count
        result[key+"_rate"] = round(100*count/n, 2) if n else None
    return result


def _attach_outcomes(ranking: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    cols = [SIGNAL_DAY, *OUTCOME_DAYS]
    px = bars[bars.date.isin(cols)].pivot_table(
        index="code", columns="date", values="close", aggfunc="last")
    px = px.reindex(columns=cols)
    out = ranking.copy()
    out["code"] = out.code.astype(str)
    out = out.merge(px, left_on="code", right_index=True, how="left")
    out["complete"] = out[cols].notna().all(axis=1)
    p0, p1, p2, p3 = (out[d] for d in cols)
    out["週三漲跌%"] = (p1/p0-1)*100
    out["週四漲跌%"] = (p2/p1-1)*100
    out["週五漲跌%"] = (p3/p2-1)*100
    out["週二至週五%"] = (p3/p0-1)*100
    out["連續三天收漲"] = (p1>p0)&(p2>p1)&(p3>p2)
    out["三天收盤均高於週二"] = (p1>p0)&(p2>p0)&(p3>p0)
    out["三天內曾高於週二"] = (p1>p0)|(p2>p0)|(p3>p0)
    out["週五收盤高於週二"] = p3>p0
    return out


def evaluate_acceptance(bars: pd.DataFrame, companies: dict) -> dict:
    """Evaluate Tuesday discovery without look-ahead, with six buy patterns.

    Three views per model: all signals available on Tuesday within two bars,
    only patterns that actually completed Tuesday, and the top 25 candidates.
    Counts may legitimately be zero. Never replace zero with winners chosen
    after the fact.
    """
    required = {SIGNAL_DAY, *OUTCOME_DAYS}
    actual_dates = set(bars.date.astype(str))
    missing = required-actual_dates
    if missing:
        raise ValueError("缺少驗收所需盤後日期：" + "、".join(sorted(missing)))
    results = {}
    for label, cats in (
        ("原 V 底三型態", V_CATS),
        ("V 底＋酒田買入型態", V_CATS+SAKATA_CATS),
    ):
        args = dict(
            min_vol_lots=0, min_turnover_100m=0.,
            min_price=10., signal_days=2, top_n=150,
            enabled_categories=cats, min_history_bars=21,
            require_all_signals=False, top_volume_n=150,
            require_week_low=False,
        )
        rank, _, diagnostics, _ = screen(
            bars, companies, SIGNAL_DAY, **args)
        all_candidates = _attach_outcomes(rank, bars)
        if not all_candidates.empty:
            fresh = all_candidates[
                all_candidates.signal_date == SIGNAL_DAY].copy()
            # The exact-day selection must not rely on a ranking where a
            # different signal (from Monday) was chosen as the 'best' one.
            strict_rank, _, _, _ = screen(
                bars, companies, SIGNAL_DAY,
                **{**args, "signal_days":1})
            fresh = _attach_outcomes(strict_rank, bars)
        else:
            fresh = pd.DataFrame()
        results[label] = {
            "volume_universe": diagnostics.get("volume_universe"),
            "available_tuesday": all_candidates,
            "finished_tuesday": fresh,
            "top25_tuesday": fresh.head(25).copy(),
            "metrics_available": _metrics(all_candidates),
            "metrics_exact": _metrics(fresh),
            "metrics_top25": _metrics(fresh.head(25)),
        }
    return results
