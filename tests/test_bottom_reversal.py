"""Regression tests for restored original K-bar screener (commit 6cec61a).

No broker network data or historical backtest is required. These tests
validate filter behavior, per-candle geometry, OR mode and default signals.
"""
import unittest

import numpy as np
import pandas as pd

from patterns import _is_regular_doji, _is_hammer_shape, screen


def sample(last="wick"):
    dates = pd.bdate_range(end="2026-11-13", periods=75)
    prices = np.linspace(130.0, 100.0, len(dates))
    rows = []
    for d, price in zip(dates, prices):
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "code": "9001", "name": "模擬股票",
            "open": float(price + 0.35),
            "high": float(price + 1),
            "low": float(price - 1),
            "close": float(price),
            "shares": 5_000_000,
            "turnover": float(price * 5_000_000),
        })
    if last == "wick":
        rows[-1].update(open=100.,high=101.,low=89.,close=99.2)
    if last == "neutral":
        rows[-1].update(open=100.,high=101.,low=99.,close=99.8)
    if last == "previous_wick":
        rows[-2].update(open=101.,high=102.,low=89.,close=100.)
        rows[-1].update(open=101.,high=102.,low=99.,close=101.2)
    return pd.DataFrame(rows)


def scan(df, cats=("低檔長下影止跌",), must_all=False,
         min_lots=0, min_money=0., days=3):
    return screen(
        df, {"9001":"模擬股票"}, df.date.max(),
        min_vol_lots=min_lots,
        min_turnover_100m=min_money,
        min_price=10., min_history_bars=65, signal_days=days,
        enabled_categories=cats, require_all_signals=must_all,
    )


class FirstVersionTests(unittest.TestCase):
    def test_regular_doji_needs_two_shadows(self):
        self.assertTrue(_is_regular_doji(100.,110.,90.,100.5))
        self.assertFalse(_is_regular_doji(100.,110.,100.,100.5))

    def test_hammer_uses_geometric_wick(self):
        self.assertTrue(_is_hammer_shape(100.,102.,80.,101.))
        self.assertFalse(_is_hammer_shape(100.,104.,100.,103.))

    def test_low_shadow_signal_on_its_own_bar(self):
        bottom, top, _, _ = scan(sample("wick"),days=1)
        self.assertEqual(len(bottom),1)
        self.assertIn("長下影止跌",bottom.iloc[0]["pattern"])
        self.assertEqual(bottom.iloc[0]["age"],0)
        self.assertEqual(bottom.iloc[0]["signal_date"],"2026-11-13")

    def test_neutral_bar_not_long_shadow(self):
        bottom, _, _, _ = scan(sample("neutral"),days=1)
        self.assertTrue(bottom.empty)

    def test_recent_three_sessions_lookback(self):
        bottom, _, _, _ = scan(sample("previous_wick"),days=3)
        self.assertEqual(len(bottom),1)
        self.assertEqual(bottom.iloc[0]["signal_date"],"2026-11-12")

    def test_or_mode_does_not_require_all_categories(self):
        data=sample("wick")
        bottom,_,_,_=scan(
            data,cats=("低檔長下影止跌","晨星反轉"),
            must_all=False,days=1)
        self.assertEqual(len(bottom),1)
        strict,_,_,_=scan(
            data,cats=("低檔長下影止跌","晨星反轉"),
            must_all=True,days=1)
        self.assertTrue(strict.empty)

    def test_liquidity_filter_remains_optional(self):
        bottom,_,_,_=scan(sample(),min_lots=100000)
        self.assertTrue(bottom.empty)

    def test_no_lookahead_for_signal_date(self):
        df=sample("previous_wick")
        prefix=df.iloc[:-1].copy()
        bottom,_,_,_=scan(prefix,days=1)
        self.assertEqual(len(bottom),1)
        self.assertEqual(bottom.iloc[0]["signal_date"],"2026-11-12")


if __name__ == "__main__":
    unittest.main()
