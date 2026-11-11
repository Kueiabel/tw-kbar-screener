"""Regression tests for the low-day + next-session confirmation screener.

Run locally with: python -m unittest discover -s tests -v
No internet, brokerage quotes, or paid services are needed.
"""
import unittest

import numpy as np
import pandas as pd

from patterns import _patterns_at, screen


def fixture(pivot_volume=6_291_000, confirm_close=4070.0, confirm_low=3785.0):
    """2383's publicly recorded last six trading bars, synthetic warm-up.

    Warm-up is deliberately synthetic: this verifies deterministic behavior
    and does NOT claim to reproduce the complete actual 2383 backtest.
    """
    dates = pd.bdate_range(end="2026-11-09", periods=75)
    pre = np.linspace(5500.0, 4900.0, len(dates))
    rows = []
    for day, price in zip(dates, pre):
        rows.append({
            "date": day.strftime("%Y-%m-%d"),
            "code": "2383", "name": "台光電",
            "open": price+10, "high": price+35,
            "low": price-35, "close": price,
            "shares": 3_500_000, "turnover": 13_000_000_000,
        })
    actual = {
        "2026-10-30": (4570, 4760, 4560, 4675, 3_862_000),
        "2026-11-02": (4500, 4835, 4500, 4705, 3_375_000),
        "2026-11-03": (4560, 4780, 4355, 4395, 3_441_000),
        "2026-11-04": (4610, 4680, 4300, 4555, 5_803_000),
        "2026-11-05": (4335, 4335, 4100, 4100, 4_701_000),
        "2026-11-06": (3690, 3920, 3690, 3820, pivot_volume),
        "2026-11-09": (3920, 4170, confirm_low, confirm_close, 4_937_000),
    }
    for row in rows:
        if row["date"] in actual:
            o, h, l, c, vol = actual[row["date"]]
            row.update(open=o, high=h, low=l, close=c,
                       shares=vol, turnover=vol*c)
    return pd.DataFrame(rows)


def run_scan(frame):
    return screen(
        frame, {"2383": "台光電"}, "2026-11-09",
        min_vol_lots=1000, min_turnover_100m=3.0,
        min_price=10., signal_days=2, min_history_bars=65,
        enabled_categories=("低檔止跌隔日確認",),
    )


class BottomReversalTest(unittest.TestCase):
    def test_2383_like_volume_and_next_day_confirm(self):
        df = fixture()
        bottom, top, diag, chart = run_scan(df)
        self.assertEqual(len(bottom), 1)
        self.assertTrue(top.empty)
        self.assertEqual(bottom.iloc[0]["pivot_date"], "2026-11-06")
        self.assertEqual(bottom.iloc[0]["signal_date"], "2026-11-09")
        self.assertEqual(bottom.iloc[0]["pivot_volume_ratio"], 1.48)
        self.assertIn("突破", bottom.iloc[0]["stage"])
        self.assertEqual(bottom.iloc[0]["matched_categories"], "低檔止跌隔日確認")

    def test_reject_if_low_day_volume_not_above_previous_five(self):
        bottom, _, _, _ = run_scan(fixture(pivot_volume=4_000_000))
        self.assertTrue(bottom.empty)

    def test_reject_if_next_day_fails_to_reclaim_pivot_close(self):
        bottom, _, _, _ = run_scan(fixture(confirm_close=3700.0, confirm_low=3690.0))
        self.assertTrue(bottom.empty)

    def test_no_doji_signal_is_emitted(self):
        df = fixture()
        result = _patterns_at(df.iloc[:74],
                              enabled_categories=("多頭十字星", "K 棒反轉"))
        self.assertNotIn("多頭十字星", [p[1] for p in result])

    def test_no_lookahead_before_confirmation_day(self):
        df = fixture().iloc[:-1].copy()
        bottom, top, diag, _ = screen(
            df, {"2383":"台光電"}, "2026-11-06",
            min_vol_lots=1000, min_turnover_100m=3.,
            min_price=10., min_history_bars=65, signal_days=1,
            enabled_categories=("低檔止跌隔日確認",))
        self.assertTrue(bottom.empty)


if __name__ == "__main__":
    unittest.main()
