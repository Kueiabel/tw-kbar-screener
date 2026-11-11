"""Regression tests for fresh 20-bar low + mandatory wick + volume confirmation.

Run: python -m unittest discover -s tests -v

All prices and volumes below are SYNTHETIC regression examples designed
to exercise code, NOT a historical 2383 backtest or buy-signal validation.
"""
import unittest

import numpy as np
import pandas as pd

from patterns import _patterns_at, screen


def fixture(pivot_volume=6_291_000, confirm_volume=4_937_000,
            confirm_close=4070.0, confirm_low=3800.0,
            wick=True, low=3690.0, decline=True):
    dates = pd.bdate_range(end="2026-11-10", periods=75)
    base = np.linspace(5200.0, 4630.0, len(dates))
    rows = []
    for day, price in zip(dates, base):
        rows.append({
            "date": day.strftime("%Y-%m-%d"), "code": "9001",
            "name": "合成案例", "open": price+10,
            "high": price+45, "low": price-45, "close": price,
            "shares": 3_500_000, "turnover": 13_000_000_000,
        })
    # These late rows create a 20-bar downtrend and a fresh low on 11/06.
    late = {
        "2026-10-30": (4570, 4760, 4560, 4675, 3_862_000),
        "2026-11-02": (4500, 4835, 4500, 4705, 3_375_000),
        "2026-11-03": (4560, 4780, 4355, 4395, 3_441_000),
        "2026-11-04": (4610, 4680, 4300, 4555, 5_803_000),
        "2026-11-05": (4335, 4335, 4100, 4100, 4_701_000),
        "2026-11-06": (
            (4520 if low > 4200 else 3890) if wick else low,
            (4580 if low > 4200 else 3980) if wick else 3920,
            low,
            (4540 if low > 4200 else 3930) if wick else 3820,
            pivot_volume
        ),
        "2026-11-09": (3910 if confirm_close < 4500 else 4560,
                       max(4170, confirm_close + 30),
                       confirm_low, confirm_close, confirm_volume),
        "2026-11-10": (4240, 4370, 4100, 4310, 3_600_000),
    }
    for row in rows:
        if row["date"] in late:
            op, hi, lo, cl, vol = late[row["date"]]
            row.update(open=op, high=hi, low=lo, close=cl,
                       shares=vol, turnover=vol*cl)
    if not decline:
        # Keep the last few signal-day shapes identical, but flatten the
        # preceding twenty bars to remove the downtrend.
        for row in rows:
            if "2026-10-12" <= row["date"] <= "2026-10-29":
                row.update(open=4185, high=4220, low=4130,
                           close=4200)
    return pd.DataFrame(rows)


def run_scan(data, signal_days=3, as_of="2026-11-10"):
    return screen(
        data, {"9001": "合成案例"}, as_of,
        min_vol_lots=1000, min_turnover_100m=3.,
        min_price=10., signal_days=signal_days, min_history_bars=65,
        enabled_categories=("低檔止跌隔日確認",)
    )


class BottomReversalTest(unittest.TestCase):
    def test_two_days_before_scan_is_wick_and_previous_day_confirms(self):
        bottom, top, _, _ = run_scan(fixture())
        self.assertEqual(len(bottom), 1)
        self.assertTrue(top.empty)
        row = bottom.iloc[0]
        self.assertEqual(row["pivot_date"], "2026-11-06")
        self.assertEqual(row["signal_date"], "2026-11-09")
        self.assertEqual(row["age"], 1)
        self.assertEqual(row["matched_categories"], "低檔止跌隔日確認")
        self.assertGreater(row["pivot_volume_ratio"], 1)
        self.assertGreater(row["two_day_volume_ratio"], 1)

    def test_missing_lower_shadow_never_qualifies(self):
        bottom, _, _, _ = run_scan(fixture(wick=False))
        self.assertTrue(bottom.empty)

    def test_pivot_volume_must_exceed_earlier_five(self):
        bottom, _, _, _ = run_scan(fixture(pivot_volume=3_000_000))
        self.assertTrue(bottom.empty)

    def test_average_of_two_signal_days_must_exceed_earlier_five(self):
        bottom, _, _, _ = run_scan(fixture(confirm_volume=500_000))
        self.assertTrue(bottom.empty)

    def test_confirmation_must_close_above_pivot(self):
        bottom, _, _, _ = run_scan(
            fixture(confirm_close=3900, confirm_low=3800))
        self.assertTrue(bottom.empty)

    def test_confirmation_must_hold_pivot_low(self):
        bottom, _, _, _ = run_scan(
            fixture(confirm_close=4070, confirm_low=3650))
        self.assertTrue(bottom.empty)

    def test_pivot_must_be_true_prior_twenty_bar_low(self):
        bottom, _, _, _ = run_scan(fixture(
            low=4300, confirm_close=4660, confirm_low=4400))
        self.assertTrue(bottom.empty)

    def test_main_signal_still_scans_when_lookback_one(self):
        bottom, _, _, _ = run_scan(fixture(), signal_days=1)
        self.assertEqual(len(bottom), 1)

    def test_scan_day_cannot_be_used_as_confirmation(self):
        df = fixture().iloc[:-1].copy()
        bottom, _, _, _ = run_scan(df, as_of="2026-11-09")
        self.assertTrue(bottom.empty)

    def test_no_doji_signal_is_emitted(self):
        result = _patterns_at(
            fixture().iloc[:-1], enabled_categories=("多頭十字星",)
        )
        self.assertFalse(any(x[1] == "多頭十字星" for x in result))


if __name__ == "__main__":
    unittest.main()
