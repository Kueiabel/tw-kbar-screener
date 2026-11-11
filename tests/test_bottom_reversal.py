"""Tests for the 3 independently selectable V-bottom scans.

python -m unittest discover -s tests -v

Fixtures are synthetic, not claims of a real-stock historical backtest.
"""
import unittest
import numpy as np
import pandas as pd

from patterns import screen


def fixture(shape="none", pivot_volume=3_500_000,
            day2_close=95., rising_context=False, day2_low=85.):
    dates = pd.bdate_range(end="2026-11-10", periods=80)
    prices = np.linspace(133, 104, len(dates))
    rows = []
    for date, price in zip(dates, prices):
        if rising_context:
            price = 100.0
        rows.append({
            "date": date.strftime("%Y-%m-%d"), "code": "9001",
            "name": "模擬標的", "open": price+.5, "high": price+2,
            "low": price-2, "close": price,
            "shares": 3_500_000, "turnover": 3_500_000*price,
        })
    # T-2: local low, T-1: first recovery, T: observation
    p = rows[-3]
    if shape == "hammer":
        p.update(open=90., high=94., low=80., close=93.)
    elif shape == "red_body":
        p.update(open=80., high=96., low=80., close=94.)
    else:
        p.update(open=92., high=97., low=80., close=84.)
    p.update(shares=pivot_volume,
             turnover=pivot_volume*p["close"])
    rows[-2].update(open=91., high=max(day2_close+2,96.),
                    low=day2_low, close=day2_close,
                    shares=3_500_000, turnover=3_500_000*day2_close)
    rows[-1].update(open=98., high=103., low=96., close=101.,
                    shares=3_500_000, turnover=353_500_000)
    return pd.DataFrame(rows)


def scan(df, selected=("V型反轉","V底K棒","V底放量")):
    return screen(
        df, {"9001": "模擬標的"}, "2026-11-10",
        min_vol_lots=1000, min_turnover_100m=3.,
        min_price=10., enabled_categories=selected,
        require_all_signals=False, signal_days=1,
        min_history_bars=65)


class IndependentVSignalsTest(unittest.TestCase):
    def test_price_only_qualifies_by_itself(self):
        bottom, _, _, _ = scan(fixture(day2_close=105.))
        self.assertEqual(len(bottom), 1)
        self.assertEqual(bottom.iloc[0]["matched_categories"], "V型反轉")

    def test_hammer_only_qualifies_by_itself(self):
        bottom, _, _, _ = scan(fixture(shape="hammer"))
        self.assertEqual(len(bottom), 1)
        self.assertEqual(bottom.iloc[0]["matched_categories"], "V底K棒")

    def test_solid_red_body_qualifies_as_shape(self):
        bottom, _, _, _ = scan(fixture(shape="red_body"))
        self.assertEqual(len(bottom), 1)
        self.assertIn("V底K棒", bottom.iloc[0]["matched_categories"])

    def test_volume_only_qualifies_by_itself(self):
        bottom, _, _, _ = scan(fixture(pivot_volume=6_000_000))
        self.assertEqual(len(bottom), 1)
        self.assertEqual(bottom.iloc[0]["matched_categories"], "V底放量")

    def test_all_three_are_listed_with_or_mode(self):
        bottom, _, _, _ = scan(fixture(
            shape="hammer", pivot_volume=6_000_000, day2_close=105.))
        self.assertEqual(len(bottom), 1)
        cats = bottom.iloc[0]["matched_categories"]
        for x in ("V型反轉","V底K棒","V底放量"):
            self.assertIn(x, cats)

    def test_no_confirmation_bar_excluded(self):
        bottom, _, _, _ = scan(fixture(day2_close=80.,day2_low=78.))
        self.assertTrue(bottom.empty)

    def test_without_prior_downtrend_excluded(self):
        bottom, _, _, _ = scan(fixture(
            shape="hammer",pivot_volume=6_000_000,
            rising_context=True))
        self.assertTrue(bottom.empty)

    def test_not_allowed_before_third_day(self):
        df = fixture().iloc[:-1]
        bottom, _, _, _ = screen(
            df, {"9001":"模擬標的"}, "2026-11-09",
            enabled_categories=("V型反轉","V底K棒","V底放量"),
            min_vol_lots=1000,min_turnover_100m=3.,
            min_history_bars=65,signal_days=1)
        # The scan date -2/-1 is then an earlier pair, not 11/06+11/09.
        self.assertTrue(bottom.empty)

    def test_price_range_and_liquidity_filters_unchanged(self):
        bottom, _, _, _ = screen(
            fixture(shape="hammer"), {"9001":"模擬標的"},
            "2026-11-10", min_price=200., min_history_bars=65,
            enabled_categories=("V底K棒",),signal_days=1)
        self.assertTrue(bottom.empty)


if __name__ == "__main__":
    unittest.main()
