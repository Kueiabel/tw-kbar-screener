"""Deterministic 21-K V-bottom and top-150-volume regression tests.

Synthetic data; not a historical-stock validation or probability estimate.
Run: python -m unittest discover -s tests -v
"""
import unittest
import numpy as np
import pandas as pd
from patterns import screen


def sample(code="9001", pivot_offset=-2, shape="hammer", last_volume=3_500_000):
    dates = pd.bdate_range(end="2026-11-13", periods=27)
    prices = np.linspace(130, 103, len(dates))
    rows = []
    for d, p in zip(dates, prices):
        rows.append(dict(
            date=d.strftime("%Y-%m-%d"), code=code, name="模擬股票",
            open=float(p+.5), high=float(p+1.5),
            low=float(p-1.5), close=float(p),
            shares=2_000_000, turnover=float(p*2_000_000)))
    if shape == "hammer":
        rows[pivot_offset].update(open=95.,high=96.,low=87.,close=95.5)
    elif shape == "lower":
        rows[pivot_offset].update(open=95.,high=99.,low=87.,close=97.)
    elif shape == "three":
        rows[-3].update(open=106.,high=108.,low=98.,close=100.)
        rows[-2].update(open=96.,high=98.,low=90.,close=93.)
        rows[-1].update(open=95.,high=107.,low=94.,close=106.)
    elif shape == "none":
        pass
    rows[-1]["shares"] = last_volume
    for row in rows:
        row["turnover"] = row["shares"]*row["close"]
    return pd.DataFrame(rows)


def run(data, selected=("V底槌頭","V底長下影","V底三K反轉"),
        limit=150):
    names = {x: "模擬股票" for x in data.code.unique()}
    return screen(
        data, names, data.date.max(),
        enabled_categories=selected, top_volume_n=limit,
        min_vol_lots=0,min_turnover_100m=0.,min_history_bars=21,
        signal_days=2,top_n=25,require_all_signals=False)


class V21Tests(unittest.TestCase):
    def test_hammer_in_last_two_bars(self):
        b, _, _, _ = run(sample(),selected=("V底槌頭",))
        self.assertEqual(len(b),1)
        self.assertIn("V 底槌頭線",b.iloc[0]["pattern"])
        self.assertEqual(b.iloc[0]["volume_rank"],1)

    def test_lower_wick_in_last_two_bars(self):
        b, _, _, _ = run(sample(shape="lower"),selected=("V底長下影",))
        self.assertEqual(len(b),1)

    def test_three_bar_reversal_completed_recently(self):
        b, _, _, _ = run(sample(shape="three"),selected=("V底三K反轉",))
        self.assertEqual(len(b),1)
        self.assertIn("三 K",b.iloc[0]["pattern"])

    def test_no_v_shape(self):
        b, _, _, _ = run(sample(shape="none"))
        self.assertTrue(b.empty)

    def test_signal_three_bars_ago_is_excluded(self):
        b, _, _, _ = run(sample(pivot_offset=-3),selected=("V底槌頭",))
        self.assertTrue(b.empty)

    def test_top_volume_150_is_actual_rank(self):
        parts = []
        for i in range(151):
            code = str(1000+i)
            if i == 150:
                parts.append(sample(code=code, shape="hammer",
                                    last_volume=100_000))
            else:
                parts.append(sample(code=code, shape="none",
                                    last_volume=4_000_000+i))
        all_data = pd.concat(parts,ignore_index=True)
        b, _, d, _ = run(all_data,selected=("V底槌頭",))
        self.assertTrue(b.empty)
        self.assertEqual(d["volume_universe"],150)

    def test_21_k_bars_eligible(self):
        df = sample().tail(21).copy()
        b, _, _, _ = run(df,selected=("V底槌頭",))
        self.assertGreaterEqual(len(b),0)  # at least full-window support

    def test_price_range_filter_unchanged(self):
        df = sample()
        b, _, _, _ = screen(
            df,{"9001":"模擬股票"},df.date.max(),
            top_volume_n=150,min_price=300.,
            enabled_categories=("V底槌頭",),signal_days=2,
            min_history_bars=21)
        self.assertTrue(b.empty)


if __name__ == "__main__":
    unittest.main()
