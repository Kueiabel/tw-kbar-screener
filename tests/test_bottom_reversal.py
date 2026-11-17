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
            open=float(p+.5), high=float(p+2.0),
            low=float(p-1.0), close=float(p),
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
    # Reversal needs a bullish CONFIRMATION bar, not only a pivot wick.
    # When a pivot was selected four days ago, confirm it on day -3.
    if shape in ("hammer", "lower"):
        ci = pivot_offset+1 if pivot_offset < -1 else None
        if ci is not None:
            rows[ci].update(open=96.,high=103.,low=93.,close=102.)
    rows[-1]["shares"] = last_volume
    for row in rows:
        row["turnover"] = row["shares"]*row["close"]
    return pd.DataFrame(rows)


def run(data, selected=("V底槌頭","V底長下影","V底三K反轉"),
        limit=150, signal_days=2, week_low=False):
    names = {x: "模擬股票" for x in data.code.unique()}
    return screen(
        data, names, data.date.max(),
        enabled_categories=selected, top_volume_n=limit,
        min_vol_lots=0,min_turnover_100m=0.,min_history_bars=21,
        signal_days=signal_days,top_n=25,require_all_signals=False,
        require_week_low=week_low)


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
        # A current-day signal can use exactly 21 bars. A yesterday
        # signal needs 22 bars: 21 available AS OF yesterday plus today.
        df = sample().tail(21).copy()
        b, _, _, _ = run(df,selected=("V底槌頭",))
        self.assertTrue(b.empty)  # pivot occurred one bar ago: needs 22
        enough = sample().tail(22).copy()
        b2, _, _, _ = run(enough,selected=("V底槌頭",))
        self.assertEqual(len(b2),1)

    def test_week_window_detects_earlier_v_bottom(self):
        df = sample(pivot_offset=-4)
        old, _, _, _ = run(df,selected=("V底槌頭",),signal_days=2)
        new, _, _, _ = run(
            df,selected=("V底槌頭",),signal_days=5,week_low=True)
        self.assertTrue(old.empty)
        self.assertEqual(len(new),1)
        self.assertEqual(new.iloc[0]["age"],2)

    def test_week_window_rejects_if_another_day_is_lower(self):
        df = sample(pivot_offset=-4)
        # Break the previous pivot low on the scan date; signal should be
        # invalidated, without creating a new confirmed bottom.
        ix = df.index[-1]
        df.loc[ix, ["open","high","low","close"]] = [88.,106.,86.,101.]
        new, _, _, _ = run(
            df,selected=("V底槌頭",),signal_days=5,week_low=True)
        self.assertTrue(new.empty)

    def test_week_low_scan_respects_top_150(self):
        df = sample()
        new, _, _, _ = run(df,selected=("V底槌頭",),
                           limit=0,signal_days=5,week_low=True)
        self.assertTrue(new.empty)


    def test_unconfirmed_latest_hammer_is_excluded(self):
        df = sample(pivot_offset=-1, shape="hammer")
        b, _, _, _ = run(df,selected=("V底槌頭",))
        self.assertTrue(b.empty, "Single hammer without next K cannot be V rebound")

    def test_gigabyte_like_continued_slide_is_excluded(self):
        df = sample(shape="none")
        # Last three bars still make progressively lower closes;
        # a continuing decline is not a V reversal.
        df.loc[df.index[-3], ["open","high","low","close"]] = [102.,103.,96.,97.]
        df.loc[df.index[-2], ["open","high","low","close"]] = [96.,97.,90.,91.]
        df.loc[df.index[-1], ["open","high","low","close"]] = [91.,92.,85.,87.]
        b, _, _, _ = run(df)
        self.assertTrue(b.empty)

    def test_pivot_hammer_with_weak_next_red_is_excluded(self):
        df = sample()
        # A small positive candle that fails to recover 1% is not confirmed.
        df.loc[df.index[-1], ["open","high","low","close"]] = [95.3,96.,94.,95.6]
        b, _, _, _ = run(df,selected=("V底槌頭",))
        self.assertTrue(b.empty)

    def test_price_range_filter_unchanged(self):
        df = sample()
        b, _, _, _ = screen(
            df,{"9001":"模擬股票"},df.date.max(),
            top_volume_n=150,min_price=300.,
            enabled_categories=("V底槌頭",),signal_days=2,
            min_history_bars=21)
        self.assertTrue(b.empty)



class SakataBuyOnlyTests(unittest.TestCase):
    @staticmethod
    def bars():
        dates = pd.bdate_range(end="2026-11-13", periods=21)
        # Default: mild downward background, no bullish three-bar patterns.
        prices = np.linspace(125, 98, len(dates))
        return pd.DataFrame([
            {"date":d.strftime("%Y-%m-%d"), "code":"9001",
             "name":"模擬股票", "open":float(p+.7),
             "high":float(p+1.3), "low":float(p-1.3),
             "close":float(p), "shares":2_000_000,
             "turnover":float(p*2_000_000)}
            for d,p in zip(dates,prices)])

    def test_morning_star_bullish_only(self):
        from patterns import _v21_patterns_at
        df = self.bars()
        df.loc[18, ["open","high","low","close"]] = [105.,106.,92.,93.]
        df.loc[19, ["open","high","low","close"]] = [92.,94.,90.,92.5]
        df.loc[20, ["open","high","low","close"]] = [93.,104.,92.,103.]
        got = _v21_patterns_at(df, ["酒田三川晨星"])
        self.assertIn("酒田三川晨星", [x[5] for x in got])
        self.assertTrue(all(x[0] == "bottom" for x in got))

    def test_three_red_soldiers(self):
        from patterns import _v21_patterns_at
        df = self.bars()
        for k, op,cl in ((18,95.,98.),(19,97.,100.),(20,99.,102.)):
            df.loc[k, ["open","high","low","close"]] = [
                op,cl+.5,op-.6,cl]
        got = _v21_patterns_at(df, ["酒田紅三兵"])
        self.assertIn("酒田紅三兵", [x[5] for x in got])

    def test_rising_three_methods_continuation(self):
        from patterns import _v21_patterns_at
        df = self.bars()
        candles = {
            16:(100.,111.,99.,110.),
            17:(108.,109.,105.,106.),
            18:(107.,108.,103.,105.),
            19:(105.,107.,102.,104.),
            20:(105.,116.,104.,115.),
        }
        for k, vals in candles.items():
            df.loc[k, ["open","high","low","close"]] = vals
        got = _v21_patterns_at(df, ["酒田上升三法"])
        self.assertNotIn("酒田上升三法", [x[5] for x in got])

    def test_no_bearish_sakata_results(self):
        from patterns import _v21_patterns_at
        df = self.bars()
        got = _v21_patterns_at(df, [
            "酒田三川晨星","酒田紅三兵","酒田上升三法"])
        self.assertFalse(any(x[0] == "top" for x in got))

    def test_sakata_optional_does_not_change_v_default(self):
        from patterns import _v21_patterns_at
        df = self.bars()
        got = _v21_patterns_at(df, ["V底槌頭","V底長下影","V底三K反轉"])
        self.assertFalse(any(str(x[5]).startswith("酒田") for x in got))


if __name__ == "__main__":
    unittest.main()
