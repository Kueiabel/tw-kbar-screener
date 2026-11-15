"""Synthetic checks of Tuesday signal screening and T+1/T+3 outcomes."""
import unittest

import numpy as np
import pandas as pd

from backtest_20261110 import (
    _attach_outcomes, _metrics, evaluate_acceptance,
    SIGNAL_DAY, OUTCOME_DAYS,
)


class AcceptanceBacktestTests(unittest.TestCase):
    def test_three_separate_performance_definitions(self):
        rows = pd.DataFrame([
            ["A", 100., 101., 102., 103.],
            ["B", 100., 107., 105., 106.],
            ["C", 100., 99., 101., 90.],
        ], columns=["code",SIGNAL_DAY,*OUTCOME_DAYS])
        bars = rows.melt(id_vars="code",var_name="date",value_name="close")
        ranking = pd.DataFrame({
            "code":["A","B","C"],"name":["A","B","C"],
            "pattern":["mock"]*3})
        result = _attach_outcomes(ranking,bars)
        stats = _metrics(result)
        self.assertEqual(stats["valid"],3)
        self.assertEqual(stats["all_three_daily_up"],1)
        self.assertEqual(stats["ever_above_tuesday"],3)
        self.assertEqual(stats["friday_above_tuesday"],2)
        self.assertEqual(stats["all_three_vs_tuesday"],2)

    def test_missing_friday_bar_excluded_from_denominator(self):
        rows = pd.DataFrame([
            ["A",SIGNAL_DAY,100.],
            ["A",OUTCOME_DAYS[0],110.],
            ["A",OUTCOME_DAYS[1],120.],
            ["B",SIGNAL_DAY,100.],
            ["B",OUTCOME_DAYS[0],110.],
            ["B",OUTCOME_DAYS[1],120.],
            ["B",OUTCOME_DAYS[2],130.],
        ], columns=["code","date","close"])
        ranking = pd.DataFrame({"code":["A","B"]})
        table = _attach_outcomes(ranking,rows)
        stats = _metrics(table)
        self.assertEqual(stats["candidates"],2)
        self.assertEqual(stats["valid"],1)
        self.assertEqual(stats["missing"],1)
        self.assertEqual(stats["all_three_daily_up_rate"],100.)

    def test_missing_required_market_date_is_error(self):
        base = pd.DataFrame({
            "date":[SIGNAL_DAY],"code":["1001"],
            "close":[100.],"open":[99.],"high":[101.],
            "low":[98.],"shares":[1000.],"turnover":[100000.]})
        with self.assertRaisesRegex(ValueError,"缺少"):
            evaluate_acceptance(base,{"1001":"test"})


if __name__ == "__main__":
    unittest.main()
