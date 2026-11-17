"""Run the official Tuesday Nov 10 / Friday Nov 13 backtest in GitHub Actions.

The only data source is TWSE official after-trading daily market OHLCV.
Output is committed to reports/acceptance_20261110 for auditability.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from backtest_20261110 import evaluate_acceptance
from twse_source import ensure_history, load_companies, open_db

OUTPUT = ROOT / "reports" / "acceptance_20261110"
OUTPUT.mkdir(parents=True, exist_ok=True)


def save_json(name: str, payload: dict):
    (OUTPUT/name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    print("Retrieving official TWSE 2026-08-15...2026-11-13 after-hours prices.",flush=True)
    db = open_db()
    try:
        with requests.Session() as session:
            companies = load_companies(db, session)
            info = ensure_history(
                db, session, date(2026,11,13),
                history_calendar_days=90, throttle=.35)
    finally:
        db.close()

    bars = info["data"]
    print(f"TWSE bars: {len(bars):,}; sessions: {info['history_sessions']}, "
          f"companies in current master: {len(companies)}",flush=True)
    report = evaluate_acceptance(bars, companies)
    summary = {
        "signal_date":"2026-11-10",
        "future_days":["2026-11-11","2026-11-12","2026-11-13"],
        "data_source":"Taiwan Stock Exchange MI_INDEX after-trading",
        "historical_sessions":int(info["history_sessions"]),
        "listed_company_master":len(companies),
        "source_qualification":"current company list; unadjusted OHLCV; survivorship bias possible",
        "groups":{},
    }
    for name, group in report.items():
        summary["groups"][name] = {
            "volume_universe":group["volume_universe"],
            "two_session_signals":group["metrics_available"],
            "tuesday_only_signals":group["metrics_exact"],
            "tuesday_top25":group["metrics_top25"],
        }
        for mode, key in (
            ("discovered","available_tuesday"),
            ("exact","finished_tuesday"),
            ("top25","top25_tuesday"),
        ):
            df=group[key]
            out=OUTPUT/f"{'v_only' if name.startswith('原') else 'v_sakata'}_{mode}.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
    save_json("summary.json", summary)

    # Visual false-positive audit from the owner's 2026-11-13 screenshots.
    # Do not use future candles; verify from official as-of Nov 13 prices.
    from patterns import screen
    audit_day = "2026-11-13"
    cats = ("V底槌頭", "V底長下影", "V底三K反轉",
            "酒田三川晨星", "酒田紅三兵")
    focus = ("2376", "2891", "2886", "2412", "2409")
    audit = {"as_of":audit_day, "focus_codes":list(focus), "modes":{}}
    todays = bars[bars.date == audit_day].sort_values(
        "shares",ascending=False).reset_index(drop=True)
    for code in focus:
        stock = bars[(bars.code == code) & (bars.date <= audit_day)].tail(21)
        audit[code] = {
            "volume_rank":(
                int(todays.index[todays.code == code][0]) + 1
                if (todays.code == code).any() else None),
            "last_6_K":stock.tail(6)[
                ["date","open","high","low","close","shares"]
            ].to_dict(orient="records"),
        }
    for label,days,week_low in (
        ("最近兩根",2,False),
        ("最近五根且週低",5,True),
    ):
        selected, _, diag, _ = screen(
            bars, companies, audit_day,
            enabled_categories=cats,
            min_vol_lots=0, min_turnover_100m=0,
            min_history_bars=21, min_price=10,
            signal_days=days, require_week_low=week_low,
            top_n=150, top_volume_n=150,
        )
        audit["modes"][label] = {
            "total":int(len(selected)),
            "focus_hits":[
                {
                    "code":str(row.code),
                    "name":str(row.name),
                    "pattern":str(row.pattern),
                    "signal_date":str(row.signal_date),
                } for row in selected.itertuples(index=False)
                if str(row.code) in focus
            ],
        }
    save_json("audit_20261113.json", audit)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print("Visual false-positive case check:",json.dumps(
        audit["modes"],ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
