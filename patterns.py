"""Deterministic, explainable price-action / K-bar pattern screener.

No stochastic KD indicator is used. All decisions are point-in-time: a signal
at index i is decided only using OHLCV rows through i. Recent swing pivots are
acknowledged *two bars after* the extremum, never before.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd


@dataclass
class Signal:
    side: str
    code: str
    name: str
    pattern: str
    signal_date: str
    age: int
    score: int
    close: float
    change_pct: float
    volume_lots: int
    avg_volume_lots: int
    avg_turnover_100m: float
    volume_ratio: float
    sma20: float
    explanation: str
    stage: str

    def as_dict(self):
        return asdict(self)


def _pivots(arr, kind: str, left: int = 2, right: int = 2):
    piv = []
    n = len(arr)
    for i in range(left, n - right):
        leftpart, rightpart = arr[i-left:i], arr[i+1:i+right+1]
        if kind == "low" and arr[i] <= min(leftpart) and arr[i] < min(rightpart):
            piv.append(i)
        elif kind == "high" and arr[i] >= max(leftpart) and arr[i] > max(rightpart):
            piv.append(i)
    return piv


def _pair_pattern(o, h, l, c, side: str):
    """Strict double bottom/top; returns [(name,base_score,reason,stage)]."""
    if len(c) < 36:
        return []
    is_bottom = side == 'bottom'
    arr = l if is_bottom else h
    pivots = _pivots(arr, 'low' if is_bottom else 'high')
    pivots = [p for p in pivots if p >= max(2, len(c)-55)]
    for b in reversed(pivots):
        for a in reversed([p for p in pivots if 6 <= b-p <= 29]):
            x, y = arr[a], arr[b]
            if abs(x-y)/max(x,y) > 0.055:
                continue
            if is_bottom:
                neck = float(np.max(h[a:b+1]))
                depth = neck / min(x,y) - 1
                if depth < .055:
                    continue
                confirmed = c[-1] > neck*1.005 and c[-2] <= neck*1.005
                if confirmed and b >= len(c)-24:
                    return [("W 底突破頸線", 55, f"雙低點接近（差 {abs(x-y)/max(x,y):.1%}），突破頸線 {neck:.2f}", "已確認")]
                if len(c)-1-b <= 6 and min(x,y)*1.025 < c[-1] <= neck*1.005:
                    return [("W 底雛形", 27, f"雙低點接近，頸線約 {neck:.2f} 尚未突破", "提前預警")]
            else:
                neck = float(np.min(l[a:b+1]))
                depth = max(x,y)/neck - 1
                if depth < .055:
                    continue
                confirmed = c[-1] < neck*.995 and c[-2] >= neck*.995
                if confirmed and b >= len(c)-24:
                    return [("M 頭跌破頸線", 55, f"雙高點接近（差 {abs(x-y)/max(x,y):.1%}），跌破頸線 {neck:.2f}", "已確認")]
                if len(c)-1-b <= 6 and neck*.995 <= c[-1] < max(x,y)*.975:
                    return [("M 頭雛形", 27, f"雙高點接近，頸線約 {neck:.2f} 尚未跌破", "提前預警")]
    return []


def _patterns_at(df: pd.DataFrame):
    """Pattern tuples (side,name,base_score,reason,stage) for the FINAL bar."""
    if len(df) < 32:
        return []
    a = df[['open','high','low','close','shares','turnover']].to_numpy(dtype=float)
    o,h,l,c,shares,turnover = (a[:,i] for i in range(6))
    op,hi,lo,cl = o[-1],h[-1],l[-1],c[-1]
    span = max(.00001, hi-lo)
    body = abs(cl-op)
    lower = min(op,cl)-lo
    upper = hi-max(op,cl)
    r20_low = float(np.min(l[-21:-1]))
    r20_high = float(np.max(h[-21:-1]))
    prior15 = c[-16:-1]
    # Context: in lower/upper 40% of the 30-session trading range, or preceding trend.
    r30_low, r30_high = float(np.min(l[-31:-1])), float(np.max(h[-31:-1]))
    pos = (cl-r30_low) / max(1e-8, r30_high-r30_low)
    prior_down = c[-2] < c[-9] * .975
    prior_up = c[-2] > c[-9] * 1.025
    near_bottom = lo <= r20_low*1.035 or pos <= .4 or prior_down
    near_top = hi >= r20_high*.965 or pos >= .6 or prior_up
    res = []
    def add(side, label, score, reason, stage="已確認"):
        res.append((side,label,score,reason,stage))
    # One-bar candlestick patterns with context; a tiny doji gets minimal body denom.
    if lower >= 2*max(body,span*.08) and lower >= span*.52 and upper <= span*.23 and cl >= lo+span*.60 and near_bottom:
        add("bottom", "錘子線／長下影", 37, "下影線明顯長於實體，收盤收回區間上半部")
    if upper >= 2*max(body,span*.08) and upper >= span*.52 and lower <= span*.23 and cl <= lo+span*.4 and near_top:
        add("top", "流星線／長上影", 37, "上影線明顯長於實體，收盤回到區間下半部")
    # Engulfing (body engulfment; previous candle opposite and real body material).
    if c[-2] < o[-2] and cl > op and op <= c[-2]*1.002 and cl >= o[-2]*.998 and near_bottom:
        add("bottom", "多頭吞噬", 43, "紅 K 實體涵蓋前一根黑 K 實體")
    if c[-2] > o[-2] and cl < op and op >= c[-2]*.998 and cl <= o[-2]*1.002 and near_top:
        add("top", "空頭吞噬", 43, "黑 K 實體涵蓋前一根紅 K 實體")
    # 3-candle morning/evening star. Avoid mandatory opening gaps (TWSE trades continuously).
    b0 = abs(c[-3]-o[-3]); b1 = abs(c[-2]-o[-2]); b2 = abs(cl-op)
    if b0 > 0 and b2 > 0 and b1 < b0*.5 and b2 > b1*1.15:
        if c[-3] < o[-3] and cl > op and cl > (o[-3]+c[-3])/2 and near_bottom:
            add("bottom", "晨星三 K 反轉", 47, "長黑 K、小實體、紅 K 收回第一根實體中點")
        if c[-3] > o[-3] and cl < op and cl < (o[-3]+c[-3])/2 and near_top:
            add("top", "夜星三 K 反轉", 47, "長紅 K、小實體、黑 K 跌破第一根實體中點")
    if lo < r20_low*.992 and cl > r20_low*1.004 and cl > op:
        add("bottom", "跌破前低後收回", 45, f"盤中跌破近 20 日低點 {r20_low:.2f}，收盤站回")
    if hi > r20_high*1.008 and cl < r20_high*.996 and cl < op:
        add("top", "突破前高後跌回", 45, f"盤中突破近 20 日高點 {r20_high:.2f}，收盤跌回")
    # Breakout/breakdown may be continuation, keep labelled as such (not guaranteed reversal).
    if cl > r20_high*1.006 and c[-2] <= float(np.max(h[-22:-2]))*1.006:
        add("bottom", "20 日區間向上突破", 32, f"收盤突破前 20 日高點 {r20_high:.2f}；屬轉強型態，不保證是底部")
    if cl < r20_low*.994 and c[-2] >= float(np.min(l[-22:-2]))*.994:
        add("top", "20 日區間向下跌破", 32, f"收盤跌破前 20 日低點 {r20_low:.2f}；屬轉弱型態，不保證是頂部")
    for side in ("bottom","top"):
        for label, score, reason, stage in _pair_pattern(o,h,l,c,side):
            add(side,label,score,reason,stage)
    # Confirmed swing: pivot is two bars ago; signal date = confirmation day.
    if len(df) >= 35:
        test_l = l[-3]
        if (test_l < min(l[-5],l[-4]) and test_l <= min(l[-2],l[-1])
            and test_l <= min(l[-23:-3])*1.025 and cl >= c[-3]*1.018):
            add("bottom", "局部低點二 K 確認", 30, "兩根後續 K 棒未創新低，且收盤自低點回升；確認日為今天")
        test_h = h[-3]
        if (test_h > max(h[-5],h[-4]) and test_h >= max(h[-2],h[-1])
            and test_h >= max(h[-23:-3])*.975 and cl <= c[-3]*.982):
            add("top", "局部高點二 K 確認", 30, "兩根後續 K 棒未創新高，且收盤自高點回落；確認日為今天")
    return res


def _score(df: pd.DataFrame, i: int, side: str, base: int, age: int):
    current = df.iloc[i]
    window = df.iloc[max(0,i-19):i+1]
    turn20 = float(window.turnover.mean())
    vol20 = float(window.shares.mean())
    # previous 20 sessions, excluding signal date, as signal volume denominator
    prev = df.iloc[max(0,i-20):i]
    prev_vol = float(prev.shares.mean()) if len(prev) else vol20
    vol_ratio = float(current.shares) / max(1, prev_vol)
    liquidity = min(12, 4+4*math.log10(max(1., turn20/1e8)))
    volume_bonus = min(15, max(-4, (vol_ratio-1)*12))
    closes = df.close.to_numpy(dtype=float)
    sma20 = float(closes[max(0,i-19):i+1].mean())
    prev_sma20 = float(closes[max(0,i-24):max(1,i-4)].mean())
    slope = (sma20/prev_sma20 - 1) if prev_sma20 else 0
    ma_bonus = 0
    if side == "bottom":
        ma_bonus += (6 if current.close >= sma20 else 0)
        ma_bonus += (5 if slope >= 0 else 0)
    else:
        ma_bonus += (6 if current.close <= sma20 else 0)
        ma_bonus += (5 if slope <= 0 else 0)
    score = max(1, min(100, round(base + liquidity + volume_bonus + ma_bonus - age*5)))
    return score, vol_ratio


def screen(data: pd.DataFrame, companies: dict[str, str],
           as_of: str, min_vol_lots: int = 1000, min_turnover_100m: float = 1.0,
           min_price: float = 10.0, signal_days: int = 3,
           top_n: int = 25):
    """Returns (bottom dataframe, top dataframe, diagnostics, selected price dataframe).

    No substitution of unqualified stocks to reach 25; displays fewer if necessary.
    """
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), {"covered": 0, "liquid": 0}, data
    data = data[data.code.isin(companies.keys()) & (data.date <= as_of)].copy()
    data = data.sort_values(["code","date"])
    result = {"bottom":[], "top":[]}
    covered, liquid, with_patterns = 0, 0, 0
    for code, grp in data.groupby('code', sort=False):
        grp = grp.reset_index(drop=True)
        if len(grp) < 65 or str(grp.date.iloc[-1]) != as_of:
            continue
        covered += 1
        vols = grp.shares.tail(20).mean()
        turns = grp.turnover.tail(20).mean()
        if (vols < min_vol_lots*1000 or turns < min_turnover_100m*1e8
            or grp.close.iloc[-1] < min_price or grp.shares.iloc[-1] <= 0):
            continue
        liquid += 1
        code_signals = 0
        for age in range(min(signal_days,len(grp)-64)):
            i = len(grp) - 1 - age
            partial = grp.iloc[:i+1]
            for side, name, base, reason, stage in _patterns_at(partial):
                code_signals += 1
                score, ratio = _score(grp,i,side,base,age)
                prev_close = float(grp.close.iloc[-2]) if len(grp) > 1 else float(grp.close.iloc[-1])
                pct = (grp.close.iloc[-1] / prev_close-1)*100 if prev_close else 0
                signal = Signal(
                    side=side, code=code, name=companies.get(code, grp.name.iloc[-1]),
                    pattern=name, signal_date=str(grp.date.iloc[i]), age=age,
                    score=score, close=round(float(grp.close.iloc[-1]),2),
                    change_pct=round(pct,2), volume_lots=int(grp.shares.iloc[-1]//1000),
                    avg_volume_lots=int(vols//1000), avg_turnover_100m=round(float(turns)/1e8,2),
                    volume_ratio=round(float(ratio),2),
                    sma20=round(float(grp.close.tail(20).mean()),2),
                    explanation=reason, stage=stage,
                )
                result[side].append(signal.as_dict())
        if code_signals:
            with_patterns += 1
    final = []
    for side in ("bottom", "top"):
        frame = pd.DataFrame(result[side])
        if not frame.empty:
            frame = (frame.sort_values(["score","age","avg_turnover_100m"],ascending=[False,True,False])
                     .drop_duplicates("code").head(top_n).reset_index(drop=True))
            frame.insert(0,"rank",np.arange(1,len(frame)+1))
        final.append(frame)
    diag = {"covered": covered, "liquid": liquid, "matched":with_patterns, "listed":len(companies),
            "bottom_candidates":len(result['bottom']),"top_candidates":len(result['top'])}
    return final[0], final[1], diag, data
