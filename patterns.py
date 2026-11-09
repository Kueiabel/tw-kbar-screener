"""Deterministic, explainable price-action / K-bar pattern screener.

Optional KD signals are available; all decisions are point-in-time: a signal
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
    category: str
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


def _hammer_at(o, h, l, c, idx: int) -> bool:
    """Detect a bottom hammer at idx using only data available through idx.

    The hammer shape is a *candidate* reversal, not proof of an uptrend.
    A separate next-session close confirms a potential turn.
    """
    end = len(c) + idx if idx < 0 else idx
    if end < 31:
        return False
    op, hi, lo, cl = (float(arr[end]) for arr in (o, h, l, c))
    span = max(0.00001, hi - lo)
    body = abs(cl - op)
    lower = min(op, cl) - lo
    upper = hi - max(op, cl)
    prior_down = c[end-1] < c[end-8] * .988
    prior_up = c[end-1] > c[end-8] * 1.012
    low20 = float(np.min(l[end-20:end]))
    low30 = float(np.min(l[end-30:end]))
    high30 = float(np.max(h[end-30:end]))
    pos = (cl-low30) / max(1e-8, high30-low30)
    near_bottom = prior_down or ((lo <= low20*1.035 or pos <= .33)
                                 and not prior_up)
    return bool(
        lower >= 2*max(body, span*.08)
        and lower >= span*.52
        and upper <= span*.23
        and cl >= lo + span*.60
        and near_bottom
    )


def _patterns_at(df: pd.DataFrame, enabled_categories=("K 棒反轉",), ma_windows=(20,52)):
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
    # Context: in lower/upper 40% of the 30-session trading range, or preceding trend.
    r30_low, r30_high = float(np.min(l[-31:-1])), float(np.max(h[-31:-1]))
    pos = (cl-r30_low) / max(1e-8, r30_high-r30_low)
    prior_down = c[-2] < c[-9] * .988
    prior_up = c[-2] > c[-9] * 1.012
    # Do not call a tall upper shadow a top in the middle of an obvious fall,
    # or a long lower shadow a bottom amid an established rise.
    near_bottom = prior_down or ((lo <= r20_low*1.035 or pos <= .33) and not prior_up)
    near_top = prior_up or ((hi >= r20_high*.965 or pos >= .67) and not prior_down)
    res = []
    allowed = set(enabled_categories)
    def add(side, label, score, reason, stage="已確認", category="K 棒反轉"):
        if category in allowed:
            res.append((side, label, score, reason, stage, category))
    # One-bar candlestick patterns with context; a tiny doji gets minimal body denom.
    if _hammer_at(o, h, l, c, -1):
        add("bottom", "錘頭出現／長下影", 35,
            "下跌後長下影與小實體，顯示盤中承接；仍待下一交易日確認",
            "提前預警")
    # Confirm on the *following* completed daily bar, never on the hammer day.
    # The bullish close must exceed the hammer's real-body high by 0.3%.
    if (_hammer_at(o, h, l, c, -2)
            and cl > op
            and cl > max(o[-2], c[-2])*1.003):
        volume_note = ("；量能高於錘頭當日" if shares[-1] > shares[-2]
                       else "；量能未超過錘頭當日")
        add("bottom", "錘頭確認轉強", 68,
            "前一交易日出現低檔錘頭，今日紅 K 收盤突破錘頭實體高點"
            + volume_note + "，仍須自行驗證", "已確認")
    if upper >= 2*max(body,span*.08) and upper >= span*.52 and lower <= span*.23 and cl <= lo+span*.4 and near_top:
        add("top", "流星線／長上影", 37, "上影線明顯長於實體，收盤回到區間下半部")
    # Candlestick families shown in the reference examples; neutral/ambiguous
    # shapes require preceding trend and are labelled early warnings.
    is_doji = body <= .12*span
    if is_doji and lower >= .65*span and upper <= .15*span and near_bottom and prior_down:
        add("bottom", "蜻蜓十字星", 42, "跌勢末端的長下影十字星；仍需後續紅 K 確認", "提前預警")
    if is_doji and upper >= .65*span and lower <= .15*span and near_top and prior_up:
        add("top", "墓碑十字星", 42, "漲勢末端的長上影十字星；仍需後續黑 K 確認", "提前預警")
    if is_doji and lower >= .20*span and upper >= .20*span:
        if near_bottom and prior_down:
            add("bottom", "多頭十字星", 31, "下跌後出現多空猶豫十字星；尚未確認止跌", "提前預警")
        if near_top and prior_up:
            add("top", "空頭十字星", 31, "上漲後出現多空猶豫十字星；尚未確認見頂", "提前預警")
    if upper >= 2*max(body, span*.08) and upper >= span*.52 and lower <= span*.23 and near_bottom and prior_down:
        add("bottom", "倒錘子線", 34, "下跌後出現長上影小實體，待後續 K 棒確認", "提前預警")
    if lower >= 2*max(body, span*.08) and lower >= span*.52 and upper <= span*.23 and near_top and prior_up:
        add("top", "上吊線", 34, "上漲後出現長下影小實體，待後續 K 棒確認", "提前預警")
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
    # Optional, separately selectable patterns (off by default).
    if "W 底／M 頭" in allowed:
        for side in ("bottom", "top"):
            for label, score, reason, stage in _pair_pattern(o,h,l,c,side):
                add(side, label, score, reason, stage, "W 底／M 頭")
    if "均線突破／跌破" in allowed:
        for period in sorted(set(int(x) for x in ma_windows if int(x) in (5, 10, 20, 52))):
            if len(c) < period + 2:
                continue
            ma_now = float(np.mean(c[-period:]))
            ma_prev = float(np.mean(c[-period-1:-1]))
            # Current close confirms the cross; prior close was on the other side.
            if c[-2] <= ma_prev*1.001 and cl > ma_now*1.003 and cl > op:
                add("bottom", f"站上 MA{period}", 30 if period < 20 else 36,
                    f"收盤價由均線下方站上 MA{period} ({ma_now:.2f})；不代表一定築底",
                    category="均線突破／跌破")
            if c[-2] >= ma_prev*.999 and cl < ma_now*.997 and cl < op:
                add("top", f"跌破 MA{period}", 30 if period < 20 else 36,
                    f"收盤價由均線上方跌破 MA{period} ({ma_now:.2f})；不代表一定見頂",
                    category="均線突破／跌破")
    # Optional popular indicators. They are all OFF by default in the UI.
    # MACD 12/26/9: crossover of DIF and signal line, not a price/MA crossover.
    if "MACD 交叉" in allowed and len(c) >= 55:
        close_s = pd.Series(c)
        dif = (close_s.ewm(span=12, adjust=False).mean()
               - close_s.ewm(span=26, adjust=False).mean())
        dea = dif.ewm(span=9, adjust=False).mean()
        if dif.iloc[-2] <= dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]:
            add("bottom", "MACD 黃金交叉", 36,
                "DIF 由下往上穿越 MACD 訊號線（12/26/9）",
                category="MACD 交叉")
        if dif.iloc[-2] >= dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]:
            add("top", "MACD 死亡交叉", 36,
                "DIF 由上往下跌破 MACD 訊號線（12/26/9）",
                category="MACD 交叉")
    # KD 9/3/3: focus on crosses in the weaker/stronger half.
    if "KD 交叉" in allowed and len(c) >= 36:
        low9 = pd.Series(l).rolling(9, min_periods=9).min()
        high9 = pd.Series(h).rolling(9, min_periods=9).max()
        rsv = ((pd.Series(c) - low9) / (high9 - low9).replace(0, np.nan)
               * 100).fillna(50)
        kval = rsv.ewm(alpha=1/3, adjust=False).mean()
        dval = kval.ewm(alpha=1/3, adjust=False).mean()
        if (kval.iloc[-2] <= dval.iloc[-2] and kval.iloc[-1] > dval.iloc[-1]
                and kval.iloc[-1] <= 50):
            add("bottom", "KD 低檔黃金交叉", 35,
                "9 日 KD 的 K 值向上穿越 D 值，且 K 值不高於 50",
                category="KD 交叉")
        if (kval.iloc[-2] >= dval.iloc[-2] and kval.iloc[-1] < dval.iloc[-1]
                and kval.iloc[-1] >= 50):
            add("top", "KD 高檔死亡交叉", 35,
                "9 日 KD 的 K 值向下跌破 D 值，且 K 值不低於 50",
                category="KD 交叉")
    # RSI (14): trigger only on exiting oversold/overbought territory.
    if "RSI 回升／轉弱" in allowed and len(c) >= 40:
        changes = pd.Series(c).diff()
        avg_gain = changes.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        avg_loss = (-changes.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(100)
        if rsi.iloc[-2] <= 30 < rsi.iloc[-1]:
            add("bottom", "RSI 脫離超賣", 35,
                "14 日 RSI 由 30 以下回升至 30 以上，仍需其他條件驗證",
                category="RSI 回升／轉弱")
        if rsi.iloc[-2] >= 70 > rsi.iloc[-1]:
            add("top", "RSI 脫離超買", 35,
                "14 日 RSI 由 70 以上回落至 70 以下，仍需其他條件驗證",
                category="RSI 回升／轉弱")
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
           min_price: float = 10.0, max_price: float | None = None, signal_days: int = 3,
           top_n: int = 25, enabled_categories=("K 棒反轉",),
           ma_windows=(20,52)):
    """Returns (bottom dataframe, top dataframe, diagnostics, selected price dataframe).

    No substitution of unqualified stocks to reach 25; displays fewer if necessary.
    """
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), {"covered": 0, "liquid": 0}, data
    if not enabled_categories:
        raise ValueError("請至少選取一種搜尋目標")
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
            or grp.close.iloc[-1] < min_price
            or (max_price is not None and grp.close.iloc[-1] > max_price)
            or grp.shares.iloc[-1] <= 0):
            continue
        liquid += 1
        code_signals = 0
        for age in range(min(signal_days,len(grp)-64)):
            i = len(grp) - 1 - age
            partial = grp.iloc[:i+1]
            for side, name, base, reason, stage, category in _patterns_at(
                    partial, enabled_categories=enabled_categories, ma_windows=ma_windows):
                code_signals += 1
                score, ratio = _score(grp,i,side,base,age)
                # K-bars are the main objective even when optional scans are enabled.
                if category == "K 棒反轉":
                    score = min(100, score + 12)
                prev_close = float(grp.close.iloc[-2]) if len(grp) > 1 else float(grp.close.iloc[-1])
                pct = (grp.close.iloc[-1] / prev_close-1)*100 if prev_close else 0
                signal = Signal(
                    side=side, category=category, code=code, name=companies.get(code, grp['name'].iloc[-1]),
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
    # The UI needs price bars only for the displayed Top 25 charts.
    selected_codes = set()
    for frame in final:
        if not frame.empty:
            selected_codes.update(frame['code'].astype(str).tolist())
    chart_data = data[data.code.isin(selected_codes)].copy()
    return final[0], final[1], diag, chart_data
