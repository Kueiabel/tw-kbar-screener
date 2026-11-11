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


def _candle_geometry(op: float, hi: float, lo: float, cl: float):
    """Shape of a single K bar, independent of its absolute stock price.

    Return relative lengths of body, upper wick and lower wick as coordinates
    in a normalized candle drawing. Zero-range bars have no useful shape.
    """
    height = float(hi-lo)
    if height <= 0 or not np.isfinite(height):
        return None
    body = abs(float(cl-op))
    upper = max(0.0, float(hi-max(op, cl)))
    lower = max(0.0, float(min(op, cl)-lo))
    return np.array((body, upper, lower), dtype=float) / height


def _shape_match(parts, templates, max_distance: float):
    """Weighted template similarity in normalized K-line drawing space."""
    if parts is None:
        return False
    return min(float(np.sum(np.abs(parts - template)))
               for template in templates) <= max_distance


# Compare geometric layout rather than prices or % daily changes.
_DOJI_TEMPLATES = (
    np.array((.02, .49, .49)),
    np.array((.05, .38, .57)),
    np.array((.05, .57, .38)),
)
_HAMMER_TEMPLATES = (
    np.array((.16, .06, .78)),
    np.array((.24, .04, .72)),
    np.array((.08, .10, .82)),
)
_INVERTED_HAMMER_TEMPLATES = tuple(
    np.array((v[0], v[2], v[1])) for v in _HAMMER_TEMPLATES
)


def _is_regular_doji(op, hi, lo, cl) -> bool:
    """A cross with both visible shadows, never a one-sided dragonfly."""
    p = _candle_geometry(op, hi, lo, cl)
    return bool(p is not None
                and p[1] > 0 and p[2] > 0
                and p[0] < min(p[1], p[2])*.35
                and _shape_match(p, _DOJI_TEMPLATES, .32))


def _is_hammer_shape(op, hi, lo, cl, inverted: bool = False) -> bool:
    """Body at one end of the bar with a clearly dominant opposite wick."""
    p = _candle_geometry(op, hi, lo, cl)
    if p is None:
        return False
    body, upper, lower = p
    long_wick, short_wick = (upper, lower) if inverted else (lower, upper)
    return bool(long_wick > 0 and
                long_wick > 2*max(body, 1e-8) and
                short_wick < long_wick*.30 and
                _shape_match(p, (_INVERTED_HAMMER_TEMPLATES if inverted
                                 else _HAMMER_TEMPLATES), .34))


def _hammer_at(o, h, l, c, idx: int) -> bool:
    """Shape-first bottom hammer with preceding decline/low-zone context."""
    end = len(c) + idx if idx < 0 else idx
    if end < 31:
        return False
    op, hi, lo, cl = (float(arr[end]) for arr in (o, h, l, c))
    if not _is_hammer_shape(op, hi, lo, cl):
        return False
    prior_down = c[end-1] < c[end-8] * .988
    prior_up = c[end-1] > c[end-8] * 1.012
    low20 = float(np.min(l[end-20:end]))
    low30 = float(np.min(l[end-30:end]))
    high30 = float(np.max(h[end-30:end]))
    pos = (cl-low30) / max(1e-8, high30-low30)
    return bool(prior_down or ((lo <= low20*1.035 or pos <= .33)
                               and not prior_up))


def _multi_test_reversal(h, l, c):
    """A breakout after >=3 confirmed similar lows; signal on breakout bar.

    Each swing is known only after two following bars. Compare the *close*
    with the neckline, avoiding hindsight selection of a breakout.
    """
    n = len(c)
    if n < 42 or c[-1] <= c[-2] or c[-1] <= 0:
        return None
    # First bullish breakout, not a marginal close over resistance.
    candle_range = max(float(h[-1]-l[-1]), 1e-8)
    if (c[-1]-l[-1])/candle_range < .66:
        return None
    bottoms = [i for i in _pivots(l, "low") if n-49 <= i <= n-4]
    if len(bottoms) < 3:
        return None
    # Only candidate lows, excluding a last-session unconfirmed dip.
    for k in reversed(range(2, len(bottoms))):
        b = bottoms[k]
        for j in reversed(range(1, k)):
            a = bottoms[j]
            if not 4 <= b-a <= 23:
                continue
            for i in reversed(range(j)):
                first = bottoms[i]
                if not 4 <= a-first <= 23 or b-first > 44:
                    continue
                levels = [float(l[idx]) for idx in (first, a, b)]
                low0 = min(levels)
                if low0 <= 0 or max(levels)/low0 > 1.065:
                    continue
                resistance = float(np.max(h[first:b+1]))
                if resistance / low0 < 1.035:
                    continue
                # A cluster of 3 tests, then the first completed bullish
                # close above intervening highs. Limit time after last test.
                if not 3 <= n-1-b <= 14:
                    continue
                if (c[-1] > resistance*1.005
                        and c[-2] <= resistance*1.005
                        and c[-1] > c[-5:].mean()):
                    return low0, resistance, (first, a, b)
    return None


def _rising_channel_breakout(o, h, l, c):
    """Look for an upward-sloping swing channel after an earlier selloff.

    Use only past confirmed swing highs/lows to project today's upper rail.
    A close crossing the rail for the first time is the signal; no lookahead.
    """
    n = len(c)
    if n < 58 or c[-1] <= o[-1]:
        return None
    body_ratio = (c[-1]-o[-1]) / max(1e-8, h[-1]-l[-1])
    if body_ratio < .50 or (c[-1]-l[-1])/max(1e-8,h[-1]-l[-1]) < .70:
        return None
    for window in (18, 22, 27, 32):
        start, last = n-1-window, n-1
        # The channel follows an observable correction/decline.
        before = c[max(0, start-11)]
        early = float(np.min(l[start:start+5]))
        if before <= 0 or (early > before*.97
                           and early > float(np.max(h[start-11:start]))*.93):
            continue
        highs = [start+i for i in _pivots(h[start:last], "high", 2, 2)]
        lows = [start+i for i in _pivots(l[start:last], "low", 2, 2)]
        if len(highs) < 2 or len(lows) < 2:
            continue
        h1, h2 = highs[-2:]
        l1, l2 = lows[-2:]
        # At least two higher highs and two higher lows, without an
        # excessive single-bar rally masquerading as a channel.
        if not (h[h2] > h[h1]*1.006 and l[l2] > l[l1]*1.006):
            continue
        high_slope = (h[h2]-h[h1])/(h2-h1)
        low_slope = (l[l2]-l[l1])/(l2-l1)
        if high_slope <= 0 or low_slope <= 0:
            continue
        rail_now = h[h2] + high_slope*(n-1-h2)
        rail_prev = h[h2] + high_slope*(n-2-h2)
        # Require a meaningful first close outside the upper rail, rather
        # than a wick-only breach or a tiny move above the projected line.
        if (c[-1] > rail_now*1.006
                and c[-2] <= rail_prev*1.003
                and c[-1] > np.mean(c[-5:])):
            return float(rail_now), float(l[l2]), (h1,h2,l1,l2)
    return None


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
    # Day 1: hammer. Day 2: bullish candle CLOSES above the hammer's
    # entire high by 0.3%. The signal is available as soon as day-2 EOD data
    # is published, so users can review it on the morning of trading day 3.
    # Never treat a hammer by itself as a confirmed entry signal.
    if _hammer_at(o, h, l, c, -2):
        hammer_high = float(h[-2])
        if cl > op and cl > hammer_high * 1.003:
            hammer_date = str(df.date.iloc[-2])
            breakout_date = str(df.date.iloc[-1])
            volume_note = ("；突破日量增" if shares[-1] > shares[-2]
                           else "；突破日量未增")
            add("bottom", "錘頭隔日突破・第三日觀察", 68,
                f"{hammer_date} 錘頭；{breakout_date} 紅 K 收盤突破錘頭最高價 "
                f"{hammer_high:.2f} 元 0.3% 以上{volume_note}；"
                "供下一交易日觀察，尚未代表後續一定上漲",
                "隔日突破已確認", category="錘頭隔日突破")
    # User-marked chart examples (2834 / 1802 / 6269 / 3019):
    # select the rejection candle ITSELF rather than a later breakout.
    # A single candle with meaningful rejection from a recent low or moving-
    # average support, on either red or green body, is an early warning only.
    if "低檔長下影止跌" in allowed and len(c) >= 35:
        candle_range = max(float(hi-lo), 1e-8)
        rejection = float(lower) / candle_range
        upper_ratio = float(upper) / candle_range
        close_position = (float(cl)-float(lo)) / candle_range
        # Price previously moved down, OR it is a recognizable pullback from
        # a local higher trading range. Never require a bullish candle body.
        recent_high = float(np.max(h[-16:-1]))
        prev_decline = (c[-2] < c[-5]*.985 or
                        c[-2] < c[-9]*.97 or
                        c[-2] < recent_high*.965)
        # Must be a fresh downside test: new/near 12-bar low, OR a pullback
        # directly into MA20/MA52 support after an earlier retreat.
        fresh_low = lo <= float(np.min(l[-13:-1]))*1.012
        ma20 = float(np.mean(c[-20:]))
        ma52 = float(np.mean(c[-52:])) if len(c) >= 52 else ma20
        support_test = any(
            level > 0 and abs(lo-level)/level <= .025
            and cl >= level*.995
            for level in (ma20, ma52)
        )
        near_low_zone = (lo <= float(np.min(l[-21:-1]))*1.028 or
                         fresh_low or support_test)
        # Avoid tiny/noisy wicks and unrecovered long bearish closes.
        wick_valid = (rejection >= .48 and lower >= max(body*1.25, span*.48)
                      and upper_ratio <= .24 and close_position >= .66)
        if wick_valid and prev_decline and near_low_zone:
            reason = ("盤中向下測試後收回，長下影占本根 K 棒約"
                      f"{rejection:.0%}；"
                      + ("接近近期低點" if fresh_low else "接近均線支撐區")
                      + "；屬止跌觀察，未代表隔日一定上漲")
            add("bottom", "長下影止跌／支撐反彈", 52,
                reason, "訊號當根／提前觀察",
                category="低檔長下影止跌")
    # User-shared example: several separate, similar low tests followed by
    # a close above the intervening neckline. Do not signal on the lows alone.
    if "多次探底突破" in allowed:
        triple = _multi_test_reversal(h, l, c)
        if triple is not None and cl > op:
            base_low, neckline, pivots = triple
            add("bottom", "多次探底後突破頸線", 64,
                f"三次低點接近 {base_low:.2f} 元，收盤首次突破頸線 "
                f"{neckline:.2f} 元（至少 0.4%）；請確認量價與後續走勢",
                "突破已確認", category="多次探底突破")
    # User-shared example: post-selloff higher-low/higher-high channel whose
    # upper rail is first broken by a strong closing bullish candle.
    if "上升通道突破" in allowed:
        channel = _rising_channel_breakout(o, h, l, c)
        if channel is not None:
            rail, channel_low, pivots = channel
            add("bottom", "上升通道上緣突破", 62,
                f"回檔後形成逐步墊高的高低點，紅 K 收盤首次站上"
                f"通道上緣約 {rail:.2f} 元；最近確認低點約 "
                f"{channel_low:.2f} 元，仍須自行核對通道畫法",
                "突破已確認", category="上升通道突破")
    # Strict early-bottom breakout, not a breakout after a long sideways base:
    # 1) the pivot must be the genuine recent low in an established decline;
    # 2) today's confirmation is nearby in price/time and must reclaim the
    # highest high since the pivot, not merely a tiny pivot candle's high.
    if "低點止跌突破" in allowed and len(c) >= 42 and cl > op:
        today_range = max(float(hi-lo), .00001)
        for bars_after_low in range(1, 5):
            pivot = len(c) - 1 - bars_after_low
            pivot_low = float(l[pivot])
            pivot_high = float(h[pivot])
            if pivot_low <= 0 or pivot_high <= 0:
                continue
            pre8 = float(c[pivot-9])
            pre4 = float(c[pivot-5])
            pivot_close = float(c[pivot])
            downtrend = (float(c[pivot-1]) <= pre8*.965
                         and pivot_close <= pre4*.985
                         and pivot_low <= float(np.min(l[pivot-20:pivot]))*1.003)
            # A fresh reversal must be close to the bottom, not after
            # substantial recovery or weeks of sideways trading.
            near_pivot = cl <= pivot_low*1.085
            held_low = float(np.min(l[pivot+1:])) >= pivot_low*.998
            prior_high = float(np.max(h[pivot:-1]))
            breakout = (cl >= prior_high*1.003
                        and c[-2] <= prior_high*1.003
                        and (cl-op) >= today_range*.45)
            if downtrend and near_pivot and held_low and breakout:
                pivot_date = str(df.date.iloc[pivot])
                add("bottom", "低點止跌後突破", 61,
                    f"{pivot_date} 急跌後創近 20 期低點 {pivot_low:.2f} 元；"
                    f"近期未破低且本期紅 K 收盤突破止跌區間高點 {prior_high:.2f} 元"
                    " 0.3% 以上，仍需核對後續走勢",
                    "突破已確認", category="低點止跌突破")
                break
    if upper >= 2*max(body,span*.08) and upper >= span*.52 and lower <= span*.23 and cl <= lo+span*.4 and near_top:
        add("top", "流星線／長上影", 37, "上影線明顯長於實體，收盤回到區間下半部")
    # Shape-first doji: evaluate the actual cross structure with both
    # upper and lower shadows. The trend/near-low context stays unchanged.
    is_doji = _is_regular_doji(op, hi, lo, cl)
    dragonfly = _is_hammer_shape(op, hi, lo, cl) and body < span*.12
    gravestone = _is_hammer_shape(op, hi, lo, cl, inverted=True) and body < span*.12
    if dragonfly and near_bottom and prior_down:
        add("bottom", "蜻蜓十字星", 42,
            "低檔出現單側長下影十字型，與上下影線均有的多頭十字星分開判斷",
            "提前預警")
    if gravestone and near_top and prior_up:
        add("top", "墓碑十字星", 42,
            "高檔出現單側長上影十字型，與一般十字星分開判斷",
            "提前預警")
    if is_doji and near_bottom and prior_down:
        add("bottom", "多頭十字星", 35,
            "低檔小實體且上下影線清楚的十字形 K 棒；屬提前觀察",
            "提前觀察", category="多頭十字星")
    if is_doji and near_top and prior_up:
        add("top", "空頭十字星", 31,
            "高檔小實體且上下影線清楚的十字形 K 棒；屬提前觀察",
            "提前預警")
    if _is_hammer_shape(op, hi, lo, cl, inverted=True) and near_bottom and prior_down:
        add("bottom", "倒錘子線", 34,
            "下跌後出現小實體與長上影，仍待下一根確認", "提前預警")
    if _is_hammer_shape(op, hi, lo, cl) and near_top and prior_up:
        add("top", "上吊線", 34,
            "上漲後小實體與長下影，仍待下一根確認", "提前預警")
    # Engulfing (body engulfment; previous candle opposite and real body material).
    if c[-2] < o[-2] and cl > op and op <= c[-2]*1.002 and cl >= o[-2]*.998 and near_bottom:
        add("bottom", "多頭吞噬", 43, "紅 K 實體涵蓋前一根黑 K 實體")
    if c[-2] > o[-2] and cl < op and op >= c[-2]*.998 and cl <= o[-2]*1.002 and near_top:
        add("top", "空頭吞噬", 43, "黑 K 實體涵蓋前一根紅 K 實體")
    # 3-candle morning/evening star. Avoid mandatory opening gaps (TWSE trades continuously).
    b0 = abs(c[-3]-o[-3]); b1 = abs(c[-2]-o[-2]); b2 = abs(cl-op)
    if b0 > 0 and b2 > 0 and b1 < b0*.5 and b2 > b1*1.15:
        # Morning star confirmed in a fresh decline (rather than a mid-range
        # three-bar color coincidence). Red confirmation closes above the
        # midpoint of the first bearish body with a meaningful real body.
        # Morning star = decline + bearish candle + small indecision candle
        # + bullish close above the first candle's body midpoint.
        # Avoid excessive hard filters that reject legitimate near-bottom stars.
        star_near_low = float(np.min(l[-3:])) <= float(np.min(l[-23:-3]))*1.03
        star_prior_decline = c[-4] <= c[-12]*.98
        star_first_bear = c[-3] < o[-3] and b0 >= (h[-3]-l[-3])*.4
        star_small = b1 <= b0*.5
        star_confirm = (cl > op and b2 >= b0*.4
                        and cl > (o[-3]+c[-3])/2
                        and cl > max(o[-2],c[-2]))
        if (star_first_bear and star_small and star_confirm
                and star_near_low and star_prior_decline):
            add("bottom", "晨星三 K 確認", 62,
                "跌勢低檔：長黑 K、小實體止跌 K、紅 K 收復首根實體中點；需自行驗證",
                "三 K 確認", category="晨星反轉")
        if c[-3] > o[-3] and cl < op and cl < (o[-3]+c[-3])/2 and near_top:
            add("top", "夜星三 K 反轉", 47, "長紅 K、小實體、黑 K 跌破第一根實體中點")
    if lo < r20_low*.992 and cl > r20_low*1.004 and cl > op:
        add("bottom", "跌破前低後收回", 45, f"盤中跌破近 20 日低點 {r20_low:.2f}，收盤站回")
    if hi > r20_high*1.008 and cl < r20_high*.996 and cl < op:
        add("top", "突破前高後跌回", 45, f"盤中突破近 20 日高點 {r20_high:.2f}，收盤跌回")
    # Independently selectable article candlestick patterns. All of these
    # have distinct categories; no label is exposed without real OHLC logic.
    prev_body = abs(c[-2] - o[-2])
    prev_span = max(float(h[-2] - l[-2]), 1e-8)
    prev_bear = c[-2] < o[-2]
    prev_bull = c[-2] > o[-2]
    cur_bull = cl > op
    cur_bear = cl < op
    down_context = prior_down and near_bottom
    up_context = prior_up and near_top
    if "低檔垂頭線" in allowed and _hammer_at(o,h,l,c,-1):
        add("bottom", "低檔垂頭線", 38,
            "低檔短實體、長下影；僅為反轉候選，還沒有隔日突破確認",
            "提前觀察", category="低檔垂頭線")
    if "高檔上吊線" in allowed and up_context and _is_hammer_shape(op,hi,lo,cl):
        add("top", "高檔上吊線", 37,
            "上漲後長下影與短實體，仍待後續轉弱確認",
            "提前觀察", category="高檔上吊線")
    if "低檔倒垂頭線" in allowed and down_context and _is_hammer_shape(op,hi,lo,cl, inverted=True):
        add("bottom", "低檔倒垂頭線", 36,
            "跌勢末端長上影小實體，尚未確認上漲",
            "提前觀察", category="低檔倒垂頭線")
    if "高檔流星線" in allowed and up_context and _is_hammer_shape(op,hi,lo,cl, inverted=True):
        add("top", "高檔流星線", 37,
            "漲勢末端長上影小實體，仍待轉弱確認",
            "提前觀察", category="高檔流星線")
    if "多頭吞噬" in allowed and down_context and prev_bear and cur_bull and (
            op <= c[-2]*1.002 and cl >= o[-2]*.998
            and body >= prev_body*.95):
        add("bottom", "多頭吞噬", 47,
            "低檔紅 K 實體包覆前一根黑 K 實體",
            category="多頭吞噬")
    if "空頭吞噬" in allowed and up_context and prev_bull and cur_bear and (
            op >= c[-2]*.998 and cl <= o[-2]*1.002
            and body >= prev_body*.95):
        add("top", "空頭吞噬", 47,
            "高檔黑 K 實體包覆前一根紅 K 實體",
            category="空頭吞噬")
    # Harami (mother-child body containment).
    if "多頭孕育線" in allowed and down_context and prev_bear and cur_bull and (
            prev_body >= prev_span*.45 and body <= prev_body*.6
            and min(op,cl) >= c[-2] and max(op,cl) <= o[-2]):
        add("bottom", "多頭孕育線", 41,
            "下跌長黑 K 後出現包在前根實體內的小紅 K（子母線）",
            "提前觀察", category="多頭孕育線")
    if "空頭孕育線" in allowed and up_context and prev_bull and cur_bear and (
            prev_body >= prev_span*.45 and body <= prev_body*.6
            and min(op,cl) >= o[-2] and max(op,cl) <= c[-2]):
        add("top", "空頭孕育線", 41,
            "上漲長紅 K 後出現包在前根實體內的小黑 K（子母線）",
            "提前觀察", category="空頭孕育線")
    # Piercing / dark-cloud cover: first body midpoint is reclaimed/lost.
    if "穿刺線" in allowed and down_context and prev_bear and cur_bull and (
            prev_body >= prev_span*.45 and op <= c[-2]*1.005
            and cl > (o[-2]+c[-2])/2 and cl < o[-2]*1.004):
        add("bottom", "穿刺線", 46,
            "下跌長黑 K 後紅 K 收回前根實體中點以上",
            category="穿刺線")
    if "烏雲蓋頂" in allowed and up_context and prev_bull and cur_bear and (
            prev_body >= prev_span*.45 and op >= c[-2]*.995
            and cl < (o[-2]+c[-2])/2 and cl > o[-2]*.996):
        add("top", "烏雲蓋頂", 46,
            "上漲長紅 K 後黑 K 跌破前根實體中點",
            category="烏雲蓋頂")
    # Standalone evening star (night star), same family as morning star.
    if "夜星反轉" in allowed and len(c) >= 35:
        body0, body1, body2 = (
            abs(c[-3]-o[-3]), abs(c[-2]-o[-2]), abs(c[-1]-o[-1]))
        if (c[-4] >= c[-12]*1.02 and prev_body > 0
                and c[-3] > o[-3]
                and body0 >= (h[-3]-l[-3])*.4
                and body1 <= body0*.5
                and cur_bear and body2 >= body0*.4
                and cl < (o[-3]+c[-3])/2):
            add("top", "夜星三 K 確認", 62,
                "漲勢末端長紅 K、小實體、長黑 K 跌破第一根實體中點",
                "三 K 確認", category="夜星反轉")
    # Price gaps use two completed bars; no look-ahead.
    if "向上跳空" in allowed and h[-2] > 0 and lo > h[-2]*1.001 and cur_bull:
        add("bottom", "向上跳空缺口", 43,
            "本期最低價高於前期最高價，出現尚未回補的上方缺口",
            category="向上跳空")
    if "向下跳空" in allowed and l[-2] > 0 and hi < l[-2]*.999 and cur_bear:
        add("top", "向下跳空缺口", 43,
            "本期最高價低於前期最低價，出現尚未回補的下方缺口",
            category="向下跳空")
    # Volume-backed bullish price action (NOT institutional or dealer net buying).
    # Only the completed signal bar and the preceding 20 completed bars are used.
    # On weekly charts these are weekly volume bars, not 20 daily sessions.
    if "放量買盤轉強" in allowed and len(c) >= 32:
        prev20_avg_shares = float(np.mean(shares[-21:-1]))
        volume_multiple = float(shares[-1]) / max(prev20_avg_shares, 1.0)
        candle_span = max(float(hi-lo), 1e-8)
        close_loc = (float(cl)-float(lo)) / candle_span
        gain = (float(cl)/float(c[-2])-1) if c[-2] > 0 else 0.0
        # Meaningful volume expansion and confirmed price strength, rather
        # than a high-volume selloff or simply elevated activity.
        if (volume_multiple >= 1.8 and cl > op and gain >= .01
                and close_loc >= .75 and cl >= float(h[-2])*1.003):
            add("bottom", "放量紅 K 突破", 56,
                f"本期成交量為前 20 期平均 {volume_multiple:.2f} 倍；"
                f"收盤漲幅 {gain:.1%}、收於當根高低區間上方，"
                "且收盤突破前一根高點 0.3% 以上；"
                "屬量價轉強訊號，非法人／主力買超統計",
                "量價已確認", category="放量買盤轉強")
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
           ma_windows=(20,52), hammer_only_bottom: bool = False,
           min_history_bars: int = 65, daily_liquidity: dict | None = None,
           require_above_ma5: bool = False,
           require_all_signals: bool = False):
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
        if len(grp) < min_history_bars or str(grp.date.iloc[-1]) != as_of:
            continue
        covered += 1
        if daily_liquidity is not None:
            if code not in daily_liquidity:
                continue
            vols, turns = daily_liquidity[code]
        else:
            vols = grp.shares.tail(20).mean()
            turns = grp.turnover.tail(20).mean()
        # MA5 is an optional latest-close filter, not a crossover signal.
        # For weekly bars it naturally means the most recent 5 weekly closes.
        if require_above_ma5 and not (
                float(grp.close.iloc[-1]) > float(grp.close.tail(5).mean())):
            continue
        if (vols < min_vol_lots*1000 or turns < min_turnover_100m*1e8
            or grp.close.iloc[-1] < min_price
            or (max_price is not None and grp.close.iloc[-1] > max_price)
            or grp.shares.iloc[-1] <= 0):
            continue
        liquid += 1
        code_signals = 0
        for age in range(min(signal_days, len(grp)-min_history_bars+1)):
            i = len(grp) - 1 - age
            partial = grp.iloc[:i+1]
            for side, name, base, reason, stage, category in _patterns_at(
                    partial, enabled_categories=enabled_categories, ma_windows=ma_windows):
                # Show this breakout only when it is the most recent
                # completed trading day, ready for next-session review.
                if name in ("錘頭隔日突破・第三日觀察",
                            "低點止跌後突破") and age != 0:
                    continue
                # Strict hammer scan: the bottom list must not contain
                # unrelated MA, engulfing or unconfirmed reversal signals.
                # Top-side screening is unchanged.
                if hammer_only_bottom and side == "bottom" and name not in (
                        "錘頭隔日突破・第三日觀察", "低點止跌後突破",
                        "晨星三 K 確認"):
                    continue
                # An optional rebound is included only when its category is
                # independently selected in the UI; hammer-only remains the default.
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
    chosen_categories = set(enabled_categories)
    # Intersection is evaluated stock-wide, across both bullish and bearish
    # signals, not independently in each of the two ranking tabs.
    stock_categories = {}
    if require_all_signals:
        for entries in result.values():
            for entry in entries:
                stock_categories.setdefault(entry["code"], set()).add(entry["category"])
    for side in ("bottom", "top"):
        frame = pd.DataFrame(result[side])
        if not frame.empty:
            # Gather every distinct signal for a stock across the selected
            # lookback window; do not lose additional patterns during ranking.
            frame = frame.sort_values(
                ["score", "age", "avg_turnover_100m"],
                ascending=[False, True, False])
            combined = []
            for code, matches in frame.groupby("code", sort=False):
                if (require_all_signals and not chosen_categories.issubset(
                        stock_categories.get(code, set()))):
                    continue
                best = matches.iloc[0].to_dict()
                unique_patterns = list(dict.fromkeys(matches["pattern"].astype(str)))
                unique_cats = list(dict.fromkeys(matches["category"].astype(str)))
                best["matched_patterns"] = "、".join(unique_patterns)
                best["matched_categories"] = "、".join(unique_cats)
                best["matched_count"] = len(unique_patterns)
                best["pattern"] = best["matched_patterns"]
                # The best-scoring signal sets the score, while the explanation
                # includes the latest dates for all different matched patterns.
                notes = []
                seen = set()
                for match in matches.sort_values("age").itertuples(index=False):
                    if match.pattern not in seen:
                        notes.append(f"{match.pattern}（{match.signal_date}）：{match.explanation}")
                        seen.add(match.pattern)
                best["explanation"] = "；".join(notes)
                combined.append(best)
            frame = pd.DataFrame(combined)
            if not frame.empty:
                frame = frame.sort_values(
                    ["score", "matched_count", "age", "avg_turnover_100m"],
                    ascending=[False, False, True, False]
                ).head(top_n).reset_index(drop=True)
                frame.insert(0, "rank", np.arange(1, len(frame) + 1))
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
