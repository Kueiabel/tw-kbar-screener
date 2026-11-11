"""台股上市 K 棒頂底型態搜尋器 — Streamlit UI."""
from __future__ import annotations

from datetime import datetime, timedelta
from html import escape
from math import isfinite
import os
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from threading import Lock
from time import monotonic
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from patterns import screen
from twse_source import ensure_history, load_companies, open_db, today_taiwan

st.set_page_config(page_title="台股 K 棒快速篩選", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
/* Native Streamlit theme handles light-background and text contrast. */
html,body,.stApp,[data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"],button,input,textarea{
 font-family:"PingFang TC",-apple-system,BlinkMacSystemFont,"Noto Sans TC","Microsoft JhengHei",sans-serif !important;
 -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
}
.block-container{max-width:980px;padding-top:3.6rem !important;padding-bottom:4rem;}
h1,h2,h3{letter-spacing:.005em !important;line-height:1.42 !important;}
h1{font-size:1.83rem !important;font-weight:620 !important;}
h2{font-weight:610 !important;}h3{font-weight:580 !important;}
p,li{line-height:1.58;}
[data-testid="stWidgetLabel"] p{font-weight:540 !important;}
.stButton>button{border-radius:13px !important;font-weight:620 !important;min-height:3rem;}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input{border-radius:11px !important;}
div[data-testid="stAlert"]{border-radius:14px;}
div[data-testid="stMetric"]{border:1px solid rgba(95,115,140,.18);border-radius:14px;padding:.68rem .86rem;}
[data-testid="stMetricValue"]{font-weight:640 !important;}
[data-testid="stDataFrame"]{max-width:100%;overflow:hidden;border-radius:12px;}
[data-testid="stTabs"] button{font-weight:600;white-space:nowrap;}
div[data-testid="stExpander"]>details{
 border:1.5px solid rgba(37,99,235,.36) !important;border-radius:14px !important;
 background:rgba(37,99,235,.035) !important;
}
div[data-testid="stExpander"]>details>summary{
 padding:.82rem .92rem !important;font-size:1rem !important;font-weight:660 !important;
}
@media(max-width:640px){
 .block-container{max-width:100% !important;padding:5.65rem .88rem 5rem !important;}
 h1{font-size:clamp(1.35rem,5.4vw,1.74rem) !important;}
 h2{font-size:1.19rem !important;}h3{font-size:1.05rem !important;}
 [data-testid="stMetricValue"]{font-size:1.24rem !important;}
 [data-testid="stMetricLabel"]{font-size:.8rem !important;}
 [data-testid="stHorizontalBlock"]{gap:.55rem;}
 [data-testid="stVerticalBlock"]{gap:.60rem;}
 [data-testid="stPlotlyChart"]{max-width:100%;overflow:hidden;}
 .stButton>button{min-height:3.2rem !important;}
 div[data-testid="stMetric"]{padding:.53rem .68rem;}
 div[data-testid="stExpander"]>details>summary{padding:1rem .85rem !important;}
}

/* Branded, accessible mobile cards. Only scoped classes are colored. */
.qs-brand{padding:.1rem 0 .35rem;}
.qs-kicker{font-size:.75rem;font-weight:720;letter-spacing:.11em;color:#526988;margin:0 0 .24rem;}
.qs-subtitle{font-size:.9rem;line-height:1.58;color:#64748b;margin:.15rem 0 .25rem;}
.qs-status{display:inline-flex;gap:.42rem;align-items:center;padding:.36rem .65rem;border-radius:999px;
 background:#eef5ff;color:#285a9b;border:1px solid #dce9fa;font-size:.78rem;font-weight:640;margin:.4rem 0;}
.qs-summary{margin:.45rem 0 .9rem;padding:.85rem 1rem;border-radius:16px;
 background:linear-gradient(115deg,#f8fafc,#e8eef6);border:1px solid #dbe7f5;}
.qs-summary-label{color:#61718b;font-size:.77rem;font-weight:660;letter-spacing:.04em;}
.qs-summary-date{color:#182f50;font-size:1.2rem;font-weight:720;margin:.18rem 0;}
.qs-summary-note{color:#64748b;font-size:.78rem;line-height:1.5;}
.qs-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem;margin:0 0 1.1rem;}
.qs-stat{min-width:0;border-radius:14px;padding:.8rem .85rem;background:#fafbfd;
 border:1px solid #dce4ee;box-shadow:0 3px 14px rgba(25,49,85,.035);}
.qs-stat-label{font-size:.78rem;color:#65768e;letter-spacing:.012em;}
.qs-stat-value{font-size:1.4rem;font-weight:740;color:#182f50;margin-top:.17rem;line-height:1.3;}
.qs-stat-unit{font-size:.76rem;font-weight:550;color:#728196;margin-left:.25rem;}
.qs-rows{display:grid;gap:.55rem;margin:.55rem 0 1rem;}
.qs-stock{border:1px solid #e0e8f2;background:#fafbfd;border-radius:13px;
 padding:.75rem .88rem;box-shadow:0 2px 10px rgba(22,49,84,.03);}
.qs-stock-top{display:flex;gap:.35rem;align-items:center;justify-content:space-between;}
.qs-stock-name{font-size:.99rem;font-weight:720;color:#20334e;min-width:0;}
.qs-stock-code{font-size:.78rem;font-weight:600;color:#64748b;margin-right:.42rem;}
.qs-stock-score{font-size:.83rem;font-weight:760;color:#2563eb;white-space:nowrap;}
.qs-stock-bottom{font-size:.8rem;color:#64748b;margin-top:.28rem;line-height:1.45;overflow-wrap:anywhere;}
.qs-stock-industry{color:#335c87;font-weight:660;}
.qs-divider{height:1px;background:#e7edf4;margin:.75rem 0;}
@media(max-width:640px){
 .qs-summary{padding:.76rem .85rem;}
 .qs-stats{gap:.5rem;}
 .qs-stat{padding:.72rem .7rem;}
 .qs-stat-value{font-size:1.26rem;}
 .qs-stock{padding:.72rem .78rem;}
}
</style>
""", unsafe_allow_html=True)

# In-process anonymous presence estimate; not unique people or exact traffic.
@st.cache_resource(show_spinner=False)
def _presence_store():
    return {"lock": Lock(), "sessions": {}}


@st.fragment(run_every="30s")
def show_online_count():
    store = _presence_store()
    if "_presence_id" not in st.session_state:
        st.session_state["_presence_id"] = uuid4().hex
    now = monotonic()
    with store["lock"]:
        sessions = store["sessions"]
        for sid in [sid for sid, seen in sessions.items() if now - seen > 90]:
            del sessions[sid]
        sessions[st.session_state["_presence_id"]] = now
        count = len(sessions)
    st.caption(f"約 {count:,} 個活躍瀏覽工作階段 · 最近 90 秒估算，非精確訪客數")


# Owner-supplied product promotion URL; shared by the banner and button.
SHOPEE_OWNER_LINK = "https://s.shopee.tw/AKc0xL7uUk"


def shopee_affiliate_url() -> str:
    # Keep the displayed product destination in sync with the owner's latest URL.
    return SHOPEE_OWNER_LINK


HAIRCARE_PROMOTIONS = (
    ("haircare_banner.webp.b64", SHOPEE_OWNER_LINK,
     "柳屋養髮液", "每天研究大盤，也別忘了頭頂這盤。"),
    ("haircare_ampoule_banner.webp.b64", "https://s.shopee.tw/AUvR9e7H9n",
     "柳屋髮根精華液安瓶", "投資分散配置，保養隨身配置。"),
)


@st.cache_data(show_spinner=False)
def _haircare_banner_data(filename: str) -> str:
    from pathlib import Path
    try:
        return (Path(__file__).resolve().parent / filename).read_text(encoding="ascii").strip()
    except OSError:
        return ""


def render_affiliate_promotion(compact: bool = False):
    # Each product keeps its own destination. Columns stack naturally on mobile.
    columns = st.columns(2)
    for column, (filename, affiliate, title, tagline) in zip(columns, HAIRCARE_PROMOTIONS):
        with column:
            banner = _haircare_banner_data(filename)
            if banner:
                st.markdown(
                    '<a href="' + escape(affiliate, quote=True) + '" target="_blank" '
                    'rel="sponsored noopener noreferrer" aria-label="' +
                    escape("廣告：" + title + "，前往蝦皮查看商品", quote=True) + '" '
                    'style="display:block;border-radius:14px;overflow:hidden;line-height:0;margin:.4rem 0;">'
                    '<img src="data:image/webp;base64,' + banner + '" '
                    'alt="' + escape("廣告：" + title + "。" + tagline, quote=True) + '" '
                    'width="1536" height="1024" '
                    'style="display:block;width:100%;height:auto;" /></a>',
                    unsafe_allow_html=True)
            else:
                st.caption("廣告 · " + title)
            st.link_button("蝦皮查看" + title + " ↗", affiliate,
                           use_container_width=True)


@st.cache_resource(show_spinner=False)
def _search_pool():
    # SQLite daily caches are shared inside each Streamlit process.
    return {"executor": ThreadPoolExecutor(max_workers=2), "lock": Lock()}


def weekly_bars_from_daily(daily: pd.DataFrame, latest_date: str) -> tuple[pd.DataFrame, str]:
    """Build completed weekly OHLCV from official TWSE daily bars, never synthetic minute bars.

    Current Mon-Thu week is excluded; a Friday date can only be used after
    the TWSE daily closing prices are published.
    """
    daily = daily.copy()
    dates = pd.to_datetime(daily['date'])
    weeks = dates.dt.to_period('W-FRI')
    latest = pd.Timestamp(latest_date)
    # The week containing latest_date is usable only when Friday's complete
    # daily bar is available. Otherwise use the last completed Friday week.
    newest_period = latest.to_period('W-FRI')
    if latest.weekday() != 4:
        daily = daily[weeks < newest_period].copy()
        weeks = weeks[weeks < newest_period]
    if daily.empty:
        raise ValueError('目前沒有完整的週 K 資料，請改用日 K。')
    daily['_week'] = weeks.astype(str)
    weekly = (daily.sort_values(['code','date'])
              .groupby(['code','_week'], sort=False)
              .agg(date=('date','last'), name=('name','last'),
                   open=('open','first'), high=('high','max'),
                   low=('low','min'), close=('close','last'),
                   shares=('shares','sum'), turnover=('turnover','sum'))
              .reset_index().drop(columns='_week'))
    return weekly, str(weekly.date.max())


def daily_liquidity_for_week(daily: pd.DataFrame, weekly_date: str) -> dict:
    """Same 20-TRADING-DAY liquidity threshold used by daily screening."""
    actual = daily[daily.date <= weekly_date].sort_values(['code','date'])
    values = {}
    for code, grp in actual.groupby('code', sort=False):
        latest = grp.tail(20)
        if len(latest) == 20 and latest.date.iloc[-1] == weekly_date:
            values[code] = (float(latest.shares.mean()),
                            float(latest.turnover.mean()))
    return values


def run_stock_scan(opts):
    """Run I/O and computation off the UI thread; no Streamlit calls here."""
    pool = _search_pool()
    with pool["lock"]:
        db = None
        try:
            db = open_db()
            with requests.Session() as session:
                companies = load_companies(db, session)
                industry_map = dict(db.execute(
                    "SELECT code,industry FROM listed_companies"))
                days = max(opts["history_days"], 420) if opts["timeframe"] == "W" else opts["history_days"]
                info = ensure_history(
                    db, session, opts["query_date"], days,
                    progress=None, throttle=.85)
            search_data = info["data"]
            search_date = info["effective_date"]
            liquidity = None
            required_bars = 65
            if opts["timeframe"] == "W":
                search_data, search_date = weekly_bars_from_daily(search_data, search_date)
                liquidity = daily_liquidity_for_week(info["data"], search_date)
                required_bars = 55
            bottom, top, diag, covered = screen(
                search_data, companies, search_date,
                min_vol_lots=opts["min_vol"],
                min_turnover_100m=opts["min_money"],
                min_price=opts["min_price"], max_price=opts["max_price"],
                signal_days=opts["signal_days"],
                top_n=25, enabled_categories=opts["targets"],
                ma_windows=opts["ma_windows"],
                hammer_only_bottom=opts["hammer_only_bottom"],
                require_above_ma5=opts["require_above_ma5"],
                require_all_signals=opts["require_all_signals"],
                min_history_bars=required_bars,
                daily_liquidity=liquidity)
            for ranking in (bottom, top):
                if not ranking.empty:
                    ranking["industry"] = ranking["code"].map(
                        industry_map).fillna("未分類")
            return {
                "bottom": bottom, "top": top, "diag": diag,
                "info": {**{k:v for k,v in info.items() if k != "data"},
                         "effective_date": search_date,
                         "history_sessions": int(search_data['date'].nunique())},
                "data": covered,
                "criteria": {
                    "date": str(opts["query_date"]),
                    "timeframe": opts["timeframe"],
                    "min_vol": opts["min_vol"],
                    "min_money": opts["min_money"],
                    "min_price": opts["min_price"],
                    "max_price": opts["max_price"],
                    "signal_days": opts["signal_days"],
                    "targets": list(opts["targets"]),
                    "hammer_only_bottom": opts["hammer_only_bottom"],
                    "require_above_ma5": opts["require_above_ma5"],
                    "require_all_signals": opts["require_all_signals"],
                    "ma_windows": list(opts["ma_windows"]),
                },
            }
        finally:
            if db is not None:
                db.close()


@st.fragment(run_every="2s")
def render_waiting_ad():
    job = st.session_state["scan_future"]
    if job.done():
        try:
            st.session_state["results"] = job.result()
            st.session_state.pop("scan_error", None)
        except Exception as exc:
            st.session_state["scan_error"] = str(exc)
        finally:
            st.session_state.pop("scan_future", None)
        st.rerun(scope="app")
    st.subheader("正在篩選股票，請稍候")
    st.caption("依照已選條件比對盤後行情，完成後立即顯示候選股票。")
    render_affiliate_promotion()
    st.caption("首次下載歷史資料可能較久，請保持網頁開啟。")


EXPORT_FIELDS = {
    'rank':'排名','code':'代號','name':'公司','industry':'產業類別','category':'搜尋類別','pattern':'型態','stage':'階段',
    'score':'評分','signal_date':'訊號日期','pivot_date':'止跌日期',
    'pivot_volume_ratio':'止跌日量／前5期均量','age':'距今交易日',
    'close':'最新收盤價','change_pct':'當日漲跌幅(%)','volume_lots':'當日成交量(張)',
    'avg_volume_lots':'20日均量(張)','avg_turnover_100m':'20日均成交金額(億元)',
    'volume_ratio':'訊號日量比','sma20':'20日均線',
    'matched_patterns':'符合型態','matched_categories':'符合搜尋項目',
    'matched_count':'符合型態數','explanation':'判斷依據'
}


def last_weekday():
    """Default to the latest likely published trading session."""
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    d = now.date()
    if d.weekday() < 5 and now.hour >= 18:
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def to_csv(frame: pd.DataFrame) -> bytes:
    if frame.empty:
        return '沒有符合條件的股票\n'.encode('utf-8-sig')
    cols = [k for k in EXPORT_FIELDS if k in frame.columns]
    return frame[cols].rename(columns=EXPORT_FIELDS).to_csv(index=False).encode('utf-8-sig')


def render_chart(bars: pd.DataFrame, item, side: str, timeframe: str = 'D'):
    code = str(item['code'])
    series = bars[bars.code == code].sort_values('date').tail(100).copy()
    if series.empty:
        st.warning('無法取得此股票的歷史 K 線。')
        return
    series['ma5'] = series['close'].rolling(5).mean()
    series['ma10'] = series['close'].rolling(10).mean()
    series['ma20'] = series['close'].rolling(20).mean()
    series['ma52'] = series['close'].rolling(52).mean()
    x = series.date.astype(str).to_list()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.055,
                        row_heights=[.74,.26])
    up = '#dc3545'  # 台股慣例：紅漲綠跌
    down = '#13a781'
    fig.add_trace(go.Candlestick(x=x, open=series.open, high=series.high,
        low=series.low, close=series.close,
        name='週 K 線' if timeframe == 'W' else '日 K 線',
        increasing_line_color=up, decreasing_line_color=down,
        increasing_fillcolor=up, decreasing_fillcolor=down), row=1,col=1)
    for col, clr, name in [('ma5','#e6ad35','MA5'),('ma10','#57a8f3','MA10'),
                           ('ma20','#2abb75','MA20'),('ma52','#ca9ee7','MA52')]:
        fig.add_trace(go.Scatter(x=x,y=series[col],mode='lines',name=name,
                                 line=dict(width=1.65,color=clr)),row=1,col=1)
    vol_colors = [up if c >= o else down for o,c in zip(series.open,series.close)]
    fig.add_trace(go.Bar(x=x, y=series.shares / 1000, name='成交張數',
                         marker_color=vol_colors,opacity=.65),row=2,col=1)
    date = str(item['signal_date'])
    if date in x:
        marked = series[series.date == date].iloc[-1]
        y = float(marked.low if side == 'bottom' else marked.high)
        axy = 55 if side == 'bottom' else -60
        fig.add_annotation(x=date,y=y,text='◆ 型態訊號',font=dict(color=up if side=='top' else down,size=12),
            arrowcolor=up if side=='top' else down,arrowhead=2,arrowsize=1.2,arrowwidth=2,
            ax=0,ay=axy,showarrow=True,row=1,col=1,bgcolor='rgba(255,255,255,.78)',borderpad=3)
        fig.add_vline(x=x.index(date),line_dash='dot',line_color='#919191',opacity=.35,row=1,col=1)
    fig.update_layout(template="plotly_white",height=500,margin=dict(t=15,b=25,r=18,l=15),
       xaxis_rangeslider_visible=False,legend=dict(orientation='h',y=1.07,x=0),
       hovermode='x unified',plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(type='category',tickangle=-40,nticks=12,showgrid=False)
    fig.update_yaxes(title_text='價格（元）',row=1,col=1,gridcolor='rgba(128,128,128,.16)')
    fig.update_yaxes(title_text='成交（張）',row=2,col=1,gridcolor='rgba(128,128,128,.14)')
    st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})


def render_rankings(frame: pd.DataFrame, side: str, full_data: pd.DataFrame,
                    effective_date: str, timeframe: str = 'D'):
    if frame.empty:
        st.info('這個日期與條件下，沒有符合定義的型態。可以延長訊號觀察天數或稍微放寬流動性門檻；不會以不符合的股票湊滿 25 名。')
        return
    shown = frame.rename(columns={
        'rank':'排名','code':'代號','name':'股票名稱','industry':'產業類別','category':'搜尋類別','pattern':'型態','stage':'階段',
        'matched_patterns':'符合型態','matched_categories':'符合搜尋項目',
        'matched_count':'符合型態數',
        'score':'分數','signal_date':'訊號日期','pivot_date':'止跌日期',
        'pivot_volume_ratio':'止跌日量／前5期均量',
        'age':'距今天數(交易日)',
        'close':'收盤價','change_pct':'漲跌幅 %',
        'volume_lots':'今日量(張)','avg_volume_lots':'20日均量(張)',
        'avg_turnover_100m':'20日均成交額(億)','volume_ratio':'量比'
    })
    # Older Streamlit sessions may still hold cached rankings from before the
    # extra pivot fields existed. Keep the display stable until a fresh scan.
    for optional_col, default in (
            ("止跌日期", ""), ("止跌日量／前5期均量", 0.0)):
        if optional_col not in shown.columns:
            shown[optional_col] = default
    # First five as readable, thumb-friendly rows, without horizontal swiping.
    # The original full 25-stock table, CSV, and chart remain available below.
    st.caption('優先查看前 5 檔；完整 25 檔榜單可在下方展開。新量能條件需重新搜尋才會套用。')
    quick_rows = []
    for _, item in frame.head(5).iterrows():
        code = escape(str(item['code']))
        name = escape(str(item['name']))
        pattern = escape(str(item.get('matched_patterns', item['pattern'])))
        stage = escape(str(item['stage']))
        signal_date = escape(str(item['signal_date']))
        pivot_date = escape(str(item.get('pivot_date') or ''))
        pivot_volume_ratio = float(item.get('pivot_volume_ratio') or 0)
        volume_info = (f' · 止跌日量比 {pivot_volume_ratio:.2f} 倍'
                       if pivot_date and pivot_volume_ratio > 0 else '')
        date_label = (f'止跌 {pivot_date} → 確認 {signal_date}{volume_info}'
                      if pivot_date else f'訊號日期：{signal_date}')
        industry_value = item.get('industry', '未分類')
        industry = escape(str(industry_value)) if pd.notna(industry_value) else '未分類'
        score = int(item['score'])
        price = float(item['close'])
        quick_rows.append(
            '<div class="qs-stock"><div class="qs-stock-top">'
            f'<span class="qs-stock-name"><span class="qs-stock-code">{code}</span>{name}</span>'
            f'<span class="qs-stock-score">{score} 分</span>'
            '</div><div class="qs-stock-bottom">'
            f'<span class="qs-stock-industry">產業：{industry}</span> · 收盤 {price:,.2f} 元'
            f'<div>符合型態：{pattern}</div><div>{date_label} · {stage}</div>'
            '</div></div>'
        )
    st.markdown('<div class="qs-rows">' + ''.join(quick_rows) + '</div>',
                unsafe_allow_html=True)
    with st.expander('查看完整榜單（最多 25 檔）', expanded=False):
        st.dataframe(shown[['排名','代號','股票名稱','產業類別','符合型態','分數','收盤價']],
            hide_index=True,use_container_width=True,height=min(35*len(frame)+44,900),
            column_config={'分數':st.column_config.ProgressColumn(
                '分數',min_value=0,max_value=100,format='%d'),
                '收盤價':st.column_config.NumberColumn('收盤價',format='%.2f')})
    with st.expander('更多欄位（成交量、成交金額、日期等）'):
        st.dataframe(shown[['排名','代號','股票名稱','產業類別',
            '搜尋類別','符合型態','符合搜尋項目','符合型態數',
            '階段','分數','止跌日期','訊號日期','止跌日量／前5期均量','收盤價',
            '漲跌幅 %','今日量(張)','20日均量(張)','20日均成交額(億)','量比']],
            hide_index=True,use_container_width=True,
            column_config={'漲跌幅 %':st.column_config.NumberColumn(
                '漲跌幅 %',format='%.2f')})
    st.download_button('⬇ 下載此榜單 CSV（Excel 可開啟）',data=to_csv(frame),
        file_name=f'TWSE_{side}_{effective_date}.csv',mime='text/csv',key=f'download_{side}')
    with st.expander('需要驗證？展開個股 K 線與訊號說明', expanded=False):
        st.divider()
        st.subheader('查看股票 K 線與訊號位置')
        choices = list(range(len(frame)))
        selected = st.selectbox('選擇排行股票',choices,
            format_func=lambda i: f"#{i+1:02d}  {frame.iloc[i]['code']} {frame.iloc[i]['name']}｜{frame.iloc[i]['pattern']}",
            key=f'select_{side}')
        rec = frame.iloc[selected]
        a,b = st.columns(2)
        a.metric('收盤',f"{rec['close']:.2f} 元",f"{rec['change_pct']:+.2f}%")
        b.metric('型態分數',f"{rec['score']}/100")
        c,d = st.columns(2)
        c.metric('20 日均成交額',f"{rec['avg_turnover_100m']:.2f} 億")
        d.metric('訊號日期',str(rec['signal_date']))
        st.markdown(f"**{rec['code']} {rec['name']}**　｜　產業：**{rec.get('industry','未分類')}**")
        st.markdown(f"**符合型態：{rec.get('matched_patterns', rec['pattern'])}**")
        st.markdown(f"**主要訊號階段：{rec['stage']}**　｜　{rec['explanation']}")
        render_chart(full_data,rec,side,timeframe)
        st.caption('標記位置為型態「辨識／確認」日期。紅 K 為上漲、綠 K 為下跌。'
                   + ('週 K 採完整交易週，均線為週均線。' if timeframe == 'W'
                      else '均線為日均線。'))


# Mobile home-screen users have no obvious browser refresh control.
# This reruns the Streamlit page without interrupting an active scan.
st.markdown('<div class="qs-kicker">TW STOCK SCREENER · 日／週 K 篩選</div>',
            unsafe_allow_html=True)
st.title('台股 K 棒快速篩選')
st.markdown('<p class="qs-subtitle">快速找出可能反轉的上市股票，再用自己的交易工具驗證。</p>',
            unsafe_allow_html=True)
if st.button('↻ 重新整理畫面', key='refresh_page', type='secondary',
             use_container_width=True,
             disabled='scan_future' in st.session_state,
             help='更新頁面狀態；若需最新盤後篩選，請重新執行搜尋。'):
    st.rerun()

if "scan_future" in st.session_state:
    render_waiting_ad()
    st.stop()

show_online_count()
# Visible on normal page load too, not only during scanning or at the
# very bottom of long results. Keep it compact and clearly disclosed.
render_affiliate_promotion(compact=True)
if "scan_error" in st.session_state:
    st.error("搜尋中斷：" + st.session_state.pop("scan_error"))

# Keep the editable search panel immediately below the header/refresh button.
# When results exist, collapse it by default so the rankings remain visible
# without a long scroll on iPhone home-screen mode.
if 'results' in st.session_state:
    st.caption('要重新選股？展開下方條件即可調整。')
search_panel = (st.expander('🔎 調整條件・重新選股', expanded=False)
                if 'results' in st.session_state else st.container(border=True))
with search_panel:
    st.subheader('設定篩選條件')
    st.caption('使用預設條件即可搜尋；需要時再調整。')
    frame_choice = st.selectbox(
        'K 線搜尋週期', ['日 K', '週 K'], index=0,
        help='預設日 K；週 K 由證交所盤後日 K 合成，僅採用已完成交易週。')
    timeframe = 'W' if frame_choice == '週 K' else 'D'
    if timeframe == 'W':
        st.caption('週 K 使用完整交易週；流動性仍依近 20 個交易日計算。首次搜尋可能需要下載較長歷史資料。')
    c_date, c_volume = st.columns(2)
    with c_date:
        query_date = st.date_input('查詢日期', value=last_weekday(),
            min_value=today_taiwan()-timedelta(days=1400),
            max_value=today_taiwan(), format='YYYY/MM/DD')
    with c_volume:
        min_vol = st.number_input('20 日均量下限（張）', min_value=0,
            max_value=100000, value=1000, step=500)
    min_money = st.number_input('20 日平均每日成交金額下限（億元）',
        min_value=0.0, max_value=500.0, value=3.0, step=.5, format='%.1f',
        help='預設 3 億元；可自行調整為 1 億、5 億等。此為股票成交金額，非公司營收。')
    c_low, c_high = st.columns(2)
    with c_low:
        price_low_text = st.text_input('最低股價（元）', value='',
            placeholder='空白＝原預設 10 元')
    with c_high:
        price_high_text = st.text_input('最高股價（元）', value='',
            placeholder='空白＝不限上限')
    price_error = None
    try:
        min_price = float(price_low_text.strip()) if price_low_text.strip() else 10.0
        max_price = float(price_high_text.strip()) if price_high_text.strip() else None
        if not isfinite(min_price) or min_price < 0:
            price_error = '最低股價必須是 0 以上的有限數字。'
        elif max_price is not None and (not isfinite(max_price) or max_price < 0):
            price_error = '最高股價必須是 0 以上的有限數字。'
        elif max_price is not None and max_price < min_price:
            price_error = '最高股價不能小於最低股價。'
    except ValueError:
        min_price, max_price = 10.0, None
        price_error = '股價請輸入有效數字（例如 30、150.5），也可以留白。'
    if price_error:
        st.error(price_error)
    st.markdown('**基本篩選條件**')
    require_above_ma5 = st.checkbox(
        '最新收盤價站上 MA5（五期均線）',
        value=False,
        help='勾選後，只保留最新收盤價高於最近五根 K 棒收盤均價的股票。日 K 是五日線；週 K 是五週線。這是價格位置條件，不是黃金交叉。')
    st.caption('日 K＝站上五日線；週 K＝站上五週線。可自行勾選，不影響其他型態設定。')
    st.markdown('**K 棒訊號**')
    st.caption('主要訊號：低檔止跌 K ＋ 當日量 > 前 5 期均量 ＋ 隔日守低轉強。其他型態皆選填。')
    select_early_confirmation = st.checkbox(
        '低檔止跌 → 隔日轉強確認（主要搜尋）', value=True,
        help='前一交易日位於近期低點、出現槌頭／長下影／開低收回，且當日成交量高於之前五個交易日平均量；下一交易日守住低點且收盤高於止跌 K 收盤。榜單會列出止跌日期、確認日期、量比。')
    extra_pattern_options = [
        '低檔長下影止跌', '低檔開低收回',
        '低檔垂頭線', '低檔倒垂頭線',
        '晨星反轉', '多頭吞噬', '多頭孕育線', '穿刺線',
        '錘頭隔日突破', '低點止跌突破',
        '多次探底突破', '上升通道突破', '放量買盤轉強',
        '高檔上吊線', '高檔流星線',
        '空頭吞噬', '空頭孕育線', '烏雲蓋頂',
        '夜星反轉', '向上跳空', '向下跳空',
        'K 棒反轉', 'W 底／M 頭', '均線突破／跌破',
        'MACD 交叉', 'KD 交叉', 'RSI 回升／轉弱',
    ]
    extra_signals = st.multiselect(
        '其他型態（選填）',
        options=extra_pattern_options, default=[],
        format_func=lambda x: {
            'K 棒反轉': '其他 K 棒反轉（綜合）',
            '均線突破／跌破': '股價穿越均線 MA5／10／20／52',
            '低檔長下影止跌': '長下影止跌（當根提前觀察）',
            '低檔開低收回': '開低收回（當根提前觀察）',
            '晨星反轉': '晨星三 K 確認',
            '低檔垂頭線': '槌頭棒（當根提前觀察）',
            '低檔倒垂頭線': '倒槌頭（當根提前觀察）',
            '高檔上吊線': '上吊線（高檔長下影）',
            '高檔流星線': '流星線（高檔長上影）',
            '多次探底突破': '多次探底 → 頸線突破',
            '上升通道突破': '上升通道 → 突破上緣',
            '放量買盤轉強': '放量紅 K 突破（非買超統計）',
        }.get(x, x),
        help='其他型態預設不啟用，十字星已從全部選項及訊號清單移除。')
    require_all_signals = st.checkbox(
        '必須符合全部勾選訊號', value=False,
        help='預設符合任一勾選訊號就列入；開啟後同一股票須在觀察天數內符合所有選取的訊號。')
    targets = []
    if select_early_confirmation:
        targets.append('低檔止跌隔日確認')
    targets.extend(extra_signals)
    ma_windows = [5, 10, 20, 52]
    # Avoid nesting expanders: Streamlit does not support an expander inside another.
    advanced_panel = (st.container() if 'results' in st.session_state
                      else st.expander('更多設定：訊號天數'))
    with advanced_panel:
        if 'results' in st.session_state:
            st.caption('進階設定 · 訊號天數')
        signal_days = st.select_slider('最近幾個交易日出現訊號',
            options=[1,2,3,4,5],value=3)
        history_days = st.slider('歷史回溯天數',120,250,165,5)
    valid_targets = bool(targets) and price_error is None
    if not targets:
        st.warning('請至少勾選一種搜尋型態。')
    run = st.button('開始篩選股票',type='primary',
        use_container_width=True, disabled=not valid_targets)
    st.caption('型態精準度須經逐筆標記與歷史回測驗證；篩選結果不等於未來上漲機率。')

if run:
    criteria = {
        "query_date": query_date, "timeframe": timeframe,
        "min_vol": min_vol,
        "min_money": min_money, "min_price": min_price,
        "max_price": max_price,
        "targets": list(targets), "ma_windows": list(ma_windows),
        "hammer_only_bottom": False,
        "require_above_ma5": require_above_ma5,
        "require_all_signals": require_all_signals,
        "signal_days": signal_days, "history_days": history_days,
    }
    st.session_state["scan_future"] = _search_pool()["executor"].submit(
        run_stock_scan, criteria)
    st.rerun()

# The search panel stays collapsed after scanning; results immediately follow it.
if 'results' in st.session_state:
    r = st.session_state['results']
    effective_date=r['info']['effective_date']
    if effective_date != r['criteria']['date']:
        if r['criteria'].get('timeframe') == 'W':
            st.info(f"週 K 僅搜尋已完成的交易週，最後使用 **{effective_date}** 交易日資料。")
        else:
            st.warning(f"選擇 {r['criteria']['date']}，該日無完整盤後行情；實際採用 **{effective_date}** 交易日資料。")
    frame_label = '週 K' if r['criteria'].get('timeframe') == 'W' else '日 K'
    st.markdown(
        '<div class="qs-summary">'
        f'<div class="qs-summary-label">篩選完成 · 盤後{frame_label}</div>'
        f'<div class="qs-summary-date">資料截至 {escape(str(effective_date))}</div>'
        f'<div class="qs-summary-note">歷史資料 {r["info"]["history_sessions"]:,} 個{("週" if frame_label == "週 K" else "交易日")} · '
        f'上市名冊 {r["diag"]["listed"]:,} 檔</div>'
        '</div>',
        unsafe_allow_html=True)
    stats = [
        ('可分析股票', r['diag']['covered'], '檔'),
        ('符合流動性', r['diag']['liquid'], '檔'),
        ('底部候選', len(r['bottom']), '/ 25 檔'),
        ('頂部候選', len(r['top']), '/ 25 檔'),
    ]
    stat_html = ''.join(
        '<div class="qs-stat">'
        f'<div class="qs-stat-label">{escape(label)}</div>'
        f'<div class="qs-stat-value">{int(value):,}'
        f'<span class="qs-stat-unit">{escape(unit)}</span></div>'
        '</div>' for label, value, unit in stats
    )
    st.markdown('<div class="qs-stats">' + stat_html + '</div>',
                unsafe_allow_html=True)
    mode_label = ('全部勾選型態均符合' if r['criteria'].get('require_all_signals')
                  else '符合任一勾選型態即可')
    st.caption(f'{mode_label} · 盤後{frame_label}。圈選型態會標示訊號當根日期；評分不是勝率。')
    tabs=st.tabs(['底部候選 Top 25','頂部候選 Top 25','判斷規則'])
    with tabs[0]:
        render_rankings(r['bottom'],'bottom',r['data'],effective_date,
                        r['criteria'].get('timeframe','D'))
    with tabs[1]:
        render_rankings(r['top'],'top',r['data'],effective_date,
                        r['criteria'].get('timeframe','D'))
    with tabs[2]:
        st.markdown('''
**判斷型態**

- **底部預設：** 先有下跌且止跌 K 出現在近 20 期低檔，形狀可為槌頭、長下影或開低收回；**止跌當日成交量必須大於前五個已完成交易日平均量**；下一交易日守住止跌 K 低點、收盤高於止跌 K 收盤。搜尋結果列出止跌日、確認日與止跌日量比。十字星不參與搜尋。
- **組合方式：** 預設符合任一勾選型態就列入；勾選「必須符合所有」時，同一股票需在選定觀察天數內符合全部勾選訊號，可在不同交易日成立。成交量、成交金額、股價及可選 MA5 屬於必須符合的基本過濾條件。
- **其他可選：** 晨星、當根槌頭或長下影、低檔開低收回、多次探底、上升通道、放量紅 K、均線等；未勾選不參與搜尋。
- **頂部：** 流星線（長上影）、空頭吞噬、夜星三 K、突破前高後跌回、局部高點二 K 確認、M 頭雛形／跌破頸線、20 日區間向下跌破。
- **評分：** 型態基本分＋流動性＋訊號日量比＋均線位置／方向－訊號距今交易日的時間折減，最高 100 分。分數不能解讀為上漲或下跌的機率。
- **榜單：** 每檔最多入選一次，但會列出全部匹配的型態；分數以最高分訊號為準。只有符合條件才列入，可能不足 25 檔。

**務必留意**

1. 資料為**盤後**日 K 或完整週 K，不是即時交易訊號。週 K 由官方日 K 合成；本週尚未完成時使用上一個完整交易週。
2. TWSE 官方 OHLC 為**未還原權息**價格。除權息、減資、分割等因素可能形成假性跳空與技術型態；請回到券商圖表人工核對。
3. W 底、M 頭與局部高低點等定義為可重現的幾何規則，並非市場公認的唯一標準。底部／頂部是**候選型態**，不代表一定見底或見頂。
4. 股票名冊以目前上市公司為準，搜尋較早歷史日不等於嚴格的歷史成分股回測；不適合直接用來計算真實策略勝率。
5. 流動性門檻是近 20 日的平均量與成交金額，不保證任何時刻都能以理想價格成交。
6. 歷史數據保存在程式資料夾的 `data/twse_cache.sqlite3`，不需上傳至第三方雲端平台。
        ''')
    st.caption('本工具僅供研究與選股輔助，不構成投資建議。週 K 使用完整交易週，不包含本週尚未結束的 K 棒。')
    # The clearly disclosed Shopee link is already available above the form.
    # Avoid repeated ad cards after the rankings.

