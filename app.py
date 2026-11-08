"""台股上市 K 棒頂底型態搜尋器 — Streamlit UI."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from threading import Lock
from time import monotonic
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from patterns import screen
from twse_source import ensure_history, load_companies, open_db, today_taiwan

st.set_page_config(page_title="台股 K 棒頂底搜尋器", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
html, body, .stApp, [data-testid="stAppViewContainer"] {
  font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Heiti TC","Noto Sans TC",sans-serif;
  -webkit-font-smoothing:antialiased;
}
.block-container{padding-top:1.15rem;max-width:1100px;padding-bottom:3rem;}
h1{font-size:2rem;line-height:1.3;font-weight:700;}
[data-testid="stWidgetLabel"] p{font-weight:550;}
div[data-testid="stAlert"]{border-radius:.75rem;}
h1,h2,h3{letter-spacing:0;overflow-wrap:anywhere;}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.20);padding:.55rem .7rem;border-radius:.65rem;}
[data-testid="stDataFrame"]{max-width:100%;overflow:hidden;}
@media(max-width:640px){
  .block-container{padding:1rem .8rem 5rem !important;max-width:100% !important;}
  h1{font-size:clamp(1.35rem,5.7vw,1.8rem) !important;line-height:1.3 !important;}
  h2{font-size:1.25rem !important;}
  h3{font-size:1.1rem !important;}
  p,li{overflow-wrap:anywhere;}
  [data-testid="stMetricValue"]{font-size:1.25rem !important;}
  [data-testid="stMetricLabel"]{font-size:.78rem !important;}
  [data-testid="stHorizontalBlock"]{gap:.65rem;}
  [data-testid="stVerticalBlock"]{gap:.55rem;}
  [data-testid="stPlotlyChart"]{max-width:100%;overflow:hidden;}
  [data-testid="stTabs"] button{font-size:.82rem;white-space:nowrap;}
  .stButton>button{min-height:2.85rem;border-radius:.65rem;}
  [data-testid="stWidgetLabel"] p{font-size:.9rem;}
  div[data-testid="stMetric"]{padding:.52rem .65rem;}
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
    st.caption(f"🟢 目前約 {count:,} 人在線 · 每 30 秒更新")
    st.caption("最近 90 秒有活動的瀏覽工作階段估算，不收集姓名或 IP。")


# Sponsor content can be configured once a real advertiser has been approved.
# Empty URL/image keeps a clearly labelled sponsorship placeholder.
SPONSOR = {
    "name": "廣告合作招募中",
    "message": "台股 K 棒搜尋器｜歡迎洽詢贊助合作",
    "image_url": "",
    "landing_url": "",
}
AD_SECONDS = 40


@st.fragment(run_every="1s")
def search_ad_countdown():
    remaining = max(0, int(__import__("math").ceil(
        st.session_state["search_ad_deadline"] - monotonic())))
    if remaining <= 0:
        st.session_state.pop("search_ad_deadline", None)
        st.session_state["pending_search"] = True
        st.rerun(scope="app")
    st.caption("廣告 · 贊助內容")
    with st.container(border=True):
        st.subheader(SPONSOR["name"])
        if SPONSOR["image_url"]:
            st.image(SPONSOR["image_url"], use_container_width=True)
        st.write(SPONSOR["message"])
        if SPONSOR["landing_url"]:
            st.link_button("了解贊助內容", SPONSOR["landing_url"],
                           use_container_width=True)
    st.progress((AD_SECONDS - remaining) / AD_SECONDS,
                text=f"廣告展示剩餘 {remaining} 秒")
    st.caption("倒數結束後會自動開始搜尋，請保持網頁開啟。")


EXPORT_FIELDS = {
    'rank':'排名','code':'代號','name':'公司','industry':'產業類別','category':'搜尋類別','pattern':'型態','stage':'階段',
    'score':'評分','signal_date':'訊號日期','age':'距今交易日',
    'close':'最新收盤價','change_pct':'當日漲跌幅(%)','volume_lots':'當日成交量(張)',
    'avg_volume_lots':'20日均量(張)','avg_turnover_100m':'20日均成交金額(億元)',
    'volume_ratio':'訊號日量比','sma20':'20日均線','explanation':'判斷依據'
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


def render_chart(bars: pd.DataFrame, item, side: str):
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
        low=series.low, close=series.close, name='日 K 線',
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
    fig.update_layout(height=565,margin=dict(t=15,b=25,r=18,l=15),
       xaxis_rangeslider_visible=False,legend=dict(orientation='h',y=1.07,x=0),
       hovermode='x unified',plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(type='category',tickangle=-40,nticks=12,showgrid=False)
    fig.update_yaxes(title_text='價格（元）',row=1,col=1,gridcolor='rgba(128,128,128,.16)')
    fig.update_yaxes(title_text='成交（張）',row=2,col=1,gridcolor='rgba(128,128,128,.14)')
    st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})


def render_rankings(frame: pd.DataFrame, side: str, full_data: pd.DataFrame, effective_date: str):
    if frame.empty:
        st.info('這個日期與條件下，沒有符合定義的型態。可以延長訊號觀察天數或稍微放寬流動性門檻；不會以不符合的股票湊滿 25 名。')
        return
    shown = frame.rename(columns={
        'rank':'排名','code':'代號','name':'股票名稱','industry':'產業類別','category':'搜尋類別','pattern':'型態','stage':'階段',
        'score':'分數','signal_date':'訊號日期','age':'距今天數(交易日)',
        'close':'收盤價','change_pct':'漲跌幅 %',
        'volume_lots':'今日量(張)','avg_volume_lots':'20日均量(張)',
        'avg_turnover_100m':'20日均成交額(億)','volume_ratio':'量比'
    })
    # Keep ranking readable on a narrow screen. Extra fields remain available
    # in the detailed expandable table and the CSV download.
    st.dataframe(shown[['排名','代號','股票名稱','產業類別','型態','分數','收盤價']],
        hide_index=True,use_container_width=True,height=min(35*len(frame)+44,900),
        column_config={'分數':st.column_config.ProgressColumn(
            '分數',min_value=0,max_value=100,format='%d'),
            '收盤價':st.column_config.NumberColumn('收盤價',format='%.2f')})
    with st.expander('展開完整榜單（成交量、成交金額、日期等）'):
        st.dataframe(shown[['排名','代號','股票名稱','產業類別',
            '搜尋類別','型態','階段','分數','訊號日期','收盤價',
            '漲跌幅 %','今日量(張)','20日均量(張)','20日均成交額(億)','量比']],
            hide_index=True,use_container_width=True,
            column_config={'漲跌幅 %':st.column_config.NumberColumn(
                '漲跌幅 %',format='%.2f')})
    st.download_button('⬇ 下載此榜單 CSV（Excel 可開啟）',data=to_csv(frame),
        file_name=f'TWSE_{side}_{effective_date}.csv',mime='text/csv',key=f'download_{side}')
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
    st.markdown(f"**{rec['pattern']}**　｜　{rec['stage']}　｜　{rec['explanation']}")
    render_chart(full_data,rec,side)
    st.caption('標記位置為型態「辨識／確認」日期，不一定是實際高低點當天。紅 K 為上漲、綠 K 為下跌。52 日均線在資料不足時不顯示。')


# Replace the search form with a sponsor screen during the 40-second slot.
if "search_ad_deadline" in st.session_state:
    st.title('📊 台股 K 棒搜尋器')
    search_ad_countdown()
    st.stop()

st.title('📊 台股 K 棒搜尋器')
show_online_count()
st.caption('台灣上市普通股｜K 棒反轉＋MA5／10／20／52 固定搜尋｜頂底各 Top 25')

# Search controls belong in the page itself (not the sidebar) to avoid
# Safari's narrow overlay and oversized text clipping on iPhone.
with st.container(border=True):
    st.subheader('🔎 搜尋條件')
    c_date, c_volume = st.columns(2)
    with c_date:
        query_date = st.date_input('查詢日期', value=last_weekday(),
            min_value=today_taiwan()-timedelta(days=1400),
            max_value=today_taiwan(), format='YYYY/MM/DD')
    with c_volume:
        min_vol = st.number_input('20 日均量下限（張）', min_value=0,
            max_value=100000, value=1000, step=500)
    c_money, c_price = st.columns(2)
    with c_money:
        min_money = st.number_input('20 日均成交金額（億元）',
            min_value=0.0, max_value=500.0, value=1.0, step=.5, format='%.1f')
    with c_price:
        min_price = st.number_input('最低股價（元）',
            min_value=0.0, max_value=3000.0, value=10.0, step=5.0, format='%.1f')
    st.caption('固定搜尋：K 棒反轉＋均線突破／跌破（MA5、MA10、MA20、MA52）')
    include_wm = st.checkbox('另外搜尋 W 底／M 頭', value=False)
    targets = ['K 棒反轉', '均線突破／跌破']
    if include_wm:
        targets.append('W 底／M 頭')
    ma_windows = [5, 10, 20, 52]
    with st.expander('更多設定：訊號天數'):
        signal_days = st.select_slider('最近幾個交易日出現訊號',
            options=[1,2,3,4,5],value=3)
        history_days = st.slider('歷史回溯天數',120,250,165,5)
    run = st.button('🔍 開始搜尋上市股票',type='primary',use_container_width=True)
    st.caption('每次搜尋前展示 40 秒贊助版位；倒數完畢自動開始搜尋。')

if run:
    # Save immutable inputs before the form is temporarily hidden.
    st.session_state["queued_criteria"] = {
        "query_date": query_date, "min_vol": min_vol,
        "min_money": min_money, "min_price": min_price,
        "targets": list(targets), "ma_windows": list(ma_windows),
        "signal_days": signal_days, "history_days": history_days,
    }
    st.session_state["search_ad_deadline"] = monotonic() + AD_SECONDS
    st.rerun()

run = st.session_state.pop("pending_search", False)
if run:
    opts = st.session_state.pop("queued_criteria")
    query_date = opts["query_date"]
    min_vol, min_money, min_price = opts["min_vol"], opts["min_money"], opts["min_price"]
    targets, ma_windows = opts["targets"], opts["ma_windows"]
    signal_days, history_days = opts["signal_days"], opts["history_days"]
    bar = st.progress(0,text='準備搜尋…')
    details = st.empty()
    db = None
    try:
        db = open_db()
        with requests.Session() as session:
            details.info('確認 TWSE 官方上市公司名冊…')
            companies = load_companies(db,session)
            industry_map = dict(db.execute("SELECT code,industry FROM listed_companies"))
            def progress(i,total,msg):
                bar.progress(min(1.0,i/max(1,total)),text=msg)
            info = ensure_history(db,session,query_date,history_days,progress,throttle=.85)
        bar.progress(1.0,text='下載完成，正在分析 K 棒頂底型態…')
        bottom,top,diag,covered = screen(info['data'],companies,info['effective_date'],
            min_vol_lots=min_vol,min_turnover_100m=min_money,
            min_price=min_price,signal_days=signal_days,top_n=25,
            enabled_categories=targets,ma_windows=ma_windows)
        for ranking in (bottom, top):
            if not ranking.empty:
                ranking['industry'] = ranking['code'].map(industry_map).fillna('未分類')
        st.session_state['results'] = {
            'bottom':bottom,'top':top,'diag':diag,'info':{k:v for k,v in info.items() if k!='data'},
            'data':covered,
            'criteria':{'date':str(query_date),'min_vol':min_vol,'min_money':min_money,
                        'min_price':min_price,'signal_days':signal_days,
                        'targets':list(targets),'ma_windows':list(ma_windows)}
        }
        details.empty()
        bar.empty()
    except Exception as e:
        bar.empty()
        st.error(f'搜尋中斷：{e}')
        st.info('已成功下載的交易日會保存在本機快取。檢查網路、等待證交所恢復服務後，再按「開始手動搜尋」即可接續，不需全部重抓。')
    finally:
        if db:
            db.close()

if 'results' not in st.session_state:
    st.info('選擇日期與條件後，按「開始搜尋上市股票」即可查看底部、頂部各 Top 25。')
else:
    r = st.session_state['results']
    effective_date=r['info']['effective_date']
    if effective_date != r['criteria']['date']:
        st.warning(f"選擇 {r['criteria']['date']}，該日無完整盤後行情；實際採用 **{effective_date}** 交易日資料。")
    st.success(f"完成搜尋：資料截止 **{effective_date}**｜上市公司名冊 {r['diag']['listed']:,} 檔｜歷史日 K {r['info']['history_sessions']:,} 個交易日")
    a,b=st.columns(2)
    a.metric('可分析上市股票',f"{r['diag']['covered']:,} 檔")
    b.metric('符合流動性',f"{r['diag']['liquid']:,} 檔")
    c,d=st.columns(2)
    c.metric('底部榜單',f"{len(r['bottom'])} / 25 檔")
    d.metric('頂部榜單',f"{len(r['top'])} / 25 檔")
    st.caption(f"本次新下載／檢查交易日 {r['info']['downloaded_days']} 天。產業類別採證交所上市公司官方分類；未分類表示官方欄位缺漏或尚未取得。排行分數並非勝率。")
    tabs=st.tabs(['🟢 底部型態 Top 25','🔴 頂部型態 Top 25','ℹ️ 評分規則與注意事項'])
    with tabs[0]:
        render_rankings(r['bottom'],'bottom',r['data'],effective_date)
    with tabs[1]:
        render_rankings(r['top'],'top',r['data'],effective_date)
    with tabs[2]:
        st.markdown('''
**判斷型態**

- **底部：** 錘子線（長下影）、多頭吞噬、晨星三 K、跌破前低後收回、局部低點二 K 確認、W 底雛形／突破頸線、20 日區間向上突破。
- **頂部：** 流星線（長上影）、空頭吞噬、夜星三 K、突破前高後跌回、局部高點二 K 確認、M 頭雛形／跌破頸線、20 日區間向下跌破。
- **評分：** 型態基本分＋流動性＋訊號日量比＋均線位置／方向－訊號距今交易日的時間折減，最高 100 分。分數不能解讀為上漲或下跌的機率。
- **榜單：** 每檔最多入選一次，採用該檔最符合條件的訊號。只有符合條件才列入，可能不足 25 檔。

**務必留意**

1. 資料為**盤後**日 K，不是即時交易訊號。當天行情尚未公布時，會使用前一個可取得的交易日。
2. TWSE 官方 OHLC 為**未還原權息**價格。除權息、減資、分割等因素可能形成假性跳空與技術型態；請回到券商圖表人工核對。
3. W 底、M 頭與局部高低點等定義為可重現的幾何規則，並非市場公認的唯一標準。底部／頂部是**候選型態**，不代表一定見底或見頂。
4. 股票名冊以目前上市公司為準，搜尋較早歷史日不等於嚴格的歷史成分股回測；不適合直接用來計算真實策略勝率。
5. 流動性門檻是近 20 日的平均量與成交金額，不保證任何時刻都能以理想價格成交。
6. 歷史數據保存在程式資料夾的 `data/twse_cache.sqlite3`，不需上傳至第三方雲端平台。
        ''')
    st.caption('本工具僅供研究與選股輔助，不構成投資建議。')
