"""TWSE listed-company daily OHLCV downloader + persistent SQLite cache.

All quote rows come from official TWSE daily all-stock data. One request covers
an entire trading day; no per-stock requests are needed. Prices are *unadjusted*
TWSE reported closes (dividends/splits can affect chart patterns).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import re
import sqlite3
import time
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

TZ = ZoneInfo("Asia/Taipei")
API_DAILY = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
API_LISTED = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
DB_PATH = Path(__file__).resolve().parent / "data" / "twse_cache.sqlite3"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.twse.com.tw/",
}
COLUMNS = ("date", "code", "name", "open", "high", "low", "close", "shares", "turnover")


def today_taiwan() -> date:
    return datetime.now(TZ).date()


def open_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS bars(
        date TEXT NOT NULL, code TEXT NOT NULL, name TEXT,
        open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
        shares INTEGER NOT NULL, turnover REAL NOT NULL,
        PRIMARY KEY(date,code))""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_bars_code_date ON bars(code,date)")
    db.execute("""CREATE TABLE IF NOT EXISTS fetched_days(
        date TEXT PRIMARY KEY, status TEXT NOT NULL, fetched_at TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS listed_companies(
        code TEXT PRIMARY KEY, name TEXT NOT NULL)""")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)")
    db.commit()
    return db


def request_json(session: requests.Session, url: str, params: dict | None = None):
    err = None
    for n in range(3):
        try:
            resp = session.get(url, params=params, timeout=(10, 35), headers=HEADERS)
            resp.raise_for_status()
            obj = resp.json()
            if not isinstance(obj, (dict, list)):
                raise ValueError("資料格式非 JSON 物件或陣列")
            return obj
        except (requests.RequestException, ValueError) as exc:
            err = exc
            if n < 2:
                time.sleep((n + 1) * 2.0)
    raise RuntimeError(f"證交所資料下載失敗：{url}（{err}）") from err


def number(value):
    """TWSE fields contain commas, dashes, HTML fragments and non-numbers."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = re.sub(r"<[^>]+>", "", str(value)).strip().replace(",", "").replace("＋", "+")
    if s in ("", "--", "---", "－", "—", "N/A", "X", "除權", "除息"):
        return None
    try:
        x = float(s)
        return x if x == x and abs(x) < float("inf") else None
    except ValueError:
        return None


def _norm_field(x: str) -> str:
    return re.sub(r"[\s\u3000()（）]", "", str(x)).replace("股數", "股數")


def parse_daily(payload: dict, day: date) -> list[tuple]:
    """Locate stock table by header, never by fixed index (TWSE can change)."""
    if payload.get("stat") != "OK":
        stat = str(payload.get("stat", ""))
        if any(w in stat for w in ("沒有符合條件", "查無資料", "無資料")):
            return []
        raise RuntimeError(f"TWSE 回覆異常：{stat[:160]}")
    tables = payload.get("tables", [])
    if not isinstance(tables, list):
        tables = []
    # Some older endpoint formats expose numbered arrays (fields9/data9).
    for i in range(4, 20):
        if f"fields{i}" in payload and f"data{i}" in payload:
            tables.append({"fields": payload[f"fields{i}"], "data": payload[f"data{i}"]})
    for table in tables:
        fields = [_norm_field(f) for f in table.get("fields", [])]
        needed = ["證券代號", "證券名稱", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價"]
        if not all(f in fields for f in needed):
            continue
        pos = {f: fields.index(f) for f in needed}
        ans = []
        for row in table.get("data", []):
            if not isinstance(row, (tuple, list)) or len(row) <= max(pos.values()):
                continue
            code = str(row[pos["證券代號"]]).strip()
            # The listed-companies universe is verified against TWSE's master later.
            if not re.fullmatch(r"[1-9][0-9]{3}", code):
                continue
            vals = [number(row[pos[f]]) for f in ("開盤價", "最高價", "最低價", "收盤價", "成交股數", "成交金額")]
            if any(v is None for v in vals):
                continue
            o, h, l, c, shares, turnover = vals
            if min(o, h, l, c) <= 0 or l > min(o, c) or h < max(o, c) or shares <= 0 or turnover < 0:
                continue
            ans.append((day.isoformat(), code, str(row[pos["證券名稱"]]).strip(), o, h, l, c, int(shares), turnover))
        if len(ans) < 100:
            raise RuntimeError(f"{day} 個股資料僅解析到 {len(ans)} 檔，可能是證交所欄位改版，已停止防止錯誤選股。")
        return ans
    raise RuntimeError(f"{day} 找不到完整的個股行情欄位；請確認證交所 API 格式。")


def load_companies(db: sqlite3.Connection, session: requests.Session) -> dict[str, str]:
    now = datetime.now(TZ)
    stamp = db.execute("SELECT value FROM metadata WHERE key='companies_fetched_at'").fetchone()
    existing = dict(db.execute("SELECT code,name FROM listed_companies"))
    if existing and stamp:
        try:
            if (now - datetime.fromisoformat(stamp[0])).total_seconds() < 7 * 86400:
                return existing
        except ValueError:
            pass
    try:
        obj = request_json(session, API_LISTED)
        if not isinstance(obj, list):
            raise ValueError("上市公司名冊不是 JSON 陣列")
        companies = {}
        for item in obj:
            code = str(item.get("公司代號", "")).strip()
            name = str(item.get("公司簡稱") or item.get("公司名稱") or "").strip()
            if re.fullmatch(r"[1-9][0-9]{3}", code) and name:
                companies[code] = name
        if len(companies) < 800:
            raise ValueError(f"上市公司名冊只有 {len(companies)} 檔")
        with db:
            db.executemany("INSERT OR REPLACE INTO listed_companies(code,name) VALUES(?,?)", companies.items())
            db.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", ("companies_fetched_at", now.isoformat()))
        return companies
    except Exception as e:
        if existing:
            return existing
        raise RuntimeError(f"無法取得上市公司名冊。請檢查網路連線後重試。原始訊息：{e}") from e


def fetch_day(db: sqlite3.Connection, session: requests.Session, d: date, refresh_today: bool = False):
    """Fetch a whole TWSE market day. Return ('ok'|'closed'|'cached', count)."""
    existing = db.execute("SELECT status FROM fetched_days WHERE date=?", (d.isoformat(),)).fetchone()
    if existing and not (refresh_today and d == today_taiwan()):
        return "cached", 0
    payload = request_json(session, API_DAILY, {"date": d.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"})
    bars = parse_daily(payload, d)
    status = "ok" if bars else "closed"
    # Do not cache the current day as closed before publishing is finished.
    with db:
        db.execute("DELETE FROM bars WHERE date=?", (d.isoformat(),))
        if bars:
            db.executemany(f"INSERT OR REPLACE INTO bars({','.join(COLUMNS)}) VALUES({','.join(['?'] * len(COLUMNS))})", bars)
        if d != today_taiwan() or status == "ok":
            db.execute("INSERT OR REPLACE INTO fetched_days(date,status,fetched_at) VALUES(?,?,?)", (d.isoformat(), status, datetime.now(TZ).isoformat()))
    return status, len(bars)


def ensure_history(db: sqlite3.Connection, session: requests.Session, as_of: date,
                   history_calendar_days: int = 165,
                   progress: Callable[[int, int, str], None] | None = None,
                   throttle: float = 0.9) -> dict:
    if as_of > today_taiwan():
        raise ValueError("查詢日期不能超過今天（台灣時間）。")
    if history_calendar_days < 90 or history_calendar_days > 750:
        raise ValueError("歷史資料日數必須介於 90–750 天。")
    start = as_of - timedelta(days=history_calendar_days)
    existing = {r[0] for r in db.execute("SELECT date FROM fetched_days WHERE date BETWEEN ? AND ?", (start.isoformat(), as_of.isoformat()))}
    dates = [start + timedelta(days=i) for i in range((as_of - start).days + 1)]
    dates = [d for d in dates if d.weekday() < 5 and (d.isoformat() not in existing or d == today_taiwan())]
    fetched = 0
    sessions = 0
    for i, d in enumerate(dates, 1):
        if progress:
            progress(i - 1, len(dates), f"正在下載 {d} 的上市股票盤後資料…（{i}/{len(dates)}）")
        status, rows = fetch_day(db, session, d, refresh_today=True)
        if status == "ok":
            sessions += 1
        if status != "cached":
            fetched += 1
        if progress:
            progress(i, len(dates), f"完成 {d}，新增 {rows:,} 筆行情；已處理 {i}/{len(dates)} 天")
        if throttle and i < len(dates):
            time.sleep(throttle)
    query = "SELECT MAX(date) FROM bars WHERE date<=?"
    effective_date = db.execute(query, (as_of.isoformat(),)).fetchone()[0]
    if not effective_date:
        raise RuntimeError(f"截至 {as_of} 沒有任何有效上市股票交易資料，請改選較近交易日。")
    data = pd.read_sql_query(
        f"SELECT {','.join(COLUMNS)} FROM bars WHERE date BETWEEN ? AND ? ORDER BY code,date",
        db, params=(start.isoformat(), effective_date),
    )
    if data.empty:
        raise RuntimeError("找不到可分析的行情資料")
    return {"data": data, "effective_date": effective_date, "calendar_days": history_calendar_days,
            "downloaded_days": fetched, "new_trade_days": sessions,
            "history_sessions": int(data['date'].nunique()), "total_rows": len(data)}
