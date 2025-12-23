import json
import hashlib
import math
import calendar
import io
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf


# ======================
# 공통 설정
# ======================
TICKER_LIST = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY", "IWD", "GLD", "QQQ", "BIL"]

# ✅ 사용자가 직접 보유 입력하는 티커(=BIL 입력칸 제거)
INPUT_TICKERS = [t for t in TICKER_LIST if t != "BIL"]

# ✅ VAA 모멘텀 표기(7개) + 선택도 7개 중에서
VAA_UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"]

# ✅ VAA 룰 분기(리스크온/디펜시브)
VAA_RISK_ON = ["SPY", "EFA", "EEM", "AGG"]
VAA_DEFENSIVE = ["LQD", "IEF", "SHY"]

st.set_page_config(page_title="Rebalance (Private)", layout="wide")
st.title("PORTFOLIO")


# ======================
# 상단 모드 버튼 (사이드바 제거 버전)
# ======================
def _set_mode(m: str):
    st.session_state["mode"] = m


if "mode" not in st.session_state:
    st.session_state["mode"] = "Monthly"

mode = st.session_state["mode"]

# ✅ Backtest 버튼 추가 (Monthly/Annual 최대한 유지)
c_m, c_a, c_b, c_sp, c_r = st.columns([1.4, 1.4, 1.4, 4.8, 1.4])
with c_m:
    st.button(
        "Monthly",
        type="primary" if mode == "Monthly" else "secondary",
        use_container_width=True,
        on_click=_set_mode,
        args=("Monthly",),
    )
with c_a:
    st.button(
        "Annual",
        type="primary" if mode == "Annual" else "secondary",
        use_container_width=True,
        on_click=_set_mode,
        args=("Annual",),
    )
with c_b:
    st.button(
        "Backtest",
        type="primary" if mode == "Backtest" else "secondary",
        use_container_width=True,
        on_click=_set_mode,
        args=("Backtest",),
    )
with c_r:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        # 실행 결과/편집 상태도 같이 초기화
        for k in list(st.session_state.keys()):
            if k.startswith(
                (
                    "annual_result",
                    "monthly_result",
                    "exec_annual_",
                    "exec_monthly_",
                    "monthly_file_sig",
                    "backtest_result",
                )
            ):
                del st.session_state[k]
        st.rerun()

st.divider()


# ======================
# 숫자 입력(콤마 표기) 유틸
# ======================
def parse_money(text: str, allow_decimal: bool) -> float:
    if text is None:
        return 0.0
    s = str(text).strip()
    if s == "":
        return 0.0
    s = s.replace(",", "").replace("₩", "").replace("$", "").replace(" ", "")
    v = float(s)
    return float(v) if allow_decimal else float(int(v))


def format_money(value: float, allow_decimal: bool) -> str:
    if allow_decimal:
        s = f"{float(value):,.2f}"
        s = s.rstrip("0").rstrip(".")
        return s
    return f"{int(value):,}"


def money_input(
    label: str,
    key: str,
    default: float = 0.0,
    allow_decimal: bool = False,
    help_text: str = "",
) -> float:
    if key not in st.session_state:
        st.session_state[key] = format_money(default, allow_decimal)

    def _fmt():
        try:
            v = parse_money(st.session_state.get(key, ""), allow_decimal)
            st.session_state[key] = format_money(v, allow_decimal)
        except Exception:
            pass

    st.text_input(label, key=key, help=help_text, on_change=_fmt)

    try:
        return parse_money(st.session_state.get(key, ""), allow_decimal)
    except Exception:
        st.error(f"'{label}' 숫자 입력이 이상해. 예: 1,000,000 / 1000 / 1,000.50")
        st.stop()


# ✅ Backtest 전용(영문 에러) - 기존 money_input은 건드리지 않음
def money_input_en(
    label: str,
    key: str,
    default: float = 0.0,
    allow_decimal: bool = False,
) -> float:
    if key not in st.session_state:
        st.session_state[key] = format_money(default, allow_decimal)

    def _fmt():
        try:
            v = parse_money(st.session_state.get(key, ""), allow_decimal)
            st.session_state[key] = format_money(v, allow_decimal)
        except Exception:
            pass

    st.text_input(label, key=key, on_change=_fmt)

    try:
        return parse_money(st.session_state.get(key, ""), allow_decimal)
    except Exception:
        st.error(f"Invalid number for '{label}'. Examples: 1,000,000 / 1000 / 1,000.50")
        st.stop()


# ======================
# Robust FRED CSV loader (BOM/whitespace/HTML guard)
# ======================
def _fred_csv(url: str) -> pd.DataFrame:
    """
    Robust CSV loader for FRED.
    - Strips BOM/whitespace in headers
    - Detects HTML response (blocked) early
    """
    text = None
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
        text = raw.decode("utf-8", errors="ignore")
        if "<html" in (text or "").lower():
            raise RuntimeError("FRED returned HTML instead of CSV.")
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        df = pd.read_csv(url)

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip().lstrip("\ufeff").lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    for c in df.columns:
        cl = str(c).lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    return None


# ======================
# yfinance / FRED (캐시)
# ======================
@st.cache_data(ttl=900, show_spinner=False)
def _download_hist_one(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} 가격 데이터를 못 가져옴 (history empty).")

    df = df.copy()

    # ✅ MultiIndex 방어
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    # tz 제거(있으면)
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    return df


@st.cache_data(ttl=300, show_spinner=False)
def last_adj_close(ticker: str) -> float:
    df = yf.download(ticker, period="7d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} 최근 가격을 못 가져옴 (history empty).")

    df = df.copy()

    # ✅ MultiIndex 방어
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    v = df[col].iloc[-1]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return float(v)


@st.cache_data(ttl=300, show_spinner=False)
def fx_usdkrw() -> float:
    ticker = "USDKRW=X"
    df = yf.download(ticker, period="7d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("USDKRW=X 환율을 못 가져옴 (history empty).")

    df = df.copy()

    # ✅ MultiIndex 방어
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(0)

    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    v = df[col].iloc[-1]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return float(v)


@st.cache_data(ttl=3600, show_spinner=False)
def _unrate_info(today: datetime):
    """
    ✅ FRED(UNRATE) robust fetch (BOM/whitespace/HTML guard)
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    df = _fred_csv(url)

    date_col = _pick_col(df, ["DATE", "observation_date"])
    val_col = _pick_col(df, ["UNRATE"])
    if not date_col or not val_col:
        raise RuntimeError("UNRATE columns not found")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col, val_col])

    start = today - timedelta(days=400)
    df = df[(df[date_col] >= start) & (df[date_col] <= today)].copy()
    if df.empty:
        raise RuntimeError("UNRATE 데이터가 비어있음")

    unrate_now = float(df[val_col].iloc[-1])
    unrate_ma = float(df[val_col].tail(12).mean())  # 최근 12개월 평균
    return unrate_now, unrate_ma


def price_asof_or_before(df: pd.DataFrame, dt: datetime) -> float:
    s = df.loc[df.index <= dt, "Adj Close"]
    if s.empty:
        return float(df["Adj Close"].iloc[0])
    v = s.iloc[-1]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return float(v)


# ======================
# 전략 로직
# ======================
def momentum_score(t: str, prices: dict, d_1m, d_3m, d_6m, d_12m) -> float:
    """가중합 모멘텀 (Adj Close): 1m*12 + 3m*4 + 6m*2 + 12m*1"""
    try:
        hist = _download_hist_one(t, period="2y")
        p = float(prices[t])

        p1 = price_asof_or_before(hist, d_1m)
        p3 = price_asof_or_before(hist, d_3m)
        p6 = price_asof_or_before(hist, d_6m)
        p12 = price_asof_or_before(hist, d_12m)

        return (p / p1 - 1) * 12 + (p / p3 - 1) * 4 + (p / p6 - 1) * 2 + (p / p12 - 1) * 1
    except Exception:
        return -9999


def strategy_value_holdings(holdings: dict, price_map: dict) -> float:
    total = 0.0
    for t, q in holdings.items():
        if t not in price_map:
            continue
        total += float(price_map[t]) * int(q)
    return float(total)


def buy_all_in(asset: str, budget_usd: float, prices: dict):
    p = float(prices[asset])
    q = int(float(budget_usd) // p)
    cash = float(float(budget_usd) - q * p)
    return {asset: q}, cash


def buy_all_in_if_affordable(asset: str, budget_usd: float, prices: dict):
    p = float(prices[asset])
    if float(budget_usd) < p:
        return {}, float(budget_usd)
    return buy_all_in(asset, budget_usd, prices)


def buy_equal_split_round(assets: list, budget_usd: float, prices: dict):
    n = len(assets)
    each = float(budget_usd) / n
    holdings = {a: 0 for a in assets}
    spent = 0.0

    for a in assets:
        p = float(prices[a])
        q = int(each // p)
        holdings[a] += q
        spent += q * p

    cash = float(float(budget_usd) - spent)
    return holdings, cash


def buy_equal_split_min_cash(assets: list, budget_usd: float, prices: dict):
    total_hold = {a: 0 for a in assets}
    cash = float(budget_usd)

    while True:
        round_hold, new_cash = buy_equal_split_round(assets, cash, prices)

        bought_any = any(q > 0 for q in round_hold.values())
        for a, q in round_hold.items():
            total_hold[a] += int(q)

        cash = float(new_cash)

        if not bought_any:
            break
        if cash < min(float(prices[a]) for a in assets):
            break

    cheapest = min(assets, key=lambda a: float(prices[a]))
    if cash >= float(prices[cheapest]):
        q = int(cash // float(prices[cheapest]))
        total_hold[cheapest] += q
        cash = float(cash - q * float(prices[cheapest]))

    total_hold = {a: int(q) for a, q in total_hold.items() if int(q) != 0}
    return total_hold, float(cash)


def safe_laa_asset(today: datetime, prices: dict) -> str:
    try:
        unrate_now, unrate_ma = _unrate_info(today)
    except Exception:
        return "QQQ"

    spy_hist = _download_hist_one("SPY", period="2y")
    spy_200ma = spy_hist["Adj Close"].rolling(200).mean().iloc[-1]
    if spy_200ma != spy_200ma:
        return "QQQ"

    return "SHY" if (unrate_now > unrate_ma and float(prices["SPY"]) < float(spy_200ma)) else "QQQ"


def odm_choice(prices: dict, d_12m: datetime) -> str:
    def r12(t):
        hist = _download_hist_one(t, period="2y")
        p0 = price_asof_or_before(hist, d_12m)
        return float(prices[t]) / float(p0) - 1

    bil_r = r12("BIL")
    spy_r = r12("SPY")
    efa_r = r12("EFA")

    if bil_r > spy_r:
        return "AGG"
    return "SPY" if spy_r >= efa_r else "EFA"


def merge_holdings(*holding_dicts):
    merged = {}
    for h in holding_dicts:
        for t, q in h.items():
            merged[t] = merged.get(t, 0) + int(q)
    return merged


def vaa_scores_df(vaa: dict) -> pd.DataFrame:
    scores = vaa.get("scores", {})
    rows = [{"Ticker": t, "Momentum Score": float(scores.get(t, -9999))} for t in VAA_UNIVERSE]
    df = pd.DataFrame(rows).sort_values("Momentum Score", ascending=False, ignore_index=True)
    return df


def pick_vaa_asset(scores: dict) -> str:
    """
    ✅ 네가 말한 VAA 룰:
    - SPY/EFA/EEM/AGG 모멘텀스코어가 모두 > 0 이면: 그 4개 중 1등 올인
    - 아니면: LQD/IEF/SHY 중 1등 올인
    """
    risk_ok = all(float(scores.get(t, -9999)) > 0 for t in VAA_RISK_ON)
    universe = VAA_RISK_ON if risk_ok else VAA_DEFENSIVE
    return max(universe, key=lambda t: float(scores.get(t, -9999)))


# ======================
# 결과 표시(UI 정리 버전)
# ======================
def show_result(result: dict, current_holdings: dict, layout: str = "side"):
    rate = float(result["meta"]["usdkrw_rate"])
    price_map = result["meta"]["prices_adj_close"]

    vaa = result["VAA"]
    laa = result["LAA"]
    odm = result["ODM"]

    vaa_h = vaa["holdings"]
    laa_h = laa["holdings"]
    odm_h = odm["holdings"]

    vaa_cash = float(vaa.get("cash_usd", 0.0))
    laa_cash = float(laa.get("cash_usd", 0.0))
    odm_cash = float(odm.get("cash_usd", 0.0))

    def holdings_value_usd(h):
        return sum(float(price_map[t]) * int(q) for t, q in h.items() if t in price_map)

    total_holdings_usd = holdings_value_usd(vaa_h) + holdings_value_usd(laa_h) + holdings_value_usd(odm_h)
    total_cash_usd = vaa_cash + laa_cash + odm_cash
    total_usd = total_holdings_usd + total_cash_usd

    total_krw = total_usd * rate
    cash_krw = total_cash_usd * rate

    a, b, c = st.columns(3)
    a.metric("총자산(₩)", f"₩{total_krw:,.0f}")
    b.metric("현금(₩)", f"₩{cash_krw:,.0f}")
    c.metric("달러환율(₩/$)", f"₩{rate:,.2f}")

    all_target = merge_holdings(vaa_h, laa_h, odm_h)

    def render_target_clean():
        st.subheader("목표 보유자산")
        items = [(t, int(q)) for t, q in all_target.items() if int(q) != 0 and t != "BIL"]
        items.sort(key=lambda x: x[0])

        if not items:
            st.write("-")
            return

        cols = st.columns(5)
        for i, (t, q) in enumerate(items):
            with cols[i % 5]:
                st.metric(t, f"{q}주")

    def render_trades_clean():
        st.subheader("매도/매수")
        rows = []
        for t in sorted(set(current_holdings.keys()) | set(all_target.keys())):
            if t == "BIL":
                continue
            cur = int(current_holdings.get(t, 0))
            tar = int(all_target.get(t, 0))
            delta = tar - cur
            if delta != 0:
                rows.append((t, delta))

        if not rows:
            st.write("-")
            return

        rows.sort(key=lambda x: (abs(x[1]), x[0]), reverse=True)

        sells = [(t, -d) for t, d in rows if d < 0]
        buys = [(t, d) for t, d in rows if d > 0]

        left, right = st.columns(2)
        with left:
            st.markdown("**매도**")
            if not sells:
                st.write("-")
            else:
                for t, q in sells:
                    st.write(f"{t} {q}주 매도")
        with right:
            st.markdown("**매입**")
            if not buys:
                st.write("-")
            else:
                for t, q in buys:
                    st.write(f"{t} {q}주 매입")

    def render_scores_bar():
        st.subheader("모멘텀스코어")
        df = vaa_scores_df(vaa)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Ticker:N", sort=alt.SortField(field="Momentum Score", order="descending")),
                y=alt.Y("Momentum Score:Q"),
                tooltip=["Ticker:N", alt.Tooltip("Momentum Score:Q", format=".4f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

    if layout == "side":
        left, right = st.columns([2, 1], gap="large")
        with right:
            render_scores_bar()
        with left:
            render_target_clean()
            render_trades_clean()
    else:
        render_scores_bar()
        render_target_clean()
        render_trades_clean()


# ======================
# 실행본 편집 + 저장(JSON: ETF만)
# ======================
def _clear_keys_with_prefix(prefix: str):
    for k in list(st.session_state.keys()):
        if k.startswith(prefix):
            del st.session_state[k]


def export_holdings_only(executed: dict, timestamp: str) -> dict:
    """
    ✅ 저장 JSON에는 현금 제외, ETF holdings만 저장
    executed = {"VAA":{...}, "LAA":{...}, "ODM":{...}} (각각 holdings dict)
    """
    payload = {
        "timestamp": timestamp,
        "schema_version": "holdings_only_v1",
        "VAA": {"holdings": {t: int(q) for t, q in executed["VAA"]["holdings"].items() if int(q) != 0}},
        "LAA": {"holdings": {t: int(q) for t, q in executed["LAA"]["holdings"].items() if int(q) != 0}},
        "ODM": {"holdings": {t: int(q) for t, q in executed["ODM"]["holdings"].items() if int(q) != 0}},
    }
    return payload


def render_execution_editor(result: dict, editor_prefix: str):
    st.subheader("실제 보유자산")

    executed = {"VAA": {"holdings": {}}, "LAA": {"holdings": {}}, "ODM": {"holdings": {}}}

    for strat in ["VAA", "LAA", "ODM"]:
        rec = result[strat]["holdings"]

        with st.expander(strat, expanded=(strat == "VAA")):
            cols = st.columns(5)
            for i, t in enumerate(INPUT_TICKERS):
                default_q = int(rec.get(t, 0))
                key = f"{editor_prefix}{strat}_{t}"
                with cols[i % 5]:
                    q = st.number_input(t, min_value=0, value=default_q, step=1, key=key)

                if int(q) != 0:
                    executed[strat]["holdings"][t] = int(q)

    return executed


# ======================
# 날짜 기준
# ======================
today = datetime.today()
today_naive = today.replace(tzinfo=None)
d_1m, d_3m, d_6m, d_12m = [today_naive - timedelta(days=d) for d in [30, 90, 180, 365]]


# ======================
# 시장 데이터 로드
# ======================
with st.spinner("가격/환율 불러오는 중..."):
    prices = {t: last_adj_close(t) for t in TICKER_LIST}
    usdkrw_rate = fx_usdkrw()


# ======================
# 실행 함수
# ======================
def compute_vaa_scores(prices: dict) -> dict:
    return {t: float(momentum_score(t, prices, d_1m, d_3m, d_6m, d_12m)) for t in VAA_UNIVERSE}


def run_year(amounts: dict, cash_usd: float):
    total_usd = sum(float(amounts.get(t, 0.0)) * float(prices[t]) for t in INPUT_TICKERS) + float(cash_usd)
    budget = float(total_usd) / 3.0

    scores = compute_vaa_scores(prices)
    best_vaa = pick_vaa_asset(scores)  # ✅ FIXED
    vaa_hold, vaa_cash_usd = buy_all_in_if_affordable(best_vaa, budget, prices)

    laa_safe = safe_laa_asset(today, prices)
    laa_assets = ["IWD", "IEF", "GLD", laa_safe]
    laa_hold, laa_cash_usd = buy_equal_split_min_cash(laa_assets, budget, prices)

    odm_asset = odm_choice(prices, d_12m)
    odm_hold, odm_cash_usd = buy_all_in_if_affordable(odm_asset, budget, prices)

    return {
        "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "usdkrw_rate": float(usdkrw_rate),
            "prices_adj_close": {t: float(prices[t]) for t in TICKER_LIST},
            "input_cash_usd": float(cash_usd),
            "cash_rule": "Annual: total_usd includes input cash; budget split 1/3 per strategy. (Cash is NOT saved to JSON.)",
        },
        "VAA": {"holdings": vaa_hold, "cash_usd": float(vaa_cash_usd), "picked": best_vaa, "scores": scores},
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_safe},
        "ODM": {"holdings": odm_hold, "cash_usd": float(odm_cash_usd), "picked": odm_asset},
    }


def run_month(prev: dict, cash_usd: float):
    cash_each = float(cash_usd) / 3.0

    vaa_prev_hold = prev["VAA"]["holdings"]
    laa_prev_hold = prev["LAA"]["holdings"]
    odm_prev_hold = prev["ODM"]["holdings"]

    vaa_budget = strategy_value_holdings(vaa_prev_hold, prices) + cash_each
    laa_budget = strategy_value_holdings(laa_prev_hold, prices) + cash_each
    odm_budget = strategy_value_holdings(odm_prev_hold, prices) + cash_each

    scores = compute_vaa_scores(prices)
    best_vaa = pick_vaa_asset(scores)  # ✅ FIXED
    vaa_hold, vaa_cash_usd = buy_all_in_if_affordable(best_vaa, vaa_budget, prices)

    laa_safe = safe_laa_asset(today, prices)
    laa_assets = ["IWD", "IEF", "GLD", laa_safe]
    laa_hold, laa_cash_usd = buy_equal_split_min_cash(laa_assets, laa_budget, prices)

    odm_asset = odm_choice(prices, d_12m)
    odm_hold, odm_cash_usd = buy_all_in_if_affordable(odm_asset, odm_budget, prices)

    return {
        "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "usdkrw_rate": float(usdkrw_rate),
            "prices_adj_close": {t: float(prices[t]) for t in TICKER_LIST},
            "input_cash_usd": float(cash_usd),
            "cash_rule": "Monthly: prev holdings value + cash_each(=input cash/3). (Cash is NOT saved to JSON.)",
        },
        "VAA": {"holdings": vaa_hold, "cash_usd": float(vaa_cash_usd), "picked": best_vaa, "scores": scores},
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_safe},
        "ODM": {"holdings": odm_hold, "cash_usd": float(odm_cash_usd), "picked": odm_asset},
    }


# ======================
# Backtest helpers (UPDATED)
# ======================
BT_FX_TICKER = "USDKRW=X"

BT_BENCHMARKS = {
    "Nasdaq 100 (QQQ) [USD]": ("QQQ", "USD"),
    "S&P 500 (SPY) [USD]": ("SPY", "USD"),
    "KOSPI (^KS11) [KRW]": ("^KS11", "KRW"),
    "KOSDAQ (^KQ11) [KRW]": ("^KQ11", "KRW"),
}
BT_ALL_BENCH_LABEL = "All benchmarks"


def bt_parse_start_month(s: str) -> datetime:
    s = (s or "").strip()
    if len(s) != 7 or s[4] != "-":
        raise ValueError("Start month must be YYYY-MM.")
    y = int(s[:4])
    m = int(s[5:7])
    return datetime(y, m, 1)


def bt_parse_end_date(s: str) -> datetime:
    s = (s or "").strip()
    if len(s) == 7 and s[4] == "-":  # YYYY-MM
        y = int(s[:4])
        m = int(s[5:7])
        last_day = calendar.monthrange(y, m)[1]
        return datetime(y, m, last_day)
    return datetime.strptime(s, "%Y-%m-%d")


def bt_next_trading_day(nominal: datetime, trading_index: pd.DatetimeIndex) -> pd.Timestamp:
    ts = pd.Timestamp(nominal.date())
    i = trading_index.searchsorted(ts, side="left")
    if i >= len(trading_index):
        raise RuntimeError(f"Could not find next trading day for {nominal.date()}.")
    return trading_index[i]


def bt_last_trading_day_on_or_before(d: datetime, trading_index: pd.DatetimeIndex) -> pd.Timestamp:
    ts = pd.Timestamp(d.date())
    i = trading_index.searchsorted(ts, side="right") - 1
    if i < 0:
        raise RuntimeError(f"Could not find previous trading day for {d.date()}.")
    return trading_index[i]


# ✅ 변경: 리밸런싱 "일자" 선택 가능 (없는 날짜면 말일로 보정)
def bt_month_day_schedule(start_month: datetime, end_date: datetime, rebalance_day: int):
    cur = start_month.replace(day=1)
    while True:
        y, m = cur.year, cur.month
        last_day = calendar.monthrange(y, m)[1]
        day = min(int(rebalance_day), int(last_day))
        dt = datetime(y, m, day)
        if dt > end_date:
            break
        yield dt
        cur = (pd.Timestamp(cur) + pd.DateOffset(months=1)).to_pydatetime()


def bt_is_annual_rebalance(month_idx: int) -> bool:
    # ✅ 최초 리밸런싱(month_idx=0)부터 12개월 단위로 "전체 포트" 리밸런싱
    return (month_idx % 12) == 0


@st.cache_data(ttl=3600, show_spinner=False)
def bt_download_adj_close(tickers: tuple, start_str: str, end_str: str) -> pd.DataFrame:
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    df = yf.download(
        list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance download returned empty data.")

    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = set(df.columns.get_level_values(0))
        lv1 = set(df.columns.get_level_values(1))

        # case A: (TICKER, FIELD)
        if set(tickers).issubset(lv0):
            for t in tickers:
                if (t, "Adj Close") in df.columns:
                    out[t] = df[(t, "Adj Close")]
                elif (t, "Close") in df.columns:
                    out[t] = df[(t, "Close")]
        # case B: (FIELD, TICKER)
        elif set(tickers).issubset(lv1):
            for t in tickers:
                try:
                    sub = df.xs(t, axis=1, level=1, drop_level=True)
                    if "Adj Close" in sub.columns:
                        out[t] = sub["Adj Close"]
                    elif "Close" in sub.columns:
                        out[t] = sub["Close"]
                except Exception:
                    pass
        else:
            # fallback: try best-effort extraction
            for t in tickers:
                try:
                    sub = df.xs(t, axis=1, level=0, drop_level=True)
                    if "Adj Close" in sub.columns:
                        out[t] = sub["Adj Close"]
                    elif "Close" in sub.columns:
                        out[t] = sub["Close"]
                except Exception:
                    continue
    else:
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        out[list(tickers)[0]] = df[col]

    adj = pd.DataFrame(out)
    adj.index = pd.to_datetime(adj.index)
    try:
        if getattr(adj.index, "tz", None) is not None:
            adj.index = adj.index.tz_localize(None)
    except Exception:
        pass

    return adj.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def bt_unrate_series(start_str: str, end_str: str) -> pd.Series:
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    df = _fred_csv(url)

    date_col = _pick_col(df, ["DATE", "observation_date"])
    val_col = _pick_col(df, ["UNRATE"])

    # 못 찾으면 빈 시리즈 반환 (LAA safe는 자동으로 QQQ로 fallback)
    if not date_col or not val_col:
        return pd.Series(dtype=float)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col, val_col])

    df = df[(df[date_col] >= (start - timedelta(days=30))) & (df[date_col] <= (end + timedelta(days=30)))].copy()
    s = df.set_index(date_col)[val_col]
    s.index = pd.to_datetime(s.index)
    try:
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
    except Exception:
        pass
    return s.sort_index()


def bt_px_on_or_before(prices_ff: pd.DataFrame, asof: pd.Timestamp, t: str) -> float:
    if t not in prices_ff.columns:
        return float("nan")
    if asof not in prices_ff.index:
        idx = prices_ff.index.searchsorted(asof, side="right") - 1
        if idx < 0:
            return float("nan")
        asof = prices_ff.index[idx]
    return float(prices_ff.at[asof, t])


def bt_momentum_score(asof: pd.Timestamp, prices_ff: pd.DataFrame, t: str) -> float:
    try:
        p = bt_px_on_or_before(prices_ff, asof, t)

        def p_at(days_back: int) -> float:
            target = asof - pd.Timedelta(days=days_back)
            idx = prices_ff.index.searchsorted(target, side="right") - 1
            if idx < 0:
                return float("nan")
            dt = prices_ff.index[idx]
            return bt_px_on_or_before(prices_ff, dt, t)

        p1 = p_at(30)
        p3 = p_at(90)
        p6 = p_at(180)
        p12 = p_at(365)

        if any(math.isnan(x) or x <= 0 for x in [p, p1, p3, p6, p12]):
            return -9999.0

        return (p / p1 - 1) * 12 + (p / p3 - 1) * 4 + (p / p6 - 1) * 2 + (p / p12 - 1) * 1
    except Exception:
        return -9999.0


def bt_buy_all_in_if_affordable(asset: str, budget_usd: float, price: float):
    if budget_usd < price or price <= 0 or math.isnan(price):
        return {}, float(budget_usd)
    q = int(budget_usd // price)
    cash = float(budget_usd - q * price)
    return {asset: q}, cash


def bt_buy_equal_split_min_cash(assets: list, budget_usd: float, px_map: dict):
    total_hold = {a: 0 for a in assets}
    cash = float(budget_usd)

    def one_round(cash_in: float):
        n = len(assets)
        each = cash_in / n
        hold = {a: 0 for a in assets}
        spent = 0.0
        for a in assets:
            p = float(px_map[a])
            q = int(each // p) if p > 0 else 0
            hold[a] += q
            spent += q * p
        return hold, float(cash_in - spent)

    while True:
        rh, new_cash = one_round(cash)
        bought_any = any(q > 0 for q in rh.values())
        for a, q in rh.items():
            total_hold[a] += q
        cash = float(new_cash)

        if not bought_any:
            break
        if cash < min(float(px_map[a]) for a in assets):
            break

    cheapest = min(assets, key=lambda a: float(px_map[a]))
    if cash >= float(px_map[cheapest]):
        q = int(cash // float(px_map[cheapest]))
        total_hold[cheapest] += q
        cash = float(cash - q * float(px_map[cheapest]))

    total_hold = {a: int(q) for a, q in total_hold.items() if int(q) != 0}
    return total_hold, float(cash)


def bt_safe_laa_asset(asof: pd.Timestamp, prices_ff: pd.DataFrame, unrate: pd.Series):
    try:
        if unrate is None or unrate.empty:
            return "QQQ"

        unrate_now = float(unrate.loc[:asof].iloc[-1])
        tail12 = unrate.loc[:asof].tail(12)
        if len(tail12) < 12:
            return "QQQ"
        unrate_ma = float(tail12.mean())

        spy_series = prices_ff["SPY"].dropna()
        spy_tail = spy_series.loc[:asof].tail(200)
        if len(spy_tail) < 200:
            return "QQQ"

        spy_200ma = float(spy_tail.mean())
        spy_px = float(spy_series.loc[:asof].iloc[-1])

        return "SHY" if (unrate_now > unrate_ma and spy_px < spy_200ma) else "QQQ"
    except Exception:
        return "QQQ"


def bt_odm_choice(asof: pd.Timestamp, prices_ff: pd.DataFrame):
    def r12(t: str) -> float:
        p = bt_px_on_or_before(prices_ff, asof, t)
        target = asof - pd.Timedelta(days=365)
        idx = prices_ff.index.searchsorted(target, side="right") - 1
        if idx < 0:
            return -9999.0
        dt = prices_ff.index[idx]
        p0 = bt_px_on_or_before(prices_ff, dt, t)
        if p0 <= 0 or math.isnan(p0) or math.isnan(p):
            return -9999.0
        return p / p0 - 1

    bil_r = r12("BIL")
    spy_r = r12("SPY")
    efa_r = r12("EFA")

    if bil_r > spy_r:
        return "AGG"
    return "SPY" if spy_r >= efa_r else "EFA"


def bt_compute_twr_cagr(port_events: list, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if len(port_events) < 2:
        return 0.0

    first_post = port_events[0][2]
    if first_post is None or first_post <= 0:
        return 0.0

    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return 0.0

    twr = 1.0
    prev_post = first_post

    for i in range(1, len(port_events)):
        pre_i = port_events[i][1]
        if pre_i is None or prev_post <= 0:
            continue
        twr *= (pre_i / prev_post)

        post_i = port_events[i][2]
        if post_i is not None:
            prev_post = post_i

    return (twr ** (1 / years) - 1) * 100


# ✅ UPDATED: 벤치마크 정수 매수 고정 + 성과 통화(USD/KRW) 변환 지원
def bt_simulate_benchmark_events(
    ticker: str,
    native_ccy: str,      # "USD" or "KRW"  (해당 벤치마크 자산의 거래 통화)
    out_ccy: str,         # "USD" or "KRW"  (성과 계산/표시 기준 통화)
    rebalance_dates,
    eval_asof,
    bench_ff: pd.DataFrame,
    fx_series: pd.Series,
    initial_total_usd: float,
    monthly_add_krw: float,
    monthly_add_usd: float,
    fractional: bool = False,  # ✅ 기본 False (정수 매수)
):
    shares = 0.0
    cash_native = 0.0  # native_ccy 기준 현금
    events = []

    def value_out(asof: pd.Timestamp) -> float:
        fx_asof = float(fx_series.loc[:asof].iloc[-1])
        price = bt_px_on_or_before(bench_ff, asof, ticker)
        v_native = float(shares * price + cash_native)

        if out_ccy == native_ccy:
            return v_native

        # native -> out 변환
        if native_ccy == "KRW" and out_ccy == "USD":
            return float(v_native / fx_asof)
        if native_ccy == "USD" and out_ccy == "KRW":
            return float(v_native * fx_asof)

        return v_native

    for m_idx, asof in enumerate(rebalance_dates):
        fx_asof = float(fx_series.loc[:asof].iloc[-1])
        price = bt_px_on_or_before(bench_ff, asof, ticker)
        if math.isnan(price) or price <= 0:
            raise RuntimeError(f"No benchmark price for {ticker} at {asof.date()}.")

        pre = value_out(asof)

        # ✅ 투자금(USD 환산) -> native 통화로 변환해 투입
        if m_idx == 0:
            add_usd = float(initial_total_usd)
        else:
            add_usd = float(monthly_add_usd + (monthly_add_krw / fx_asof))

        add_native = add_usd if native_ccy == "USD" else add_usd * fx_asof

        if fractional:
            # (이번 요구사항에선 안 씀)
            shares += add_native / price
        else:
            # ✅ 정수 매수
            cash_native += add_native
            q = int(cash_native // price)
            shares += q
            cash_native -= q * price

        post = value_out(asof)
        events.append((asof, float(pre), float(post)))

    end_value = value_out(eval_asof)
    events.append((eval_asof, float(end_value), None))
    return events


# ✅ UPDATED: USD/KRW 성과 둘 다 계산해서 반환 + 벤치마크 정수 매수 고정
def bt_run_backtest(
    start_month_ym: str,
    end_date_in: str,
    initial_krw: float,
    initial_usd: float,
    monthly_add_krw: float,
    monthly_add_usd: float,
    bench_label: str,
    rebalance_day: int,
):
    start_month = bt_parse_start_month(start_month_ym)
    end_date = bt_parse_end_date(end_date_in)

    data_start = start_month - timedelta(days=900)
    data_end = end_date + timedelta(days=10)

    # ✅ All benchmarks 지원
    if bench_label == BT_ALL_BENCH_LABEL:
        bench_items = list(BT_BENCHMARKS.items())
        bench_tickers = [v[0] for _, v in bench_items]
    else:
        if bench_label not in BT_BENCHMARKS:
            raise RuntimeError("Unknown benchmark label.")
        bench_items = [(bench_label, BT_BENCHMARKS[bench_label])]
        bench_tickers = [BT_BENCHMARKS[bench_label][0]]

    all_tickers = sorted(set(TICKER_LIST + [BT_FX_TICKER] + bench_tickers))

    prices_all = bt_download_adj_close(
        tickers=tuple(all_tickers),
        start_str=data_start.strftime("%Y-%m-%d"),
        end_str=data_end.strftime("%Y-%m-%d"),
    )

    if BT_FX_TICKER not in prices_all.columns:
        raise RuntimeError("Missing USDKRW=X data.")
    fx = prices_all[BT_FX_TICKER].dropna()

    prices_raw = prices_all[[t for t in TICKER_LIST if t in prices_all.columns]].copy()
    if prices_raw.empty:
        raise RuntimeError("ETF price data is empty.")

    cal_ticker = "SPY" if "SPY" in prices_raw.columns else prices_raw.columns[0]
    trading_index = prices_raw[cal_ticker].dropna().index
    prices_ff = prices_raw.reindex(trading_index).ffill()

    # ✅ 벤치마크 가격도 같은 trading_index로 정렬/ffill
    bench_ff_map = {}
    for label, (ticker, _native_ccy) in bench_items:
        if ticker not in prices_all.columns:
            raise RuntimeError(f"Missing benchmark price data: {ticker}")
        bench_raw = prices_all[[ticker]].copy()
        bench_ff_map[label] = bench_raw.reindex(trading_index).ffill()

    unrate = bt_unrate_series(
        start_str=data_start.strftime("%Y-%m-%d"),
        end_str=data_end.strftime("%Y-%m-%d"),
    )

    # ✅ 리밸런싱일: 사용자가 고른 day
    nominal_dates = list(bt_month_day_schedule(start_month, end_date, rebalance_day=rebalance_day))
    rebalance_dates = [bt_next_trading_day(nd, trading_index) for nd in nominal_dates]
    if not rebalance_dates:
        raise RuntimeError("No rebalance dates in the given range.")

    start_asof = rebalance_dates[0]
    fx0 = float(fx.loc[:start_asof].iloc[-1])

    # ✅ 투자금 기준(USD/KRW) 둘 다 추적
    initial_total_usd = float(initial_usd + (initial_krw / fx0))
    initial_total_krw = float(initial_krw + (initial_usd * fx0))

    state = {
        "VAA": {"holdings": {}, "cash": 0.0},
        "LAA": {"holdings": {}, "cash": 0.0},
        "ODM": {"holdings": {}, "cash": 0.0},
    }

    def value_of(holdings: dict, asof: pd.Timestamp) -> float:
        if not holdings:
            return 0.0
        return sum(int(q) * bt_px_on_or_before(prices_ff, asof, t) for t, q in holdings.items())

    def total_value(asof: pd.Timestamp) -> float:
        return sum(value_of(state[k]["holdings"], asof) + float(state[k]["cash"]) for k in state.keys())

    def rebalance_one_date(asof: pd.Timestamp, annual: bool, add_total_usd: float):
        px_map = {t: bt_px_on_or_before(prices_ff, asof, t) for t in TICKER_LIST}

        if annual:
            total_after = total_value(asof) + float(add_total_usd)
            budgets = {k: total_after / 3.0 for k in ["VAA", "LAA", "ODM"]}
        else:
            add_each = float(add_total_usd) / 3.0
            budgets = {}
            for k in ["VAA", "LAA", "ODM"]:
                budgets[k] = value_of(state[k]["holdings"], asof) + float(state[k]["cash"]) + add_each

        scores = {t: bt_momentum_score(asof, prices_ff, t) for t in VAA_UNIVERSE}
        best_vaa = pick_vaa_asset(scores)
        vaa_hold, vaa_cash = bt_buy_all_in_if_affordable(best_vaa, budgets["VAA"], px_map[best_vaa])

        laa_safe = bt_safe_laa_asset(asof, prices_ff, unrate)
        laa_assets = ["IWD", "IEF", "GLD", laa_safe]
        laa_hold, laa_cash = bt_buy_equal_split_min_cash(laa_assets, budgets["LAA"], px_map)

        odm_asset = bt_odm_choice(asof, prices_ff)
        odm_hold, odm_cash = bt_buy_all_in_if_affordable(odm_asset, budgets["ODM"], px_map[odm_asset])

        state["VAA"] = {"holdings": vaa_hold, "cash": vaa_cash, "picked": best_vaa}
        state["LAA"] = {"holdings": laa_hold, "cash": laa_cash, "safe": laa_safe}
        state["ODM"] = {"holdings": odm_hold, "cash": odm_cash, "picked": odm_asset}

    logs = []
    port_events_usd = []  # (date, pre_value, post_value) in USD
    port_events_krw = []  # (date, pre_value, post_value) in KRW

    total_added_usd = 0.0
    total_added_krw = 0.0

    for m_idx, asof in enumerate(rebalance_dates):
        fx_asof = float(fx.loc[:asof].iloc[-1])

        pre_value_usd = total_value(asof)

        if m_idx == 0:
            add_total_usd = float(initial_total_usd)
            add_total_krw = float(initial_total_krw)
        else:
            add_total_usd = float(monthly_add_usd + (monthly_add_krw / fx_asof))
            add_total_krw = float(monthly_add_krw + (monthly_add_usd * fx_asof))
            total_added_usd += add_total_usd
            total_added_krw += add_total_krw

        annual = bt_is_annual_rebalance(m_idx)
        rebalance_one_date(asof, annual=annual, add_total_usd=add_total_usd)

        post_value_usd = total_value(asof)

        port_events_usd.append((asof, float(pre_value_usd), float(post_value_usd)))
        port_events_krw.append((asof, float(pre_value_usd * fx_asof), float(post_value_usd * fx_asof)))

        logs.append(
            {
                "asof": asof,
                "month_idx": m_idx,
                "annual": annual,
                "fx": fx_asof,
                "total_usd": post_value_usd,
                "total_krw": post_value_usd * fx_asof,
                "VAA_picked": state["VAA"].get("picked"),
                "LAA_safe": state["LAA"].get("safe"),
                "ODM_picked": state["ODM"].get("picked"),
            }
        )

    eval_asof = bt_last_trading_day_on_or_before(end_date, trading_index)
    fx_end = float(fx.loc[:eval_asof].iloc[-1])
    final_usd = float(total_value(eval_asof))
    final_krw = float(final_usd * fx_end)

    port_events_usd.append((eval_asof, float(final_usd), None))
    port_events_krw.append((eval_asof, float(final_krw), None))

    total_invested_usd = float(initial_total_usd + total_added_usd)
    total_invested_krw = float(initial_total_krw + total_added_krw)

    ret_total_usd = (final_usd / total_invested_usd - 1) * 100 if total_invested_usd > 0 else 0.0
    ret_total_krw = (final_krw / total_invested_krw - 1) * 100 if total_invested_krw > 0 else 0.0

    port_cagr_usd = bt_compute_twr_cagr(port_events=port_events_usd, start_date=rebalance_dates[0], end_date=eval_asof)
    port_cagr_krw = bt_compute_twr_cagr(port_events=port_events_krw, start_date=rebalance_dates[0], end_date=eval_asof)

    df_log = pd.DataFrame(logs)

    # 포트 시계열 (USD/KRW 둘 다)
    port_points_usd = [(row["asof"], float(row["total_usd"])) for _, row in df_log.iterrows()]
    port_points_usd.append((eval_asof, float(final_usd)))
    port_series_usd = pd.Series([v for _, v in port_points_usd], index=pd.to_datetime([d for d, _ in port_points_usd]))

    port_points_krw = [(row["asof"], float(row["total_krw"])) for _, row in df_log.iterrows()]
    port_points_krw.append((eval_asof, float(final_krw)))
    port_series_krw = pd.Series([v for _, v in port_points_krw], index=pd.to_datetime([d for d, _ in port_points_krw]))

    def _events_to_series(events: list):
        pts = [(d, post) for (d, _pre, post) in events[:-1]]  # rebalance dates: post
        pts.append((events[-1][0], events[-1][1]))            # final eval: end_value
        return pd.Series([v for _, v in pts], index=pd.to_datetime([d for d, _ in pts]))

    # ✅ 벤치마크 (USD/KRW 둘 다) + ✅ 정수매수 고정(fractional=False)
    bench_results = {}
    bench_series_map_usd = {}
    bench_series_map_krw = {}

    for label, (ticker, native_ccy) in bench_items:
        ev_usd = bt_simulate_benchmark_events(
            ticker=ticker,
            native_ccy=native_ccy,
            out_ccy="USD",
            rebalance_dates=list(df_log["asof"]),
            eval_asof=eval_asof,
            bench_ff=bench_ff_map[label],
            fx_series=fx,
            initial_total_usd=float(initial_total_usd),
            monthly_add_krw=float(monthly_add_krw),
            monthly_add_usd=float(monthly_add_usd),
            fractional=False,  # ✅ 정수만
        )
        ev_krw = bt_simulate_benchmark_events(
            ticker=ticker,
            native_ccy=native_ccy,
            out_ccy="KRW",
            rebalance_dates=list(df_log["asof"]),
            eval_asof=eval_asof,
            bench_ff=bench_ff_map[label],
            fx_series=fx,
            initial_total_usd=float(initial_total_usd),
            monthly_add_krw=float(monthly_add_krw),
            monthly_add_usd=float(monthly_add_usd),
            fractional=False,  # ✅ 정수만
        )

        bench_final_usd = float(ev_usd[-1][1])
        bench_final_krw = float(ev_krw[-1][1])

        bench_ret_usd = (bench_final_usd / total_invested_usd - 1) * 100 if total_invested_usd > 0 else 0.0
        bench_ret_krw = (bench_final_krw / total_invested_krw - 1) * 100 if total_invested_krw > 0 else 0.0

        bench_cagr_usd = bt_compute_twr_cagr(port_events=ev_usd, start_date=rebalance_dates[0], end_date=eval_asof)
        bench_cagr_krw = bt_compute_twr_cagr(port_events=ev_krw, start_date=rebalance_dates[0], end_date=eval_asof)

        bench_series_map_usd[label] = _events_to_series(ev_usd)
        bench_series_map_krw[label] = _events_to_series(ev_krw)

        bench_results[label] = {
            "label": label,
            "ticker": ticker,
            "native_ccy": native_ccy,
            "fractional": False,  # ✅ 고정
            "final_usd": bench_final_usd,
            "final_krw": bench_final_krw,
            "return_pct_usd": float(bench_ret_usd),
            "return_pct_krw": float(bench_ret_krw),
            "cagr_twr_pct_usd": float(bench_cagr_usd),
            "cagr_twr_pct_krw": float(bench_cagr_krw),
        }

    # 단일/전체 벤치 payload
    if bench_label != BT_ALL_BENCH_LABEL:
        benchmark_payload = bench_results[bench_label]
    else:
        benchmark_payload = {"mode": "all", "items": bench_results}

    return {
        "rebalance_day": int(rebalance_day),
        "start_rebalance_asof": rebalance_dates[0],
        "end_eval_asof": eval_asof,

        "initial_total_usd": float(initial_total_usd),
        "initial_total_krw": float(initial_total_krw),

        "total_invested_usd": float(total_invested_usd),
        "total_invested_krw": float(total_invested_krw),

        "final_usd": float(final_usd),
        "final_krw": float(final_krw),

        "return_pct_usd": float(ret_total_usd),
        "return_pct_krw": float(ret_total_krw),

        "cagr_twr_pct_usd": float(port_cagr_usd),
        "cagr_twr_pct_krw": float(port_cagr_krw),

        "benchmark_label": bench_label,
        "benchmark": benchmark_payload,

        "log": df_log,

        "port_series_usd": port_series_usd,
        "port_series_krw": port_series_krw,

        "bench_series_map_usd": bench_series_map_usd,
        "bench_series_map_krw": bench_series_map_krw,

        "bench_results_map": bench_results,
    }


# ======================
# 화면: Annual / Monthly / Backtest
# ======================
if mode == "Annual":
    st.header("Annual Rebalancing")

    st.subheader("Assets")
    amounts = {}

    # ✅ 10개 티커 + 현금 = 11개 → 6칸 그리드
    fields = INPUT_TICKERS + ["현금($)"]
    cols = st.columns(6)

    cash_usd = 0.0
    for i, f in enumerate(fields):
        with cols[i % 6]:
            if f == "현금($)":
                cash_usd = money_input("현금($)", key="y_cash_usd", default=0, allow_decimal=True)
            else:
                amounts[f] = st.number_input(f, min_value=0, value=0, step=1, key=f"y_amt_{f}")

    run_btn = st.button("Rebalance", type="primary")
    if run_btn:
        try:
            with st.spinner("계산 중..."):
                result = run_year(amounts, cash_usd)
            st.session_state["annual_result"] = result
            _clear_keys_with_prefix("exec_annual_")  # 실행본 편집 키 초기화
            st.success("Completed")
        except Exception as e:
            st.error(str(e))

    if "annual_result" in st.session_state:
        result = st.session_state["annual_result"]
        current_holdings = {t: int(amounts.get(t, 0)) for t in INPUT_TICKERS}

        show_result(result, current_holdings, layout="side")
        st.divider()

        executed = render_execution_editor(result, editor_prefix="exec_annual_")
        payload = export_holdings_only(executed, timestamp=result["timestamp"])

        st.download_button(
            label="✅ File Download",
            data=json.dumps(payload, indent=2),
            file_name=f"rebalance_exec_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
        )

elif mode == "Monthly":
    st.header("Monthly Rebalancing")

    uploaded = st.file_uploader("File Upload", type=["json"])
    if not uploaded:
        st.stop()

    raw_bytes = uploaded.getvalue()
    file_sig = hashlib.md5(raw_bytes).hexdigest()

    if st.session_state.get("monthly_file_sig") != file_sig:
        st.session_state["monthly_file_sig"] = file_sig
        if "monthly_result" in st.session_state:
            del st.session_state["monthly_result"]
        _clear_keys_with_prefix("exec_monthly_")

    try:
        prev_raw = json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        st.error("업로드 파일이 JSON 파싱에 실패했어. (파일 깨짐/형식 오류)")
        st.stop()

    for k in ["VAA", "LAA", "ODM"]:
        if k not in prev_raw or "holdings" not in prev_raw[k]:
            st.error("이 JSON은 예상 형식이 아니야. (VAA/LAA/ODM 안에 holdings가 필요)")
            st.stop()

    prev = json.loads(json.dumps(prev_raw))  # deep copy

    st.subheader("")
    cash_usd = money_input("현금($)", key="m_cash_usd", default=0, allow_decimal=True)

    with st.expander("Previous", expanded=False):
        merged_prev = merge_holdings(prev["VAA"]["holdings"], prev["LAA"]["holdings"], prev["ODM"]["holdings"])
        items = [(t, int(q)) for t, q in merged_prev.items() if int(q) != 0]
        items.sort(key=lambda x: x[0])
        if not items:
            st.write("-")
        else:
            cols = st.columns(5)
            for i, (t, q) in enumerate(items):
                with cols[i % 5]:
                    st.metric(t, f"{q}주")

    run_btn = st.button("REBALANCE", type="primary")
    if run_btn:
        try:
            with st.spinner("Calculating..."):
                result = run_month(prev, cash_usd)
            st.session_state["monthly_result"] = result
            _clear_keys_with_prefix("exec_monthly_")
            st.success("Completed")
        except Exception as e:
            st.error(str(e))

    if "monthly_result" in st.session_state:
        result = st.session_state["monthly_result"]
        current_holdings = merge_holdings(prev["VAA"]["holdings"], prev["LAA"]["holdings"], prev["ODM"]["holdings"])

        show_result(result, current_holdings, layout="side")
        st.divider()

        executed = render_execution_editor(result, editor_prefix="exec_monthly_")
        payload = export_holdings_only(executed, timestamp=result["timestamp"])

        st.download_button(
            label="✅ File Download",
            data=json.dumps(payload, indent=2),
            file_name=f"rebalance_exec_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    # ======================
    # Backtest UI (UPDATED: return currency selector + benchmark integer-only)
    # ======================
    st.header("Backtest")

    # ✅ 5칸: Start / End / Benchmark / Return CCY / Rebalance day
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        bt_start_ym = st.text_input(
            "Start month (YYYY-MM)",
            value=st.session_state.get("bt_start_ym", "2000-01"),
            key="bt_start_ym",
        )
    with c2:
        bt_end_in = st.text_input(
            "End date (YYYY-MM-DD or YYYY-MM)",
            value=st.session_state.get("bt_end_in", today.strftime("%Y-%m-%d")),
            key="bt_end_in",
        )
    with c3:
        bench_options = [BT_ALL_BENCH_LABEL] + list(BT_BENCHMARKS.keys())
        default_bench = st.session_state.get("bt_bench", "S&P 500 (SPY) [USD]")
        idx = bench_options.index(default_bench) if default_bench in bench_options else 1
        bt_bench = st.selectbox("Benchmark", options=bench_options, index=idx, key="bt_bench")
    with c4:
        # ✅ NEW: 수익률/성과 계산 통화
        bt_ret_ccy = st.selectbox("Return currency", options=["USD", "KRW"], index=0, key="bt_ret_ccy")
    with c5:
        # ✅ 리밸런싱일 선택
        bt_reb_day = st.number_input(
            "Rebalance day of month (1~31)",
            min_value=1,
            max_value=31,
            value=int(st.session_state.get("bt_reb_day", 10)),
            step=1,
            key="bt_reb_day",
        )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        bt_initial_krw = money_input_en("Initial KRW", key="bt_initial_krw", default=0, allow_decimal=False)
    with r2:
        bt_initial_usd = money_input_en("Initial USD", key="bt_initial_usd", default=0, allow_decimal=True)
    with r3:
        bt_add_krw = money_input_en("Monthly add KRW", key="bt_add_krw", default=0, allow_decimal=False)
    with r4:
        bt_add_usd = money_input_en("Monthly add USD", key="bt_add_usd", default=0, allow_decimal=True)

    # ✅ fractional 버튼 삭제: 벤치마크는 무조건 정수 매수
    st.caption("Benchmark shares: integer-only (fractional disabled).")

    run_bt = st.button("Run Backtest", type="primary", use_container_width=True)

    if run_bt:
        try:
            with st.spinner("Running backtest..."):
                out = bt_run_backtest(
                    start_month_ym=bt_start_ym,
                    end_date_in=bt_end_in,
                    initial_krw=float(bt_initial_krw),
                    initial_usd=float(bt_initial_usd),
                    monthly_add_krw=float(bt_add_krw),
                    monthly_add_usd=float(bt_add_usd),
                    bench_label=bt_bench,
                    rebalance_day=int(bt_reb_day),
                )
            st.session_state["backtest_result"] = out
            st.success("Completed")
        except Exception as e:
            st.error(str(e))

    if "backtest_result" in st.session_state:
        out = st.session_state["backtest_result"]

        # ✅ 현재 UI 선택(USD/KRW)에 따라 보여줄 값 선택
        ret_ccy = st.session_state.get("bt_ret_ccy", "USD")
        if ret_ccy == "USD":
            total_invested = float(out["total_invested_usd"])
            final_value = float(out["final_usd"])
            ret_pct = float(out["return_pct_usd"])
            cagr = float(out["cagr_twr_pct_usd"])
            money_label = "USD"
            invested_str = f"${total_invested:,.2f}"
            final_str = f"${final_value:,.2f}"
        else:
            total_invested = float(out["total_invested_krw"])
            final_value = float(out["final_krw"])
            ret_pct = float(out["return_pct_krw"])
            cagr = float(out["cagr_twr_pct_krw"])
            money_label = "KRW"
            invested_str = f"₩{total_invested:,.0f}"
            final_str = f"₩{final_value:,.0f}"

        a, b, c, d, e0 = st.columns(5)
        a.metric("Rebalance day", f"{int(out.get('rebalance_day', 10))}")
        b.metric("Start rebalance (asof)", str(pd.to_datetime(out["start_rebalance_asof"]).date()))
        c.metric("End eval (asof)", str(pd.to_datetime(out["end_eval_asof"]).date()))
        d.metric(f"Total invested ({money_label})", invested_str)
        e0.metric(f"Final ({money_label})", final_str)

        e, f, g, h = st.columns(4)
        e.metric("Return vs invested", f"{ret_pct:.2f}%")
        f.metric("CAGR (TWR)", f"{cagr:.2f}%")

        # 벤치 CAGR도 선택 통화로 표시
        if out.get("benchmark_label") != BT_ALL_BENCH_LABEL:
            bench = out["benchmark"]
            bench_cagr = float(bench["cagr_twr_pct_usd"] if ret_ccy == "USD" else bench["cagr_twr_pct_krw"])
            g.metric("Benchmark CAGR (TWR)", f"{bench_cagr:.2f}%")
        else:
            items = out.get("bench_results_map", {})
            key = "cagr_twr_pct_usd" if ret_ccy == "USD" else "cagr_twr_pct_krw"
            cagr_list = [(k, float(v.get(key, 0.0))) for k, v in items.items()]
            cagr_list.sort(key=lambda x: x[1], reverse=True)
            if cagr_list:
                best_k, best_v = cagr_list[0]
                worst_k, worst_v = cagr_list[-1]
                g.metric("Bench best/worst CAGR", f"{best_v:.2f}% / {worst_v:.2f}%")
            else:
                g.metric("Bench best/worst CAGR", "-")

        # 참고용(반대 통화도 같이 보여줌)
        other_ccy = "KRW" if ret_ccy == "USD" else "USD"
        other_ret = float(out["return_pct_krw"] if ret_ccy == "USD" else out["return_pct_usd"])
        other_cagr = float(out["cagr_twr_pct_krw"] if ret_ccy == "USD" else out["cagr_twr_pct_usd"])
        h.metric(f"({other_ccy}) Return/CAGR", f"{other_ret:.2f}% / {other_cagr:.2f}%")

        # ✅ 차트: 선택 통화 기준으로 PORT + BENCH(또는 All)
        port_series = (out["port_series_usd"] if ret_ccy == "USD" else out["port_series_krw"]).copy()

        series_map = {"PORT": port_series}

        if out.get("benchmark_label") != BT_ALL_BENCH_LABEL:
            bench_map = out["bench_series_map_usd"] if ret_ccy == "USD" else out["bench_series_map_krw"]
            bs = bench_map.get(out.get("benchmark_label"))
            if bs is not None:
                series_map["BENCH"] = bs.copy()
        else:
            bench_map = out["bench_series_map_usd"] if ret_ccy == "USD" else out["bench_series_map_krw"]
            for label, s in bench_map.items():
                short = label.split(" (")[0]
                key2 = f"BENCH: {short}"
                series_map[key2] = s.copy()

        df_chart = pd.concat([v.rename(k) for k, v in series_map.items()], axis=1).dropna()

        if not df_chart.empty:
            df_chart = df_chart / df_chart.iloc[0]
            df_chart = df_chart.reset_index().rename(columns={"index": "Date"})
            value_cols = [c for c in df_chart.columns if c != "Date"]
            df_melt = df_chart.melt(id_vars=["Date"], value_vars=value_cols, var_name="Series", value_name="Value")

            chart = (
                alt.Chart(df_melt)
                .mark_line()
                .encode(
                    x=alt.X("Date:T"),
                    y=alt.Y("Value:Q"),
                    color=alt.Color("Series:N"),
                    tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", format=".4f")],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)

        # ✅ All 모드일 때 벤치마크 성과표 (선택 통화 기준)
        if out.get("benchmark_label") == BT_ALL_BENCH_LABEL:
            items = out.get("bench_results_map", {})
            if items:
                rows = []
                for label, v in items.items():
                    if ret_ccy == "USD":
                        rows.append(
                            {
                                "Benchmark": label,
                                "CAGR (TWR) %": float(v.get("cagr_twr_pct_usd", 0.0)),
                                "Return vs invested %": float(v.get("return_pct_usd", 0.0)),
                                "Final (USD)": float(v.get("final_usd", 0.0)),
                                "Ticker": v.get("ticker", ""),
                                "Native CCY": v.get("native_ccy", ""),
                            }
                        )
                    else:
                        rows.append(
                            {
                                "Benchmark": label,
                                "CAGR (TWR) %": float(v.get("cagr_twr_pct_krw", 0.0)),
                                "Return vs invested %": float(v.get("return_pct_krw", 0.0)),
                                "Final (KRW)": float(v.get("final_krw", 0.0)),
                                "Ticker": v.get("ticker", ""),
                                "Native CCY": v.get("native_ccy", ""),
                            }
                        )

                df_b = pd.DataFrame(rows).sort_values("CAGR (TWR) %", ascending=False, ignore_index=True)
                with st.expander("Benchmarks (All) - summary", expanded=True):
                    showb = df_b.copy()
                    showb["CAGR (TWR) %"] = showb["CAGR (TWR) %"].map(lambda x: f"{x:.2f}%")
                    showb["Return vs invested %"] = showb["Return vs invested %"].map(lambda x: f"{x:.2f}%")
                    for col in ["Final (USD)", "Final (KRW)"]:
                        if col in showb.columns:
                            showb[col] = showb[col].map(lambda x: f"{x:,.2f}" if ret_ccy == "USD" else f"{x:,.0f}")
                    st.dataframe(showb, use_container_width=True, hide_index=True)

        df_log = out["log"].copy()
        with st.expander("Log (last 6)", expanded=True):
            show = df_log.tail(6).copy()
            show["asof"] = pd.to_datetime(show["asof"]).dt.date.astype(str)
            show["total_usd"] = show["total_usd"].map(lambda x: f"{x:,.2f}")
            show["total_krw"] = show["total_krw"].map(lambda x: f"{x:,.0f}")
            st.dataframe(
                show[["asof", "annual", "total_usd", "total_krw", "VAA_picked", "LAA_safe", "ODM_picked"]],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Log (full)", expanded=False):
            tmp = df_log.copy()
            tmp["asof"] = pd.to_datetime(tmp["asof"])
            st.dataframe(tmp, use_container_width=True, hide_index=True)

        st.download_button(
            "Download log (CSV)",
            data=df_log.to_csv(index=False).encode("utf-8"),
            file_name="backtest_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ✅ summary JSON: USD/KRW 둘 다 담기
        summary_payload = {
            "rebalance_day": int(out.get("rebalance_day", 10)),
            "start_rebalance_asof": str(pd.to_datetime(out["start_rebalance_asof"]).date()),
            "end_eval_asof": str(pd.to_datetime(out["end_eval_asof"]).date()),

            "total_invested_usd": out["total_invested_usd"],
            "total_invested_krw": out["total_invested_krw"],

            "final_usd": out["final_usd"],
            "final_krw": out["final_krw"],

            "return_pct_usd": out["return_pct_usd"],
            "return_pct_krw": out["return_pct_krw"],

            "cagr_twr_pct_usd": out["cagr_twr_pct_usd"],
            "cagr_twr_pct_krw": out["cagr_twr_pct_krw"],

            "benchmark_label": out.get("benchmark_label"),
            "benchmark": out.get("benchmark"),
            "bench_results_map": out.get("bench_results_map"),

            "benchmark_integer_only": True,
        }
        st.download_button(
            "Download summary (JSON)",
            data=json.dumps(summary_payload, indent=2),
            file_name="backtest_summary.json",
            mime="application/json",
            use_container_width=True,
        )
