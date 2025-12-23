# app.py
import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf


# ======================
# 공통 설정
# ======================
TICKER_LIST = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY", "IWD", "GLD", "QQQ", "BIL"]
VAA_UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"]
FX_TICKER = "USDKRW=X"

st.set_page_config(page_title="Rebalance (Private)", layout="wide")
st.title("리밸런싱 웹앱 (개인용)")
st.caption("입력은 폼(form)으로 묶고, 보유수량은 표로 한 번에 편집 → 결과는 탭에서 확인")


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


def money_input(label: str, key: str, default: float = 0.0, allow_decimal: bool = False, help_text: str = "") -> float:
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


# ======================
# yfinance batch helpers
# ======================
def _drop_tz_index(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df


def _split_download_by_ticker(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    yfinance multi download 형태가 환경에 따라 달라져서 최대한 방어적으로 분리.
    - MultiIndex columns: (ticker, field) 또는 (field, ticker) 둘 다 대응
    - 단일 티커면 그대로 반환
    """
    out: dict[str, pd.DataFrame] = {}

    if df is None or df.empty:
        return out

    df = df.copy()
    df = _drop_tz_index(df)

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        lvln = list(df.columns.get_level_values(-1))

        # 케이스 A: level0 = ticker
        if any(t in lvl0 for t in tickers):
            for t in tickers:
                if t in df.columns.get_level_values(0):
                    sub = df[t].copy()
                    out[t] = _drop_tz_index(sub)
            return out

        # 케이스 B: last level = ticker
        if any(t in lvln for t in tickers):
            for t in tickers:
                if t in df.columns.get_level_values(-1):
                    sub = df.xs(t, axis=1, level=-1, drop_level=True).copy()
                    out[t] = _drop_tz_index(sub)
            return out

        # 그래도 못 찾으면 그냥 첫 레벨만 단일처럼
        # (이 경우는 거의 없지만 안전빵)
        for t in tickers:
            out[t] = df.copy()
        return out

    # 단일 티커 형태
    if len(tickers) >= 1:
        out[tickers[0]] = df
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_last_adj_close_map(tickers: list[str]) -> dict[str, float]:
    """
    여러 티커를 한 번에 받아서 마지막 Adj Close를 dict로 반환 (속도 체감 개선).
    """
    df = yf.download(tickers, period="7d", auto_adjust=False, progress=False, group_by="ticker")
    pieces = _split_download_by_ticker(df, tickers)

    out: dict[str, float] = {}
    for t in tickers:
        sub = pieces.get(t)
        if sub is None or sub.empty:
            continue

        col = "Adj Close" if "Adj Close" in sub.columns else "Close"
        v = sub[col].iloc[-1]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        out[t] = float(v)
    return out


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

    df = _drop_tz_index(df)

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    return df


# ======================
# FRED UNRATE (pandas_datareader 없이)
# ======================
@st.cache_data(ttl=3600, show_spinner=False)
def _unrate_info(today: datetime):
    """
    FRED 제공 CSV:
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    df = pd.read_csv(url)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["UNRATE"] = pd.to_numeric(df["UNRATE"], errors="coerce")
    df = df.dropna(subset=["DATE", "UNRATE"])

    start = today - timedelta(days=400)
    df = df[(df["DATE"] >= start) & (df["DATE"] <= today)].copy()

    if df.empty:
        raise RuntimeError("UNRATE 데이터가 비어있음")

    unrate_now = float(df["UNRATE"].iloc[-1])
    unrate_ma = float(df["UNRATE"].tail(12).mean())  # 최근 12개월 평균
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
        total += float(price_map.get(t, 0.0)) * int(q)
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
    # 실패 시 fallback: QQQ
    try:
        unrate_now, unrate_ma = _unrate_info(today)
    except Exception:
        return "QQQ"

    spy_hist = _download_hist_one("SPY", period="2y")
    spy_200ma = spy_hist["Adj Close"].rolling(200).mean().iloc[-1]
    if spy_200ma != spy_200ma:  # NaN check
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


def trade_plan_df(current: dict, target: dict, price_map: dict) -> pd.DataFrame:
    tickers = sorted(set(current.keys()) | set(target.keys()))
    rows = []
    for t in tickers:
        cur = int(current.get(t, 0))
        tar = int(target.get(t, 0))
        delta = tar - cur
        action = "BUY" if delta > 0 else "SELL" if delta < 0 else "-"
        est_usd = float(delta) * float(price_map.get(t, 0.0))
        rows.append(
            {
                "Ticker": t,
                "Current Qty": cur,
                "Target Qty": tar,
                "Delta": delta,
                "Action": action,
                "Est. Cash Impact (USD)": est_usd,
            }
        )
    return pd.DataFrame(rows)


def vaa_scores_df(vaa: dict) -> pd.DataFrame:
    scores = vaa.get("scores", {})
    rows = [{"Ticker": t, "Momentum Score": float(scores.get(t, -9999))} for t in VAA_UNIVERSE]
    df = pd.DataFrame(rows).sort_values("Momentum Score", ascending=False, ignore_index=True)
    return df


# ======================
# UI helpers
# ======================
def holdings_editor(
    key: str,
    tickers: list[str],
    prices: dict[str, float],
    default_qty: dict[str, int] | None = None,
    help_text: str = "",
) -> dict[str, int]:
    default_qty = default_qty or {}
    base = pd.DataFrame(
        {
            "Ticker": tickers,
            "Qty": [int(default_qty.get(t, 0)) for t in tickers],
            "Price(USD)": [float(prices.get(t, 0.0)) for t in tickers],
        }
    ).set_index("Ticker")
    base["Value(USD)"] = base["Qty"] * base["Price(USD)"]

    edited = st.data_editor(
        base,
        key=key,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Qty": st.column_config.NumberColumn("Qty", min_value=0, step=1, help=help_text),
            "Price(USD)": st.column_config.NumberColumn("Price(USD)", format="%.2f", disabled=True),
            "Value(USD)": st.column_config.NumberColumn("Value(USD)", format="%.2f", disabled=True),
        },
        disabled=["Price(USD)", "Value(USD)"],
    )

    # Value 재계산해서 아래에 요약 보여주기
    edited = edited.copy()
    edited["Value(USD)"] = edited["Qty"] * edited["Price(USD)"]

    total = float(edited["Value(USD)"].sum())
    st.caption(f"보유 평가금액 합계(USD): **{total:,.2f}**")

    qty_map = {t: int(edited.loc[t, "Qty"]) for t in tickers}
    return qty_map


def normalize_weights(w_vaa: float, w_laa: float, w_odm: float) -> tuple[float, float, float]:
    total = float(w_vaa) + float(w_laa) + float(w_odm)
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (float(w_vaa) / total, float(w_laa) / total, float(w_odm) / total)


def style_trades(df: pd.DataFrame):
    def _row(r):
        if r["Action"] == "BUY":
            return ["background-color: rgba(0, 200, 0, 0.12)"] * len(r)
        if r["Action"] == "SELL":
            return ["background-color: rgba(200, 0, 0, 0.12)"] * len(r)
        return [""] * len(r)

    return (
        df.style.apply(_row, axis=1)
        .format({"Est. Cash Impact (USD)": "{:+,.2f}"})
    )


def orders_text_from_trades(df_trades: pd.DataFrame) -> str:
    lines = []
    for _, r in df_trades.iterrows():
        if int(r["Delta"]) == 0:
            continue
        side = "BUY" if int(r["Delta"]) > 0 else "SELL"
        qty = abs(int(r["Delta"]))
        lines.append(f"{side} {r['Ticker']} {qty}")
    return "\n".join(lines) if lines else "(변경 없음)"


# ======================
# 날짜 기준 (룩백은 사이드바에서 설정)
# ======================
today = datetime.today()
today_naive = today.replace(tzinfo=None)


# ======================
# 사이드바: 모드 + 고급설정 + 캐시리셋
# ======================
with st.sidebar:
    st.subheader("모드")
    mode = st.radio("리밸런싱 타입", ["Year (Y)", "Month (M)"], index=0)

    st.divider()
    with st.expander("고급설정", expanded=False):
        st.caption("비중은 합이 100이 아니어도 자동 정규화함.")
        w_vaa = st.number_input("VAA 비중", min_value=0.0, value=1.0, step=0.1)
        w_laa = st.number_input("LAA 비중", min_value=0.0, value=1.0, step=0.1)
        w_odm = st.number_input("ODM 비중", min_value=0.0, value=1.0, step=0.1)

        st.caption("룩백(일): 기본 30/90/180/365")
        lb_1m = st.number_input("1M 룩백(일)", min_value=7, value=30, step=1)
        lb_3m = st.number_input("3M 룩백(일)", min_value=14, value=90, step=1)
        lb_6m = st.number_input("6M 룩백(일)", min_value=30, value=180, step=1)
        lb_12m = st.number_input("12M 룩백(일)", min_value=90, value=365, step=1)

    if st.button("시장데이터 새로고침(캐시 삭제)"):
        st.cache_data.clear()
        st.rerun()

wV, wL, wO = normalize_weights(w_vaa, w_laa, w_odm)
d_1m = today_naive - timedelta(days=int(lb_1m))
d_3m = today_naive - timedelta(days=int(lb_3m))
d_6m = today_naive - timedelta(days=int(lb_6m))
d_12m = today_naive - timedelta(days=int(lb_12m))


# ======================
# 시장 데이터 로드 (batch)
# ======================
with st.spinner("가격/환율 불러오는 중..."):
    last_map = fetch_last_adj_close_map(TICKER_LIST + [FX_TICKER])

# 환율 실패하면 수동입력 fallback
usdkrw_rate = float(last_map.get(FX_TICKER, 0.0))
prices = {t: float(last_map.get(t, 0.0)) for t in TICKER_LIST}

if usdkrw_rate <= 0:
    st.warning("USDKRW=X 환율을 못 가져왔어. 수동으로 입력해줘.")
    usdkrw_rate = st.number_input("수동 환율 입력 (원/달러)", min_value=1.0, value=1300.0, step=1.0)

# 가격 누락 체크
missing = [t for t in TICKER_LIST if prices.get(t, 0.0) <= 0]
if missing:
    st.warning(f"일부 티커 가격을 못 가져왔어: {', '.join(missing)} (해당 티커는 계산이 이상할 수 있음)")


# ======================
# 실행 함수
# ======================
def compute_vaa_scores(prices: dict) -> dict:
    return {t: float(momentum_score(t, prices, d_1m, d_3m, d_6m, d_12m)) for t in VAA_UNIVERSE}


def run_year(amounts: dict, krw_cash: float, usd_cash: float, krw_add: float, usd_add: float):
    total_usd = (
        sum(float(amounts.get(t, 0.0)) * float(prices[t]) for t in TICKER_LIST)
        + (float(krw_cash) / float(usdkrw_rate)) + float(usd_cash)
        + (float(krw_add) / float(usdkrw_rate)) + float(usd_add)
    )

    budget_vaa = float(total_usd) * wV
    budget_laa = float(total_usd) * wL
    budget_odm = float(total_usd) * wO

    scores = compute_vaa_scores(prices)
    best_vaa = max(scores, key=scores.get)
    vaa_hold, vaa_cash_usd = buy_all_in_if_affordable(best_vaa, budget_vaa, prices)

    laa_safe = safe_laa_asset(today, prices)
    laa_assets = ["IWD", "IEF", "GLD", laa_safe]
    laa_hold, laa_cash_usd = buy_equal_split_min_cash(laa_assets, budget_laa, prices)

    odm_asset = odm_choice(prices, d_12m)
    odm_hold, odm_cash_usd = buy_all_in_if_affordable(odm_asset, budget_odm, prices)

    result = {
        "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "usdkrw_rate": float(usdkrw_rate),
            "weights": {"VAA": float(wV), "LAA": float(wL), "ODM": float(wO)},
            "lookback_days": {"1m": int(lb_1m), "3m": int(lb_3m), "6m": int(lb_6m), "12m": int(lb_12m)},
            "prices_adj_close": {t: float(prices[t]) for t in TICKER_LIST},
        },
        "VAA": {"holdings": vaa_hold, "cash_usd": float(vaa_cash_usd), "picked": best_vaa, "scores": scores},
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_safe},
        "ODM": {"holdings": odm_hold, "cash_usd": float(odm_cash_usd), "picked": odm_asset},
    }
    return result


def run_month(prev: dict, krw_add: float, usd_add: float):
    add_total_usd = (float(krw_add) / float(usdkrw_rate)) + float(usd_add)

    add_vaa = float(add_total_usd) * wV
    add_laa = float(add_total_usd) * wL
    add_odm = float(add_total_usd) * wO

    vaa_prev_hold = prev["VAA"]["holdings"]
    laa_prev_hold = prev["LAA"]["holdings"]
    odm_prev_hold = prev["ODM"]["holdings"]

    vaa_prev_cash = float(prev["VAA"].get("cash_usd", 0.0))
    laa_prev_cash = float(prev["LAA"].get("cash_usd", 0.0))
    odm_prev_cash = float(prev["ODM"].get("cash_usd", 0.0))

    vaa_budget = strategy_value_holdings(vaa_prev_hold, prices) + vaa_prev_cash + add_vaa
    laa_budget = strategy_value_holdings(laa_prev_hold, prices) + laa_prev_cash + add_laa
    odm_budget = strategy_value_holdings(odm_prev_hold, prices) + odm_prev_cash + add_odm

    scores = compute_vaa_scores(prices)
    best_vaa = max(scores, key=scores.get)
    vaa_hold, vaa_cash_usd = buy_all_in_if_affordable(best_vaa, vaa_budget, prices)

    laa_safe = safe_laa_asset(today, prices)
    laa_assets = ["IWD", "IEF", "GLD", laa_safe]
    laa_hold, laa_cash_usd = buy_equal_split_min_cash(laa_assets, laa_budget, prices)

    odm_asset = odm_choice(prices, d_12m)
    odm_hold, odm_cash_usd = buy_all_in_if_affordable(odm_asset, odm_budget, prices)

    result = {
        "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "usdkrw_rate": float(usdkrw_rate),
            "weights": {"VAA": float(wV), "LAA": float(wL), "ODM": float(wO)},
            "lookback_days": {"1m": int(lb_1m), "3m": int(lb_3m), "6m": int(lb_6m), "12m": int(lb_12m)},
            "prices_adj_close": {t: float(prices[t]) for t in TICKER_LIST},
        },
        "VAA": {"holdings": vaa_hold, "cash_usd": float(vaa_cash_usd), "picked": best_vaa, "scores": scores},
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_safe},
        "ODM": {"holdings": odm_hold, "cash_usd": float(odm_cash_usd), "picked": odm_asset},
    }
    return result


# ======================
# 결과 표시 (탭 친화)
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
        return sum(float(price_map.get(t, 0.0)) * int(q) for t, q in h.items())

    total_holdings_usd = holdings_value_usd(vaa_h) + holdings_value_usd(laa_h) + holdings_value_usd(odm_h)
    total_cash_usd = vaa_cash + laa_cash + odm_cash
    total_usd = total_holdings_usd + total_cash_usd

    total_krw = total_usd * rate
    cash_krw = total_cash_usd * rate

    a, b, c, d = st.columns(4)
    a.metric("총자산(원)", f"₩{total_krw:,.0f}")
    b.metric("현금(원)", f"₩{cash_krw:,.0f}")
    c.metric("총자산(USD)", f"${total_usd:,.2f}")
    d.metric("환율(원/달러)", f"₩{rate:,.2f}")

    st.write(
        f"**VAA picked:** {vaa.get('picked')}  |  "
        f"**LAA safe:** {laa.get('safe')}  |  "
        f"**ODM picked:** {odm.get('picked')}"
    )

    all_target = merge_holdings(vaa_h, laa_h, odm_h)

    # ---- VAA 모멘텀
    def render_scores():
        st.subheader("VAA 모멘텀 스코어 (7개)")
        st.dataframe(vaa_scores_df(vaa), use_container_width=True)

    # ---- 목표 보유(통합)
    def render_target_total():
        rows = []
        for t in sorted(all_target.keys()):
            qty = int(all_target[t])
            if qty == 0:
                continue
            px = float(price_map.get(t, 0.0))
            val_usd = px * qty
            val_krw = val_usd * rate
            rows.append({"Ticker": t, "Qty": qty, "Price(USD)": px, "Value(USD)": val_usd, "Value(KRW)": val_krw})

        st.subheader("목표 보유 ETF (TOTAL)")
        df_total = pd.DataFrame(rows)
        st.dataframe(df_total, use_container_width=True)

        # 다운로드(목표 보유)
        st.download_button(
            "목표 보유표 CSV 다운로드",
            data=df_total.to_csv(index=False).encode("utf-8-sig"),
            file_name="target_holdings.csv",
            mime="text/csv",
        )

    # ---- 전략별 요약
    def render_strategy_summary():
        st.subheader("전략별 요약")
        strat_rows = []
        for name, h, csh in [("VAA", vaa_h, vaa_cash), ("LAA", laa_h, laa_cash), ("ODM", odm_h, odm_cash)]:
            hv = holdings_value_usd(h)
            tv = hv + csh
            w = (tv / total_usd * 100) if total_usd > 0 else 0.0
            strat_rows.append({"Strategy": name, "Total(USD)": tv, "Holdings(USD)": hv, "Cash(USD)": csh, "Weight(%)": w})

        df = pd.DataFrame(strat_rows)
        st.dataframe(df, use_container_width=True)

    # ---- 매매표 + 주문용 텍스트
    def render_trade_plan():
        st.subheader("매수/매도 계획 (현재 vs 목표)")

        df_trades = trade_plan_df(current_holdings, all_target, price_map)
        df_trades["AbsImpact"] = df_trades["Est. Cash Impact (USD)"].abs()
        df_trades = df_trades.sort_values(["AbsImpact", "Ticker"], ascending=[False, True]).drop(columns=["AbsImpact"])

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            show_only = st.selectbox("필터", ["전체", "BUY만", "SELL만", "변경 없음 숨김"], index=3)
        with col2:
            sort_by = st.selectbox("정렬", ["현금영향 큰 순", "Delta 큰 순"], index=0)
        with col3:
            st.caption("Est. Cash Impact(USD): BUY면 +, SELL면 - (대략값)")

        df_view = df_trades.copy()
        if show_only == "BUY만":
            df_view = df_view[df_view["Action"] == "BUY"]
        elif show_only == "SELL만":
            df_view = df_view[df_view["Action"] == "SELL"]
        elif show_only == "변경 없음 숨김":
            df_view = df_view[df_view["Action"] != "-"]

        if sort_by == "Delta 큰 순":
            df_view["AbsDelta"] = df_view["Delta"].abs()
            df_view = df_view.sort_values(["AbsDelta", "Ticker"], ascending=[False, True]).drop(columns=["AbsDelta"])

        st.dataframe(style_trades(df_view), use_container_width=True)

        # 주문용 텍스트
        st.subheader("주문용 텍스트(복붙)")
        st.code(orders_text_from_trades(df_view), language="text")

        # 다운로드(매매표)
        st.download_button(
            "매매표 CSV 다운로드",
            data=df_view.to_csv(index=False).encode("utf-8-sig"),
            file_name="trade_plan.csv",
            mime="text/csv",
        )

    # 레이아웃
    if layout == "side":
        left, right = st.columns([2, 1], gap="large")
        with right:
            render_scores()
        with left:
            render_target_total()
            render_strategy_summary()
            render_trade_plan()
    else:
        render_scores()
        render_target_total()
        render_strategy_summary()
        render_trade_plan()


# ======================
# 메인: 탭 구성
# ======================
tab_in, tab_out, tab_dbg = st.tabs(["입력", "결과", "데이터/진단"])

# ---- 데이터/진단 탭
with tab_dbg:
    st.subheader("현재가(Adj Close) / 환율")
    px_df = pd.DataFrame([{"Ticker": t, "Adj Close(USD)": float(prices.get(t, 0.0))} for t in TICKER_LIST]).set_index("Ticker")
    st.dataframe(px_df, use_container_width=True)
    st.write(f"달러환율(원/달러): **₩{usdkrw_rate:,.2f}**")

    st.subheader("UNRATE 상태(가능하면)")
    try:
        un_now, un_ma = _unrate_info(today)
        st.success(f"UNRATE 최신: {un_now:.2f} / 12개월 평균: {un_ma:.2f}")
    except Exception as e:
        st.info(f"UNRATE 가져오기 실패(무시 가능): {e}")

    st.subheader("현재 설정 요약")
    st.write(
        {
            "weights(normalized)": {"VAA": wV, "LAA": wL, "ODM": wO},
            "lookback_days": {"1m": int(lb_1m), "3m": int(lb_3m), "6m": int(lb_6m), "12m": int(lb_12m)},
        }
    )

# ---- 입력 탭
with tab_in:
    if mode.startswith("Year"):
        st.header("Year (Y) — 연 리밸런싱")

        with st.form("year_form"):
            st.subheader("현재 보유 수량(주) — 표에서 한 번에 입력")
            default_qty = {t: 0 for t in TICKER_LIST}
            amounts = holdings_editor(
                key="year_holdings_editor",
                tickers=TICKER_LIST,
                prices=prices,
                default_qty=default_qty,
                help_text="Qty 컬럼만 수정하면 됨. (복붙 가능)",
            )

            st.subheader("현금/추가투자 (콤마 입력 가능)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                krw_cash = money_input("현금(₩)", key="y_krw_cash", default=0, allow_decimal=False)
            with c2:
                usd_cash = money_input("현금($)", key="y_usd_cash", default=0, allow_decimal=True)
            with c3:
                krw_add = money_input("추가투자(₩)", key="y_krw_add", default=0, allow_decimal=False)
            with c4:
                usd_add = money_input("추가투자($)", key="y_usd_add", default=0, allow_decimal=True)

            submitted = st.form_submit_button("리밸런싱 계산", type="primary")

        if submitted:
            try:
                with st.spinner("계산 중..."):
                    result = run_year(amounts, krw_cash, usd_cash, krw_add, usd_add)

                st.session_state["last_result"] = result
                st.session_state["last_current_holdings"] = {t: int(amounts.get(t, 0)) for t in TICKER_LIST}
                st.session_state["last_layout"] = "side"

                st.success("완료. 이제 '결과' 탭에서 확인하면 돼.")

                st.download_button(
                    label="✅ 결과 JSON 다운로드(저장)",
                    data=json.dumps(result, indent=2),
                    file_name=f"rebalance_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(str(e))

    else:
        st.header("Month (M) — 월 리밸런싱")
        st.write("지난번 JSON 업로드 → (선택) 보유/현금 수정 → 이번달 추가투자 입력 → 계산")

        uploaded = st.file_uploader("이전 결과 JSON 업로드", type=["json"])
        if not uploaded:
            st.info("먼저 Year(Y)로 결과를 계산하고 JSON을 다운로드한 다음, 여기서 업로드해.")
        else:
            try:
                prev_raw = json.loads(uploaded.getvalue().decode("utf-8"))
            except Exception:
                st.error("업로드 파일이 JSON 파싱에 실패했어. (파일 깨짐/형식 오류)")
                st.stop()

            needed = ["VAA", "LAA", "ODM", "meta"]
            if any(k not in prev_raw for k in needed):
                st.error("이 JSON은 예상 형식이 아니야. (VAA/LAA/ODM/meta 키가 필요)")
                st.stop()

            st.caption(f"업로드된 이전 결과 timestamp: {prev_raw.get('timestamp', '(no timestamp)')}")

            # deep copy
            prev = json.loads(json.dumps(prev_raw))

            edit_prev = st.checkbox("업로드된 내용을 수정할게 (보유/현금 직접 수정)", value=False)

            with st.form("month_form"):
                if edit_prev:
                    st.subheader("업로드 내용 수정 — 전략별로 표 편집")
                    strat_tabs = st.tabs(["VAA", "LAA", "ODM"])
                    for strat, ttab in zip(["VAA", "LAA", "ODM"], strat_tabs):
                        with ttab:
                            default_cash = float(prev[strat].get("cash_usd", 0.0))
                            cash_val = money_input(
                                f"{strat} 현금(USD)",
                                key=f"m_edit_{strat}_cash",
                                default=default_cash,
                                allow_decimal=True,
                            )
                            default_qty = {t: int(prev[strat]["holdings"].get(t, 0)) for t in TICKER_LIST}
                            new_qty = holdings_editor(
                                key=f"m_edit_{strat}_holdings_editor",
                                tickers=TICKER_LIST,
                                prices=prices,
                                default_qty=default_qty,
                                help_text="Qty만 수정하면 됨",
                            )
                            prev[strat]["cash_usd"] = float(cash_val)
                            prev[strat]["holdings"] = {t: int(q) for t, q in new_qty.items() if int(q) != 0}

                    st.caption("수정된 값이 이번달 계산의 '현재 보유/현금' 기준으로 사용됨.")

                st.subheader("이번달 추가투자 (콤마 입력 가능)")
                c1, c2 = st.columns(2)
                with c1:
                    krw_add = money_input("이번달 추가투자(₩)", key="m_krw_add", default=0, allow_decimal=False)
                with c2:
                    usd_add = money_input("이번달 추가투자($)", key="m_usd_add", default=0, allow_decimal=True)

                submitted = st.form_submit_button("월 리밸런싱 계산", type="primary")

            if submitted:
                try:
                    with st.spinner("계산 중..."):
                        result = run_month(prev, krw_add, usd_add)

                    current_holdings = merge_holdings(
                        prev["VAA"]["holdings"], prev["LAA"]["holdings"], prev["ODM"]["holdings"]
                    )

                    st.session_state["last_result"] = result
                    st.session_state["last_current_holdings"] = current_holdings
                    st.session_state["last_layout"] = "stack"

                    st.success("완료. 이제 '결과' 탭에서 확인하면 돼.")

                    st.download_button(
                        label="✅ 이번달 결과 JSON 다운로드(저장)",
                        data=json.dumps(result, indent=2),
                        file_name=f"rebalance_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.error(str(e))

# ---- 결과 탭
with tab_out:
    st.header("결과")
    if "last_result" not in st.session_state:
        st.info("아직 계산 결과가 없어. '입력' 탭에서 계산부터 해.")
    else:
        show_result(
            st.session_state["last_result"],
            st.session_state.get("last_current_holdings", {}),
            layout=st.session_state.get("last_layout", "side"),
        )
