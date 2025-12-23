import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf
from pandas_datareader import data as web


# ======================
# 공통 설정
# ======================
TICKER_LIST = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY", "IWD", "GLD", "QQQ", "BIL"]

# ✅ VAA 모멘텀 표기(7개) + 선택도 7개 중에서
VAA_UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"]

st.set_page_config(page_title="Rebalance (Private)", layout="wide")
st.title("PORTFOLIO REBALANCING")


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


@st.cache_data(ttl=3600, show_spinner=False)
def _unrate_info(today: datetime):
    unrate = web.DataReader("UNRATE", "fred", today - timedelta(days=400), today)
    unrate_now = float(unrate["UNRATE"].iloc[-1])
    unrate_ma = float(unrate["UNRATE"].tail(12).mean())
    return unrate_now, unrate_ma


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
# 결과 표시
# ======================
def show_result(result: dict, current_holdings: dict, layout: str = "stack"):
    """
    layout:
      - "side": (연 리밸런싱) 좌/우로: 오른쪽에 VAA 모멘텀표
      - "stack": (월 리밸런싱) 기존처럼 아래로
    """
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
        return sum(float(price_map[t]) * int(q) for t, q in h.items())

    total_holdings_usd = holdings_value_usd(vaa_h) + holdings_value_usd(laa_h) + holdings_value_usd(odm_h)
    total_cash_usd = vaa_cash + laa_cash + odm_cash
    total_usd = total_holdings_usd + total_cash_usd

    # ✅ (3) 상단 요약: 총자산/현금 = 원화, 보유자산 표시 제거
    total_krw = total_usd * rate
    cash_krw = total_cash_usd * rate

    a, b, c = st.columns(3)
    a.metric("총자산(₩)", f"₩{total_krw:,.0f}")
    b.metric("현금잔액(₩)", f"₩{cash_krw:,.0f}")
    c.metric("환율(₩/$)", f"₩{rate:,.2f}")

    st.write(
        f"**VAA picked:** {vaa.get('picked')}  |  "
        f"**LAA safe:** {laa.get('safe')}  |  "
        f"**ODM picked:** {odm.get('picked')}"
    )

    all_target = merge_holdings(vaa_h, laa_h, odm_h)

    def render_main_tables():
        # 목표 보유 ETF
        rows = []
        for t in sorted(all_target.keys()):
            qty = int(all_target[t])
            if qty == 0:
                continue
            px = float(price_map.get(t, 0.0))
            val_usd = px * qty
            val_krw = val_usd * rate
            rows.append({"Ticker": t, "Qty": qty, "Price(USD)": px, "Value(USD)": val_usd, "Value(KRW)": val_krw})

        st.subheader("목표 보유 ETF")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # 전략별 요약
        st.subheader("전략별 요약")
        strat_rows = []
        for name, h, csh in [("VAA", vaa_h, vaa_cash), ("LAA", laa_h, laa_cash), ("ODM", odm_h, odm_cash)]:
            hv = holdings_value_usd(h)
            tv = hv + csh
            w = (tv / total_usd * 100) if total_usd > 0 else 0.0
            strat_rows.append(
                {"Strategy": name, "Total(USD)": tv, "Holdings(USD)": hv, "Cash(USD)": csh, "Weight(%)": w}
            )
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True)

        # 매수/매도 수량
        st.subheader("매수/매도 수량")
        df_trades = trade_plan_df(current_holdings, all_target, price_map)
        df_trades["AbsDelta"] = df_trades["Delta"].abs()
        df_trades = df_trades.sort_values(["AbsDelta", "Ticker"], ascending=[False, True]).drop(columns=["AbsDelta"])
        st.dataframe(df_trades, use_container_width=True)

    def render_scores():
        st.subheader("모멘텀스코어")
        st.dataframe(vaa_scores_df(vaa), use_container_width=True)

    # ✅ (1) 연 리밸런싱: 모멘텀표를 좌우로
    if layout == "side":
        left, right = st.columns([2, 1], gap="large")
        with right:
            render_scores()
        with left:
            render_main_tables()
    else:
        render_scores()
        render_main_tables()


# ======================
# 사이드바: 모드 + 캐시리셋
# ======================
with st.sidebar:
    st.subheader("모드")
    mode = st.radio("TYPE", ["Year", "Month"], index=0)

    if st.button("데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()


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

with st.expander("현재가/환율"):
    px_df = pd.DataFrame([{"Ticker": t, "Adj Close(USD)": float(prices[t])} for t in TICKER_LIST]).set_index("Ticker")
    st.dataframe(px_df, use_container_width=True)

    # ✅ (2) 표기 변경 + 환율 앞 ₩
    st.write(f"환율(₩/$): **₩{usdkrw_rate:,.2f}**")


# ======================
# 실행 함수: Year / Month
# ======================
def compute_vaa_scores(prices: dict) -> dict:
    return {t: float(momentum_score(t, prices, d_1m, d_3m, d_6m, d_12m)) for t in VAA_UNIVERSE}


def run_year(amounts: dict, krw_cash: float, usd_cash: float, krw_add: float, usd_add: float):
    total_usd = (
        sum(float(amounts.get(t, 0.0)) * float(prices[t]) for t in TICKER_LIST)
        + (float(krw_cash) / float(usdkrw_rate)) + float(usd_cash)
        + (float(krw_add) / float(usdkrw_rate)) + float(usd_add)
    )

    budget = float(total_usd) / 3.0

    scores = compute_vaa_scores(prices)
    best_vaa = max(scores, key=scores.get)
    vaa_hold, vaa_cash_usd = buy_all_in_if_affordable(best_vaa, budget, prices)

    laa_safe = safe_laa_asset(today, prices)
    laa_assets = ["IWD", "IEF", "GLD", laa_safe]
    laa_hold, laa_cash_usd = buy_equal_split_min_cash(laa_assets, budget, prices)

    odm_asset = odm_choice(prices, d_12m)
    odm_hold, odm_cash_usd = buy_all_in_if_affordable(odm_asset, budget, prices)

    result = {
        "timestamp": today.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "usdkrw_rate": float(usdkrw_rate),
            "prices_adj_close": {t: float(prices[t]) for t in TICKER_LIST},
        },
        "VAA": {"holdings": vaa_hold, "cash_usd": float(vaa_cash_usd), "picked": best_vaa, "scores": scores},
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_safe},
        "ODM": {"holdings": odm_hold, "cash_usd": float(odm_cash_usd), "picked": odm_asset},
    }
    return result


def run_month(prev: dict, krw_add: float, usd_add: float):
    add_total_usd = (float(krw_add) / float(usdkrw_rate)) + float(usd_add)
    add_each = float(add_total_usd) / 3.0

    vaa_prev_hold = prev["VAA"]["holdings"]
    laa_prev_hold = prev["LAA"]["holdings"]
    odm_prev_hold = prev["ODM"]["holdings"]

    vaa_prev_cash = float(prev["VAA"].get("cash_usd", 0.0))
    laa_prev_cash = float(prev["LAA"].get("cash_usd", 0.0))
    odm_prev_cash = float(prev["ODM"].get("cash_usd", 0.0))

    vaa_budget = strategy_value_holdings(vaa_prev_hold, prices) + vaa_prev_cash + add_each
    laa_budget = strategy_value_holdings(laa_prev_hold, prices) + laa_prev_cash + add_each
    odm_budget = strategy_value_holdings(odm_prev_hold, prices) + odm_prev_cash + add_each

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
            "prices_adj_close": {t: float(prices[t]) for t in TICKER_LIST},
        },
        "VAA": {"holdings": vaa_hold, "cash_usd": float(vaa_cash_usd), "picked": best_vaa, "scores": scores},
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_safe},
        "ODM": {"holdings": odm_hold, "cash_usd": float(odm_cash_usd), "picked": odm_asset},
    }
    return result


# ======================
# 화면: Year / Month
# ======================
if mode.startswith("Year"):
    st.header("연 리밸런싱")

    st.subheader("현재 보유")
    amounts = {}
    cols = st.columns(4)
    for i, t in enumerate(TICKER_LIST):
        with cols[i % 4]:
            amounts[t] = st.number_input(t, min_value=0, value=0, step=1, key=f"y_amt_{t}")

    st.subheader("현금잔액/추가투자금액")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        krw_cash = money_input("현금(₩)", key="y_krw_cash", default=0, allow_decimal=False)
    with c2:
        usd_cash = money_input("현금($)", key="y_usd_cash", default=0, allow_decimal=True)
    with c3:
        krw_add = money_input("추가투자(₩)", key="y_krw_add", default=0, allow_decimal=False)
    with c4:
        usd_add = money_input("추가투자($)", key="y_usd_add", default=0, allow_decimal=True)

    if st.button("리밸런싱 계산", type="primary"):
        try:
            with st.spinner("계산 중..."):
                result = run_year(amounts, krw_cash, usd_cash, krw_add, usd_add)

            st.success("완료")

            current_holdings = {t: int(amounts.get(t, 0)) for t in TICKER_LIST}

            # ✅ (1) Year는 side 레이아웃
            show_result(result, current_holdings, layout="side")

            st.download_button(
                label="✅ 결과 JSON 다운로드(저장)",
                data=json.dumps(result, indent=2),
                file_name=f"rebalance_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
                mime="application/json",
            )
        except Exception as e:
            st.error(str(e))

else:
    st.header("월 리밸런싱")
    st.write("지난번에 다운로드한 JSON을 업로드하면, 그걸 기준으로 이번달 리밸런싱을 계산해.")

    uploaded = st.file_uploader("이전 결과 JSON 업로드", type=["json"])
    if not uploaded:
        st.info("먼저 Year(Y)로 결과를 계산하고 JSON을 다운로드한 다음, 여기서 업로드해.")
        st.stop()

    try:
        prev_raw = json.loads(uploaded.getvalue().decode("utf-8"))
    except Exception:
        st.error("업로드 파일이 JSON 파싱에 실패했어. (파일 깨짐/형식 오류)")
        st.stop()

    needed = ["VAA", "LAA", "ODM", "meta"]
    if any(k not in prev_raw for k in needed):
        st.error("이 JSON은 예상 형식이 아니야. (VAA/LAA/ODM/meta 키가 필요)")
        st.stop()

    st.write("업로드된 이전 결과 timestamp:", prev_raw.get("timestamp", "(no timestamp)"))

    edit_prev = st.checkbox("업로드된 내용을 수정할게 (보유/현금 직접 수정)", value=False)

    prev = json.loads(json.dumps(prev_raw))  # deep copy

    if edit_prev:
        st.subheader("업로드 내용 수정")

        for strat in ["VAA", "LAA", "ODM"]:
            with st.expander(f"{strat} 보유/현금 수정", expanded=False):
                default_cash = float(prev[strat].get("cash_usd", 0.0))
                cash_val = money_input(
                    f"{strat} 현금(USD)",
                    key=f"m_edit_{strat}_cash",
                    default=default_cash,
                    allow_decimal=True,
                )

                new_hold = {}
                cols = st.columns(4)
                for i, t in enumerate(TICKER_LIST):
                    default_q = int(prev[strat]["holdings"].get(t, 0))
                    with cols[i % 4]:
                        q = st.number_input(f"{t}", min_value=0, value=default_q, step=1, key=f"m_edit_{strat}_{t}")
                    if int(q) != 0:
                        new_hold[t] = int(q)

                prev[strat]["cash_usd"] = float(cash_val)
                prev[strat]["holdings"] = new_hold

        st.caption("수정된 값이 이번달 계산의 '현재 보유/현금' 기준으로 사용됨.")

    st.subheader("이번달 추가투자")
    c1, c2 = st.columns(2)
    with c1:
        krw_add = money_input("이번달 추가투자(₩)", key="m_krw_add", default=0, allow_decimal=False)
    with c2:
        usd_add = money_input("이번달 추가투자($)", key="m_usd_add", default=0, allow_decimal=True)

    if st.button("월 리밸런싱 계산", type="primary"):
        try:
            with st.spinner("계산 중..."):
                result = run_month(prev, krw_add, usd_add)

            st.success("완료")

            current_holdings = merge_holdings(prev["VAA"]["holdings"], prev["LAA"]["holdings"], prev["ODM"]["holdings"])

            # Month는 stack 유지
            show_result(result, current_holdings, layout="stack")

            st.download_button(
                label="✅ 이번달 결과 JSON 다운로드(저장)",
                data=json.dumps(result, indent=2),
                file_name=f"rebalance_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
                mime="application/json",
            )
        except Exception as e:
            st.error(str(e))
