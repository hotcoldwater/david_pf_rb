import json
import hashlib
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

c_m, c_a, c_sp, c_r = st.columns([1.4, 1.4, 6.2, 1.4])
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
with c_r:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        # 실행 결과/편집 상태도 같이 초기화
        for k in list(st.session_state.keys()):
            if k.startswith(("annual_result", "monthly_result", "exec_annual_", "exec_monthly_", "monthly_file_sig")):
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
    ✅ pandas_datareader 없이 FRED(UNRATE) 가져오기 (Python 3.12 호환)
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


# ======================
# 결과 표시(UI 정리 버전)
# ======================
def show_result(result: dict, current_holdings: dict, layout: str = "side"):
    """
    ✅ 여기서는 '추천안(result)' 기반으로만 보여줌.
    저장은 아래 '실행본 편집'에서 받은 holdings로 따로 저장.
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
        st.subheader("목표 보유자산 제안")
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
        st.subheader("BUY/SELL")
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
            st.markdown("**SELL**")
            if not sells:
                st.write("-")
            else:
                for t, q in sells:
                    st.write(f"{t} {q}주 SELL")
        with right:
            st.markdown("**BUY**")
            if not buys:
                st.write("-")
            else:
                for t, q in buys:
                    st.write(f"{t} {q}주 BUY")

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
    """
    result(추천안)에서 VAA/LAA/ODM holdings를 기본값으로 깔고,
    사용자가 전략별로 INPUT_TICKERS(10개) 수량을 조정해서 '실행본 holdings'를 만든다.
    """
    st.subheader("실제 보유자산")
    st.caption("여기서 네가 실제로 매매한 수량대로 조정하고 저장하면, 다음 달 Monthly에서 수정 없이 그대로 쓸 수 있어. (현금은 저장 안 함)")

    executed = {"VAA": {"holdings": {}}, "LAA": {"holdings": {}}, "ODM": {"holdings": {}}}

    for strat in ["VAA", "LAA", "ODM"]:
        rec = result[strat]["holdings"]
        with st.expander(f"{strat} 실행본 수량 조정", expanded=(strat == "VAA")):
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
    best_vaa = max(scores, key=scores.get)
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
    best_vaa = max(scores, key=scores.get)
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
# 화면: Annual / Monthly
# ======================
if mode == "Annual":
    st.header("Annual Rebalancing")

    st.subheader("Assets (현재 보유 수량)")
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
            st.success("완료")
        except Exception as e:
            st.error(str(e))

    # 결과가 있으면 표시 + 실행본 편집 + 저장
    if "annual_result" in st.session_state:
        result = st.session_state["annual_result"]
        current_holdings = {t: int(amounts.get(t, 0)) for t in INPUT_TICKERS}

        show_result(result, current_holdings, layout="side")
        st.divider()

        executed = render_execution_editor(result, editor_prefix="exec_annual_")
        payload = export_holdings_only(executed, timestamp=result["timestamp"])

        st.download_button(
            label="✅ 파일 다운로드",
            data=json.dumps(payload, indent=2),
            file_name=f"rebalance_exec_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    st.header("Monthly Rebalancing")

    uploaded = st.file_uploader("File Upload", type=["json"])
    if not uploaded:
        st.stop()

    raw_bytes = uploaded.getvalue()
    file_sig = hashlib.md5(raw_bytes).hexdigest()

    # 파일 바뀌면 월간 결과/편집키 초기화
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

    # 최소 요구: VAA/LAA/ODM + holdings
    for k in ["VAA", "LAA", "ODM"]:
        if k not in prev_raw or "holdings" not in prev_raw[k]:
            st.error("이 JSON은 예상 형식이 아니야. (VAA/LAA/ODM 안에 holdings가 필요)")
            st.stop()

    prev = json.loads(json.dumps(prev_raw))  # deep copy

    st.subheader("현금($)")
    cash_usd = money_input(key="m_cash_usd", default=0, allow_decimal=True)

    # 이전 실행본 요약(선택)
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
            _clear_keys_with_prefix("exec_monthly_")  # 새 추천안이면 실행본 편집 초기화
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
            label="✅ 실행본(ETF만) 다운로드",
            data=json.dumps(payload, indent=2),
            file_name=f"rebalance_exec_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
        )
