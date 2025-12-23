import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf


# ======================
# UI TEXT REGISTRY (여기만 보면 화면 문구를 한번에 다 볼 수 있음)
# ======================
UI_DEFAULT = {
    # page
    "page_title": "Rebalance (Private)",
    "app_title": "리밸런싱 웹앱 (개인용)",
    "app_caption": "저장 방식: 결과 JSON 다운로드 → 다음달에 JSON 업로드로 이어가기",

    # sidebar
    "sidebar_mode_header": "모드",
    "sidebar_mode_label": "리밸런싱 타입",
    "mode_year": "Year (Y)",
    "mode_month": "Month (M)",
    "sidebar_cache_clear": "시장데이터 새로고침(캐시 삭제)",
    "sidebar_ui_editor_header": "UI 문구 편집",
    "sidebar_ui_upload": "UI 문구 JSON 불러오기",
    "sidebar_ui_toggle": "문구 편집 모드 켜기",
    "sidebar_ui_apply": "✅ 수정 반영",
    "sidebar_ui_reset": "↩︎ 기본값으로 리셋",
    "sidebar_ui_download": "UI 문구 JSON 다운로드",
    "sidebar_ui_loaded": "UI 문구 로드됨",

    # expander / market info
    "expander_price_fx": "현재가(Adj Close) / 달러환율 보기",
    "fx_label": "달러환율(원/달러):",

    # headers
    "header_year": "Year (Y) — 연 리밸런싱",
    "header_month": "Month (M) — 월 리밸런싱",
    "month_desc": "지난번에 다운로드한 JSON을 업로드하면, 그걸 기준으로 이번달 리밸런싱을 계산해.",
    "month_need_json_info": "먼저 Year(Y)로 결과를 계산하고 JSON을 다운로드한 다음, 여기서 업로드해.",
    "uploaded_prev_timestamp": "업로드된 이전 결과 timestamp:",

    # sections
    "sub_current_qty": "현재 보유 수량(주)",
    "sub_cash_add": "현금/추가투자 (콤마 입력 가능)",
    "sub_month_add": "이번달 추가투자 (콤마 입력 가능)",
    "sub_edit_uploaded": "업로드 내용 수정",
    "edit_caption": "수정된 값이 이번달 계산의 '현재 보유/현금' 기준으로 사용됨.",

    # inputs
    "label_krw_cash": "현금(₩)",
    "label_usd_cash": "현금($)",
    "label_krw_add": "추가투자(₩)",
    "label_usd_add": "추가투자($)",
    "label_month_krw_add": "이번달 추가투자(₩)",
    "label_month_usd_add": "이번달 추가투자($)",
    "upload_prev_json": "이전 결과 JSON 업로드",
    "edit_prev_checkbox": "업로드된 내용을 수정할게 (보유/현금 직접 수정)",

    # buttons
    "btn_calc_year": "리밸런싱 계산",
    "btn_calc_month": "월 리밸런싱 계산",
    "download_year": "✅ 결과 JSON 다운로드(저장)",
    "download_month": "✅ 이번달 결과 JSON 다운로드(저장)",

    # results
    "done": "완료",
    "metric_total": "총자산(원)",
    "metric_cash": "현금(원)",
    "metric_fx": "달러환율(원/달러)",
    "sub_momentum_table": "VAA 모멘텀 스코어 (7개)",
    "sub_target_total": "목표 보유 ETF (TOTAL)",
    "sub_strategy_summary": "전략별 요약",
    "sub_trades": "매수/매도 수량 (현재 보유 vs 목표)",

    # errors / infos
    "err_money_input": "'{label}' 숫자 입력이 이상해. 예: 1,000,000 / 1000 / 1,000.50",
    "err_json_parse": "업로드 파일이 JSON 파싱에 실패했어. (파일 깨짐/형식 오류)",
    "err_json_format": "이 JSON은 예상 형식이 아니야. (VAA/LAA/ODM/meta 키가 필요)",
}

def _load_ui_override() -> dict:
    raw = st.session_state.get("UI_OVERRIDE", {})
    return raw if isinstance(raw, dict) else {}

def t(key: str) -> str:
    """
    override가 있으면 그 값을 그대로 사용 (빈 문자열도 허용)
    override가 없으면 default 사용
    """
    override = _load_ui_override()
    if key in override:
        return str(override[key])
    return str(UI_DEFAULT.get(key, key))

def tf(key: str, **kwargs) -> str:
    """format placeholder 지원"""
    try:
        return t(key).format(**kwargs)
    except Exception:
        return t(key)

def ui_text_editor():
    """
    사이드바에서 UI 문구를 표로 한 번에 보고 수정 + JSON 저장/로드
    """
    with st.sidebar:
        st.divider()
        st.subheader(t("sidebar_ui_editor_header"))

        up = st.file_uploader(t("sidebar_ui_upload"), type=["json"], key="ui_text_upload")
        if up is not None:
            try:
                loaded = json.loads(up.getvalue().decode("utf-8"))
                if isinstance(loaded, dict):
                    # 알 수 없는 키가 섞여도 그냥 저장 (필요시 걸러도 됨)
                    st.session_state["UI_OVERRIDE"] = loaded
                    st.success(t("sidebar_ui_loaded"))
                    st.rerun()
                else:
                    st.error("JSON 최상단은 dict여야 함")
            except Exception as e:
                st.error(f"로드 실패: {e}")

        edit_on = st.toggle(t("sidebar_ui_toggle"), value=False, key="ui_text_edit_toggle")

        if edit_on:
            override = _load_ui_override()
            rows = []
            for k in UI_DEFAULT.keys():
                cur = override.get(k, UI_DEFAULT[k])
                rows.append({"key": k, "text": cur, "default": UI_DEFAULT[k]})

            df = pd.DataFrame(rows)

            edited = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "key": st.column_config.TextColumn("key", disabled=True),
                    "text": st.column_config.TextColumn("현재 문구(수정가능)"),
                    "default": st.column_config.TextColumn("기본값", disabled=True),
                },
                key="ui_text_editor_table",
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button(t("sidebar_ui_apply"), type="primary"):
                    # default와 다른 것만 override로 저장 (파일 깔끔해짐)
                    new_override = {}
                    for _, r in edited.iterrows():
                        k = str(r["key"])
                        txt = "" if pd.isna(r["text"]) else str(r["text"])
                        dft = "" if pd.isna(r["default"]) else str(r["default"])
                        if txt != dft:
                            new_override[k] = txt
                    st.session_state["UI_OVERRIDE"] = new_override
                    st.success("반영됨")
                    st.rerun()

            with c2:
                if st.button(t("sidebar_ui_reset")):
                    st.session_state["UI_OVERRIDE"] = {}
                    st.success("리셋됨")
                    st.rerun()

        override = _load_ui_override()
        st.download_button(
            t("sidebar_ui_download"),
            data=json.dumps(override, ensure_ascii=False, indent=2),
            file_name="ui_text_override.json",
            mime="application/json",
            key="ui_text_download_btn",
        )


# ======================
# 공통 설정
# ======================
TICKER_LIST = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY", "IWD", "GLD", "QQQ", "BIL"]

# ✅ VAA 모멘텀 표기(7개) + 선택도 7개 중에서
VAA_UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"]

st.set_page_config(page_title=t("page_title"), layout="wide")
st.title(t("app_title"))
cap = t("app_caption")
if cap != "":
    st.caption(cap)


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
        st.error(tf("err_money_input", label=label))
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
    FRED 제공 CSV:
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    df = pd.read_csv(url)

    # 컬럼: DATE, UNRATE
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
def momentum_score(ticker: str, prices: dict, d_1m, d_3m, d_6m, d_12m) -> float:
    """가중합 모멘텀 (Adj Close): 1m*12 + 3m*4 + 6m*2 + 12m*1"""
    try:
        hist = _download_hist_one(ticker, period="2y")
        p = float(prices[ticker])

        p1 = price_asof_or_before(hist, d_1m)
        p3 = price_asof_or_before(hist, d_3m)
        p6 = price_asof_or_before(hist, d_6m)
        p12 = price_asof_or_before(hist, d_12m)

        return (p / p1 - 1) * 12 + (p / p3 - 1) * 4 + (p / p6 - 1) * 2 + (p / p12 - 1) * 1
    except Exception:
        return -9999


def strategy_value_holdings(holdings: dict, price_map: dict) -> float:
    total = 0.0
    for tkr, q in holdings.items():
        total += float(price_map[tkr]) * int(q)
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
    def r12(tkr):
        hist = _download_hist_one(tkr, period="2y")
        p0 = price_asof_or_before(hist, d_12m)
        return float(prices[tkr]) / float(p0) - 1

    bil_r = r12("BIL")
    spy_r = r12("SPY")
    efa_r = r12("EFA")

    if bil_r > spy_r:
        return "AGG"
    return "SPY" if spy_r >= efa_r else "EFA"


def merge_holdings(*holding_dicts):
    merged = {}
    for h in holding_dicts:
        for tkr, q in h.items():
            merged[tkr] = merged.get(tkr, 0) + int(q)
    return merged


def trade_plan_df(current: dict, target: dict, price_map: dict) -> pd.DataFrame:
    tickers = sorted(set(current.keys()) | set(target.keys()))
    rows = []
    for tkr in tickers:
        cur = int(current.get(tkr, 0))
        tar = int(target.get(tkr, 0))
        delta = tar - cur
        action = "BUY" if delta > 0 else "SELL" if delta < 0 else "-"
        est_usd = float(delta) * float(price_map.get(tkr, 0.0))
        rows.append(
            {
                "Ticker": tkr,
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
    rows = [{"Ticker": tkr, "Momentum Score": float(scores.get(tkr, -9999))} for tkr in VAA_UNIVERSE]
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
        return sum(float(price_map[tkr]) * int(q) for tkr, q in h.items())

    total_holdings_usd = holdings_value_usd(vaa_h) + holdings_value_usd(laa_h) + holdings_value_usd(odm_h)
    total_cash_usd = vaa_cash + laa_cash + odm_cash
    total_usd = total_holdings_usd + total_cash_usd

    total_krw = total_usd * rate
    cash_krw = total_cash_usd * rate

    a, b, c = st.columns(3)
    a.metric(t("metric_total"), f"₩{total_krw:,.0f}")
    b.metric(t("metric_cash"), f"₩{cash_krw:,.0f}")
    c.metric(t("metric_fx"), f"₩{rate:,.2f}")

    st.write(
        f"**VAA picked:** {vaa.get('picked')}  |  "
        f"**LAA safe:** {laa.get('safe')}  |  "
        f"**ODM picked:** {odm.get('picked')}"
    )

    all_target = merge_holdings(vaa_h, laa_h, odm_h)

    def render_main_tables():
        rows = []
        for tkr in sorted(all_target.keys()):
            qty = int(all_target[tkr])
            if qty == 0:
                continue
            px = float(price_map.get(tkr, 0.0))
            val_usd = px * qty
            val_krw = val_usd * rate
            rows.append({"Ticker": tkr, "Qty": qty, "Price(USD)": px, "Value(USD)": val_usd, "Value(KRW)": val_krw})

        st.subheader(t("sub_target_total"))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader(t("sub_strategy_summary"))
        strat_rows = []
        for name, h, csh in [("VAA", vaa_h, vaa_cash), ("LAA", laa_h, laa_cash), ("ODM", odm_h, odm_cash)]:
            hv = holdings_value_usd(h)
            tv = hv + csh
            w = (tv / total_usd * 100) if total_usd > 0 else 0.0
            strat_rows.append(
                {"Strategy": name, "Total(USD)": tv, "Holdings(USD)": hv, "Cash(USD)": csh, "Weight(%)": w}
            )
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True)

        st.subheader(t("sub_trades"))
        df_trades = trade_plan_df(current_holdings, all_target, price_map)
        df_trades["AbsDelta"] = df_trades["Delta"].abs()
        df_trades = df_trades.sort_values(["AbsDelta", "Ticker"], ascending=[False, True]).drop(columns=["AbsDelta"])
        st.dataframe(df_trades, use_container_width=True)

    def render_scores():
        st.subheader(t("sub_momentum_table"))
        st.dataframe(vaa_scores_df(vaa), use_container_width=True)

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
# 사이드바: 모드 + 캐시리셋 + UI 문구 편집기
# ======================
with st.sidebar:
    st.subheader(t("sidebar_mode_header"))

    # ✅ 표시 문구는 바꿔도 내부키는 고정이라 안전하게 동작
    mode_key = st.radio(
        t("sidebar_mode_label"),
        options=["Year", "Month"],
        index=0,
        format_func=lambda k: t("mode_year") if k == "Year" else t("mode_month"),
        key="mode_radio_key",
    )

    if st.button(t("sidebar_cache_clear")):
        st.cache_data.clear()
        st.rerun()

# 사이드바 하단에 문구 편집기
ui_text_editor()


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
    prices = {tkr: last_adj_close(tkr) for tkr in TICKER_LIST}
    usdkrw_rate = fx_usdkrw()

with st.expander(t("expander_price_fx")):
    px_df = pd.DataFrame([{"Ticker": tkr, "Adj Close(USD)": float(prices[tkr])} for tkr in TICKER_LIST]).set_index("Ticker")
    st.dataframe(px_df, use_container_width=True)
    st.write(f"{t('fx_label')} **₩{usdkrw_rate:,.2f}**")


# ======================
# 실행 함수: Year / Month
# ======================
def compute_vaa_scores(prices_: dict) -> dict:
    return {tkr: float(momentum_score(tkr, prices_, d_1m, d_3m, d_6m, d_12m)) for tkr in VAA_UNIVERSE}


def run_year(amounts: dict, krw_cash: float, usd_cash: float, krw_add: float, usd_add: float):
    total_usd = (
        sum(float(amounts.get(tkr, 0.0)) * float(prices[tkr]) for tkr in TICKER_LIST)
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
            "prices_adj_close": {tkr: float(prices[tkr]) for tkr in TICKER_LIST},
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
            "prices_adj_close": {tkr: float(prices[tkr]) for tkr in TICKER_LIST},
        },
        "VAA": {"holdings": vaa_hold, "cash_usd": float(vaa_cash_usd), "picked": best_vaa, "scores": scores},
        "LAA": {"holdings": laa_hold, "cash_usd": float(laa_cash_usd), "safe": laa_safe},
        "ODM": {"holdings": odm_hold, "cash_usd": float(odm_cash_usd), "picked": odm_asset},
    }
    return result


# ======================
# 화면: Year / Month
# ======================
if mode_key == "Year":
    st.header(t("header_year"))

    st.subheader(t("sub_current_qty"))
    amounts = {}
    cols = st.columns(4)
    for i, tkr in enumerate(TICKER_LIST):
        with cols[i % 4]:
            amounts[tkr] = st.number_input(tkr, min_value=0, value=0, step=1, key=f"y_amt_{tkr}")

    st.subheader(t("sub_cash_add"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        krw_cash = money_input(t("label_krw_cash"), key="y_krw_cash", default=0, allow_decimal=False)
    with c2:
        usd_cash = money_input(t("label_usd_cash"), key="y_usd_cash", default=0, allow_decimal=True)
    with c3:
        krw_add = money_input(t("label_krw_add"), key="y_krw_add", default=0, allow_decimal=False)
    with c4:
        usd_add = money_input(t("label_usd_add"), key="y_usd_add", default=0, allow_decimal=True)

    if st.button(t("btn_calc_year"), type="primary"):
        try:
            with st.spinner("계산 중..."):
                result = run_year(amounts, krw_cash, usd_cash, krw_add, usd_add)

            st.success(t("done"))
            current_holdings = {tkr: int(amounts.get(tkr, 0)) for tkr in TICKER_LIST}
            show_result(result, current_holdings, layout="side")

            st.download_button(
                label=t("download_year"),
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name=f"rebalance_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
                mime="application/json",
            )
        except Exception as e:
            st.error(str(e))

else:
    st.header(t("header_month"))
    st.write(t("month_desc"))

    uploaded = st.file_uploader(t("upload_prev_json"), type=["json"], key="prev_json_upload_main")
    if not uploaded:
        st.info(t("month_need_json_info"))
        st.stop()

    try:
        prev_raw = json.loads(uploaded.getvalue().decode("utf-8"))
    except Exception:
        st.error(t("err_json_parse"))
        st.stop()

    needed = ["VAA", "LAA", "ODM", "meta"]
    if any(k not in prev_raw for k in needed):
        st.error(t("err_json_format"))
        st.stop()

    st.write(t("uploaded_prev_timestamp"), prev_raw.get("timestamp", "(no timestamp)"))

    edit_prev = st.checkbox(t("edit_prev_checkbox"), value=False, key="edit_prev_cb")

    prev = json.loads(json.dumps(prev_raw))  # deep copy

    if edit_prev:
        st.subheader(t("sub_edit_uploaded"))

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
                for i, tkr in enumerate(TICKER_LIST):
                    default_q = int(prev[strat]["holdings"].get(tkr, 0))
                    with cols[i % 4]:
                        q = st.number_input(f"{tkr}", min_value=0, value=default_q, step=1, key=f"m_edit_{strat}_{tkr}")
                    if int(q) != 0:
                        new_hold[tkr] = int(q)

                prev[strat]["cash_usd"] = float(cash_val)
                prev[strat]["holdings"] = new_hold

        cap2 = t("edit_caption")
        if cap2 != "":
            st.caption(cap2)

    st.subheader(t("sub_month_add"))
    c1, c2 = st.columns(2)
    with c1:
        krw_add = money_input(t("label_month_krw_add"), key="m_krw_add", default=0, allow_decimal=False)
    with c2:
        usd_add = money_input(t("label_month_usd_add"), key="m_usd_add", default=0, allow_decimal=True)

    if st.button(t("btn_calc_month"), type="primary"):
        try:
            with st.spinner("계산 중..."):
                result = run_month(prev, krw_add, usd_add)

            st.success(t("done"))
            current_holdings = merge_holdings(prev["VAA"]["holdings"], prev["LAA"]["holdings"], prev["ODM"]["holdings"])
            show_result(result, current_holdings, layout="stack")

            st.download_button(
                label=t("download_month"),
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name=f"rebalance_{result['timestamp'].replace(':','-').replace(' ','_')}.json",
                mime="application/json",
            )
        except Exception as e:
            st.error(str(e))
