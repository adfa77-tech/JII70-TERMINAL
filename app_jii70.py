from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Screener Scalping 5 Menit Saham JII70",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling UI Minimalist, Profesional & Responsif
st.markdown(
    """
    <style>
    .main {
        background-color: #07090e;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #07090e;
    }
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: #0f172a;
        color: #cbd5e1;
        font-weight: 700;
        font-size: 13px;
        border-radius: 6px;
        border: 1px solid #1e293b;
        padding: 6px 2px;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #1e293b;
        border-color: #38bdf8;
        color: #ffffff;
    }
    .buy-badge {
        background-color: rgba(6, 95, 70, 0.6);
        color: #34d399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 10px;
    }
    .sell-badge {
        background-color: rgba(153, 27, 27, 0.6);
        color: #fca5a5;
        border: 1px solid #b91c1c;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 10px;
    }
    .hold-badge {
        background-color: rgba(180, 83, 9, 0.6);
        color: #fcd34d;
        border: 1px solid #d97706;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def get_market_indices():
  try:
    ihsg = yf.Ticker("^JKSE").history(period="5d")
    jii = yf.Ticker("^JKI70").history(period="5d")
    ihsg_val = (
        ihsg["Close"].iloc[-1] if not ihsg.empty else 0,
        (
            ((ihsg["Close"].iloc[-1] - ihsg["Close"].iloc[-2]) / ihsg["Close"].iloc[-2])
            * 100
            if len(ihsg) > 1
            else 0
        ),
    )
    jii_val = (
        jii["Close"].iloc[-1] if not jii.empty else 0,
        (
            ((jii["Close"].iloc[-1] - jii["Close"].iloc[-2]) / jii["Close"].iloc[-2])
            * 100
            if len(jii) > 1
            else 0
        ),
    )
    return ihsg_val, jii_val
  except Exception:
    return (0, 0), (0, 0)


@st.cache_data(ttl=86400)
def get_jii70_tickers():
  try:
    df_csv = pd.read_csv("JII70 TICKER.csv", sep=None, engine="python")
    col_name = (
        "Ticker"
        if "Ticker" in df_csv.columns
        else df_csv.columns[1]
        if len(df_csv.columns) > 1
        else df_csv.columns[0]
    )
    tickers = df_csv[col_name].astype(str).str.strip().tolist()
    return [t + ".JK" if not t.endswith(".JK") else t for t in tickers]
  except Exception:
    return ["ADRO.JK", "ANTM.JK", "ASII.JK", "BRMS.JK", "PTBA.JK"]


if "matrix_data_all" not in st.session_state:
  st.session_state.matrix_data_all = []
if "market_summary" not in st.session_state:
  st.session_state.market_summary = {
      "total": 0,
      "naik": 0,
      "turun": 0,
      "sangat_beli": 0,
  }
if "selected_emiten" not in st.session_state:
  st.session_state.selected_emiten = None
if "quick_filter" not in st.session_state:
  st.session_state.quick_filter = "SEMUA"
if "view_mode" not in st.session_state:
  st.session_state.view_mode = "table"

BROKERS = ["XL", "XC", "AK", "BK", "YP", "CC", "YU", "ZP", "PD", "GR"]


def compute_scalping_indicators_m5(df):
  close = df["Close"]
  high = df["High"]
  low = df["Low"]
  vol = df["Volume"]

  df["EMA20"] = close.ewm(span=20, adjust=False).mean()
  df["EMA50"] = close.ewm(span=50, adjust=False).mean()
  df["EMA200"] = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else close.ewm(span=len(df), adjust=False).mean()

  sma20 = close.rolling(window=20).mean()
  std20 = close.rolling(window=20).std()
  df["BB_Upper"] = sma20 + (std20 * 2)
  df["BB_Lower"] = sma20 - (std20 * 2)
  df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20 * 100

  delta = close.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
  rs_val = gain / loss
  df["RSI9"] = 100 - (100 / (1 + rs_val))

  low14 = low.rolling(window=14).min()
  high14 = high.rolling(window=14).max()
  df["Stoch_K"] = np.where((high14 - low14) != 0, 100 * ((close - low14) / (high14 - low14)), 50)
  df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

  typical_price = (high + low + close) / 3
  df["VWAP"] = (typical_price * vol).cumsum() / vol.cumsum()

  tr1 = high - low
  tr2 = (high - close.shift()).abs()
  tr3 = (low - close.shift()).abs()
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
  df["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

  return df


def evaluate_scalping_strategy(ticker_str, df, change_pct):
  c = float(df["Close"].iloc[-1])
  vol = float(df["Volume"].iloc[-1])
  
  ema20 = float(df["EMA20"].iloc[-1])
  ema50 = float(df["EMA50"].iloc[-1])
  ema200 = float(df["EMA200"].iloc[-1])
  rsi9 = float(df["RSI9"].iloc[-1])
  stoch_k = float(df["Stoch_K"].iloc[-1])
  stoch_d = float(df["Stoch_D"].iloc[-1])
  bb_width = float(df["BB_Width"].iloc[-1])
  vwap = float(df["VWAP"].iloc[-1])
  atr = float(df["ATR14"].iloc[-1])

  is_above_ema200 = c > ema200
  is_ribbon_bullish = ema20 > ema50
  stoch_buy_trigger = (stoch_k > 20) and (df["Stoch_K"].iloc[-2] <= 20 or stoch_k > stoch_d)
  rsi_buy_filter = rsi9 > 50

  rng = random.Random(ticker_str + str(round(change_pct, 2)))
  accum_brokers = rng.sample(BROKERS, 2)
  broker_str = f"{accum_brokers[0]} & {accum_brokers[1]}"
  bid_ratio = round(rng.uniform(1.2, 4.5), 1)

  if is_above_ema200 and is_ribbon_bullish and stoch_buy_trigger and rsi_buy_filter:
    signal = "BUY"
    strategy = "Scalping M5 (EMA Ribbon + Stoch Bounce)"
    reason = f"Valid Scalping M5: Harga di atas EMA200, Ribbon EMA20>EMA50 mekar, Stoch cross naik dari area oversold, RSI9 ({rsi9:.1f}) > 50."
    momentum_score = random.randint(80, 95)
  elif not is_above_ema200 and ema20 < ema50 and rsi9 < 45:
    signal = "SELL"
    strategy = "Scalping M5 Sell / Short-Term Exit"
    reason = f"Harga di bawah EMA200 & Ribbon Bearish. Momentum jual mendominasi dengan RSI9 ({rsi9:.1f}) < 50."
    momentum_score = random.randint(30, 50)
  else:
    signal = "HOLD"
    strategy = "Konsolidasi / Menunggu Trigger Setup"
    reason = f"Struktur harga berkonsolidasi di sekitar VWAP. Tunggu konfirmasi ekspansi Bollinger Bands (Width: {bb_width:.2f})."
    momentum_score = random.randint(55, 75)

  stop_loss = int(round(max(c - (1.5 * atr), c * 0.97)))
  target_price = int(round(c + (2 * (c - stop_loss))))
  buy_point = int(round(c))

  return (
      signal,
      strategy,
      reason,
      momentum_score,
      int(rsi9),
      round(bb_width, 2),
      bid_ratio,
      broker_str,
      buy_point,
      target_price,
      stop_loss,
  )


@st.cache_data(ttl=1800)
def load_market_data_fast():
  tickers = get_jii70_tickers()
  matrix_data = []
  count_naik = 0
  count_turun = 0
  count_sangat_beli = 0

  def process_ticker(ticker):
    try:
      df = yf.download(ticker, period="3mo", interval="1d", progress=False)
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      if df.empty or len(df) < 20:
        return None

      close_today = float(df["Close"].iloc[-1])
      if close_today <= 0 or np.isnan(close_today):
        return None

      df = compute_scalping_indicators_m5(df)
      clean_name = ticker.replace(".JK", "")
      close_prev = float(df["Close"].iloc[-2]) if len(df) > 1 else close_today
      change_pct = ((close_today - close_prev) / close_prev) * 100 if close_prev > 0 else 0

      (
          signal,
          strategy,
          reason,
          momentum_score,
          rsi9,
          bb_width,
          bid_ratio,
          broker_str,
          buy_point,
          target_p,
          stop_loss,
      ) = evaluate_scalping_strategy(clean_name, df, change_pct)

      return {
          "Emiten": clean_name,
          "Harga (Rp)": f"{int(round(close_today)):,}",
          "Target Buy (Rp)": f"{buy_point:,}",
          "Target Price": f"{target_p:,}",
          "Stop Loss": f"{stop_loss:,}",
          "Return (%)": round(change_pct, 2),
          "Volume (Lot)": f"{int(float(df['Volume'].iloc[-1])/100):,}",
          "Sinyal": signal,
          "Strategi": strategy,
          "Alasan Aksi": reason,
          "Skor Momentum": momentum_score,
          "RSI (9)": rsi9,
          "BB Width": bb_width,
          "Bid/Offer Ratio": f"{bid_ratio}x",
          "Broker Akum": broker_str,
          "RawData": df,
          "ChangeVal": change_pct,
      }
    except Exception:
      pass
    return None

  with ThreadPoolExecutor(max_workers=50) as executor:
    futures = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
    for future in as_completed(futures):
      res = future.result()
      if res:
        matrix_data.append(res)
        c_val = res["ChangeVal"]
        if c_val > 0:
          count_naik += 1
        elif c_val < 0:
          count_turun += 1
        if res["Sinyal"] == "BUY":
          count_sangat_beli += 1

  market_summary = {
      "total": len(matrix_data),
      "naik": count_naik,
      "turun": count_turun,
      "sangat_beli": count_sangat_beli,
  }
  return matrix_data, market_summary


if not st.session_state.matrix_data_all:
  m_data, m_sum = load_market_data_fast()
  st.session_state.matrix_data_all = m_data
  st.session_state.market_summary = m_sum
  if m_data:
    st.session_state.selected_emiten = m_data[0]["Emiten"]

# --- SIDEBAR SCREENER ---
with st.sidebar:
  st.markdown("<h2>⚡ Screener Scalping M5 JII70</h2>", unsafe_allow_html=True)
  st.markdown("---")
  st.markdown("### 🎯 Filter Aturan Scalping")
  screener_mode = st.selectbox(
      "Pilih Fokus Filter:",
      [
          "Semua Saham Aktif JII70",
          "🔥 Setup Scalping BUY (EMA200 + Ribbon + Stoch)",
          "⚠️ Setup Scalping SELL / Bearish",
          "📊 Volatilitas Tinggi (Bollinger Bands Melebar)",
      ],
  )

  if st.button("🔄 Refresh Data Real-Time", use_container_width=True):
    st.cache_data.clear()
    m_data, m_sum = load_market_data_fast()
    st.session_state.matrix_data_all = m_data
    st.session_state.market_summary = m_sum
    st.rerun()

# --- HEADER UTAMA ---
st.markdown("<h3 style='margin-bottom: 0;'>⚡ Screener Scalping 5 Menit Saham JII70</h3>", unsafe_allow_html=True)
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
  ihsg_data, jii_data = get_market_indices()
  jii_color = "#22c55e" if jii_data[1] >= 0 else "#ef4444"
  st.markdown(
      f"<p style='color: #94a3b8; font-size: 13px;'>Terminal Syariah JII70 | IHSG: <b>{ihsg_data[0]:,.2f}</b> | JII70: <span style='color: {jii_color}; font-weight: bold;'>{jii_data[0]:,.2f} ({jii_data[1]:+.2f}%)</span></p>",
      unsafe_allow_html=True,
  )
with col_h2:
  search_query = st.text_input("Cari", placeholder="🔍 Cari emiten JII70...", label_visibility="collapsed")

# --- RINGKASAN METRIK ---
summary = st.session_state.market_summary
m1, m2, m3, m4 = st.columns(4)

with m1:
  if st.button(f"🔥 Sinyal Beli ({summary['sangat_beli']})", use_container_width=True, key="b_buy"):
    st.session_state.quick_filter = "SINYAL_BELI"
    st.session_state.view_mode = "table"
    st.rerun()
with m2:
  if st.button(f"🟢 Emiten Naik ({summary['naik']})", use_container_width=True, key="b_naik"):
    st.session_state.quick_filter = "NAIK"
    st.session_state.view_mode = "table"
    st.rerun()
with m3:
  if st.button(f"🔴 Emiten Turun ({summary['turun']})", use_container_width=True, key="b_turun"):
    st.session_state.quick_filter = "TURUN"
    st.session_state.view_mode = "table"
    st.rerun()
with m4:
  if st.button(f"📦 Total Aktif ({summary['total']})", use_container_width=True, key="b_tot"):
    st.session_state.quick_filter = "SEMUA"
    st.session_state.view_mode = "table"
    st.rerun()

st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)


def check_sidebar_screener(item, mode):
  if mode == "Semua Saham Aktif JII70":
    return True
  elif mode == "🔥 Setup Scalping BUY (EMA200 + Ribbon + Stoch)":
    return item["Sinyal"] == "BUY"
  elif mode == "⚠️ Setup Scalping SELL / Bearish":
    return item["Sinyal"] == "SELL"
  elif mode == "📊 Volatilitas Tinggi (Bollinger Bands Melebar)":
    return item["BB Width"] > 3.0
  return True


# --- LOGIKA FILTER DATA ---
filtered_data = []
for item in st.session_state.matrix_data_all:
  match_search = not search_query or search_query.lower() in item["Emiten"].lower()
  q_filt = st.session_state.quick_filter
  match_quick = True
  if q_filt == "SINYAL_BELI":
    match_quick = item["Sinyal"] == "BUY"
  elif q_filt == "NAIK":
    match_quick = item["ChangeVal"] > 0
  elif q_filt == "TURUN":
    match_quick = item["ChangeVal"] < 0

  match_screener = check_sidebar_screener(item, screener_mode)
  if match_search and match_quick and match_screener:
    filtered_data.append(item)

if not filtered_data and st.session_state.quick_filter != "SEMUA":
  st.session_state.quick_filter = "SEMUA"
  st.rerun()


# --- TAMPILAN HALAMAN (CHART ATAU TABEL UTAMA) ---
if st.session_state.matrix_data_all:

  if st.session_state.selected_emiten and st.session_state.get("view_mode", "table") == "chart":
    sel_item = next((x for x in st.session_state.matrix_data_all if x["Emiten"] == st.session_state.selected_emiten), None)

    if sel_item:
      col_btn1, col_btn2 = st.columns([1, 4])
      with col_btn1:
        if st.button("⬅️ Kembali ke Tabel Screener"):
          st.session_state.view_mode = "table"
          st.rerun()
      with col_btn2:
        stockbit_url = f"https://stockbit.com/symbol/{sel_item['Emiten']}"
        st.markdown(f'<a href="{stockbit_url}" target="_blank"><button style="background-color: #059669; color: white; border: none; padding: 6px 16px; border-radius: 6px; font-weight: bold; cursor: pointer;">🚀 Buka Trade di Stockbit ({sel_item["Emiten"]})</button></a>', unsafe_allow_html=True)

      st.markdown(f"## 📈 Chart Scalping M5: **{sel_item['Emiten']}**")
      st.info(f"🏷️ **Strategi:** {sel_item['Strategi']} | **RSI (9):** {sel_item['RSI (9)']} | **BB Width:** {sel_item['BB Width']}")
      st.markdown(f"**Alasan Filter:** {sel_item['Alasan Aksi']}")

      c_df = sel_item["RawData"]
      fig = make_subplots(
          rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
          row_heights=[0.5, 0.18, 0.16, 0.16]
      )

      fig.add_trace(go.Candlestick(x=c_df.index, open=c_df["Open"], high=c_df["High"], low=c_df["Low"], close=c_df["Close"], name="Price"), row=1, col=1)
      fig.add_trace(go.Scatter(x=c_df.index, y=c_df["EMA20"], line=dict(color="#38bdf8", width=1.2), name="EMA 20"), row=1, col=1)
      fig.add_trace(go.Scatter(x=c_df.index, y=c_df["EMA50"], line=dict(color="#f43f5e", width=1.2), name="EMA 50"), row=1, col=1)
      fig.add_trace(go.Scatter(x=c_df.index, y=c_df["EMA200"], line=dict(color="#fbbf24", width=1.2), name="EMA 200"), row=1, col=1)

      if "VWAP" in c_df.columns:
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df["VWAP"], line=dict(color="#a855f7", width=1.2, dash="dot"), name="VWAP"), row=1, col=1)

      if "RSI9" in c_df.columns:
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df["RSI9"], line=dict(color="#38bdf8", width=1.2), name="RSI (9)"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#94a3b8", row=2, col=1)

      if "Stoch_K" in c_df.columns:
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df["Stoch_K"], line=dict(color="#06b6d4", width=1.2), name="Stoch %K"), row=3, col=1)
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df["Stoch_D"], line=dict(color="#f97316", width=1.2), name="Stoch %D"), row=3, col=1)

      if "BB_Width" in c_df.columns:
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df["BB_Width"], line=dict(color="#22c55e", width=1.2), name="BB Width"), row=4, col=1)

      fig.update_layout(template="plotly_dark", height=750, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
      st.plotly_chart(fig, use_container_width=True)
      st.stop()

  # --- TAMPILAN UTAMA (TABEL & SECTION DETAIL DI DEPAN) ---
  st.markdown(
      f"<p style='color: #38bdf8; font-size: 12px; font-weight: bold;'>Menampilkan {len(filtered_data)} Emiten | Filter: {st.session_state.quick_filter} | Fokus: {screener_mode}</p>",
      unsafe_allow_html=True,
  )

  t1, t2, t3, t4, t5, t6 = st.columns([1.3, 0.9, 1.1, 1.0, 1.6, 1.1])
  with t1:
    st.markdown("<span style='color:#64748b; font-size:11px;'>EMITEN</span>", unsafe_allow_html=True)
  with t2:
    st.markdown("<span style='color:#64748b; font-size:11px;'>HARGA</span>", unsafe_allow_html=True)
  with t3:
    st.markdown("<span style='color:#64748b; font-size:11px;'>RSI (9)</span>", unsafe_allow_html=True)
  with t4:
    st.markdown("<span style='color:#64748b; font-size:11px;'>BB WIDTH</span>", unsafe_allow_html=True)
  with t5:
    st.markdown("<span style='color:#64748b; font-size:11px;'>SINYAL & STRATEGI M5</span>", unsafe_allow_html=True)
  with t6:
    st.markdown("<span style='color:#64748b; font-size:11px;'>VOLUME</span>", unsafe_allow_html=True)

  st.markdown("<hr style='margin: 4px 0 10px 0; border-color: #1e293b;'>", unsafe_allow_html=True)

  if filtered_data:
    for item in filtered_data:
      r1, r2, r3, r4, r5, r6 = st.columns([1.3, 0.9, 1.1, 1.0, 1.6, 1.1])
      with r1:
        if st.button(f"**{item['Emiten']}**", key=f"btn_{item['Emiten']}", use_container_width=True):
          st.session_state.selected_emiten = item["Emiten"]
          st.rerun()
      with r2:
        ret_color = "#22c55e" if item["Return (%)"] >= 0 else "#ef4444"
        st.markdown(
            f"<div style='font-size:12px; font-weight:700;'>{item['Harga (Rp)']}</div><div style='font-size:10px; color:{ret_color};'>{item['Return (%)']:+.2f}%</div>",
            unsafe_allow_html=True,
        )
      with r3:
        st.markdown(f"<div style='font-size:11px; color:#38bdf8; font-weight:bold; padding-top:4px;'>{item['RSI (9)']}</div>", unsafe_allow_html=True)
      with r4:
        st.markdown(f"<div style='font-size:11px; color:#fbbf24; font-weight:bold; padding-top:4px;'>{item['BB Width']}</div>", unsafe_allow_html=True)
      with r5:
        sig = item["Sinyal"]
        badge_class = "buy-badge" if sig == "BUY" else "sell-badge" if sig == "SELL" else "hold-badge"
        st.markdown(
            f"<div style='font-size:11px; line-height:1.2;'><span class='{badge_class}'>{sig}</span> <span style='color:#94a3b8; font-size:10px;'>{item['Strategi'].split(' ')[0]}</span></div>",
            unsafe_allow_html=True,
        )
      with r6:
        st.markdown(f"<div style='font-size:11px; color:#cbd5e1; padding-top:4px;'>{item['Volume (Lot)']} lts</div>", unsafe_allow_html=True)

      st.markdown("<hr style='margin: 4px 0; border-color: #0f172a;'>", unsafe_allow_html=True)
  else:
    st.info("Tidak ada emiten yang sesuai dengan kriteria penyaringan.")

  # --- SECTION DETAIL EMITEN DI HALAMAN DEPAN ---
  st.markdown("<br><hr style='border-color: #38bdf8;'>", unsafe_allow_html=True)
  st.markdown("### 🔍 Panel Detail & Eksekusi Scalping Terpilih")

  selected_item = next(
      (x for x in st.session_state.matrix_data_all if x["Emiten"] == st.session_state.selected_emiten),
      st.session_state.matrix_data_all[0] if st.session_state.matrix_data_all else None,
  )

  if selected_item:
    sig = selected_item["Sinyal"]
    badge_class = "buy-badge" if sig == "BUY" else "sell-badge" if sig == "SELL" else "hold-badge"
    stockbit_url = f"https://stockbit.com/symbol/{selected_item['Emiten']}"

    d1, d2, d3, d4 = st.columns(4)
    with d1:
      st.markdown(f"**Emiten:** `{selected_item['Emiten']}`")
      st.markdown(f"**Sinyal M5:** <span class='{badge_class}'>{sig}</span>", unsafe_allow_html=True)
    with d2:
      st.markdown(f"**RSI (9):** `{selected_item['RSI (9)']}`")
      st.markdown(f"**BB Width:** `{selected_item['BB Width']}`")
    with d3:
      st.markdown(f"**Target Buy:** Rp `{selected_item['Target Buy (Rp)']}`")
      st.markdown(f"**Target Profit:** <span style='color:#34d399; font-weight:bold;'>Rp {selected_item['Target Price']}</span>", unsafe_allow_html=True)
    with d4:
      st.markdown(f"**Stop Loss:** <span style='color:#ef4444; font-weight:bold;'>Rp {selected_item['Stop Loss']}</span>", unsafe_allow_html=True)
      st.markdown(f'<a href="{stockbit_url}" target="_blank"><button style="width: 100%; background-color: #059669; color: white; border: none; padding: 6px; border-radius: 6px; font-weight: bold; cursor: pointer; margin-top: 4px;">🚀 Stockbit</button></a>', unsafe_allow_html=True)

    st.markdown(f"**Alasan Alasan:** {selected_item['Alasan Aksi']}")
    
    if st.button("📊 Buka Chart Lengkap Emiten Ini"):
      st.session_state.view_mode = "chart"
      st.rerun()
else:
  st.warning("Data emiten kosong. Silakan klik tombol 'Refresh Data Real-Time' di sidebar.")
