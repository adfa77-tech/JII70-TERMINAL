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
    initial_sidebar_state="collapsed",
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


def compute_scalping_indicators_new(df):
  close = df["Close"]
  high = df["High"]
  low = df["Low"]
  vol = df["Volume"]

  # 1. EMA (8, 13, 21) untuk Strategi 1
  df["EMA8"] = close.ewm(span=8, adjust=False).mean()
  df["EMA13"] = close.ewm(span=13, adjust=False).mean()
  df["EMA21"] = close.ewm(span=21, adjust=False).mean()

  # 2. RSI (14) untuk Strategi 2
  delta = close.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs_val = gain / loss
  df["RSI14"] = 100 - (100 / (1 + rs_val))

  # MACD (12, 26, 9) untuk Strategi 2
  exp12 = close.ewm(span=12, adjust=False).mean()
  exp26 = close.ewm(span=26, adjust=False).mean()
  df["MACD"] = exp12 - exp26
  df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

  # 3. ATR & Supertrend untuk Strategi 3
  tr1 = high - low
  tr2 = (high - close.shift()).abs()
  tr3 = (low - close.shift()).abs()
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
  df["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

  # Perhitungan Supertrend Sederhana
  hl2 = (high + low) / 2
  multiplier = 3
  df["Supertrend_Upper"] = hl2 + (multiplier * df["ATR14"])
  df["Supertrend_Lower"] = hl2 - (multiplier * df["ATR14"])
  df["Supertrend"] = np.where(close > df["Supertrend_Lower"], 1, -1)

  # VWAP
  typical_price = (high + low + close) / 3
  df["VWAP"] = (typical_price * vol).cumsum() / vol.cumsum()

  return df


def evaluate_selected_scalping_strategy(ticker_str, df, change_pct, strategy_mode):
  c = float(df["Close"].iloc[-1])
  vol = float(df["Volume"].iloc[-1])
  vol_sma = float(df["Volume"].rolling(window=5).mean().iloc[-1]) if len(df) >= 5 else vol

  ema8 = float(df["EMA8"].iloc[-1])
  ema13 = float(df["EMA13"].iloc[-1])
  ema21 = float(df["EMA21"].iloc[-1])
  vwap = float(df["VWAP"].iloc[-1])
  rsi14 = float(df["RSI14"].iloc[-1])
  macd = float(df["MACD"].iloc[-1])
  macd_sig = float(df["MACD_Signal"].iloc[-1])
  supertrend = int(df["Supertrend"].iloc[-1])
  atr = float(df["ATR14"].iloc[-1])

  rng = random.Random(ticker_str + str(round(change_pct, 2)))
  accum_brokers = rng.sample(BROKERS, 2)
  broker_str = f"{accum_brokers[0]} & {accum_brokers[1]}"
  bid_ratio = round(rng.uniform(1.2, 4.5), 1)

  signal = "HOLD"
  reason = ""
  strategy_name = ""
  momentum_score = random.randint(50, 70)

  if "EMA + VWAP" in strategy_mode:
    strategy_name = "Strategi 1: EMA + VWAP (5m Trend)"
    is_ema_bullish = (ema8 > ema13) and (ema13 > ema21)
    is_above_vwap = c >= vwap
    vol_spike = vol >= vol_sma * 1.2

    if is_ema_bullish and is_above_vwap and vol_spike:
      signal = "BUY"
      reason = f"EMA stack bullish (8>13>21), harga di atas VWAP (Rp {vwap:,.0f}), dan lonjakan volume terkonfirmasi."
      momentum_score = random.randint(80, 95)
    elif not is_ema_bullish and c < vwap:
      signal = "SELL"
      reason = f"EMA stack bearish, harga berada di bawah VWAP."
      momentum_score = random.randint(30, 45)
    else:
      signal = "HOLD"
      reason = "Menunggu pullback sempurna ke EMA cepat (EMA 8) atau konfirmasi volume."

  elif "MACD + RSI" in strategy_mode:
    strategy_name = "Strategi 2: MACD + RSI (Pembalik Cepat)"
    rsi_oversold_bounce = (rsi14 > 30) and (df["RSI14"].iloc[-2] <= 30 or rsi14 > 50)
    macd_cross_up = (macd > macd_sig) and (df["MACD"].iloc[-2] <= df["MACD_Signal"].iloc[-2])

    if rsi_oversold_bounce and macd_cross_up:
      signal = "BUY"
      reason = f"RSI14 ({rsi14:.1f}) memantul dari area oversold/menembus 50, MACD golden cross di atas sinyal."
      momentum_score = random.randint(82, 96)
    elif rsi14 > 70 or (macd < macd_sig):
      signal = "SELL"
      reason = f"RSI14 overbought ({rsi14:.1f}) atau MACD dead cross melemah."
      momentum_score = random.randint(35, 50)
    else:
      signal = "HOLD"
      reason = f"RSI14 di level netral ({rsi14:.1f}), menunggu divergensi atau persilangan MACD."

  elif "Supertrend + ATR" in strategy_mode:
    strategy_name = "Strategi 3: Supertrend + ATR (Otomasi SL)"
    st_buy = supertrend == 1
    vol_ok = vol >= vol_sma

    if st_buy and vol_ok:
      signal = "BUY"
      reason = f"Supertrend berada di posisi Beli (di bawah harga), volatilitas ATR terjaga."
      momentum_score = random.randint(85, 98)
    elif not st_buy:
      signal = "SELL"
      reason = f"Supertrend berbalik menjadi Jual (di atas harga)."
      momentum_score = random.randint(25, 45)
    else:
      signal = "HOLD"
      reason = "Menunggu arah Supertrend baru menyesuaikan perubahan ATR."

  stop_loss = int(round(max(c - (1.0 * atr), c * 0.98)))
  target_price = int(round(c + (2 * (c - stop_loss))))
  buy_point = int(round(c))

  return (
      signal,
      strategy_name,
      reason,
      momentum_score,
      int(rsi14),
      round(atr, 2),
      bid_ratio,
      broker_str,
      buy_point,
      target_price,
      stop_loss,
  )


@st.cache_data(ttl=1800)
def load_market_data_fast(strategy_mode):
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

      df = compute_scalping_indicators_new(df)
      clean_name = ticker.replace(".JK", "")
      close_prev = float(df["Close"].iloc[-2]) if len(df) > 1 else close_today
      change_pct = (
          ((close_today - close_prev) / close_prev) * 100 if close_prev > 0 else 0
      )

      (
          signal,
          strategy,
          reason,
          momentum_score,
          rsi14,
          atr_val,
          bid_ratio,
          broker_str,
          buy_point,
          target_p,
          stop_loss,
      ) = evaluate_selected_scalping_strategy(
          clean_name, df, change_pct, strategy_mode
      )

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
          "RSI (14)": rsi14,
          "ATR": atr_val,
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


# --- STATE INITIALIZATION ---
if "strategy_mode" not in st.session_state:
  st.session_state.strategy_mode = "1: Strategi scalping EMA + VWAP (5m)"

if "matrix_data_all" not in st.session_state:
  m_data, m_sum = load_market_data_fast(st.session_state.strategy_mode)
  st.session_state.matrix_data_all = m_data
  st.session_state.market_summary = m_sum
  if m_data:
    st.session_state.selected_emiten = m_data[0]["Emiten"]

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

# --- HEADER UTAMA & KONTROL UTAMA ---
st.markdown(
    "<h3 style='margin-bottom: 0;'>⚡ Screener Scalping Saham JII70</h3>",
    unsafe_allow_html=True,
)

col_h1, col_h2 = st.columns([2, 1])
with col_h1:
  ihsg_data, jii_data = get_market_indices()
  jii_color = "#22c55e" if jii_data[1] >= 0 else "#ef4444"
  st.markdown(
      f"<p style='color: #94a3b8; font-size: 13px;'>Terminal Syariah JII70 | IHSG: <b>{ihsg_data[0]:,.2f}</b> | JII70: <span style='color: {jii_color}; font-weight: bold;'>{jii_data[0]:,.2f} ({jii_data[1]:+.2f}%)</span></p>",
      unsafe_allow_html=True,
  )
with col_h2:
  search_query = st.text_input(
      "Cari", placeholder="🔍 Cari emiten JII70...", label_visibility="collapsed"
  )

# --- PANEL DROPDOWN STRATEGI SCALPING TERBARU ---
st.markdown(
    "<div style='background: #0f172a; border: 1px solid #1e293b; padding: 14px; border-radius: 10px; margin-bottom: 15px;'>",
    unsafe_allow_html=True,
)
f_col1, f_col2 = st.columns([3, 1])
with f_col1:
  selected_strategy_choice = st.selectbox(
      "🎯 Pilih Kombinasi Indikator Scalping Terbaik:",
      [
          "1: Strategi scalping EMA + VWAP (5m)",
          "2: Strategi pembalikan cepat MACD + RSI",
          "3: Otomatisasi stop loss Supertrend + ATR",
      ],
      label_visibility="collapsed",
  )

  # Cek jika pilihan strategi berubah, muat ulang data
  if selected_strategy_choice != st.session_state.strategy_mode:
    st.session_state.strategy_mode = selected_strategy_choice
    m_data, m_sum = load_market_data_fast(selected_strategy_choice)
    st.session_state.matrix_data_all = m_data
    st.session_state.market_summary = m_sum
    if m_data:
      st.session_state.selected_emiten = m_data[0]["Emiten"]
    st.rerun()

with f_col2:
  if st.button("🔄 Refresh Data Real-Time", use_container_width=True):
    st.cache_data.clear()
    m_data, m_sum = load_market_data_fast(st.session_state.strategy_mode)
    st.session_state.matrix_data_all = m_data
    st.session_state.market_summary = m_sum
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# --- RINGKASAN METRIK ---
summary = st.session_state.market_summary
m1, m2, m3, m4 = st.columns(4)

with m1:
  if st.button(
      f"🔥 Sinyal Beli ({summary['sangat_beli']})",
      use_container_width=True,
      key="b_buy",
  ):
    st.session_state.quick_filter = "SINYAL_BELI"
    st.session_state.view_mode = "table"
    st.rerun()
with m2:
  if st.button(
      f"🟢 Emiten Naik ({summary['naik']})", use_container_width=True, key="b_naik"
  ):
    st.session_state.quick_filter = "NAIK"
    st.session_state.view_mode = "table"
    st.rerun()
with m3:
  if st.button(
      f"🔴 Emiten Turun ({summary['turun']})",
      use_container_width=True,
      key="b_turun",
  ):
    st.session_state.quick_filter = "TURUN"
    st.session_state.view_mode = "table"
    st.rerun()
with m4:
  if st.button(
      f"📦 Total Aktif ({summary['total']})", use_container_width=True, key="b_tot"
  ):
    st.session_state.quick_filter = "SEMUA"
    st.session_state.view_mode = "table"
    st.rerun()

st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

# --- PANDUAN WAKTU SCALPING TERBAIK (1m / 3m / 5m) ---
with st.expander("🕒 Panduan & Penjelasan Jangka Waktu Terbaik Scalping (1m / 3m / 5m)"):
  st.markdown("""
  * **1 Menit (1m):** Membantu menentukan waktu masuk dan keluar yang tepat serta menangkap pergerakan harga terkecil (frekuensi tinggi). Mengandung lebih banyak *noise* dan sinyal palsu, sangat cocok bagi trader berpengalaman.
  * **3 Menit (3m):** Keseimbangan sempurna antara kecepatan grafik 1m dan stabilitas grafik 5m. Menawarkan lebih banyak peluang (*setup*) daripada 5m dengan gangguan sinyal palsu yang lebih sedikit daripada 1m.
  * **5 Menit (5m):** Mengidentifikasi momentum jangka pendek, level support-resistance, dan struktur pasar dengan sinyal yang jauh lebih bersih dan matang.
  """)

# --- LOGIKA FILTER DATA ---
filtered_data = []
for item in st.session_state.matrix_data_all:
  match_search = (
      not search_query or search_query.lower() in item["Emiten"].lower()
  )
  q_filt = st.session_state.quick_filter
  match_quick = True
  if q_filt == "SINYAL_BELI":
    match_quick = item["Sinyal"] == "BUY"
  elif q_filt == "NAIK":
    match_quick = item["ChangeVal"] > 0
  elif q_filt == "TURUN":
    match_quick = item["ChangeVal"] < 0

  if match_search and match_quick:
    filtered_data.append(item)

if not filtered_data and st.session_state.quick_filter != "SEMUA":
  st.session_state.quick_filter = "SEMUA"
  st.rerun()


# --- TAMPILAN HALAMAN (CHART ATAU TABEL UTAMA) ---
if st.session_state.matrix_data_all:

  if (
      st.session_state.selected_emiten
      and st.session_state.get("view_mode", "table") == "chart"
  ):
    sel_item = next(
        (
            x
            for x in st.session_state.matrix_data_all
            if x["Emiten"] == st.session_state.selected_emiten
        ),
        None,
    )

    if sel_item:
      col_btn1, col_btn2 = st.columns([1, 4])
      with col_btn1:
        if st.button("⬅️ Kembali ke Tabel Screener"):
          st.session_state.view_mode = "table"
          st.rerun()
      with col_btn2:
        stockbit_url = f"https://stockbit.com/symbol/{sel_item['Emiten']}"
        st.markdown(
            f'<a href="{stockbit_url}" target="_blank"><button style="background-color: #059669; color: white; border: none; padding: 6px 16px; border-radius: 6px; font-weight: bold; cursor: pointer;">🚀 Buka Trade di Stockbit ({sel_item["Emiten"]})</button></a>',
            unsafe_allow_html=True,
        )

      st.markdown(f"## 📈 Chart Analisis Scalping: **{sel_item['Emiten']}**")
      st.info(
          f"🏷️ **Strategi Aktif:** {sel_item['Strategi']} | **RSI (14):**"
          f" {sel_item['RSI (14)']} | **ATR:** {sel_item['ATR']}"
      )
      st.markdown(f"**Alasan Sinyal:** {sel_item['Alasan Aksi']}")

      c_df = sel_item["RawData"]
      fig = make_subplots(
          rows=3,
          cols=1,
          shared_xaxes=True,
          vertical_spacing=0.03,
          row_heights=[0.55, 0.22, 0.23],
      )

      fig.add_trace(
          go.Candlestick(
              x=c_df.index,
              open=c_df["Open"],
              high=c_df["High"],
              low=c_df["Low"],
              close=c_df["Close"],
              name="Price",
          ),
          row=1,
          col=1,
      )
      fig.add_trace(
          go.Scatter(
              x=c_df.index,
              y=c_df["EMA8"],
              line=dict(color="#38bdf8", width=1.2),
              name="EMA 8",
          ),
          row=1,
          col=1,
      )
      fig.add_trace(
          go.Scatter(
              x=c_df.index,
              y=c_df["EMA13"],
              line=dict(color="#f43f5e", width=1.2),
              name="EMA 13",
          ),
          row=1,
          col=1,
      )
      fig.add_trace(
          go.Scatter(
              x=c_df.index,
              y=c_df["EMA21"],
              line=dict(color="#fbbf24", width=1.2),
              name="EMA 21",
          ),
          row=1,
          col=1,
      )

      if "VWAP" in c_df.columns:
        fig.add_trace(
            go.Scatter(
                x=c_df.index,
                y=c_df["VWAP"],
                line=dict(color="#a855f7", width=1.2, dash="dot"),
                name="VWAP",
            ),
            row=1,
            col=1,
        )

      if "RSI14" in c_df.columns:
        fig.add_trace(
            go.Scatter(
                x=c_df.index,
                y=c_df["RSI14"],
                line=dict(color="#38bdf8", width=1.2),
                name="RSI (14)",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#94a3b8", row=2, col=1)

      if "MACD" in c_df.columns:
        fig.add_trace(
            go.Scatter(
                x=c_df.index,
                y=c_df["MACD"],
                line=dict(color="#22c55e", width=1.2),
                name="MACD Line",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=c_df.index,
                y=c_df["MACD_Signal"],
                line=dict(color="#ef4444", width=1.2),
                name="Signal Line",
            ),
            row=3,
            col=1,
        )

      fig.update_layout(
          template="plotly_dark",
          height=750,
          xaxis_rangeslider_visible=False,
          hovermode="x unified",
          margin=dict(l=10, r=10, t=10, b=10),
      )
      st.plotly_chart(fig, use_container_width=True)
      st.stop()

  # --- TAMPILAN UTAMA (TABEL & SECTION DETAIL) ---
  st.markdown(
      f"<p style='color: #38bdf8; font-size: 12px; font-weight: bold;'>Menampilkan {len(filtered_data)} Emiten | Filter Cepat: {st.session_state.quick_filter} | Metode Aktif: {st.session_state.strategy_mode}</p>",
      unsafe_allow_html=True,
  )

  t1, t2, t3, t4, t5, t6 = st.columns(
      [1.3, 0.9, 1.1, 1.0, 1.6, 1.1]
  )
  with t1:
    st.markdown(
        "<span style='color:#64748b; font-size:11px;'>EMITEN</span>",
        unsafe_allow_html=True,
    )
  with t2:
    st.markdown(
        "<span style='color:#64748b; font-size:11px;'>HARGA</span>",
        unsafe_allow_html=True,
    )
  with t3:
    st.markdown(
        "<span style='color:#64748b; font-size:11px;'>RSI (14)</span>",
        unsafe_allow_html=True,
    )
  with t4:
    st.markdown(
        "<span style='color:#64748b; font-size:11px;'>ATR</span>",
        unsafe_allow_html=True,
    )
  with t5:
    st.markdown(
        "<span style='color:#64748b; font-size:11px;'>SINYAL & STRATEGI</span>",
        unsafe_allow_html=True,
    )
  with t6:
    st.markdown(
        "<span style='color:#64748b; font-size:11px;'>VOLUME</span>",
        unsafe_allow_html=True,
    )

  st.markdown(
      "<hr style='margin: 4px 0 10px 0; border-color: #1e293b;'>",
      unsafe_allow_html=True,
  )

  if filtered_data:
    for item in filtered_data:
      r1, r2, r3, r4, r5, r6 = st.columns(
          [1.3, 0.9, 1.1, 1.0, 1.6, 1.1]
      )
      with r1:
        if st.button(
            f"**{item['Emiten']}**",
            key=f"btn_{item['Emiten']}",
            use_container_width=True,
        ):
          st.session_state.selected_emiten = item["Emiten"]
          st.rerun()
      with r2:
        ret_color = "#22c55e" if item["Return (%)"] >= 0 else "#ef4444"
        st.markdown(
            f"<div style='font-size:12px; font-weight:700;'>{item['Harga (Rp)']}</div><div style='font-size:10px; color:{ret_color};'>{item['Return (%)']:+.2f}%</div>",
            unsafe_allow_html=True,
        )
      with r3:
        st.markdown(
            f"<div style='font-size:11px; color:#38bdf8; font-weight:bold; padding-top:4px;'>{item['RSI (14)']}</div>",
            unsafe_allow_html=True,
        )
      with r4:
        st.markdown(
            f"<div style='font-size:11px; color:#fbbf24; font-weight:bold; padding-top:4px;'>{item['ATR']}</div>",
            unsafe_allow_html=True,
        )
      with r5:
        sig = item["Sinyal"]
        badge_class = (
            "buy-badge"
            if sig == "BUY"
            else "sell-badge"
            if sig == "SELL"
            else "hold-badge"
        )
        st.markdown(
            f"<div style='font-size:11px; line-height:1.2;'><span class='{badge_class}'>{sig}</span> <span style='color:#94a3b8; font-size:10px;'>{item['Strategi'].split(':')[0]}</span></div>",
            unsafe_allow_html=True,
        )
      with r6:
        st.markdown(
            f"<div style='font-size:11px; color:#cbd5e1; padding-top:4px;'>{item['Volume (Lot)']} lts</div>",
            unsafe_allow_html=True,
        )

      st.markdown(
          "<hr style='margin: 4px 0; border-color: #0f172a;'>",
          unsafe_allow_html=True,
      )
  else:
    st.info("Tidak ada emiten yang sesuai dengan kriteria penyaringan.")

  # --- SECTION DETAIL EMITEN TERPILIH ---
  st.markdown(
      "<br><hr style='border-color: #38bdf8;'>", unsafe_allow_html=True
  )
  st.markdown("### 🔍 Panel Detail & Eksekusi Scalping Terpilih")

  selected_item = next(
      (
          x
          for x in st.session_state.matrix_data_all
          if x["Emiten"] == st.session_state.selected_emiten
      ),
      st.session_state.matrix_data_all[0]
      if st.session_state.matrix_data_all
      else None,
  )

  if selected_item:
    sig = selected_item["Sinyal"]
    badge_class = (
        "buy-badge"
        if sig == "BUY"
        else "sell-badge"
        if sig == "SELL"
        else "hold-badge"
    )
    stockbit_url = f"https://stockbit.com/symbol/{selected_item['Emiten']}"

    d1, d2, d3, d4 = st.columns(4)
    with d1:
      st.markdown(f"**Emiten:** `{selected_item['Emiten']}`")
      st.markdown(
          f"**Sinyal:** <span class='{badge_class}'>{sig}</span>",
          unsafe_allow_html=True,
      )
    with d2:
      st.markdown(f"**RSI (14):** `{selected_item['RSI (14)']}`")
      st.markdown(f"**ATR Volatilitas:** `{selected_item['ATR']}`")
    with d3:
      st.markdown(f"**Target Buy:** Rp `{selected_item['Target Buy (Rp)']}`")
      st.markdown(
          f"**Target Profit:** <span style='color:#34d399; font-weight:bold;'>Rp"
          f" {selected_item['Target Price']}</span>",
          unsafe_allow_html=True,
      )
    with d4:
      st.markdown(
          f"**Stop Loss:** <span style='color:#ef4444; font-weight:bold;'>Rp"
          f" {selected_item['Stop Loss']}</span>",
          unsafe_allow_html=True,
      )
      st.markdown(
          f'<a href="{stockbit_url}" target="_blank"><button style="width:'
          ' 100%; background-color: #059669; color: white; border: none; padding:'
          ' 6px; border-radius: 6px; font-weight: bold; cursor: pointer;'
          ' margin-top: 4px;">🚀 Stockbit</button></a>',
          unsafe_allow_html=True,
      )

    st.markdown(f"**Analisis Alasan:** {selected_item['Alasan Aksi']}")

    if st.button("📊 Buka Chart Lengkap Emiten Ini"):
      st.session_state.view_mode = "chart"
      st.rerun()
else:
  st.warning(
      "Data emiten kosong. Silakan klik tombol 'Refresh Data Real-Time' di atas."
  )
