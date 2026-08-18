import os
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI Trading Intelligence V1 - Real Account", layout="wide")

# =======================================================
# INTERFACE SIDEBAR: GABUNGAN PENGATURAN & PROMPT KUSTOM
# =======================================================
with st.sidebar:
    st.header("⚙️ Pengaturan AI & Strategi")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Model", value="gpt-4o")
    symbol = st.text_input("Symbol", value="EUR/USD")
    timeframe = st.text_input("Timeframe", value="H1")
    
    st.subheader("📝 Kustomisasi Prompt Strategi AI")
    user_custom_prompt = st.text_area(
        label="Edit Prompt Aturan Analisis AI Di Sini:",
        value="""You are Trading Intelligence Analyst V1 untuk Akun Riil.

Purpose: Menganalisis data pasar secara objektif untuk memberikan keputusan trading yang valid dengan konfirmasi kuat.

Aturan Analisis & Indikator Teknikal:
1. Analisis Tren Dominan (Naik/Turun). Jika Sideways, wajib mengambil keputusan: NO TRADE.
2. Market Structure & ICT (Deteksi Break of Structure / BOS atau Change of Character / ChoCH).
3. Fibonacci Retracement (Deteksi area entri Golden Ratio 61.8% - 78.6%).

ATURAN ENTRY & FILTER KETAT (ANTI-SIDEWAYS):
- Rekomendasi ENTRY (BUY/SELL) HANYA BOLEH direkomendasikan jika minimal 2 dari 3 indikator di atas memberikan konfirmasi searah!
- Jika HANYA 1 INDIKATOR yang berhasil, atau pasar terdeteksi Sideways, Anda WAJIB membatalkan transaksi dan mengeluarkan status: NO TRADE (Konfirmasi Lemah).
- Jika Tren NAIK: cari momentum BUY. Jika Tren TURUN: cari posisi SELL.

MANAJEMEN RISIKO REAL (RASIO FIXED 1:2):
- Setiap keputusan entry wajib menyertakan target harga yang jelas dengan proporional nilai finansial berikut:
  * Stop Loss (SL): Maksimal risiko kerugian dipatok setara Rp 50.000,-
  * Take Profit (TP): Target keuntungan wajib dua kali lipat yaitu Rp 100.000,-

Output Format:
# MARKET INTELLIGENCE REPORT
## Data Quality
## Market Regime & Trend Status
## Structure & Confirmation Analysis
## Strategy Decision (BUY / SELL / NO TRADE)
## Risk Management Plan (Target SL & TP)
""",
        height=400
    )

# =======================================================
# INDIKATOR TEKNIKAL BAWAAN YANG DISEMPURNAKAN
# =======================================================
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df):
    df = df.copy()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
    
    high_max = df["high"].rolling(window=20).max()
    low_min = df["low"].rolling(window=20).min()
    diff = high_max - low_min
    df["fibo_618"] = high_max - (diff * 0.618)
    df["fibo_786"] = high_max - (diff * 0.786)
    return df

def validate_csv(df):
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns.str.lower())
    return missing

# =======================================================
# DASHBOARD UTAMA (STREAMLIT)
# =======================================================
st.title("📈 AI Trading Intelligence - V1 Real Account Ready")
st.caption("Gabungan Analisis Strategi Lama dengan Aturan Manajemen Risiko Baru.")

uploaded = st.file_uploader("Upload CSV Data Pasar (OHLC)", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
    raw.columns = [c.strip().lower() for c in raw.columns]
    
    missing = validate_csv(raw)
    if missing:
        st.error(f"Berkas CSV kekurangan kolom wajib: {', '.join(sorted(missing))}")
        st.stop()
        
    for c in ["open", "high", "low", "close"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    
    df_berindikator = add_indicators(raw)
    data_terakhir = df_berindikator.tail(5).to_string()
    
    st.success("Data pasar berhasil dimuat dan semua indikator gabungan telah dihitung!")
    st.dataframe(df_berindikator.tail(10))
    
    if st.button("🚀 Jalankan Analisis AI Gabungan"):
        if not api_key:
            st.error("Silakan masukkan OpenAI API Key Anda di sidebar terlebih dahulu!")
            st.stop()
            
        try:
            client = OpenAI(api_key=api_key)
            
            prompt_eksekusi = f"""
Data Pasar dan Indikator Terakhir (5 Baris Terbaru):
{data_terakhir}

Lakukan analisis berdasarkan data di atas menggunakan Symbol: {symbol} pada Timeframe: {timeframe}. 
Keluarkan keputusan trading yang disiplin berdasarkan instruksi sistem Anda.
"""
            
            with st.spinner("AI sedang memproses penalaran logika strategi..."):
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": user_custom_prompt},
                        {"role": "user", "content": prompt_eksekusi}
                    ],
                    temperature=0.2
                )
                
            st.subheader("📊 Hasil Laporan & Keputusan AI")
            st.markdown(response.choices.message.content)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis API: {str(e)}")
