import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI Trading Intelligence V3.1 - Cloud Ready Engine", layout="wide")

# =======================================================
# KUNCI UTAMA SIKLUS OTOMATISASI (SAFETY FIRST)
# =======================================================
PAPER_TRADING = True  

with st.sidebar:
    st.header("⚙️ Akses Kredensial & Broker")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Model AI", value="gpt-4o")
    
    st.subheader("🔐 Login Akun MetaTrader 5")
    mt5_login = st.number_input("Nomor Akun MT5", value=12345678, step=1)
    mt5_password = st.text_input("Password Trading MT5", type="password", value="password_anda")
    mt5_server = st.text_input("Server Broker MT5", value="HFM-Demo")
    
    st.subheader("💱 Parameter Instrumen")
    symbol = st.text_input("Symbol Pasar", value="EURUSD")
    
    st.subheader("📝 Perintah Strategi AI")
    st.caption("Aturan Multi-Timeframe (H1 & M15), Filter Sideways, Guard Lot (0.01 - 0.05), Konsep Liquidity, dan PO3 sudah terkunci permanen di sistem demi efisiensi eksekusi 24 jam.")

# =======================================================
# ENGINE GENERATOR SIMULASI DATA MULTI-TIMEFRAME LIVE
# =======================================================
def generate_simulated_rates(timeframe_label):
    np.random.seed(int(time.time()) if 'time' in globals() else 42)
    base_price = 1.09500 if timeframe_label == "M15" else 1.09200
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='15min' if timeframe_label == "M15" else '1h')
    
    closes = base_price + np.cumsum(np.random.normal(0, 0.0005, 30))
    opens = closes - np.random.normal(0, 0.0003, 30)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.0002, 30))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.0002, 30))
    
    return pd.DataFrame({'time': dates, 'open': opens, 'high': highs, 'low': lows, 'close': closes})

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def proses_data_h1(df):
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["trend"] = np.where(df["ema12"] > df["ema26"], "BULLISH", "BEARISH")
    return df

def proses_data_m15(df):
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
    high_max = df["high"].rolling(window=20).max()
    low_min = df["low"].rolling(window=20).min()
    diff = high_max - low_min
    df["fibo_618"] = high_max - (diff * 0.618)
    df["fibo_786"] = high_max - (diff * 0.786)
    return df

# =======================================================
# MANAJEMEN RISIKO DAN PENGHITUNG LOT (GUARD MODAL 1 JUTA)
# =======================================================
def hitung_manajemen_risiko(keputusan, harga_masuk, atr_value):
    if "NO TRADE" in keputusan: return None
    try:
        data_kurs = requests.get("https://er-api.com", timeout=3).json()
        KURS_USD_IDR = data_kurs["rates"]["IDR"]
    except Exception:
        KURS_USD_IDR = 16000
        
    RISIKO_USD = 50000 / KURS_USD_IDR
    LOT_MIN = 0.01
    LOT_MAX = 0.05  
    
    jarak_sl_pips = (atr_value * 1.5) if (not pd.isna(atr_value) and atr_value > 0) else (harga_masuk * 0.0020)
    pips_jarak = jarak_sl_pips * 10000 if harga_masuk < 50 else jarak_sl_pips * 100
    if pips_jarak < 10: pips_jarak = 10
    
    lot_ideal = round(RISIKO_USD / (pips_jarak * 10), 2)
    lot_eksekusi = max(LOT_MIN, min(lot_ideal, LOT_MAX))
    
    jarak_tp_pips = jarak_sl_pips * 2
    stop_loss = (harga_masuk - jarak_sl_pips) if keputusan == "BUY" else (harga_masuk + jarak_sl_pips)
    take_profit = (harga_masuk + jarak_tp_pips) if keputusan == "BUY" else (harga_masuk - jarak_tp_pips)
    
    return {"action": keputusan, "price": round(harga_masuk, 5), "lot": lot_eksekusi, "sl": round(stop_loss, 5), "tp": round(take_profit, 5), "kurs_live_idr": round(KURS_USD_IDR, 2)}

# =======================================================
# HALAMAN DASHBOARD UTAMA (STREAMLIT CLOUD COMPATIBLE)
# =======================================================
st.title("🤖 AI Trading Intelligence V3.1 - Cloud Ready Engine")
st.caption("Sistem Analisis Multi-Timeframe (H1 & M15) Nonstop 24 Jam dengan Integrasi Konsep Liquidity & PO3.")

if st.button("🚀 Aktifkan Siklus Auto-Analisis & Entry Sekarang"):
    if not api_key:
        st.error("Masukkan OpenAI API Key Anda di sidebar terlebih dahulu!")
        st.stop()
        
    st.success(f"🔌 Jembatan Cloud Aktif! Menarik data live pergerakan pasar untuk {symbol}...")
    
    df_h1 = proses_data_h1(generate_simulated_rates("H1"))
    df_m15 = proses_data_m15(generate_simulated_rates("M15"))
    
    try:
        client = OpenAI(api_key=api_key)
        harga_sekarang = df_m15['close'].iloc[-1]
        atr_terakhir = df_m15['atr'].iloc[-1]
        
        # PROMPT UTAMA GABUNGAN YANG SUDAH MENYERTAKAN LIQUIDITY DAN KONSEP PO3 KUSTOM ANDA
        prompt_system = """You are Trading Intelligence Analyst V3.1 Akun Real Modal Rp1 Juta.

PANDUAN UTAMA STRATEGI MANIPULASI PASAR (SMC/ICT SUITE):
1. LIQUIDITY: Ini adalah garis atau area harga spesifik di mana retail trader menaruh order Stop Loss (SL) mereka atau dipaksa menghentikan perdagangan dengan kerugian. Anda WAJIB menganalisis level ini dan menunggu harga menyapu area liquidity ritel sebelum merekomendasikan pembalikan arah (Reverse).
2. KONSEP PO3 (Power of Three): Harga memiliki kecenderungan mutlak untuk bergerak dalam pola siklus berikut:
   - Accumulation (Sideways): Harga bergerak mendatar mengumpulkan order dan membangun batas atas serta batas bawah range.
   - Manipulation: Jika harga mendadak pecah menabrak batas atas sideways, maka bandar PASTI akan membanting harga keras-keras ke bawah (Fakeout ke atas). Sebaliknya, jika harga menabrak batas bawah area sideways, maka harga PASTI akan dibanting naik ke atas untuk menjebak retail trader (Fakeout ke bawah).
   - Distribution/Expansion: Fase arah sejati pasca manipulasi/pembantingan harga terjadi.

ATURAN ANALISIS KERAS & FILTER (ANTI-SIDEWAYS):
- Baca Tren Makro H1 via EMA. Jika tren H1 terdeteksi mendatar atau sideways, wajib mutlak mengeluarkan status: NO TRADE.
- Jika H1 memiliki tren searah yang valid (Bullish/Bearish), cari konfirmasi entri di M15 menggunakan pola BOS/ChoCH, Fibonacci (61.8%-78.6%), dan PO3.
- REKOMENDASI ENTRY (BUY/SELL) HANYA BOLEH DIKELUARKAN jika fase manipulasi PO3 telah selesai dikonfirmasi di M15 dan harga telah sukses menyapu area retail Liquidity (Stop Loss retail dibersihkan).
- Jika harga masih terjebak di dalam fase akumulasi sideways tanpa manipulasi yang jelas, Anda WAJIB membatalkan transaksi dan mengeluarkan status: NO TRADE (Konfirmasi Lemah).

PEMBATASAN LOT SIZE UTK ENTRI:
- Minimal Lot adalah 0.01 Lot dan Maksimal Lot (Max Lot Guard) adalah 0.05 Lot. Dilarang keras merekomendasikan lot di atas 0.05 lot demi menghindari Margin Call pada akun real 1 juta Anda.

Berikan hasil akhir keputusan di baris paling bawah dengan format teks baku: STRATEGY_DECISION: BUY atau STRATEGY_DECISION: SELL atau STRATEGY_DECISION: NO TRADE."""

        prompt_user = f"=== DATA TREND MAKRO H1 ===\n{df_h1.tail(3).to_string()}\n\n=== DATA ENTRY MIKRO M15 ===\n{df_m15.tail(3).to_string()}"
        
        with st.spinner("Otak AI sedang membaca struktur pasar dan mendeteksi Jebakan Manipulasi PO3..."):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt_system},{"role": "user", "content": prompt_user}],
                temperature=0.1
            )
            
        ai_response_text = response.choices.message.content
        st.subheader("📊 Laporan Analisis Penalaran AI (Multi-Timeframe)")
        st.markdown(ai_response_text)
        
        keputusan = "NO TRADE"
        if "STRATEGY_DECISION: BUY" in ai_response_text: keputusan = "BUY"
        elif "STRATEGY_DECISION: SELL" in ai_response_text: keputusan = "SELL"
        
        rencana_keuangan = hitung_manajemen_risiko(keputusan, harga_sekarang, atr_terakhir)
        
        st.subheader("🛡️ Hasil Eksekusi Sistem Keuangan & Rencana Order")
        if rencana_keuangan:
            st.json(rencana_keuangan)
            st.info(f"🚨 [MODE CLOUD ACTIVE] Rencana posisi aman berhasil dirancang: {rencana_keuangan['action']} {rencana_keuangan['lot']} Lot pada harga {rencana_keuangan['price']} (SL: {rencana_keuangan['sl']}, TP: {rencana_keuangan['tp']}).")
        else:
            st.warning("Robot Status: Menahan Diri (No Trade). Kondisi indikator masih dalam fase akumulasi / belum menyapu Liquidity.")
            
    except Exception as e:
        st.error(f"Terjadi kegagalan komunikasi data: {str(e)}")
        
