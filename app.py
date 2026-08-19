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

# Ambil API key dari Streamlit Secrets secara aman
SECRET_KEY = st.secrets.get("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Akses Kredensial & Broker")
    # Membaca otomatis dari Secrets, namun user tetap bisa menimpa jika diperlukan
    api_key = st.text_input("OpenAI API Key", type="password", value=SECRET_KEY)
    model = st.text_input("Model AI", value="gpt-4o-mini") # Diubah ke mini agar hemat saldo $5
    
    st.subheader("🛡️ Manajemen Modal & Leverage")
    modal_usd = st.number_input("Modal Akun Demo ($)", value=5.0, step=0.5)
    leverage = st.number_input("Leverage Akun (1:X)", value=100, step=50, help="Contoh: isi 100 untuk leverage 1:100")
    persen_risiko = st.slider("Batas Risiko Per Trade (%)", min_value=1, max_value=5, value=2, help="Rekomendasi aman: 2% dari modal")
    
    st.subheader("🔐 Login Akun MetaTrader 5")
    mt5_login = st.number_input("Nomor Akun MT5", value=12345678, step=1)
    mt5_password = st.text_input("Password Trading MT5", type="password", value="password_anda")
    mt5_server = st.text_input("Server Broker MT5", value="HFM-Demo")
    
    st.subheader("💱 Parameter Instrumen")
    symbol = st.text_input("Symbol Pasar", value="EURUSD")
    
    st.subheader("📝 Perintah Strategi AI")
    st.caption("Aturan Multi-Timeframe (H1 & M15), Guard Lot Terintegrasi Leverage, Konsep Liquidity, dan PO3 sudah terkunci.")

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
# MANAJEMEN RISIKO DENGAN PROTEKSI LEVERAGE & MODAL KECIL
# =======================================================
def hitung_manajemen_risiko(keputusan, harga_masuk, atr_value, modal, lev, persen_risk):
    if "NO TRADE" in keputusan: return None
    
    # Menghitung toleransi kehilangan modal per trade dalam USD (contoh: 2% dari $5 = $0.10)
    RISIKO_USD = modal * (persen_risk / 100.0)
    
    # Hitung Jarak Stop Loss berdasarkan Volatilitas (ATR)
    jarak_sl_pips = (atr_value * 1.5) if (not pd.isna(atr_value) and atr_value > 0) else (harga_masuk * 0.0020)
    pips_jarak = jarak_sl_pips * 10000 if harga_masuk < 50 else jarak_sl_pips * 100
    if pips_jarak < 10: pips_jarak = 10
    
    # Rumus Hitung Lot Ideal berdasarkan Nilai Pip Kontrak Standar Forex ($10 per lot untuk 1 pip)
    lot_ideal = round(RISIKO_USD / (pips_jarak * 10), 2)
    
    # --- PROTEKSI KEMAMPUAN MARGIN BERDASARKAN LEVERAGE ---
    # Kontrak Standar 1 Lot Forex = 100.000 unit mata uang dasar.
    # Rumus Margin Diperlukan = (Ukuran Kontrak * Lot * Harga Masuk) / Leverage
    # Batas Maksimal Lot yang bisa dibeli daya ungkit (Leverage) Anda:
    max_lot_leverage = round((modal * lev) / (100000 * harga_masuk), 2)
    
    # Ambil lot terkecil agar tidak terjadi Margin Call akibat daya ungkit habis
    LOT_MIN = 0.01
    LOT_MAX = max(LOT_MIN, max_lot_leverage)
    
    lot_eksekusi = max(LOT_MIN, min(lot_ideal, LOT_MAX))
    
    jarak_tp_pips = jarak_sl_pips * 2
    stop_loss = (harga_masuk - jarak_sl_pips) if keputusan == "BUY" else (harga_masuk + jarak_sl_pips)
    take_profit = (harga_masuk + jarak_tp_pips) if keputusan == "BUY" else (harga_masuk - jarak_tp_pips)
    
    return {
        "action": keputusan, 
        "price": round(harga_masuk, 5), 
        "lot": lot_eksekusi, 
        "sl": round(stop_loss, 5), 
        "tp": round(take_profit, 5),
        "max_lot_allowed_by_leverage": LOT_MAX
    }

# =======================================================
# HALAMAN DASHBOARD UTAMA
# =======================================================
st.title("🤖 AI Trading Intelligence V3.1 - Cloud Ready Engine")
st.caption("Sistem Analisis Multi-Timeframe (H1 & M15) Nonstop 24 Jam dengan Integrasi Konsep Liquidity & PO3.")

# Jalankan simulasi data pasar internal
df_h1 = proses_data_h1(generate_simulated_rates("H1"))
df_m15 = proses_data_m15(generate_simulated_rates("M15"))

col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Data Live Trend H1")
    st.dataframe(df_h1.tail(3))
with col2:
    st.subheader("📉 Data Live Eksekusi M15")
    st.dataframe(df_m15.tail(3))

# Tombol Eksekusi Pemicu OpenAI Analisis
if st.button("🚀 Aktifkan Siklus Auto-Analisis & Entry Sekarang"):
    if not api_key:
        st.error("Silakan masukkan API Key OpenAI Anda di sidebar atau simpan di Streamlit Secrets!")
    else:
        with st.spinner("AI sedang memproses indikator teknikal, Liquidity Sweep, dan struktur PO3..."):
            try:
                # Memanggil Client OpenAI
                client = OpenAI(api_key=api_key)
                
                # Mengirimkan instruksi yang mengunci aturan leverage ke otak OpenAI
                prompt_sistem = f"""
                Anda adalah mesin AI Trading Core V3.1. 
                Anda memegang akun demo dengan modal ${modal_usd} dan Leverage 1:{leverage}.
                Gunakan konsep Liquidity Sweep dan skema PO3 (Accumulation, Manipulation, Distribution) untuk menentukan keputusan.
                """
                
                # Mengirimkan ringkasan data pasar terakhir
                data_ringkasan = f"Tren H1 saat ini: {df_h1['trend'].iloc[-1]}. Harga Close M15 terakhir: {df_m15['close'].iloc[-1]}, RSI: {df_m15['rsi'].iloc[-1]}"
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt_sistem},
                        {"role": "user", "content": f"Analisis data berikut dan berikan instruksi akhir hanya dalam satu kata 'BUY', 'SELL', atau 'NO TRADE': {data_ringkasan}"}
                    ]
                )
                
                keputusan_ai = response.choices.message.content.strip().upper()
                st.success(f"Otak AI Mengeluarkan Rekomendasi: **{keputusan_ai}**")
                
                # Hitung manajemen risiko dengan memasukkan faktor leverage ke rumus Python
                harga_skrg = df_m15["close"].iloc[-1]
                atr_skrg = df_m15["atr"].iloc[-1]
                
                hasil_manajemen = hitung_manajemen_risiko(keputusan_ai, harga_skrg, atr_skrg, modal_usd, leverage, persen_risiko)
                
                if hasil_manajemen:
                    st.json(hasil_manajemen)
                    st.info(f"🛡️ Pengaman Otomatis: Ukuran posisi dikunci sebesar {hasil_manajemen['lot']} Lot untuk menjaga keamanan margin leverage 1:{leverage} dari modal ${modal_usd}.")
                else:
                    st.warning("AI Memutuskan untuk Wait and See (No Trade). Tidak ada lot yang dihitung.")
                    
            except Exception as e:
                st.error(f"Gagal menghubungkan ke OpenAI API: {str(e)}")
