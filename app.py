import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import MetaTrader5 as mt5

st.set_page_config(page_title="AI Trading Intelligence V3 - Auto MT5", layout="wide")

# =======================================================
# KUNCI UTAMA SIKLUS OTOMATISASI (SAFETY FIRST)
# =======================================================
PAPER_TRADING = True  # TETAPKAN True UNTUK AKUN DEMO. UBAH KE False JIKA SUDAH SIAP DI AKUN REAL!

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
    st.caption("Aturan Multi-Timeframe (H1 & M15), Filter Sideways, Guard Lot (0.01 - 0.05), dan RR 1:2 sudah terkunci permanen di sistem demi efisiensi eksekusi 24 jam.")

# =======================================================
# INDIKATOR TEKNIKAL & MULTI-TIMEFRAME ENGINE
# =======================================================
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

def proses_data_h1(rates):
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["trend"] = np.where(df["ema12"] > df["ema26"], "BULLISH", "BEARISH")
    return df

def proses_data_m15(rates):
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
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
    LOT_MAX = 0.05  # Kunci Safety Lot Akun Rp 1 Juta Anda
    
    jarak_sl_pips = (atr_value * 1.5) if (not pd.isna(atr_value) and atr_value > 0) else (harga_masuk * 0.0020)
    pips_jarak = jarak_sl_pips * 10000 if harga_masuk < 50 else jarak_sl_pips * 100
    if pips_jarak < 10: pips_jarak = 10
    
    lot_ideal = round(RISIKO_USD / (pips_jarak * 10), 2)
    lot_eksekusi = max(LOT_MIN, min(lot_ideal, LOT_MAX))
    
    jarak_tp_pips = jarak_sl_pips * 2
    stop_loss = (harga_masuk - jarak_sl_pips) if keputusan == "BUY" else (harga_masuk + jarak_sl_pips)
    take_profit = (harga_masuk + jarak_tp_pips) if keputusan == "BUY" else (harga_masuk - jarak_tp_pips)
    
    return {"action": keputusan, "price": harga_masuk, "lot": lot_eksekusi, "sl": round(stop_loss, 5), "tp": round(take_profit, 5), "kurs": round(KURS_USD_IDR, 2)}

# =======================================================
# ROBOT EKSEKUSI TOMBOL ORDER LANGSUNG KE BROKER MT5
# =======================================================
def eksekusi_order_mt5(order_plan, symbol_name):
    if not order_plan: return "Sistem Menahan Diri (No Trade)."
    
    tipe_order = mt5.ORDER_TYPE_BUY if order_plan["action"] == "BUY" else mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_name,
        "volume": order_plan["lot"],
        "type": tipe_order,
        "price": order_plan["price"],
        "sl": order_plan["sl"],
        "tp": order_plan["tp"],
        "deviation": 20,
        "magic": 202608,
        "comment": "AI Trading Intelligence V3",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    if PAPER_TRADING:
        return f"🚨 [MODE DEMO AKTIF] Rencana order berhasil dibuat oleh AI: {order_plan['action']} {order_plan['lot']} Lot pada harga {order_plan['price']} (SL: {order_plan['sl']}, TP: {order_plan['tp']})"
        
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return f"❌ Eksekusi Gagal! Kode Eror Broker: {result.retcode}"
    return f"✅ Sukses! Order Akun Real Terbuka Otomatis: {order_plan['action']} {result.volume} Lot pada harga {result.price}"

# =======================================================
# HALAMAN DASHBOARD UTAMA (STREAMLIT)
# =======================================================
st.title("🤖 AI Trading Intelligence V3 - Auto Multi-Timeframe")
st.caption("Sistem Robot Otomatis: Menarik Data Live Broker ➔ Analisis MTF (H1 & M15) ➔ Auto-Entry.")

if st.button("🚀 Aktifkan Siklus Auto-Analisis & Entry Sekarang"):
    if not api_key:
        st.error("Masukkan OpenAI API Key Anda di sidebar terlebih dahulu!")
        st.stop()
        
    if not mt5.initialize():
        st.error(f"Gagal terhubung ke aplikasi MetaTrader 5. Pastikan software MT5 di komputer/VPS Anda sudah terbuka!")
        st.stop()
        
    authorized = mt5.login(int(mt5_login), password=mt5_password, server=mt5_server)
    if not authorized:
        st.error(f"Gagal Login ke Akun Broker. Periksa kembali nomor akun, password, dan server Anda! Code: {mt5.last_error()}")
        mt5.shutdown()
        st.stop()
        
    st.success(f"🔌 Berhasil Terhubung ke Akun Broker MT5! Memulai pemindaian pasar otomatis...")
    
    rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 30)
    rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 30)
    
    if rates_h1 is None or rates_m15 is None:
        st.error(f"Gagal menarik data harga untuk symbol {symbol}. Pastikan nama pasarnya sudah benar!")
        mt5.shutdown()
        st.stop()
        
    df_h1 = proses_data_h1(rates_h1)
    df_m15 = proses_data_m15(rates_m15)
    
    try:
        client = OpenAI(api_key=api_key)
        harga_sekarang = df_m15['close'].iloc[-1]
        atr_terakhir = df_m15['atr'].iloc[-1]
        
        prompt_system = """You are Trading Intelligence Analyst V3 Akun Real Modal Rp1 Juta.
Aturan Analisis Keras:
1. Baca Tren Makro H1 via EMA. Jika Sideways wajib mutlak mengeluarkan status NO TRADE.
2. Jika H1 searah (Bullish/Bearish), cari konfirmasi entri di M15 menggunakan pola BOS, ICT, dan Fibonacci (61.8%-78.6%).
3. Entry minimal didukung oleh 2 dari 3 konfirmasi di M15. Jika hanya 1 konfirmasi, wajib batalkan dan keluarkan status NO TRADE.
4. PEMBATASAN LOT SIZE UTK ENTRI: Minimal Lot adalah 0.01 Lot dan Maksimal Lot (Max Lot Guard) adalah 0.05 Lot. Dilarang keras merekomendasikan lot di atas 0.05 lot demi menghindari Margin Call.
5. Berikan hasil akhir keputusan di baris paling bawah dengan format teks baku: STRATEGY_DECISION: BUY atau STRATEGY_DECISION: SELL atau STRATEGY_DECISION: NO TRADE."""

        prompt_user = f"=== DATA TREND MAKRO H1 ===\n{df_h1.tail(3).to_string()}\n\n=== DATA ENTRY MIKRO M15 ===\n{df_m15.tail(3).to_string()}"
        
        with st.spinner("Otak AI sedang mensinkronisasikan data pasar H1 & M15..."):
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
        
        st.subheader("🛡️ Hasil Eksekusi Sistem Keuangan & Robot Broker")
        if rencana_keuangan:
            st.json(rencana_keuangan)
            status_eksekusi = eksekusi_order_mt5(rencana_keuangan, symbol)
            st.info(status_eksekusi)
        else:
            st.warning("Robot Status: Menahan Diri (No Trade). Perintah order dibatalkan demi keamanan dana.")
            
    except Exception as e:
        st.error(f"Terjadi kegagalan komunikasi data: {str(e)}")
        
    finally:
        mt5.shutdown()
  
