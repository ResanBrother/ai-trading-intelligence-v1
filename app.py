import os
import time
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# OPTIONAL / EXTERNAL
# ============================================================

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Trading Intelligence V4",
    layout="wide"
)


# ============================================================
# GLOBAL CONFIG
# ============================================================

DB_FILE = "ai_trading.db"

DEFAULT_SYMBOLS = [
    "EURUSD",
    "EURUSDm",
    "EURUSD.",
    "EURUSD.a",
    "XAUUSD",
    "XAUUSDm",
    "XAUUSD.",
    "GOLD",
    "GOLDm"
]

MAGIC_NUMBER = 26082026


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket INTEGER,
            symbol TEXT,
            decision TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            volume REAL,

            h1_trend TEXT,
            rsi REAL,
            atr REAL,
            ema12 REAL,
            ema26 REAL,

            spread REAL,
            timestamp TEXT,

            result REAL DEFAULT NULL,
            outcome INTEGER DEFAULT NULL,

            notes TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT,
            created_at TEXT,
            trades_used INTEGER,
            win_rate REAL,
            notes TEXT,
            active INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


init_database()


# ============================================================
# MT5 CONNECTION
# ============================================================

def connect_mt5(login, password, server, terminal_path=""):

    if mt5 is None:
        return False, "Package MetaTrader5 belum terinstall."

    try:

        if terminal_path.strip():

            ok = mt5.initialize(
                terminal_path,
                login=int(login),
                password=password,
                server=server,
                timeout=60000
            )

        else:

            ok = mt5.initialize(
                login=int(login),
                password=password,
                server=server,
                timeout=60000
            )

        if not ok:
            return False, f"MT5 initialize gagal: {mt5.last_error()}"

        return True, "MT5 berhasil terhubung."

    except Exception as e:

        return False, str(e)


# ============================================================
# SYMBOL DISCOVERY
# ============================================================

def find_broker_symbol(preferred_symbol):

    if mt5 is None:
        return None

    candidates = []

    preferred_symbol = preferred_symbol.strip()

    candidates.append(preferred_symbol)

    if preferred_symbol.upper() == "EURUSD":

        candidates += [
            "EURUSD",
            "EURUSDm",
            "EURUSD.",
            "EURUSD.a",
            "EURUSD_i"
        ]

    elif preferred_symbol.upper() in ["XAUUSD", "GOLD"]:

        candidates += [
            "XAUUSD",
            "XAUUSDm",
            "XAUUSD.",
            "XAUUSD.a",
            "GOLD",
            "GOLDm"
        ]

    # Direct check
    for symbol in candidates:

        info = mt5.symbol_info(symbol)

        if info is not None:

            if not info.visible:
                mt5.symbol_select(symbol, True)

            return symbol

    # Fallback: search all broker symbols
    try:

        symbols = mt5.symbols_get()

        if symbols:

            base = preferred_symbol.upper()

            for s in symbols:

                name = s.name.upper()

                if base == "EURUSD" and "EURUSD" in name:
                    mt5.symbol_select(s.name, True)
                    return s.name

                if base in ["XAUUSD", "GOLD"]:
                    if "XAUUSD" in name or "GOLD" in name:
                        mt5.symbol_select(s.name, True)
                        return s.name

    except Exception:
        pass

    return None


# ============================================================
# LIVE TICK
# ============================================================

def get_live_tick(symbol):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return None

    return {
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "spread": float(tick.ask - tick.bid),
        "time": datetime.fromtimestamp(
            tick.time,
            tz=timezone.utc
        )
    }


# ============================================================
# REAL MARKET DATA
# ============================================================

def get_rates(symbol, timeframe, bars=300):

    rates = mt5.copy_rates_from_pos(
        symbol,
        timeframe,
        0,
        bars
    )

    if rates is None:
        raise RuntimeError(
            f"Gagal mengambil data {symbol}: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)

    if df.empty:
        raise RuntimeError("Data market kosong.")

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    return result


def atr(df, period=14):

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def process_h1(df):

    df = df.copy()

    df["ema12"] = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["ema26"] = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema200"] = df["close"].ewm(
        span=200,
        adjust=False
    ).mean()

    df["trend"] = np.where(
        df["ema50"] > df["ema200"],
        "BULLISH",
        "BEARISH"
    )

    return df


def process_m15(df):

    df = df.copy()

    df["rsi"] = rsi(df["close"])

    df["atr"] = atr(df)

    df["ema20"] = df["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    high_max = df["high"].rolling(20).max()
    low_min = df["low"].rolling(20).min()

    diff = high_max - low_min

    df["fibo_618"] = high_max - diff * 0.618
    df["fibo_786"] = high_max - diff * 0.786

    return df


# ============================================================
# AI ANALYSIS
# ============================================================

def ai_analysis(
    api_key,
    model,
    symbol,
    h1,
    m15,
    tick
):

    if not api_key:
        raise RuntimeError("OpenAI API Key belum diisi.")

    if OpenAI is None:
        raise RuntimeError(
            "Package openai belum terinstall."
        )

    client = OpenAI(api_key=api_key)

    h1_last = h1.iloc[-1]
    m15_last = m15.iloc[-1]

    market_data = {

        "symbol": symbol,

        "bid": tick["bid"],
        "ask": tick["ask"],
        "spread": tick["spread"],

        "h1_trend": h1_last["trend"],

        "h1_close": float(h1_last["close"]),
        "h1_ema12": float(h1_last["ema12"]),
        "h1_ema26": float(h1_last["ema26"]),
        "h1_ema50": float(h1_last["ema50"]),
        "h1_ema200": float(h1_last["ema200"]),

        "m15_close": float(m15_last["close"]),
        "m15_rsi": float(m15_last["rsi"]),
        "m15_atr": float(m15_last["atr"]),
        "m15_ema20": float(m15_last["ema20"]),
        "m15_ema50": float(m15_last["ema50"]),

        "fibo_618": float(m15_last["fibo_618"]),
        "fibo_786": float(m15_last["fibo_786"])
    }

    system_prompt = """
Anda adalah AI Trading Analyst.

Tugas:
Analisis data market yang diberikan.

Gunakan:
- trend H1
- EMA
- RSI
- ATR
- struktur harga
- liquidity concept
- PO3 sebagai konteks tambahan

Jangan memaksakan trade.

Jika kondisi tidak jelas:
NO TRADE.

Jawaban HARUS JSON valid:

{
  "decision": "BUY",
  "confidence": 0.0,
  "reason": "alasan singkat",
  "entry_type": "MARKET"
}

decision hanya boleh:
BUY
SELL
NO TRADE

confidence antara 0 dan 1.
"""

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=json.dumps(
            market_data,
            indent=2
        )
    )

    text = response.output_text.strip()

    # Bersihkan markdown jika model mengembalikan ```json
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        result = json.loads(text)

    except Exception:

        # Fallback
        upper = text.upper()

        if "BUY" in upper:
            decision = "BUY"
        elif "SELL" in upper:
            decision = "SELL"
        else:
            decision = "NO TRADE"

        result = {
            "decision": decision,
            "confidence": 0.0,
            "reason": text,
            "entry_type": "MARKET"
        }

    return result


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calculate_position(
    symbol,
    decision,
    price,
    atr_value,
    account_balance,
    risk_percent,
    rr=2.0
):

    if decision not in ["BUY", "SELL"]:
        return None

    info = mt5.symbol_info(symbol)

    if info is None:
        raise RuntimeError(
            f"Symbol {symbol} tidak ditemukan."
        )

    # ATR based SL
    sl_distance = atr_value * 1.5

    if not np.isfinite(sl_distance) or sl_distance <= 0:
        raise RuntimeError(
            "ATR tidak valid."
        )

    if decision == "BUY":

        sl = price - sl_distance
        tp = price + sl_distance * rr

    else:

        sl = price + sl_distance
        tp = price - sl_distance * rr

    # ========================================================
    # PENTING:
    # Gunakan order_calc_profit untuk mengetahui kerugian
    # aktual berdasarkan spesifikasi broker.
    # ========================================================

    risk_money = account_balance * (
        risk_percent / 100.0
    )

    min_volume = float(info.volume_min)
    max_volume = float(info.volume_max)
    volume_step = float(info.volume_step)

    order_type = (
        mt5.ORDER_TYPE_BUY
        if decision == "BUY"
        else mt5.ORDER_TYPE_SELL
    )

    # Loss per 1 lot jika SL terkena
    loss_1_lot = mt5.order_calc_profit(
        order_type,
        symbol,
        1.0,
        price,
        sl
    )

    if loss_1_lot is None:
        raise RuntimeError(
            f"Gagal menghitung risiko: {mt5.last_error()}"
        )

    loss_1_lot = abs(float(loss_1_lot))

    if loss_1_lot <= 0:
        raise RuntimeError(
            "Nilai risiko lot tidak valid."
        )

    raw_volume = risk_money / loss_1_lot

    # Floor ke volume step
    volume = (
        np.floor(
            raw_volume / volume_step
        ) * volume_step
    )

    volume = round(volume, 8)

    # ========================================================
    # Jangan memaksakan volume minimum jika minimum lot
    # membuat risiko melebihi batas.
    # ========================================================

    min_lot_loss = mt5.order_calc_profit(
        order_type,
        symbol,
        min_volume,
        price,
        sl
    )

    if min_lot_loss is None:
        raise RuntimeError(
            "Gagal menghitung risiko minimum lot."
        )

    min_lot_loss = abs(float(min_lot_loss))

    if min_lot_loss > risk_money:

        return {
            "allowed": False,
            "reason": (
                f"Minimum lot broker {min_volume} "
                f"akan memiliki risiko sekitar "
                f"${min_lot_loss:.2f}, melebihi "
                f"batas ${risk_money:.2f}."
            ),
            "lot": min_volume,
            "sl": sl,
            "tp": tp
        }

    volume = max(
        min_volume,
        min(volume, max_volume)
    )

    # Normalize price
    digits = int(info.digits)

    sl = round(sl, digits)
    tp = round(tp, digits)
    price = round(price, digits)

    actual_risk = mt5.order_calc_profit(
        order_type,
        symbol,
        volume,
        price,
        sl
    )

    actual_risk = abs(
        float(actual_risk)
    ) if actual_risk is not None else None

    return {

        "allowed": True,

        "symbol": symbol,

        "action": decision,

        "price": price,

        "lot": volume,

        "sl": sl,

        "tp": tp,

        "risk_money": risk_money,

        "actual_risk": actual_risk,

        "sl_distance": sl_distance,

        "rr": rr,

        "min_volume": min_volume,

        "max_volume": max_volume,

        "volume_step": volume_step
    }


# ============================================================
# ORDER EXECUTION
# ============================================================

def execute_order(
    symbol,
    decision,
    volume,
    price,
    sl,
    tp,
    deviation=20
):

    info = mt5.symbol_info(symbol)

    if info is None:
        return None, "Symbol tidak ditemukan."

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return None, "Tick tidak tersedia."

    if decision == "BUY":

        order_type = mt5.ORDER_TYPE_BUY
        execution_price = tick.ask

    elif decision == "SELL":

        order_type = mt5.ORDER_TYPE_SELL
        execution_price = tick.bid

    else:

        return None, "NO TRADE."

    request = {

        "action": mt5.TRADE_ACTION_DEAL,

        "symbol": symbol,

        "volume": float(volume),

        "type": order_type,

        "price": float(execution_price),

        "sl": float(sl),

        "tp": float(tp),

        "deviation": deviation,

        "magic": MAGIC_NUMBER,

        "comment": "AI_TRADING_V4",

        "type_time": mt5.ORDER_TIME_GTC,

        "type_filling": mt5.ORDER_FILLING_RETURN
    }

    # ========================================================
    # CHECK BEFORE SEND
    # ========================================================

    check = mt5.order_check(request)

    if check is None:

        return None, (
            f"order_check gagal: "
            f"{mt5.last_error()}"
        )

    if check.retcode != 0:

        return None, (
            f"Order ditolak oleh order_check. "
            f"retcode={check.retcode}, "
            f"comment={check.comment}"
        )

    result = mt5.order_send(request)

    if result is None:

        return None, (
            f"order_send gagal: "
            f"{mt5.last_error()}"
        )

    if result.retcode != mt5.TRADE_RETCODE_DONE:

        return result, (
            f"Order tidak berhasil. "
            f"retcode={result.retcode}, "
            f"comment={result.comment}"
        )

    return result, "Order berhasil dikirim."


# ============================================================
# JOURNAL
# ============================================================

def save_trade(
    ticket,
    symbol,
    decision,
    entry,
    sl,
    tp,
    volume,
    h1_trend,
    rsi_value,
    atr_value,
    ema12,
    ema26,
    spread,
    notes
):

    conn = get_db()

    conn.execute("""
        INSERT INTO trade_journal
        (
            ticket,
            symbol,
            decision,
            entry,
            sl,
            tp,
            volume,
            h1_trend,
            rsi,
            atr,
            ema12,
            ema26,
            spread,
            timestamp,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        ticket,
        symbol,
        decision,
        entry,
        sl,
        tp,
        volume,
        h1_trend,
        rsi_value,
        atr_value,
        ema12,
        ema26,
        spread,
        datetime.now(
            timezone.utc
        ).isoformat(),
        notes
    ))

    conn.commit()
    conn.close()


def load_journal():

    conn = get_db()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM trade_journal
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    return df


# ============================================================
# LEARNING ENGINE
# ============================================================

def get_training_data():

    conn = get_db()

    df = pd.read_sql_query(
        """
        SELECT
            rsi,
            atr,
            ema12,
            ema26,
            spread,
            outcome
        FROM trade_journal
        WHERE outcome IS NOT NULL
        """,
        conn
    )

    conn.close()

    return df


def train_learning_model():

    if not SKLEARN_AVAILABLE:
        return None, (
            "scikit-learn belum tersedia."
        )

    df = get_training_data()

    if len(df) < 50:
        return None, (
            f"Data belum cukup. "
            f"Baru {len(df)} trade selesai. "
            f"Minimal 50."
        )

    df = df.dropna()

    if df["outcome"].nunique() < 2:
        return None, (
            "Belum ada kombinasi WIN dan LOSS "
            "yang cukup untuk training."
        )

    X = df[
        [
            "rsi",
            "atr",
            "ema12",
            "ema26",
            "spread"
        ]
    ]

    y = df["outcome"]

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    model.fit(X, y)

    version = (
        "ML-"
        + datetime.now(
            timezone.utc
        ).strftime("%Y%m%d%H%M")
    )

    win_rate = float(
        df["outcome"].mean()
    )

    conn = get_db()

    conn.execute(
        """
        UPDATE model_versions
        SET active = 0
        """
    )

    conn.execute(
        """
        INSERT INTO model_versions
        (
            version,
            created_at,
            trades_used,
            win_rate,
            notes,
            active
        )
        VALUES (?, ?, ?, ?, ?, 1)
        """,
        (
            version,
            datetime.now(
                timezone.utc
            ).isoformat(),
            len(df),
            win_rate,
            "RandomForest retraining"
        )
    )

    conn.commit()
    conn.close()

    return model, (
        f"Model {version} berhasil "
        f"dilatih dengan {len(df)} trade."
    )


# ============================================================
# UPDATE TRADE OUTCOMES
# ============================================================

def update_closed_trade_results():

    if mt5 is None:
        return

    conn = get_db()

    rows = conn.execute("""
        SELECT id, ticket
        FROM trade_journal
        WHERE outcome IS NULL
        AND ticket IS NOT NULL
    """).fetchall()

    for row in rows:

        ticket = row["ticket"]

        try:

            deals = mt5.history_deals_get(
                position=ticket
            )

            if deals is None:
                continue

            profit = 0.0

            found = False

            for deal in deals:

                if deal.entry in [
                    mt5.DEAL_ENTRY_OUT,
                    mt5.DEAL_ENTRY_OUT_BY
                ]:

                    profit += float(
                        deal.profit
                    )

                    found = True

            if not found:
                continue

            outcome = 1 if profit > 0 else 0

            conn.execute(
                """
                UPDATE trade_journal
                SET result = ?,
                    outcome = ?
                WHERE id = ?
                """,
                (
                    profit,
                    outcome,
                    row["id"]
                )
            )

        except Exception:
            continue

    conn.commit()
    conn.close()


# ============================================================
# UI SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ AI Trading Intelligence V4"
    )

    st.subheader(
        "🤖 OpenAI"
    )

    secret_key = st.secrets.get(
        "OPENAI_API_KEY",
        ""
    )

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=secret_key
    )

    model = st.text_input(
        "Model AI",
        value="gpt-5.6"
    )

    st.divider()

    st.subheader(
        "💱 Instrumen"
    )

    preferred_symbol = st.selectbox(
        "Pair",
        [
            "EURUSD",
            "XAUUSD"
        ]
    )

    st.divider()

    st.subheader(
        "🛡️ Risk Management"
    )

    risk_percent = st.slider(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    rr = st.number_input(
        "Risk : Reward",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5
    )

    max_spread = st.number_input(
        "Max spread",
        min_value=0.0,
        value=0.00030,
        format="%.5f"
    )

    st.divider()

    st.subheader(
        "🔐 MetaTrader 5"
    )

    mt5_login = st.number_input(
        "MT5 Login",
        value=0,
        step=1
    )

    mt5_password = st.text_input(
        "MT5 Password",
        type="password"
    )

    mt5_server = st.text_input(
        "MT5 Server",
        value="HFM-Demo"
    )

    mt5_path = st.text_input(
        "MT5 Terminal Path (optional)",
        value=""
    )

    st.divider()

    st.subheader(
        "🚦 Execution Safety"
    )

    # Default FALSE.
    # Ini penting supaya aplikasi tidak langsung
    # mengirim order hanya karena halaman dibuka.
    AUTO_EXECUTION = st.checkbox(
        "Enable automatic order execution",
        value=False
    )

    st.caption(
        "Matikan untuk mode analisis/paper trading."
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 AI Trading Intelligence V4"
)

st.caption(
    "Real Market MT5 • H1 + M15 • "
    "AI Analysis • Risk Engine • "
    "Trade Journal • Learning Engine"
)


# ============================================================
# CONNECT
# ============================================================

if mt5 is None:

    st.error(
        "MetaTrader5 Python package belum terinstall."
    )

    st.stop()


connected, message = connect_mt5(
    mt5_login,
    mt5_password,
    mt5_server,
    mt5_path
)

if not connected:

    st.error(
        f"❌ {message}"
    )

    st.info(
        "Pastikan MetaTrader 5 terinstall "
        "dan akun demo sudah login."
    )

    st.stop()


st.success(
    f"🟢 {message}"
)


# ============================================================
# ACCOUNT
# ============================================================

account = mt5.account_info()

if account is None:

    st.error(
        "Tidak bisa membaca informasi akun MT5."
    )

    st.stop()


account_balance = float(
    account.balance
)

account_equity = float(
    account.equity
)

account_currency = account.currency


# ============================================================
# SYMBOL
# ============================================================

symbol = find_broker_symbol(
    preferred_symbol
)

if symbol is None:

    st.error(
        f"❌ Pair {preferred_symbol} "
        "tidak ditemukan di broker."
    )

    st.stop()


tick = get_live_tick(symbol)

if tick is None:

    st.error(
        "Tidak mendapatkan harga live."
    )

    st.stop()


# ============================================================
# MARKET DATA
# ============================================================

try:

    h1_raw = get_rates(
        symbol,
        mt5.TIMEFRAME_H1,
        300
    )

    m15_raw = get_rates(
        symbol,
        mt5.TIMEFRAME_M15,
        300
    )

    df_h1 = process_h1(
        h1_raw
    )

    df_m15 = process_m15(
        m15_raw
    )

except Exception as e:

    st.error(
        f"Market data error: {e}"
    )

    st.stop()


# ============================================================
# ACCOUNT INFO
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Symbol",
        symbol
    )

with c2:

    st.metric(
        "Bid",
        f"{tick['bid']:.5f}"
    )

with c3:

    st.metric(
        "Ask",
        f"{tick['ask']:.5f}"
    )

with c4:

    st.metric(
        "Balance",
        f"{account_balance:.2f} {account_currency}"
    )


# ============================================================
# MARKET TABLES
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "📈 Real Market H1"
    )

    st.dataframe(
        df_h1[
            [
                "time",
                "open",
                "high",
                "low",
                "close",
                "ema12",
                "ema26",
                "ema50",
                "ema200",
                "trend"
            ]
        ].tail(10),
        use_container_width=True
    )


with col2:

    st.subheader(
        "📉 Real Market M15"
    )

    st.dataframe(
        df_m15[
            [
                "time",
                "open",
                "high",
                "low",
                "close",
                "rsi",
                "atr",
                "ema20",
                "ema50"
            ]
        ].tail(10),
        use_container_width=True
    )


# ============================================================
# MARKET STATUS
# ============================================================

h1_last = df_h1.iloc[-1]
m15_last = df_m15.iloc[-1]

st.divider()

a, b, c, d, e = st.columns(5)

with a:
    st.metric(
        "H1 Trend",
        h1_last["trend"]
    )

with b:
    st.metric(
        "M15 RSI",
        f"{m15_last['rsi']:.2f}"
    )

with c:
    st.metric(
        "M15 ATR",
        f"{m15_last['atr']:.6f}"
    )

with d:
    st.metric(
        "Spread",
        f"{tick['spread']:.5f}"
    )

with e:
    st.metric(
        "Equity",
        f"{account_equity:.2f}"
    )


# ============================================================
# AUTO RESULT UPDATE
# ============================================================

update_closed_trade_results()


# ============================================================
# ANALYSIS BUTTON
# ============================================================

st.divider()

if st.button(
    "🚀 ANALISIS MARKET SEKARANG",
    use_container_width=True
):

    if tick["spread"] > max_spread:

        st.warning(
            f"Spread terlalu besar: "
            f"{tick['spread']:.5f}"
        )

    else:

        with st.spinner(
            "AI sedang menganalisis real market..."
        ):

            try:

                ai_result = ai_analysis(
                    api_key,
                    model,
                    symbol,
                    df_h1,
                    df_m15,
                    tick
                )

                decision = (
                    ai_result
                    .get(
                        "decision",
                        "NO TRADE"
                    )
                    .upper()
                )

                confidence = float(
                    ai_result.get(
                        "confidence",
                        0
                    )
                )

                reason = ai_result.get(
                    "reason",
                    ""
                )

                st.session_state[
                    "ai_result"
                ] = ai_result

                if decision == "BUY":

                    st.success(
                        f"🟢 AI: BUY | "
                        f"Confidence {confidence:.0%}"
                    )

                elif decision == "SELL":

                    st.error(
                        f"🔴 AI: SELL | "
                        f"Confidence {confidence:.0%}"
                    )

                else:

                    st.warning(
                        f"🟡 AI: NO TRADE | "
                        f"Confidence {confidence:.0%}"
                    )

                st.write(
                    f"**Alasan AI:** {reason}"
                )

                # ==================================================
                # POSITION
                # ==================================================

                if decision in [
                    "BUY",
                    "SELL"
                ]:

                    entry_price = (
                        tick["ask"]
                        if decision == "BUY"
                        else tick["bid"]
                    )

                    risk_result = calculate_position(
                        symbol,
                        decision,
                        entry_price,
                        float(m15_last["atr"]),
                        account_balance,
                        risk_percent,
                        rr
                    )

                    st.subheader(
                        "🛡️ Risk Engine"
                    )

                    st.json(
                        risk_result
                    )

                    if not risk_result["allowed"]:

                        st.error(
                            "🚫 TRADE DIBATALKAN\n\n"
                            + risk_result["reason"]
                        )

                    else:

                        # ==================================================
                        # AUTOMATIC EXECUTION
                        # ==================================================

                        if AUTO_EXECUTION:

                            st.warning(
                                "⚠️ Automatic execution aktif."
                            )

                            result, result_message = execute_order(
                                symbol,
                                decision,
                                risk_result["lot"],
                                risk_result["price"],
                                risk_result["sl"],
                                risk_result["tp"]
                            )

                            if result is not None:

                                if (
                                    result.retcode
                                    == mt5.TRADE_RETCODE_DONE
                                ):

                                    ticket = int(
                                        result.order
                                    )

                                    save_trade(

                                        ticket,

                                        symbol,

                                        decision,

                                        risk_result[
                                            "price"
                                        ],

                                        risk_result[
                                            "sl"
                                        ],

                                        risk_result[
                                            "tp"
                                        ],

                                        risk_result[
                                            "lot"
                                        ],

                                        h1_last[
                                            "trend"
                                        ],

                                        float(
                                            m15_last[
                                                "rsi"
                                            ]
                                        ),

                                        float(
                                            m15_last[
                                                "atr"
                                            ]
                                        ),

                                        float(
                                            h1_last[
                                                "ema12"
                                            ]
                                        ),

                                        float(
                                            h1_last[
                                                "ema26"
                                            ]
                                        ),

                                        tick[
                                            "spread"
                                        ],

                                        reason
                                    )

                                    st.success(
                                        f"✅ ORDER BERHASIL | "
                                        f"Ticket {ticket}"
                                    )

                                else:

                                    st.error(
                                        result_message
                                    )

                            else:

                                st.error(
                                    result_message
                                )

                        else:

                            st.info(
                                "🧪 PAPER/ANALYSIS MODE: "
                                "Tidak ada order dikirim."
                            )

            except Exception as e:

                st.error(
                    f"AI / Execution error: {e}"
                )


# ============================================================
# LEARNING ENGINE
# ============================================================

st.divider()

st.header(
    "🧠 Learning Engine"
)

journal = load_journal()

if journal.empty:

    st.info(
        "Belum ada trade yang tersimpan."
    )

else:

    completed = journal[
        journal["outcome"].notna()
    ]

    if not completed.empty:

        wins = (
            completed["outcome"] == 1
        ).sum()

        losses = (
            completed["outcome"] == 0
        ).sum()

        winrate = (
            wins / len(completed)
        )

        x1, x2, x3 = st.columns(3)

        with x1:
            st.metric(
                "Completed Trades",
                len(completed)
            )

        with x2:
            st.metric(
                "Wins",
                wins
            )

        with x3:
            st.metric(
                "Win Rate",
                f"{winrate:.1%}"
            )

        st.write(
            "Learning Engine akan menggunakan "
            "hasil trade yang sudah selesai untuk "
            "mencari pola kondisi WIN/LOSS."
        )

    if st.button(
        "🧠 Retrain Learning Model"
    ):

        with st.spinner(
            "Learning Engine sedang belajar..."
        ):

            model_ml, msg = (
                train_learning_model()
            )

            if model_ml is not None:

                st.success(msg)

            else:

                st.warning(msg)


# ============================================================
# TRADE JOURNAL
# ============================================================

st.divider()

st.header(
    "📚 Trade Journal"
)

journal = load_journal()

if not journal.empty:

    st.dataframe(
        journal,
        use_container_width=True
    )

else:

    st.info(
        "Trade journal masih kosong."
    )


# ============================================================
# MODEL HISTORY
# ============================================================

st.divider()

st.header(
    "🧬 Model Versions"
)

conn = get_db()

models_df = pd.read_sql_query(
    """
    SELECT *
    FROM model_versions
    ORDER BY id DESC
    """,
    conn
)

conn.close()

if not models_df.empty:

    st.dataframe(
        models_df,
        use_container_width=True
    )

else:

    st.info(
        "Belum ada model ML yang dilatih."
    )


# ============================================================
# CURRENT POSITIONS
# ============================================================

st.divider()

st.header(
    "📊 Open Positions"
)

positions = mt5.positions_get(
    symbol=symbol
)

if positions:

    pos_df = pd.DataFrame(
        [
            p._asdict()
            for p in positions
        ]
    )

    st.dataframe(
        pos_df,
        use_container_width=True
    )

else:

    st.info(
        "Tidak ada posisi terbuka."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Trading Intelligence V4 | "
    "Real MT5 Market Data | "
    "Risk Engine | "
    "Trade Journal | "
    "Learning Engine"
)
