import os
import json
import requests
import pandas as pd
import numpy as np
import ta

# --- ENVIRONMENT VARIABLES (LOADED FROM GITHUB SECRETS) ---
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ACCOUNT_BALANCE = 5000.0  # Simulated account balance in USD
RISK_PER_TRADE = 0.01     # 1% strict risk rule ($50)
MAX_SPREAD_USD = 0.40     # Max allowed spread on Gold

# --- 1. FETCH MARKET DATA & COMPUTE METRICS ---
def fetch_gold_snapshot():
    # 15m Candlestick data feed
    url = "https://api.binance.com/api/v3/klines?symbol=PAXGUSDT&interval=15m&limit=100"
    res = requests.get(url, timeout=10).json()

    df = pd.DataFrame(res, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Technical Indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # Session VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    # Fair Value Gap (FVG) Check
    df['bullish_fvg'] = df['low'] > df['high'].shift(2)
    df['bearish_fvg'] = df['high'] < df['low'].shift(2)

    latest = df.iloc[-1]
    
    return {
        "current_price": round(latest['close'], 2),
        "spread": 0.25,
        "rsi_14": round(latest['rsi'], 2),
        "ema_20": round(latest['ema_20'], 2),
        "ema_50": round(latest['ema_50'], 2),
        "session_vwap": round(latest['vwap'], 2),
        "atr_14": round(latest['atr'], 2),
        "bullish_fvg": bool(df['bullish_fvg'].iloc[-3:].any()),
        "bearish_fvg": bool(df['bearish_fvg'].iloc[-3:].any()),
        "trend_15m": "Bullish" if latest['ema_20'] > latest['ema_50'] else "Bearish"
    }

# --- 2. AI REASONING ENGINE ---
def run_ai_evaluation(data):
    system_prompt = f"""
    You are 'Aegis-Gold', an institutional Quantitative Commodities Trader.
    Account Balance: ${ACCOUNT_BALANCE} | Risk Per Trade: {RISK_PER_TRADE*100}% (${ACCOUNT_BALANCE * RISK_PER_TRADE}).
    
    RULES:
    1. Require >= 75% setup confluence and at least 1:2.0 Risk-to-Reward Ratio (RRR).
    2. If market is choppy or below 75% conviction, output strictly: NO_TRADE.
    3. Structural Stop Loss only.
    
    OUTPUT FORMAT (IF VALID TRADE):
    [SIGNAL ALERT: XAU/USD INTRADAY]
    • Bias: [BUY / SELL]
    • Execution: [Market Execution @ current price OR Limit @ key zone]
    • Entry: $[Exact Price]
    • Invalidation (Stop Loss): $[Exact Price]
    • Target 1 (50% Off + Move SL to Breakeven): $[Exact Price]
    • Target 2 (Runner): $[Exact Price]
    • Risk-to-Reward Ratio: [e.g., 1:2.4]
    • Confluence Score: [__%]
    • Suggested Lot Size: [Calculated]
    
    [CORE CONFLUENCE]
    1. Structure: [FVG fill / session sweep]
    2. Indicators: [RSI / EMA / VWAP alignment]
    
    [INVALIDATION CRITERIA]
    • [Exact price level or event that invalidates trade]
    """

    payload = {
        "contents": [{
            "parts": [{
                "text": f"{system_prompt}\n\nLive Snapshot:\n{json.dumps(data, indent=2)}"
            }]
        }]
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    res = requests.post(url, json=payload, timeout=20).json()

    try:
        return res['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        return "NO_TRADE"

# --- 3. DISCORD EMBED DISPATCHER ---
def send_discord_alert(message):
    if not DISCORD_WEBHOOK_URL:
        print("Discord Webhook URL missing.")
        return

    is_buy = "BUY" in message.upper()
    color = 3066993 if is_buy else 15158332  # Green for BUY, Red for SELL

    payload = {
        "username": "Aegis-Gold Desk",
        "embeds": [{
            "title": "🚨 XAU/USD Institutional Alert",
            "description": message,
            "color": color,
            "footer": {
                "text": "Automated Intraday Risk Engine • 1% Capital Risk Rule"
            }
        }]
    }

    requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)

# --- 4. MAIN PIPELINE ---
def main():
    snapshot = fetch_gold_snapshot()

    # Pre-trade filter
    if snapshot["spread"] > MAX_SPREAD_USD:
        print(f"Spread too high: ${snapshot['spread']}. Aborting.")
        return

    ai_decision = run_ai_evaluation(snapshot)

    if "NO_TRADE" not in ai_decision:
        send_discord_alert(ai_decision)
        print("Valid trade found! Alert sent to Discord.")
    else:
        print("Market scanned: No institutional setup. Standing by.")

if __name__ == "__main__":
    main()
