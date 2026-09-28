import os
import json
import requests
import pandas as pd
import numpy as np
import ta

# --- ENVIRONMENT VARIABLES (LOADED FROM GITHUB SECRETS) ---
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ACCOUNT_BALANCE = 5000.0  
RISK_PER_TRADE = 0.01     
MAX_SPREAD_USD = 0.40     

# --- 1. FETCH MARKET DATA & COMPUTE METRICS ---
def fetch_gold_snapshot():
    url = "https://api.kraken.com/0/public/OHLC?pair=PAXGUSD&interval=15"
    res = requests.get(url, headers={"User-Agent": "AegisBot/1.0"}, timeout=10).json()

    data_list = res['result']['PAXGUSD']
    df = pd.DataFrame(data_list, columns=['time', 'open', 'high', 'low', 'close', 'vwap_kraken', 'volume', 'count'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

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
    
    You must evaluate the market and ALWAYS return a status report.
    
    If confluence is < 75% or the market is choppy, your Action is "WAITING". 
    If confluence is >= 75% and Risk-to-Reward is >= 1:2.0, your Action is "ACTIVE TRADE".
    
    OUTPUT FORMAT:
    
    **[XAU/USD 15-Min Market Scan]**
    • **Current Price:** $[Insert price from snapshot]
    • **Session Trend:** [Insert trend from snapshot]
    • **Confluence Score:** [Evaluate from 0% to 100%]
    • **Action:** [WAITING or ACTIVE TRADE]
    • **Market Context:** [1-2 concise sentences on VWAP, EMA, and RSI]

    [IF ACTION IS "ACTIVE TRADE", INCLUDE THE FOLLOWING]
    • **Bias:** [BUY / SELL]
    • **Entry:** $[Exact Price]
    • **Invalidation (Stop Loss):** $[Exact Price]
    • **Target 1 (50% Off + Breakeven SL):** $[Exact Price]
    • **Target 2 (Runner):** $[Exact Price]
    • **Risk-to-Reward:** [e.g., 1:2.2]
    • **Suggested Lot Size:** [Calculated based on 1% risk]
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
        return "**[XAU/USD 15-Min Market Scan]**\n• **Action:** WAITING (Data scan normal, no trigger)"

# --- 3. DISCORD EMBED DISPATCHER ---
def send_discord_alert(message, is_test=False):
    if not DISCORD_WEBHOOK_URL:
        return

    content_tag = ""

    if is_test:
        color = 3447003
        title = "🔧 SYSTEM TEST: Bot is Online"
    elif "ACTIVE TRADE" in message.upper():
        is_buy = "BUY" in message.upper()
        color = 3066993 if is_buy else 15158332
        title = "🚨 HIGH-CONFLUENCE INSTITUTIONAL SETUP (SCORE CHANCE)"
        content_tag = "@here 🚨 **HIGH-PROBABILITY GOLD SETUP DETECTED!**"
    else:
        color = 8421504
        title = "⏱️ XAU/USD 15-Min Market Scan"

    payload = {
        "username": "Aegis-Gold Desk",
        "content": content_tag,
        "embeds": [{
            "title": title,
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

    if snapshot["spread"] > MAX_SPREAD_USD:
        send_discord_alert(f"**[XAU/USD 15-Min Market Scan]**\n• **Current Price:** ${snapshot['current_price']}\n• **Action:** WAITING (Spread too high: ${snapshot['spread']})")
        return

    ai_decision = run_ai_evaluation(snapshot)
    send_discord_alert(ai_decision)
    print("Market scan complete. Alert sent to Discord.")

if __name__ == "__main__":
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_discord_alert("✅ System Update Complete! Priority alerts enabled for active trades.", is_test=True)
    
    main()
