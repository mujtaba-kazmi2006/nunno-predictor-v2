 Nunno AI – Confluence-Based Market Predictor (Core Module)

This is the intelligent prediction core behind Nunno — your AI-powered market assistant.

Instead of guessing “up or down,” Nunno analyzes real confluences using RSI, MACD, EMA, and other indicators to provide human-style explanations of what’s happening in the market.

---

## ✅ What This Module Does

- 📊 Detects overbought/oversold zones (RSI, Stoch RSI)
- 📉 Measures momentum and trend direction (MACD, EMA)
- 🔍 Evaluates trend strength (ADX)
- 📦 Considers volume spikes and candle structures
- 🧠 Returns clear reasoning like:
RSI < 30 → Oversold, potential bounce
MACD negative → Bearish momentum
Price below EMA 200 → Long-term downtrend bias

yaml
Copy
Edit

No “BUY NOW” or “SELL NOW” — just smart, realistic trend interpretation.

---

## 🚀 How to Use

1. Install dependencies:

```bash
pip install pandas numpy ta
Import and run:

python
Copy
Edit
from betterpridictormodule import fetch_binance_ohlcv, add_indicators, generate_reasoning

df = fetch_binance_ohlcv("BTCUSDT", "15m")
df = add_indicators(df)
latest = df.iloc[-1]

for line in generate_reasoning(latest):
    print("•", line)
