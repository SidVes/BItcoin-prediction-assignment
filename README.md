# Bitcoin Price Prediction

Chat-based Bitcoin prediction app built for a senior AI engineer interview. Ask it anything about BTC price and it runs five ML models, shows you each one's forecast with eval metrics, and uses GPT-4o to summarize what they're all saying.

---

## What's inside

The backend is FastAPI + LangGraph. When you send a query, the agent figures out what you're asking for (predict, retrain, or off-topic), fetches BTC/USDT data from Yahoo Finance, trains and evaluates five models in parallel, then calls GPT-4o to write a plain-English summary of the results.

The five models are ARIMA, XGBoost, LSTM, Prophet, and PatchTST. Each one trains on everything except the last 60 days, which are kept as a holdout for evaluation. Results include RMSE, MAPE, and directional accuracy (did it get the up/down direction right). Models are cached so they don't retrain on every request — only if the data changed or you explicitly ask.

There's a guardrail so if you ask about Ethereum or something random it just rejects the query without running anything.

The frontend is Angular 21. Dark theme, chat interface, shows a comparison table of all model outputs alongside the LLM summary.

---

## Running it locally

You need Python 3.11+ for the backend and Node 18+ for the frontend.

**Backend:**

```bash
cd Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` in the `Backend/` folder:

```
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o
```

LangSmith tracing is optional. If you want it add `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, and `LANGCHAIN_TRACING_V2=true`.

```bash
uvicorn main:app --reload --port 8000
```

The first request will be slow — it downloads about 3 years of BTC data and trains all the models from scratch. After that everything is cached so subsequent requests are much faster.

**Frontend:**

```bash
cd chat-ui
npm install
npm start
```

Go to `http://localhost:4200`. API calls are proxied to port 8000 automatically in dev.

---

## A few things worth noting

The train/test split is strictly time-aware — the 60-day holdout always comes from the end of the series. No shuffling, no leakage.

XGBoost uses around 40 engineered features — lags, rolling SMAs/EMAs, RSI, MACD, Bollinger Bands, ATR, volume ratios. The LSTM is bidirectional with 30-step input sequences. PatchTST was fine-tuned separately in the notebook and isn't retrained during normal pipeline runs.

Directional accuracy in the 55–65% range is normal for crypto. If you see 90%+ something is wrong.

---

## Docker

```bash
cd chat-ui
docker build -t btc-chat-ui .
docker run -p 80:80 btc-chat-ui
```

---

## License

MIT
