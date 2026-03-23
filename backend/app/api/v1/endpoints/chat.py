"""
app/api/v1/endpoints/chat.py
─────────────────────────────
SparkChat — AI-powered Canadian investment education chatbot.
Follows the same pattern as all other v1 endpoints in this project.

LLM: Groq (free tier) — set GROQ_API_KEY in your .env also for voice transcription.
"""

import base64
import io
import json
import os
import re
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

load_dotenv()

router = APIRouter()

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Spark, a friendly and expert financial educator specialized in Canadian banking and investment markets. You help users understand:

- Canadian tax-advantaged accounts: TFSA, RRSP, RESP, FHSA, RDSP
- Canadian stock market: TSX (Toronto Stock Exchange), TSX Venture Exchange
- Canadian ETFs, mutual funds, GICs, bonds, REITs
- Canadian banks: Big Six (RBC, TD, BMO, Scotiabank, CIBC, National Bank)
- CDIC deposit insurance, CIPF investor protection
- Canadian tax rules: capital gains, dividend tax credits, ACB tracking
- Robo-advisors in Canada: Wealthsimple, Questrade, CI Direct
- Portfolio optimization concepts: Markowitz, Sharpe ratio, diversification
- Investment forecasting concepts: EWM, time series, risk metrics

Rules:
- Always clarify you are for educational purposes only, not licensed financial advice
- Be encouraging, clear, and avoid unexplained jargon
- Use Canadian context (CAD, CRA rules, Canadian brokers)
- For complex personal situations, recommend a licensed advisor (IIROC/MFDA registered)
- Respond in the same language the user writes in (English or Spanish)
- Keep answers concise but complete — aim for 3-5 sentences unless more detail is needed
- When given portfolio or forecast data, interpret the numbers in plain language a beginner can understand
- When a user asks about the performance, price history, or chart of a specific asset or index, include a chart tag in your response using this exact format: [CHART:TICKER:PERIOD]
  where TICKER is the Yahoo Finance symbol and PERIOD is one of: 1mo, 3mo, 6mo, 1y, 2y, 5y
  Common symbols: ^GSPC (S&P 500), ^OSPTSX (TSX Composite), ^DJI (Dow Jones), ^IXIC (NASDAQ), BTC-USD (Bitcoin), ETH-USD (Ethereum), AAPL (Apple), MSFT (Microsoft)
  Example: "Here is the S&P 500 performance over the last year: [CHART:^GSPC:1y]"
  Only include a chart tag when it genuinely adds value — not for every message"""


# ── Context injector ──────────────────────────────────────────────────────────
def build_context_prompt(context: dict) -> str:
    """Convert the current page's API result into a natural language block for the LLM."""
    ctx_type = context.get("type")
    data = context.get("data", {})

    if ctx_type == "portfolio_optimize":
        symbols = data.get("symbols", [])
        weights = data.get("weights", {})
        perf = data.get("performance", {})
        risk = data.get("risk_metrics", {})
        individual = data.get("individual_stats", {})
        advanced = data.get("advanced", {})

        lines = [
            f"Symbols: {', '.join(symbols)}",
            f"Optimal weights: {', '.join(f'{s}: {round(w*100,1)}%' for s, w in weights.items())}",
            f"Expected annual return: {round(perf.get('expected_return', perf.get('expected_annual_return', 0)) * 100, 2)}%",
            f"Annual volatility: {round(perf.get('volatility', perf.get('annual_volatility', 0)) * 100, 2)}%",
            f"Sharpe ratio: {perf.get('sharpe_ratio')}",
            f"VaR (95%): {round(risk.get('var_95', 0) * 100, 2)}%",
            f"CVaR (95%): {round(risk.get('cvar_95', 0) * 100, 2)}%",
            f"Max drawdown: {round(risk.get('max_drawdown', 0) * 100, 2)}%",
        ]

        if individual:
            lines.append("\nPer-asset stats:")
            for sym, stats in individual.items():
                lines.append(
                    f"  {sym}: cumulative return {round(stats.get('cumulative_return', 0)*100, 1)}%, "
                    f"volatility {round(stats.get('annualized_volatility', 0)*100, 1)}%, "
                    f"Sharpe {stats.get('sharpe_score')}, "
                    f"max drawdown {round(stats.get('max_drawdown', 0)*100, 1)}%"
                )

        if advanced and advanced.get("correlation_matrix"):
            corr = advanced["correlation_matrix"]
            syms = list(corr.keys())
            if len(syms) == 2:
                s1, s2 = syms
                lines.append(f"\nCorrelation between {s1} and {s2}: {corr[s1].get(s2)}")

        lines.append("\nInterpret all these numbers in plain language for a beginner investor.")
        return "\n".join(lines)

    if ctx_type == "portfolio_stats":
        symbols = data.get("symbols", [])
        individual = data.get("individual", {})
        advanced = data.get("advanced", {})
        lines = [f"Portfolio Statistics for: {', '.join(symbols)}\n"]
        for sym, stats in individual.items():
            lines.append(
                f"{sym}: cumulative return {round(stats.get('cumulative_return', 0)*100, 1)}%, "
                f"volatility {round(stats.get('annualized_volatility', 0)*100, 1)}%, "
                f"Sharpe {stats.get('sharpe_score')}, "
                f"max drawdown {round(stats.get('max_drawdown', 0)*100, 1)}%"
            )
        corr = advanced.get("correlation_matrix", {})
        if len(symbols) == 2:
            s1, s2 = symbols
            c = corr.get(s1, {}).get(s2)
            if c is not None:
                lines.append(f"Correlation between {s1} and {s2}: {c}")
        lines.append("\nInterpret these stats in plain language for a beginner investor.")
        return "\n".join(lines)

    if ctx_type in ("forecast", "analyze"):
        return f"""
The user is viewing a {ctx_type} result:
{json.dumps(data, indent=2)[:800]}

Summarize the key takeaways in plain language.
"""

    if ctx_type == "portfolio_simulate":
        symbols = data.get("symbols", [])
        weights = data.get("weights", {})
        mc = data.get("mc_summary", {})
        hist = data.get("hist_summary", {})
        n_sims = data.get("n_simulations", 0)
        n_periods = data.get("n_periods", 0)
        init = data.get("initial_value", 10_000)

        lines = [
            f"Portfolio Simulation for: {', '.join(symbols)}",
            f"Weights: {', '.join(f'{s}: {round(w*100,1)}%' for s, w in weights.items())}",
            f"Starting value: ${init:,.0f} | Horizon: {n_periods} periods | Paths: {n_sims:,}",
            "",
            "Monte Carlo GBM results:",
            f"  Probability of profit: {round(mc.get('prob_positive', 0)*100, 1)}%",
            f"  Expected final value: ${mc.get('expected_terminal', 0):,.0f}",
            f"  P5 / Median / P95: ${mc.get('ci_5',0):,.0f} / ${mc.get('ci_50',0):,.0f} / ${mc.get('ci_95',0):,.0f}",
            f"  Sortino: {round(mc.get('sortino_ratio', 0), 3)} | Calmar: {round(mc.get('calmar_ratio', 0), 3)} | Omega: {round(mc.get('omega_ratio', 0), 3)}",
            f"  Max drawdown: {round(mc.get('max_drawdown', 0)*100, 2)}%",
            "",
            "Historical Bootstrap results:",
            f"  Probability of profit: {round(hist.get('prob_positive', 0)*100, 1)}%",
            f"  Expected final value: ${hist.get('expected_terminal', 0):,.0f}",
            f"  P5 / Median / P95: ${hist.get('ci_5',0):,.0f} / ${hist.get('ci_50',0):,.0f} / ${hist.get('ci_95',0):,.0f}",
            f"  Sortino: {round(hist.get('sortino_ratio', 0), 3)} | Calmar: {round(hist.get('calmar_ratio', 0), 3)} | Omega: {round(hist.get('omega_ratio', 0), 3)}",
            f"  Max drawdown: {round(hist.get('max_drawdown', 0)*100, 2)}%",
            "",
            "Interpret these simulation results in plain language for a beginner investor. "
            "Explain what the two methods mean, whether this portfolio looks promising, "
            "and what the risk metrics indicate.",
        ]
        return "\n".join(lines)

    return ""


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class PageContext(BaseModel):
    type: str
    data: Any


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: Optional[PageContext] = None


class ChatResponse(BaseModel):
    reply: str


# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Send a message to SparkChat and receive an AI-generated educational response.
    Optionally accepts page context (portfolio/forecast results) to interpret in natural language.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    # Inject page context into system prompt if provided
    system = SYSTEM_PROMPT
    if req.context:
        context_block = build_context_prompt(req.context.model_dump())
        if context_block:
            system += f"\n\n--- CURRENT PAGE DATA ---\n{context_block}"

    messages = [{"role": "system", "content": system}]
    messages += [{"role": m.role, "content": m.content} for m in req.history[-8:]]
    messages.append({"role": "user", "content": req.message})

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 600,
                "temperature": 0.65,
                "messages": messages,
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LLM error: {resp.text}")

    reply = resp.json()["choices"][0]["message"]["content"]
    return ChatResponse(reply=reply)


# ── Transcription endpoint ─────────────────────────────────────────────────────
@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict:
    """
    Transcribe voice audio using Groq Whisper.
    Accepts audio/webm (or any format Whisper supports) and returns the transcript.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    audio_bytes = await file.read()
    filename = file.filename or "audio.webm"
    content_type = file.content_type or "audio/webm"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, audio_bytes, content_type)},
            data={"model": "whisper-large-v3"},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Whisper error: {resp.text}")

    return {"transcript": resp.json().get("text", "")}


# ── TTS endpoint (Amazon Polly) ────────────────────────────────────────────────
class SpeakRequest(BaseModel):
    text: str


@router.post("/speak")
async def speak_text(request: SpeakRequest):
    """
    Convert text to speech using Amazon Polly (Matthew — neural, US English male).
    Returns MP3 audio as a streaming response.
    """
    import boto3
    from fastapi.responses import StreamingResponse

    text = request.text[:3000]  # Polly limit safety

    try:
        polly = boto3.client(
            "polly",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        response = polly.synthesize_speech(
            TextType="ssml",
            Text=f'<speak><prosody rate="108%">{text}</prosody></speak>',
            OutputFormat="mp3",
            VoiceId="Matthew",
            Engine="neural",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Polly error: {exc}")

    audio_stream = response["AudioStream"]
    return StreamingResponse(audio_stream, media_type="audio/mpeg")


# ── Chart generation endpoint ──────────────────────────────────────────────────
_VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}
_TICKER_NAMES: dict[str, str] = {
    "^GSPC": "S&P 500",
    "^OSPTSX": "TSX Composite",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "BTC-USD": "Bitcoin (USD)",
    "ETH-USD": "Ethereum (USD)",
    "BNB-USD": "BNB (USD)",
    "SOL-USD": "Solana (USD)",
    "XRP-USD": "XRP (USD)",
}


@router.get("/chart")
async def generate_chart(
    ticker: str = Query(..., description="Yahoo Finance ticker symbol"),
    period: str = Query("1y", description="One of: 1mo, 3mo, 6mo, 1y, 2y, 5y"),
) -> dict:
    """
    Generate a dark-themed price chart for any Yahoo Finance ticker.
    Returns a base64-encoded PNG as { image: 'data:image/png;base64,...' }.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import yfinance as yf

    # Validate inputs
    if not re.match(r'^[\w.\-^]+$', ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")
    if period not in _VALID_PERIODS:
        period = "1y"

    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {exc}")

    if data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for '{ticker}'")

    closes = data["Close"].squeeze()

    start_val = float(closes.iloc[0])
    end_val = float(closes.iloc[-1])
    pct = (end_val - start_val) / start_val * 100
    line_color = "#22d3ee" if pct >= 0 else "#f87171"

    display_name = _TICKER_NAMES.get(ticker.upper(), ticker.upper())

    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    fig.patch.set_facecolor("#080e1f")
    ax.set_facecolor("#0d1526")

    ax.plot(closes.index, closes.values, color=line_color, linewidth=1.6)
    ax.fill_between(closes.index, closes.values, alpha=0.10, color=line_color)

    ax.set_title(f"{display_name}  ·  {period}", color="#e2e8f0", fontsize=10, pad=6, loc="left")
    ax.tick_params(colors="#475569", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y", color="#1e293b", linewidth=0.6, linestyle="--")

    sign = "+" if pct >= 0 else ""
    ax.annotate(
        f"{sign}{pct:.1f}%",
        xy=(0.99, 0.96), xycoords="axes fraction",
        fontsize=11, fontweight="bold",
        color=line_color, ha="right", va="top",
    )

    plt.tight_layout(pad=0.6)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"image": f"data:image/png;base64,{b64}"}