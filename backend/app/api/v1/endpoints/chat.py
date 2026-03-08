"""
app/api/v1/endpoints/chat.py
─────────────────────────────
SparkChat — AI-powered Canadian investment education chatbot.
Follows the same pattern as all other v1 endpoints in this project.

LLM: Groq (free tier) — set GROQ_API_KEY in your .env also for voice transcription.
"""

import json
import os
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile
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
- When given portfolio or forecast data, interpret the numbers in plain language a beginner can understand"""


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