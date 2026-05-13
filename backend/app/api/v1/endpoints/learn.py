"""
app/api/v1/endpoints/learn.py
──────────────────────────────
Learning Hub — generates and caches educational module content via Groq.
Cache lives in Supabase `learn_modules` table, refreshed every 30 days.

One-time Supabase setup (run in SQL editor):
  CREATE TABLE IF NOT EXISTS learn_modules (
    module_id    TEXT PRIMARY KEY,
    content      JSONB NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT NOW()
  );
"""

import json
import os
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from app.api.dependencies import get_db

load_dotenv()

router = APIRouter()

CACHE_DAYS = 30

# Official links are hardcoded — LLM must never generate URLs
OFFICIAL_LINKS: dict[str, list[dict]] = {
    "banking-101": [
        {
            "org": "FCAC",
            "title": "Bank Accounts in Canada",
            "url": "https://www.canada.ca/en/financial-consumer-agency/services/banking/bank-accounts.html",
            "description": "Official FCAC guide to types of accounts, your rights, and how to choose one.",
        },
        {
            "org": "CDIC",
            "title": "How Your Deposit Is Protected",
            "url": "https://www.cdic.ca/your-coverage/protecting-your-deposit/",
            "description": "Canada Deposit Insurance Corporation — covers up to $100,000 per category.",
        },
        {
            "org": "FCAC",
            "title": "Bank Fees & Charges",
            "url": "https://www.canada.ca/en/financial-consumer-agency/services/banking/bank-fees.html",
            "description": "What fees Canadian banks can charge and how to compare them.",
        },
        {
            "org": "CBA",
            "title": "Banking for Newcomers to Canada",
            "url": "https://cba.ca/newcomers-to-canada",
            "description": "Canadian Bankers Association guide specifically written for newcomers.",
        },
    ],
    "credit": [
        {
            "org": "FCAC",
            "title": "Understanding Your Credit Report & Score",
            "url": "https://www.canada.ca/en/financial-consumer-agency/services/credit-reports-score.html",
            "description": "Official FCAC guide to credit scores in Canada — what they are and how to improve yours.",
        },
        {
            "org": "Equifax",
            "title": "Free Credit Report — Equifax Canada",
            "url": "https://www.equifax.com/personal/credit-report-services/free-credit-reports/",
            "description": "Request your free credit report from Equifax, one of Canada's two credit bureaus.",
        },
        {
            "org": "TransUnion",
            "title": "Free Credit Score — TransUnion Canada",
            "url": "https://www.transunion.ca/product/personal/free-credit-score",
            "description": "Check your free credit score and report from TransUnion Canada.",
        },
        {
            "org": "FCAC",
            "title": "Improving Your Credit Score",
            "url": "https://www.canada.ca/en/financial-consumer-agency/services/credit-reports-score/improve-credit-score.html",
            "description": "Step-by-step official guidance on rebuilding and improving your credit score.",
        },
    ],
    "tax-accounts": [
        {
            "org": "CRA",
            "title": "Tax-Free Savings Account (TFSA)",
            "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/tax-free-savings-account.html",
            "description": "Official CRA page — contribution limits, who qualifies, and how withdrawals work.",
        },
        {
            "org": "CRA",
            "title": "Registered Retirement Savings Plan (RRSP)",
            "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/rrsps-related-plans.html",
            "description": "CRA guide to RRSP — how to calculate your room, deductions, and deadlines.",
        },
        {
            "org": "CRA",
            "title": "First Home Savings Account (FHSA)",
            "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/first-home-savings-account.html",
            "description": "Canada's newest registered account — $8,000/year, $40,000 lifetime, tax-free for first home.",
        },
        {
            "org": "FCAC",
            "title": "Registered Savings & Investment Accounts",
            "url": "https://www.canada.ca/en/financial-consumer-agency/services/savings-investments.html",
            "description": "FCAC comparison of TFSA, RRSP, RESP, and FHSA — which one fits your situation.",
        },
    ],
    "investing": [
        {
            "org": "FCAC",
            "title": "Investing Basics",
            "url": "https://www.canada.ca/en/financial-consumer-agency/services/savings-investments/investing-basics.html",
            "description": "Official beginner's guide to investing in Canada — ETFs, stocks, risk, and brokers.",
        },
        {
            "org": "CIPF",
            "title": "Canadian Investor Protection Fund",
            "url": "https://www.cipf.ca/",
            "description": "Protects your investments up to $1M if a CIPF member firm becomes insolvent.",
        },
        {
            "org": "CSA",
            "title": "Investor Tools — Canadian Securities Administrators",
            "url": "https://www.securities-administrators.ca/investor-tools/",
            "description": "Check if your broker is registered and access investor education from Canada's regulators.",
        },
        {
            "org": "CIRO",
            "title": "Choosing a Broker in Canada",
            "url": "https://www.ciro.ca/investors",
            "description": "Canada's national self-regulatory organization for investment dealers — investor resources.",
        },
    ],
}

MODULE_PROMPTS: dict[str, str] = {
    "banking-101": """\
You are a financial education content generator for newcomers to Canada.
Create module content for "Canadian Banking 101" — aimed at recent arrivals (immigrants,
international students) who have never banked in Canada before.

Return ONLY a valid JSON object — no markdown, no extra text — with this exact structure:

{
  "sections": [
    {
      "heading": "section title",
      "content": "3-5 clear sentences. Plain English, no unexplained acronyms.",
      "key_points": ["point 1", "point 2", "point 3"]
    }
  ],
  "examples": [
    {
      "title": "short example title",
      "scenario": "realistic newcomer scenario in 1-2 sentences",
      "result": "outcome with real CAD numbers and bank names"
    }
  ],
  "quiz": [
    {
      "question": "question text",
      "options": ["A", "B", "C", "D"],
      "correct": 0,
      "explanation": "why the correct answer is right, in 1-2 sentences"
    }
  ]
}

Requirements:
- 5 sections: Types of Accounts | Debit vs Credit Cards | Bank Fees & How to Avoid Them | How e-Transfer Works | CDIC Deposit Insurance
- 3 examples with real CAD amounts (monthly fee ~$16.95, NSF fee ~$48, CDIC limit $100,000)
- 4 quiz questions covering: chequing vs savings purpose, CDIC limit, NSF fee consequence, ID needed to open account
- Mention the Big Six by name: RBC, TD, BMO, Scotiabank, CIBC, National Bank
- Friendly, encouraging tone suited for someone new to Canada\
""",
    "credit": """\
You are a financial education content generator for newcomers to Canada.
Create module content for "Credit & Building History" — Module 2 of the UNF Investor Learning Hub.
Target audience: newcomers in their first 6 months in Canada who have no Canadian credit history.

Return ONLY a valid JSON object with this exact structure:

{
  "sections": [
    { "heading": "string", "content": "3-5 clear sentences. Plain English.", "key_points": ["string", "string", "string"] }
  ],
  "examples": [
    { "title": "string", "scenario": "1-2 sentences", "result": "outcome with real numbers" }
  ],
  "quiz": [
    { "question": "string", "options": ["A", "B", "C", "D"], "correct": 0, "explanation": "1-2 sentences" }
  ]
}

Requirements:
- 5 sections: What Is a Credit Score in Canada (300–900 scale, Equifax & TransUnion) | How to Get Your Free Credit Report | Building Credit from Zero (secured cards, becoming an authorized user, credit-builder loans) | What Hurts Your Score (late payments, maxing cards, hard inquiries) | Credit Score Timeline for Newcomers
- 3 examples: checking your free report online, opening a secured card ($500 limit, typical annual fee $29–$59), the cost of a missed payment (score drop + interest)
- 4 quiz questions: credit score range in Canada, difference between Equifax and TransUnion, what a secured card requires, how long it takes to build a good score
- Mention real products: Scotiabank Value Visa, Home Trust Secured Visa, Capital One Guaranteed Mastercard
- Encouraging, practical tone — newcomers often feel embarrassed about having no history; normalize it\
""",
    "tax-accounts": """\
You are a financial education content generator for newcomers to Canada.
Create module content for "Tax-Advantaged Accounts" — Module 3 of the UNF Investor Learning Hub.
Target audience: newcomers who just received their first paycheck and want to save and invest tax-efficiently.

Return ONLY a valid JSON object with this exact structure:

{
  "sections": [
    { "heading": "string", "content": "3-5 clear sentences. Plain English.", "key_points": ["string", "string", "string"] }
  ],
  "examples": [
    { "title": "string", "scenario": "1-2 sentences", "result": "outcome with real CAD numbers" }
  ],
  "quiz": [
    { "question": "string", "options": ["A", "B", "C", "D"], "correct": 0, "explanation": "1-2 sentences" }
  ]
}

Requirements:
- 5 sections: TFSA — The Newcomer's Best Friend (available from day 1, 2024 limit $7,000, any Canadian resident 18+ with SIN) | RRSP — Retirement Savings (requires filed tax return, 18% of previous year income up to $31,560 for 2024, tax deduction) | FHSA — First Home Savings Account ($8,000/year, $40,000 lifetime, brand new in 2023, combines RRSP+TFSA advantages) | Which Account to Open First as a Newcomer (decision flowchart: TFSA first, then FHSA if planning to buy, then RRSP once earning well) | Key 2024/2025 Contribution Limits & Deadlines
- 3 examples: TFSA room calculation for someone who arrived in 2023, RRSP tax deduction at $60K salary saves ~$13,200 in taxes, FHSA combined with RRSP Home Buyers Plan for a $40K tax-free withdrawal
- 4 quiz questions: TFSA annual limit 2024, who qualifies for TFSA on arrival, what makes FHSA unique, RRSP contribution deadline
- Use plain language — many newcomers confuse these with regular bank accounts
- Emphasize that TFSA and FHSA are available to international students and temporary residents too\
""",
    "investing": """\
You are a financial education content generator for newcomers to Canada.
Create module content for "Investing Fundamentals" — Module 4 of the UNF Investor Learning Hub.
Target audience: newcomers with 1+ year in Canada who have opened a TFSA and want to start investing.

Return ONLY a valid JSON object with this exact structure:

{
  "sections": [
    { "heading": "string", "content": "3-5 clear sentences. Plain English.", "key_points": ["string", "string", "string"] }
  ],
  "examples": [
    { "title": "string", "scenario": "1-2 sentences", "result": "outcome with real CAD numbers" }
  ],
  "quiz": [
    { "question": "string", "options": ["A", "B", "C", "D"], "correct": 0, "explanation": "1-2 sentences" }
  ]
}

Requirements:
- 5 sections: What Is an ETF and Why Newcomers Love Them (low MER ~0.20%, instant diversification, trades like a stock) | The Simple 3-Fund Portfolio (XEQT or VEQT for Canadian all-in-one, or split: VCN + XUU + ZAG) | Risk Tolerance and Time Horizon (how to think about volatility — if you need the money in 2 years, don't invest it in stocks) | Canadian Brokers for Newcomers (Wealthsimple Trade: $0 commission; Questrade: $4.95–$9.95; big bank apps: convenient but higher fees) | Connecting to UNF Investor (how to use the platform's forecast and portfolio optimizer to practice before investing real money)
- 3 examples: buying XEQT inside a TFSA (MER 0.20%, holds 9,000+ stocks globally), the compound interest effect over 10 years ($500/month at 7% → ~$87,000), Wealthsimple vs TD Direct Investing fee comparison on a $1,000 trade
- 4 quiz questions: what MER stands for, the main benefit of an ETF over a single stock, what risk tolerance means, how UNF Investor's portfolio optimizer helps
- Mention UNF Investor's features naturally: "use the Forecasting tool to see historical patterns before you buy" and "the Portfolio optimizer helps you find the right allocation"
- Energizing, forward-looking tone — this is the module where theory becomes action\
""",
}


async def _generate_content(module_id: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    prompt = MODULE_PROMPTS[module_id]

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.4,
                "max_tokens": 3000,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial education content generator. Respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"LLM error: {resp.text}")

    content = json.loads(resp.json()["choices"][0]["message"]["content"])
    content["official_links"] = OFFICIAL_LINKS.get(module_id, [])
    return content


@router.get("/module/{module_id}")
async def get_module(module_id: str, db: Client = Depends(get_db)) -> dict:
    """
    Return educational content for a learning module.
    Serves Supabase cache if < 30 days old, otherwise regenerates via Groq.
    """
    if module_id not in MODULE_PROMPTS:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    # Try cache
    try:
        row = (
            db.table("learn_modules")
            .select("content, generated_at")
            .eq("module_id", module_id)
            .single()
            .execute()
        )
        if row.data:
            generated_at = datetime.fromisoformat(
                row.data["generated_at"].replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) - generated_at < timedelta(days=CACHE_DAYS):
                return {"module_id": module_id, "cached": True, **row.data["content"]}
    except Exception:
        pass  # No row yet — generate fresh

    content = await _generate_content(module_id)

    db.table("learn_modules").upsert(
        {
            "module_id": module_id,
            "content": content,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    ).execute()

    return {"module_id": module_id, "cached": False, **content}
