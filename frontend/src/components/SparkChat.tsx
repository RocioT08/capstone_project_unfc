"use client";

/**
 * SparkChat.tsx
 * Drop this in: frontend/src/components/SparkChat.tsx
 *
 * Uses your existing Tailwind dark-navy design system (cyan/violet accents).
 * Calls your FastAPI backend at /api/chat — no Supabase needed.
 *
 * BACKEND SETUP → see backend/app/chat_routes.py (provided separately)
 *
 * USAGE (add to Layout.tsx or App.tsx):
 *   import { SparkChat } from "./SparkChat";
 *   <SparkChat />
 */

import { useState, useRef, useEffect, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isError?: boolean;
}

interface SparkChatProps {
  context?: {
    type: "portfolio_optimize" | "portfolio_stats" | "forecast" | "analyze";
    data: unknown;
  };
}

// ─── Constants ────────────────────────────────────────────────────────────────
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const SUGGESTED = [
  "What is a TFSA?",
  "RRSP vs TFSA difference?",
  "How does the TSX work?",
  "What are Canadian ETFs?",
  "Explain capital gains tax in Canada",
  "What is portfolio diversification?",
];

const WELCOME_CONTENT =
  "Hey! I'm **Spark** ⚡ — your Canadian investment guide.\n\nI can explain **TFSA, RRSP, FHSA**, how the **TSX** works, ETFs, portfolio theory, and more.\n\n*Just for learning — not licensed financial advice!* What do you want to explore?";

// ─── Simple markdown → HTML (bold, italic, line breaks) ──────────────────────
function renderMd(text: string) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br/>");
}

// ─── SparkChat Component ──────────────────────────────────────────────────────
export function SparkChat({ context }: SparkChatProps = {}) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>(() => [
    { id: "welcome", role: "assistant", content: WELCOME_CONTENT, timestamp: new Date() },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Focus input on open
  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 280);
  }, [open]);

  // ── Send message ────────────────────────────────────────────────────────────
  const send = useCallback(
    async (text?: string) => {
      const body = (text ?? input).trim();
      if (!body || loading) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: body,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setLoading(true);

      // Build history (last 8 turns for context window)
      const history = messages
        .filter((m) => m.id !== "welcome")
        .slice(-8)
        .map((m) => ({ role: m.role, content: m.content }));

      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: body, history, context }),
        });

        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();

        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: data.reply,
            timestamp: new Date(),
          },
        ]);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "Oops! I couldn't connect to the server. Please try again.",
            timestamp: new Date(),
            isError: true,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, messages]
  );

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const fmt = (d: Date) =>
    d.toLocaleTimeString("en-CA", { hour: "2-digit", minute: "2-digit" });

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <>
      {/* ── Trigger button ── */}
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label="Toggle SparkChat"
        className={`
          fixed bottom-6 right-6 z-50
          w-14 h-14 rounded-full
          flex items-center justify-center
          border transition-all duration-200
          shadow-[0_0_24px_rgba(0,212,255,0.25)]
          ${open
            ? "bg-[hsl(222,47%,13%)] border-[hsl(222,47%,20%)] text-[hsl(215,20%,55%)] hover:text-white"
            : "bg-cyan-400 border-cyan-300 text-[hsl(222,47%,5%)] hover:bg-cyan-300 hover:shadow-[0_0_32px_rgba(0,212,255,0.45)]"
          }
        `}
      >
        {open ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <span className="text-2xl leading-none">⚡</span>
        )}
      </button>

      {/* ── Chat panel ── */}
      <div
        className={`
          fixed bottom-24 right-6 z-50
          w-[360px] max-h-[580px]
          flex flex-col
          rounded-2xl border border-[hsl(222,47%,14%)]
          bg-[hsl(222,47%,5%)]
          shadow-[0_24px_60px_rgba(0,0,0,0.5),0_0_0_1px_rgba(0,212,255,0.06)]
          overflow-hidden
          transition-all duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)]
          origin-bottom-right
          ${open ? "scale-100 opacity-100 pointer-events-auto" : "scale-90 opacity-0 pointer-events-none"}
        `}
        role="dialog"
        aria-label="SparkChat investment chatbot"
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[hsl(222,47%,14%)] bg-[hsl(222,47%,7%)]">
          <div className="w-9 h-9 rounded-full bg-[hsl(222,47%,13%)] border border-[hsl(222,47%,18%)] flex items-center justify-center text-xl flex-shrink-0">
            ⚡
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-bold text-white tracking-wide">SparkChat</p>
            <p className="text-[11px] text-[hsl(215,20%,55%)] flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
              Canadian Investment Guide
            </p>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="text-[hsl(215,20%,55%)] hover:text-white transition-colors p-1"
            aria-label="Close"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3 scroll-smooth
          [&::-webkit-scrollbar]:w-1
          [&::-webkit-scrollbar-track]:bg-transparent
          [&::-webkit-scrollbar-thumb]:bg-[hsl(222,47%,18%)]
          [&::-webkit-scrollbar-thumb]:rounded-full">

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-end gap-2 animate-[fadeSlideUp_0.22s_ease_forwards] ${msg.role === "user" ? "flex-row-reverse" : ""}`}
            >
              {/* Avatar (assistant only) */}
              {msg.role === "assistant" && (
                <div className="w-6 h-6 rounded-full bg-[hsl(222,47%,13%)] border border-[hsl(222,47%,18%)] flex items-center justify-center text-sm flex-shrink-0">
                  ⚡
                </div>
              )}

              <div
                className={`max-w-[82%] px-3.5 py-2.5 rounded-2xl
                  ${msg.role === "user"
                    ? "bg-cyan-400 text-[hsl(222,47%,5%)] rounded-br-sm"
                    : msg.isError
                      ? "bg-[hsl(0,40%,12%)] border border-[hsl(0,60%,25%)] text-[hsl(0,84%,75%)] rounded-bl-sm"
                      : "bg-[hsl(222,47%,10%)] border border-[hsl(222,47%,16%)] text-[hsl(0,0%,90%)] rounded-bl-sm"
                  }`}
              >
                <p
                  className="text-[13px] leading-relaxed m-0"
                  dangerouslySetInnerHTML={{ __html: renderMd(msg.content) }}
                />
                <span className={`block text-[10px] mt-1.5 text-right
                  ${msg.role === "user" ? "text-[hsl(222,47%,30%)]" : "text-[hsl(215,20%,40%)]"}`}>
                  {fmt(msg.timestamp)}
                </span>
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="flex items-end gap-2">
              <div className="w-6 h-6 rounded-full bg-[hsl(222,47%,13%)] border border-[hsl(222,47%,18%)] flex items-center justify-center text-sm flex-shrink-0">
                ⚡
              </div>
              <div className="px-4 py-3 rounded-2xl rounded-bl-sm bg-[hsl(222,47%,10%)] border border-[hsl(222,47%,16%)] flex gap-1.5 items-center">
                {[0, 0.18, 0.36].map((delay, i) => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-[hsl(215,20%,45%)] animate-bounce"
                    style={{ animationDelay: `${delay}s`, animationDuration: "1s" }}
                  />
                ))}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Suggestion chips — only at start */}
        {messages.length <= 1 && (
          <div className="px-4 pb-3 flex flex-wrap gap-1.5">
            {SUGGESTED.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                className="text-[11px] px-3 py-1.5 rounded-full
                  border border-[hsl(186,100%,50%,0.25)] bg-[hsl(186,100%,50%,0.07)]
                  text-cyan-400 hover:bg-[hsl(186,100%,50%,0.15)]
                  transition-all hover:-translate-y-px font-medium"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {/* Input area */}
        <div className="px-3 py-3 border-t border-[hsl(222,47%,14%)] bg-[hsl(222,47%,7%)] flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ask about TFSA, RRSP, TSX..."
            disabled={loading}
            rows={1}
            className="flex-1 resize-none rounded-xl
              bg-[hsl(222,47%,13%)] border border-[hsl(222,47%,18%)]
              text-[13px] text-white placeholder:text-[hsl(215,20%,38%)]
              px-3.5 py-2.5 outline-none max-h-24
              focus:border-cyan-400/50 focus:shadow-[0_0_0_2px_rgba(0,212,255,0.1)]
              transition-all disabled:opacity-50"
            style={{ lineHeight: "1.5" }}
          />
          <button
            onClick={() => send()}
            disabled={!input.trim() || loading}
            className="w-9 h-9 rounded-xl flex-shrink-0
              bg-cyan-400 text-[hsl(222,47%,5%)]
              flex items-center justify-center
              hover:bg-cyan-300 transition-all
              disabled:bg-[hsl(222,47%,18%)] disabled:text-[hsl(215,20%,45%)] disabled:cursor-not-allowed
              hover:shadow-[0_0_16px_rgba(0,212,255,0.4)]"
            aria-label="Send"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>

        {/* Disclaimer */}
        <p className="text-center text-[10px] text-[hsl(215,20%,35%)] py-1.5 bg-[hsl(222,47%,7%)] border-t border-[hsl(222,47%,12%)]">
          Educational only · Not licensed financial advice
        </p>
      </div>

      {/* Keyframe for message animation */}
      <style>{`
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  );
}