"use client";

import { useState, useRef, useEffect, useCallback } from "react";

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

function renderMd(text: string) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br/>");
}

function stripMd(text: string) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/<[^>]+>/g, "");
}

type VoiceStep =
  | "idle"
  | "recording"
  | "transcribing"
  | "thinking"
  | "speaking";

export function SparkChat({ context }: SparkChatProps = {}) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>(() => [
    {
      id: "welcome",
      role: "assistant",
      content: WELCOME_CONTENT,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [voiceStep, setVoiceStep] = useState<VoiceStep>("idle");

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Voice refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const voiceModeRef = useRef(false);
  const messagesRef = useRef(messages);
  const contextRef = useRef(context);

  // Function refs to break circular dependency
  const startVoiceRecordingRef = useRef<() => Promise<void>>(async () => {});

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  useEffect(() => {
    contextRef.current = context;
  }, [context]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (open && !isVoiceMode) setTimeout(() => inputRef.current?.focus(), 280);
  }, [open, isVoiceMode]);

  // ── TTS: speak reply, then restart recording if still in voice mode ───────────
  const speakAndListen = useCallback((text: string) => {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(stripMd(text));
    utter.rate = 1.05;
    setVoiceStep("speaking");
    utter.onend = () => {
      if (voiceModeRef.current) {
        startVoiceRecordingRef.current();
      } else {
        setVoiceStep("idle");
      }
    };
    utter.onerror = () => {
      setVoiceStep("idle");
      if (voiceModeRef.current) startVoiceRecordingRef.current();
    };
    window.speechSynthesis.speak(utter);
  }, []);

  // ── Send transcribed text to API, then speak the response ─────────────────────
  const sendVoiceMessage = useCallback(
    async (transcript: string) => {
      if (!transcript.trim() || !voiceModeRef.current) return;
      setVoiceStep("thinking");
      setLoading(true);

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "user",
          content: transcript,
          timestamp: new Date(),
        },
      ]);

      const history = messagesRef.current
        .filter((m) => m.id !== "welcome")
        .slice(-8)
        .map((m) => ({ role: m.role, content: m.content }));

      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: transcript,
            history,
            context: contextRef.current,
          }),
        });
        if (!res.ok) throw new Error(`${res.status}`);
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

        speakAndListen(data.reply);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content:
              "Oops! I couldn't connect to the server. Please try again.",
            timestamp: new Date(),
            isError: true,
          },
        ]);
        if (voiceModeRef.current) startVoiceRecordingRef.current();
      } finally {
        setLoading(false);
      }
    },
    [speakAndListen]
  );

  // ── Start recording with auto silence detection ───────────────────────────────
  const startVoiceRecording = useCallback(async () => {
    if (!voiceModeRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      // Silence detection via Web Audio API
      const audioCtx = new AudioContext();
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      audioCtx.createMediaStreamSource(stream).connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const SILENCE_THRESHOLD = 12; // 0-255, below = silence
      const SILENCE_DELAY_MS = 1400; // ms of silence before auto-stop
      const MIN_SPEECH_MS = 300; // must detect speech first

      let silenceTimer: ReturnType<typeof setTimeout> | null = null;
      let hasSpeech = false;
      let speechStart = 0;

      const poll = setInterval(() => {
        if (!voiceModeRef.current || recorder.state !== "recording") {
          clearInterval(poll);
          return;
        }
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

        if (avg > SILENCE_THRESHOLD) {
          // Sound detected
          if (!hasSpeech) {
            hasSpeech = true;
            speechStart = Date.now();
          }
          if (silenceTimer) {
            clearTimeout(silenceTimer);
            silenceTimer = null;
          }
        } else if (hasSpeech && Date.now() - speechStart > MIN_SPEECH_MS) {
          // Silence after speech → schedule auto-stop
          if (!silenceTimer) {
            silenceTimer = setTimeout(() => {
              clearInterval(poll);
              if (audioCtx.state !== "closed") audioCtx.close();
              if (recorder.state === "recording") recorder.stop();
            }, SILENCE_DELAY_MS);
          }
        }
      }, 80);

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        clearInterval(poll);
        if (audioCtx.state !== "closed") audioCtx.close();
        stream.getTracks().forEach((t) => t.stop());
        if (!voiceModeRef.current) return;

        setVoiceStep("transcribing");
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const form = new FormData();
        form.append("file", blob, "audio.webm");

        try {
          const res = await fetch(`${API_URL}/chat/transcribe`, {
            method: "POST",
            body: form,
          });
          if (res.ok) {
            const { transcript } = await res.json();
            if (transcript?.trim()) {
              await sendVoiceMessage(transcript.trim());
            } else {
              if (voiceModeRef.current) startVoiceRecordingRef.current();
            }
          } else {
            if (voiceModeRef.current) startVoiceRecordingRef.current();
          }
        } catch {
          if (voiceModeRef.current) startVoiceRecordingRef.current();
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setVoiceStep("recording");
    } catch {
      voiceModeRef.current = false;
      setIsVoiceMode(false);
      setVoiceStep("idle");
    }
  }, [sendVoiceMessage]);

  // Keep ref in sync so speakAndListen always calls the latest version
  useEffect(() => {
    startVoiceRecordingRef.current = startVoiceRecording;
  }, [startVoiceRecording]);

  const stopVoiceRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // ── Toggle voice mode ─────────────────────────────────────────────────────────
  const toggleVoiceMode = useCallback(() => {
    if (voiceModeRef.current) {
      // Deactivate
      voiceModeRef.current = false;
      setIsVoiceMode(false);
      setVoiceStep("idle");
      window.speechSynthesis.cancel();
      if (mediaRecorderRef.current?.state === "recording")
        mediaRecorderRef.current.stop();
    } else {
      // Activate → immediately start recording
      voiceModeRef.current = true;
      setIsVoiceMode(true);
      window.speechSynthesis.cancel();
      startVoiceRecording();
    }
  }, [startVoiceRecording]);

  // ── Text send ─────────────────────────────────────────────────────────────────
  const send = useCallback(
    async (text?: string) => {
      const body = (text ?? input).trim();
      if (!body || loading) return;

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "user",
          content: body,
          timestamp: new Date(),
        },
      ]);
      setInput("");
      setLoading(true);

      const history = messagesRef.current
        .filter((m) => m.id !== "welcome")
        .slice(-8)
        .map((m) => ({ role: m.role, content: m.content }));

      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: body, history, context }),
        });
        if (!res.ok) throw new Error(`${res.status}`);
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
            content:
              "Oops! I couldn't connect to the server. Please try again.",
            timestamp: new Date(),
            isError: true,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, context]
  );

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const fmt = (d: Date) =>
    d.toLocaleTimeString("en-CA", { hour: "2-digit", minute: "2-digit" });

  const voiceLabel: Record<VoiceStep, string> = {
    idle: "Starting…",
    recording: "Listening — will send automatically when you stop speaking",
    transcribing: "Transcribing…",
    thinking: "Thinking…",
    speaking: "Speaking…",
  };

  const voiceColor: Record<VoiceStep, string> = {
    idle: "text-[hsl(215,20%,50%)] bg-[hsl(222,47%,7%)]",
    recording: "text-red-400 bg-red-500/10",
    transcribing: "text-amber-400 bg-amber-500/10",
    thinking: "text-violet-400 bg-violet-500/10",
    speaking: "text-emerald-400 bg-emerald-500/10",
  };

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Trigger button */}
      <button
        onClick={() => {
          setOpen((v) => !v);
          window.speechSynthesis.cancel();
        }}
        aria-label="Toggle SparkChat"
        className={`fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full flex items-center justify-center border transition-all duration-200 shadow-[0_0_24px_rgba(0,212,255,0.25)]
          ${
            open
              ? "bg-[hsl(222,47%,13%)] border-[hsl(222,47%,20%)] text-[hsl(215,20%,55%)] hover:text-white"
              : "bg-cyan-400 border-cyan-300 text-[hsl(222,47%,5%)] hover:bg-cyan-300 hover:shadow-[0_0_32px_rgba(0,212,255,0.45)]"
          }`}
      >
        {open ? (
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <span className="text-2xl leading-none">⚡</span>
        )}
      </button>

      {/* Chat panel */}
      <div
        className={`fixed bottom-24 right-6 z-50 w-[360px] max-h-[580px] flex flex-col rounded-2xl border border-[hsl(222,47%,14%)] bg-[hsl(222,47%,5%)] shadow-[0_24px_60px_rgba(0,0,0,0.5),0_0_0_1px_rgba(0,212,255,0.06)] overflow-hidden transition-all duration-300 ease-[cubic-bezier(0.34,1.56,0.64,1)] origin-bottom-right
          ${
            open
              ? "scale-100 opacity-100 pointer-events-auto"
              : "scale-90 opacity-0 pointer-events-none"
          }`}
        role="dialog"
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[hsl(222,47%,14%)] bg-[hsl(222,47%,7%)]">
          <div className="w-9 h-9 rounded-full bg-[hsl(222,47%,13%)] border border-[hsl(222,47%,18%)] flex items-center justify-center text-xl flex-shrink-0">
            ⚡
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-bold text-white tracking-wide">
              SparkChat
            </p>
            <p className="text-[11px] text-[hsl(215,20%,55%)] flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
              Canadian Investment Guide
            </p>
          </div>

          {/* Mic toggle: click to activate voice mode / deactivate */}
          <button
            onClick={toggleVoiceMode}
            title={
              isVoiceMode
                ? "Desactivar modo voz"
                : "Activar conversación por voz"
            }
            className={`p-1.5 rounded-lg transition-all ${
              isVoiceMode
                ? "text-red-400 bg-red-500/10 border border-red-500/30 hover:bg-red-500/20"
                : "text-[hsl(215,20%,55%)] hover:text-cyan-400 hover:bg-cyan-400/10"
            }`}
          >
            {isVoiceMode ? (
              // Stop icon when active
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="currentColor"
                stroke="none"
              >
                <rect x="4" y="4" width="16" height="16" rx="2" />
              </svg>
            ) : (
              // Mic icon when inactive
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
              >
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            )}
          </button>

          <button
            onClick={() => {
              setOpen(false);
              window.speechSynthesis.cancel();
            }}
            className="text-[hsl(215,20%,55%)] hover:text-white transition-colors p-1"
            aria-label="Close"
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Voice status bar */}
        {isVoiceMode && (
          <div
            className={`px-4 py-2 flex items-center gap-2 text-[11px] font-semibold border-b border-[hsl(222,47%,14%)] ${voiceColor[voiceStep]}`}
          >
            {voiceStep === "recording" && (
              <span className="flex gap-0.5 items-end">
                {[8, 13, 16, 13, 8].map((h, i) => (
                  <span
                    key={i}
                    className="w-0.5 rounded-full bg-red-400 animate-bounce"
                    style={{
                      height: `${h}px`,
                      animationDelay: `${i * 0.08}s`,
                      animationDuration: "0.65s",
                    }}
                  />
                ))}
              </span>
            )}
            {voiceStep === "transcribing" && (
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
            )}
            {voiceStep === "thinking" && (
              <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
            )}
            {voiceStep === "speaking" && (
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
              >
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
              </svg>
            )}
            {voiceStep === "idle" && (
              <span className="w-2 h-2 rounded-full bg-current opacity-60" />
            )}
            <span>{voiceLabel[voiceStep]}</span>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3 scroll-smooth [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(222,47%,18%)] [&::-webkit-scrollbar-thumb]:rounded-full">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-end gap-2 animate-[fadeSlideUp_0.22s_ease_forwards] ${
                msg.role === "user" ? "flex-row-reverse" : ""
              }`}
            >
              {msg.role === "assistant" && (
                <div className="w-6 h-6 rounded-full bg-[hsl(222,47%,13%)] border border-[hsl(222,47%,18%)] flex items-center justify-center text-sm flex-shrink-0">
                  ⚡
                </div>
              )}
              <div
                className={`max-w-[82%] px-3.5 py-2.5 rounded-2xl ${
                  msg.role === "user"
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
                <span
                  className={`block text-[10px] mt-1.5 text-right ${
                    msg.role === "user"
                      ? "text-[hsl(222,47%,30%)]"
                      : "text-[hsl(215,20%,40%)]"
                  }`}
                >
                  {fmt(msg.timestamp)}
                </span>
              </div>
            </div>
          ))}

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
                    style={{
                      animationDelay: `${delay}s`,
                      animationDuration: "1s",
                    }}
                  />
                ))}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Suggestion chips (text mode only) */}
        {messages.length <= 1 && !isVoiceMode && (
          <div className="px-4 pb-3 flex flex-wrap gap-1.5">
            {SUGGESTED.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                className="text-[11px] px-3 py-1.5 rounded-full border border-[hsl(186,100%,50%,0.25)] bg-[hsl(186,100%,50%,0.07)] text-cyan-400 hover:bg-[hsl(186,100%,50%,0.15)] transition-all hover:-translate-y-px font-medium"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {/* Text input (text mode) */}
        {!isVoiceMode && (
          <div className="px-3 py-3 border-t border-[hsl(222,47%,14%)] bg-[hsl(222,47%,7%)] flex gap-2 items-end">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              placeholder="Ask about TFSA, RRSP, TSX..."
              disabled={loading}
              rows={1}
              className="flex-1 resize-none rounded-xl bg-[hsl(222,47%,13%)] border border-[hsl(222,47%,18%)] text-[13px] text-white placeholder:text-[hsl(215,20%,38%)] px-3.5 py-2.5 outline-none max-h-24 focus:border-cyan-400/50 focus:shadow-[0_0_0_2px_rgba(0,212,255,0.1)] transition-all disabled:opacity-50"
              style={{ lineHeight: "1.5" }}
            />
            <button
              onClick={() => send()}
              disabled={!input.trim() || loading}
              className="w-9 h-9 rounded-xl flex-shrink-0 bg-cyan-400 text-[hsl(222,47%,5%)] flex items-center justify-center hover:bg-cyan-300 transition-all disabled:bg-[hsl(222,47%,18%)] disabled:text-[hsl(215,20%,45%)] disabled:cursor-not-allowed hover:shadow-[0_0_16px_rgba(0,212,255,0.4)]"
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
              >
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        )}

        {/* Voice mode footer */}
        {isVoiceMode && (
          <div className="px-3 py-3 border-t border-[hsl(222,47%,14%)] bg-[hsl(222,47%,7%)] flex items-center justify-center">
            <p className="text-[11px] text-[hsl(215,20%,40%)]">
              {voiceStep === "recording"
                ? "Speak — sends automatically when you stop"
                : ""}
            </p>
          </div>
        )}

        <p className="text-center text-[10px] text-[hsl(215,20%,35%)] py-1.5 bg-[hsl(222,47%,7%)] border-t border-[hsl(222,47%,12%)]">
          Educational only · Not licensed financial advice
        </p>
      </div>

      <style>{`
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  );
}
