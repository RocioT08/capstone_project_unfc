"use client";

import { useState } from "react";
import { CheckCircle2, XCircle, RotateCcw } from "lucide-react";

interface QuizQuestion {
  question: string;
  options: string[];
  correct: number;
  explanation: string;
}

interface Props {
  quiz: QuizQuestion[];
  accentCorrect?: string;
  submitBtn?: string;
}

export default function QuizSection({
  quiz,
  accentCorrect = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
  submitBtn = "bg-emerald-600 hover:bg-emerald-500",
}: Props) {
  const [selected, setSelected] = useState<(number | null)[]>(quiz.map(() => null));
  const [submitted, setSubmitted] = useState(false);

  const score = submitted
    ? selected.filter((ans, i) => ans === quiz[i].correct).length
    : 0;

  function reset() {
    setSelected(quiz.map(() => null));
    setSubmitted(false);
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white">Quick Quiz</h2>
        {submitted && (
          <button
            onClick={reset}
            className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-white transition-colors"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Try again
          </button>
        )}
      </div>

      {submitted && (
        <div
          className={`rounded-xl px-5 py-3 text-sm font-semibold ${
            score === quiz.length
              ? accentCorrect
              : score >= quiz.length / 2
              ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
              : "bg-red-500/20 text-red-400 border border-red-500/30"
          }`}
        >
          {score === quiz.length
            ? `Perfect score! ${score}/${quiz.length} — you're ready to open your first account.`
            : score >= quiz.length / 2
            ? `Good effort — ${score}/${quiz.length}. Review the sections above for the ones you missed.`
            : `${score}/${quiz.length} — read through the module again, then try once more.`}
        </div>
      )}

      <div className="space-y-5">
        {quiz.map((q, qi) => {
          const answered = selected[qi] !== null;
          const isCorrect = submitted && selected[qi] === q.correct;
          const isWrong = submitted && answered && selected[qi] !== q.correct;

          return (
            <div
              key={qi}
              className={`rounded-xl border p-5 space-y-3 transition-colors ${
                isCorrect
                  ? "border-emerald-500/40 bg-emerald-500/5"
                  : isWrong
                  ? "border-red-500/40 bg-red-500/5"
                  : "border-white/10 bg-white/3"
              }`}
            >
              <p className="text-white font-medium text-sm">
                <span className="text-emerald-400 font-bold mr-2">{qi + 1}.</span>
                {q.question}
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {q.options.map((opt, oi) => {
                  const isSelected = selected[qi] === oi;
                  const isAnswer = oi === q.correct;

                  let style =
                    "rounded-lg border px-3 py-2 text-sm text-left transition-colors cursor-pointer ";

                  if (submitted) {
                    if (isAnswer) {
                      style += "border-emerald-500/60 bg-emerald-500/15 text-emerald-300";
                    } else if (isSelected && !isAnswer) {
                      style += "border-red-500/60 bg-red-500/10 text-red-400";
                    } else {
                      style += "border-white/8 text-gray-500 cursor-default";
                    }
                  } else if (isSelected) {
                    style += "border-emerald-500/50 bg-emerald-500/10 text-white";
                  } else {
                    style += "border-white/10 text-gray-300 hover:border-emerald-500/30 hover:text-white";
                  }

                  return (
                    <button
                      key={oi}
                      disabled={submitted}
                      className={style}
                      onClick={() => {
                        const next = [...selected];
                        next[qi] = oi;
                        setSelected(next);
                      }}
                    >
                      {opt}
                    </button>
                  );
                })}
              </div>

              {submitted && (
                <div
                  className={`flex items-start gap-2 text-sm rounded-lg px-3 py-2 ${
                    isCorrect
                      ? "bg-emerald-500/10 text-emerald-300"
                      : "bg-red-500/10 text-red-300"
                  }`}
                >
                  {isCorrect ? (
                    <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0" />
                  ) : (
                    <XCircle className="h-4 w-4 mt-0.5 shrink-0" />
                  )}
                  <span>{q.explanation}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {!submitted && (
        <button
          disabled={selected.some((s) => s === null)}
          onClick={() => setSubmitted(true)}
          className={`w-full rounded-xl disabled:opacity-40 disabled:cursor-not-allowed transition-colors py-3 text-white font-semibold text-sm ${submitBtn}`}
        >
          {selected.some((s) => s === null)
            ? `Answer all ${quiz.length} questions to submit`
            : "Check My Answers"}
        </button>
      )}
    </div>
  );
}
