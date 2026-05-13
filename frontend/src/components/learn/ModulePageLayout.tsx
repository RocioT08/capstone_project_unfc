import { ArrowLeft, ExternalLink, CheckCircle, MessageCircle, AlertCircle } from "lucide-react";
import Link from "next/link";
import QuizSection from "./QuizSection";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000/api/v1";

export interface ModuleAccent {
  circle: string;       // bg class for numbered circle
  text: string;         // primary text color
  badge: string;        // inline badge classes (bg + text)
  sectionHover: string; // card hover border
  exampleCard: string;  // example card border + bg
  exampleTitle: string; // example card title color
  linkBadge: string;    // org badge bg + text
  checkIcon: string;    // check icon color
  submitBtn: string;    // quiz submit button classes
  quizCorrect: string;  // correct answer highlight
  nextCircle: string;   // next module circle classes
}

export interface ModulePageConfig {
  id: string;
  number: string;
  timing: string;
  title: string;
  subtitle: string;
  stat: string;
  statDescription: string;
  accent: ModuleAccent;
  next?: {
    number: string;
    title: string;
    href: string | null;
  };
}

interface Section { heading: string; content: string; key_points: string[]; }
interface Example { title: string; scenario: string; result: string; }
interface OfficialLink { org: string; title: string; url: string; description: string; }
interface QuizQuestion { question: string; options: string[]; correct: number; explanation: string; }
interface ModuleData { sections: Section[]; examples: Example[]; official_links: OfficialLink[]; quiz: QuizQuestion[]; }

async function fetchModuleData(id: string): Promise<ModuleData | null> {
  try {
    const res = await fetch(`${API_URL}/learn/module/${id}`, {
      next: { revalidate: 60 * 60 * 24 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function ModulePageLayout({ config }: { config: ModulePageConfig }) {
  const data = await fetchModuleData(config.id);
  const { accent } = config;

  return (
    <div className="min-h-screen py-10 space-y-12 max-w-3xl mx-auto px-4">

      {/* Back */}
      <Link
        href="/learn"
        className={`inline-flex items-center gap-2 text-sm text-gray-400 hover:${accent.text} transition-colors`}
      >
        <ArrowLeft className="h-4 w-4" />
        Learning Hub
      </Link>

      {/* Module header */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full ${accent.circle} flex items-center justify-center text-white font-bold text-sm`}>
            {config.number}
          </div>
          <span className={`text-[11px] font-semibold tracking-widest uppercase ${accent.text}`}>
            {config.timing}
          </span>
        </div>
        <h1 className="text-4xl font-extrabold text-white">{config.title}</h1>
        <p className="text-gray-400 text-lg leading-relaxed">{config.subtitle}</p>
        <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${accent.badge}`}>
          Addresses: {config.stat} — {config.statDescription}
        </div>
      </div>

      {/* Error state */}
      {!data && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-5 py-4 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-amber-300 font-medium text-sm">Content unavailable</p>
            <p className="text-amber-400/70 text-xs">
              Make sure the backend is running and the{" "}
              <code className="font-mono">learn_modules</code> Supabase table exists.
            </p>
          </div>
        </div>
      )}

      {data && (
        <>
          {/* Sections */}
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-white border-b border-white/10 pb-2">
              What You'll Learn
            </h2>
            {data.sections.map((section, i) => (
              <div
                key={i}
                className={`rounded-xl border border-white/8 bg-white/3 p-6 space-y-4 transition-colors ${accent.sectionHover}`}
              >
                <h3 className="text-white font-bold text-base">{section.heading}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{section.content}</p>
                <ul className="space-y-1.5">
                  {section.key_points.map((point, pi) => (
                    <li key={pi} className="flex items-start gap-2 text-sm text-gray-300">
                      <CheckCircle className={`h-4 w-4 ${accent.checkIcon} shrink-0 mt-0.5`} />
                      {point}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Examples */}
          <div className="space-y-4">
            <h2 className="text-xl font-bold text-white border-b border-white/10 pb-2">
              Real Examples
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {data.examples.map((ex, i) => (
                <div key={i} className={`rounded-xl border p-4 space-y-2 ${accent.exampleCard}`}>
                  <p className={`font-semibold text-sm ${accent.exampleTitle}`}>{ex.title}</p>
                  <p className="text-gray-400 text-xs leading-relaxed">{ex.scenario}</p>
                  <p className="text-white text-sm font-medium">{ex.result}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Official links */}
          <div className="space-y-4">
            <h2 className="text-xl font-bold text-white border-b border-white/10 pb-2">
              Official Sources
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {data.official_links.map((link, i) => (
                <a
                  key={i}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`group rounded-xl border border-white/8 bg-white/3 transition-colors p-4 flex items-start gap-3 ${accent.sectionHover}`}
                >
                  <div className={`flex-shrink-0 rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide mt-0.5 ${accent.linkBadge}`}>
                    {link.org}
                  </div>
                  <div className="space-y-0.5 min-w-0">
                    <p className={`text-white text-sm font-medium transition-colors flex items-center gap-1 group-hover:${accent.text}`}>
                      {link.title}
                      <ExternalLink className="h-3 w-3 opacity-50" />
                    </p>
                    <p className="text-gray-500 text-xs leading-relaxed">{link.description}</p>
                  </div>
                </a>
              ))}
            </div>
          </div>

          {/* Quiz */}
          <div className="rounded-2xl border border-white/8 bg-white/2 p-6">
            <QuizSection quiz={data.quiz} accentCorrect={accent.quizCorrect} submitBtn={accent.submitBtn} />
          </div>
        </>
      )}

      {/* Spark CTA */}
      <div className="rounded-2xl border border-cyan-400/20 bg-gradient-to-r from-slate-900/60 to-slate-800/40 px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="space-y-1 text-center sm:text-left">
          <p className="text-white font-semibold">Have a specific question about this topic?</p>
          <p className="text-gray-400 text-sm">
            Ask Spark — it answers in your language with Canadian context.
          </p>
        </div>
        <Link
          href="/"
          className="flex-shrink-0 inline-flex items-center gap-2 rounded-full bg-cyan-500 hover:bg-cyan-400 transition-colors px-5 py-2.5 text-sm font-semibold text-black"
        >
          <MessageCircle className="h-4 w-4" />
          Ask Spark
        </Link>
      </div>

      {/* Next module teaser */}
      {config.next && (
        <div className="rounded-xl border border-white/8 bg-white/2 px-5 py-4 flex items-center justify-between">
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-widest mb-0.5">Up next</p>
            <p className="text-white font-medium text-sm">
              Module {config.next.number} — {config.next.title}
            </p>
            {!config.next.href && <p className="text-gray-500 text-xs">Coming soon</p>}
          </div>
          {config.next.href ? (
            <Link href={config.next.href} className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${accent.nextCircle}`}>
              →
            </Link>
          ) : (
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${accent.nextCircle}`}>
              {config.next.number}
            </div>
          )}
        </div>
      )}

    </div>
  );
}
