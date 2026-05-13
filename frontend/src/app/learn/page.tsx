import {
  BookOpen,
  Building2,
  CreditCard,
  Receipt,
  TrendingUp,
  PieChart,
  LineChart,
  MessageCircle,
  Lock,
  ArrowRight,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import Link from "next/link";

const newcomerModules = [
  {
    number: "01",
    timing: "NEWCOMERS · DAY 1",
    title: "Canadian Banking 101",
    description:
      "How accounts work, debit vs credit, fees, e-Transfers, and reading a bank statement — in plain language.",
    stat: "38%",
    statLabel: "don't understand the system",
    accent: {
      border: "border-emerald-500/50",
      bg: "bg-emerald-500/10",
      circle: "bg-emerald-600",
      text: "text-emerald-400",
      stat: "text-emerald-400",
      hover: "hover:border-emerald-500/70",
    },
    icon: Building2,
    available: true,
    href: "/learn/banking-101",
  },
  {
    number: "02",
    timing: "FIRST 6 MONTHS",
    title: "Credit & Building History",
    description:
      "Why credit score matters in Canada, how to build it from zero, secured cards, and avoiding common traps.",
    stat: "80%",
    statLabel: "face credit barriers",
    accent: {
      border: "border-amber-500/50",
      bg: "bg-amber-500/10",
      circle: "bg-amber-600",
      text: "text-amber-400",
      stat: "text-amber-400",
      hover: "hover:border-amber-500/70",
    },
    icon: CreditCard,
    available: true,
    href: "/learn/credit",
  },
  {
    number: "03",
    timing: "AFTER FIRST PAYCHECK",
    title: "Tax-Advantaged Accounts",
    description:
      "TFSA, FHSA, RRSP — what they are, who qualifies, contribution rules, and which to open first.",
    stat: "$40K",
    statLabel: "FHSA — most underused",
    accent: {
      border: "border-orange-500/50",
      bg: "bg-orange-500/10",
      circle: "bg-orange-700",
      text: "text-orange-400",
      stat: "text-orange-400",
      hover: "hover:border-orange-500/70",
    },
    icon: Receipt,
    available: true,
    href: "/learn/tax-accounts",
  },
  {
    number: "04",
    timing: "1+ YEAR IN CANADA",
    title: "Investing Fundamentals",
    description:
      "ETFs, diversification, risk tolerance — connected to Spark and your live portfolio for personalized practice.",
    stat: "→ Spark",
    statLabel: "ties into the AI agent",
    accent: {
      border: "border-violet-500/50",
      bg: "bg-violet-500/10",
      circle: "bg-violet-700",
      text: "text-violet-400",
      stat: "text-violet-400",
      hover: "hover:border-violet-500/70",
    },
    icon: TrendingUp,
    available: true,
    href: "/learn/investing",
  },
];

const platformTopics = [
  {
    icon: TrendingUp,
    title: "Price Forecasting",
    description:
      "EWM, Prophet, and Prophet+XGBoost — how the platform predicts prices using historical patterns.",
  },
  {
    icon: PieChart,
    title: "Portfolio Theory",
    description:
      "Modern Portfolio Theory, diversification, and the efficient frontier — maximizing returns for a given risk.",
  },
  {
    icon: LineChart,
    title: "Risk Metrics",
    description:
      "Sharpe Ratio, Max Drawdown, Volatility, Skewness, and Kurtosis — the numbers every investor should know.",
  },
  {
    icon: BookOpen,
    title: "Optimization Methods",
    description:
      "How PyPortfolioOpt solves for ideal portfolio weights using convex optimization under real-world constraints.",
  },
];

export default function LearnPage() {
  return (
    <div className="relative min-h-[80vh] py-12 space-y-16">

      {/* Header */}
      <div className="text-center space-y-4 max-w-2xl mx-auto">
        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-4 py-1.5 text-sm font-medium text-cyan-400">
          <BookOpen className="h-4 w-4" />
          The Learning Hub
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight text-white">
          A guided journey from{" "}
          <span className="bg-gradient-to-r from-emerald-400 to-violet-500 bg-clip-text text-transparent">
            arrival to investing
          </span>
        </h1>
        <p className="text-lg text-gray-400">
          Four sequential modules that meet newcomers where they are — and walk them step by step toward financial confidence.
        </p>
      </div>

      {/* Newcomer Journey — timeline + cards */}
      <div className="max-w-6xl mx-auto px-4">

        {/* Timeline connector row */}
        <div className="hidden md:flex items-center justify-between mb-4 px-[9%]">
          {newcomerModules.map((mod, i) => (
            <div key={mod.number} className="flex items-center flex-1">
              <div
                className={`flex-shrink-0 w-11 h-11 rounded-full ${mod.accent.circle} flex items-center justify-center text-white font-bold text-sm shadow-lg`}
              >
                {mod.number}
              </div>
              {i < newcomerModules.length - 1 && (
                <div className="flex-1 border-t-2 border-dashed border-white/15 mx-2" />
              )}
            </div>
          ))}
        </div>

        {/* Module cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {newcomerModules.map((mod) => {
            const cardContent = (
              <>
                {/* Mobile number badge */}
                <div className={`md:hidden w-9 h-9 rounded-full ${mod.accent.circle} flex items-center justify-center text-white font-bold text-xs`}>
                  {mod.number}
                </div>

                {/* Timing label */}
                <p className={`text-[10px] font-semibold tracking-widest uppercase ${mod.accent.text}`}>
                  {mod.timing}
                </p>

                {/* Title */}
                <h3 className="text-white font-bold text-base leading-snug">{mod.title}</h3>

                {/* Description */}
                <p className="text-gray-400 text-sm leading-relaxed flex-1">{mod.description}</p>

                {/* Divider */}
                <div className="border-t border-white/10 pt-3 mt-auto">
                  <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-0.5">Addresses</p>
                  <p className={`text-2xl font-extrabold ${mod.accent.stat}`}>{mod.stat}</p>
                  <p className="text-xs text-gray-500">{mod.statLabel}</p>
                </div>

                {/* Available: start arrow / unavailable: lock */}
                {mod.available ? (
                  <div className={`absolute top-3 right-3 ${mod.accent.text}`}>
                    <ArrowRight className="h-4 w-4" />
                  </div>
                ) : (
                  <div className="absolute top-3 right-3">
                    <Lock className="h-3.5 w-3.5 text-gray-600" />
                  </div>
                )}
              </>
            );

            const baseClass = `relative rounded-xl border ${mod.accent.border} ${mod.accent.bg} ${mod.accent.hover} backdrop-blur p-5 flex flex-col gap-3 transition-colors`;

            return mod.available && mod.href ? (
              <Link key={mod.number} href={mod.href} className={`${baseClass} cursor-pointer`}>
                {cardContent}
              </Link>
            ) : (
              <div key={mod.number} className={baseClass}>
                {cardContent}
              </div>
            );
          })}
        </div>
      </div>

      {/* Spark CTA banner */}
      <div className="max-w-4xl mx-auto px-4">
        <div className="rounded-2xl border border-cyan-400/20 bg-gradient-to-r from-emerald-900/30 to-violet-900/30 px-8 py-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center md:text-left">
            <p className="text-white font-bold text-lg">
              Spark answers what you ask.{" "}
              <span className="text-cyan-400">The Hub teaches you what to ask.</span>
            </p>
            <p className="text-gray-400 text-sm">
              Use the Hub for structured learning — then ask Spark for instant answers while you invest.
            </p>
          </div>
          <Link
            href="/"
            className="flex-shrink-0 inline-flex items-center gap-2 rounded-full bg-cyan-500 hover:bg-cyan-400 transition-colors px-5 py-2.5 text-sm font-semibold text-black"
          >
            <MessageCircle className="h-4 w-4" />
            Ask Spark now
          </Link>
        </div>
      </div>

      {/* Platform Concepts */}
      <div className="max-w-4xl mx-auto px-4 space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold text-white">Platform Concepts</h2>
          <p className="text-gray-400 text-sm">
            The science behind every tool — understand the models you're using.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {platformTopics.map(({ icon: Icon, title, description }) => (
            <Card
              key={title}
              className="border border-white/8 bg-white/3 backdrop-blur hover:border-cyan-400/30 transition-colors"
            >
              <CardHeader>
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-cyan-400/10 mb-3">
                  <Icon className="h-5 w-5 text-cyan-400" />
                </div>
                <CardTitle className="text-lg text-white">{title}</CardTitle>
                <CardDescription className="text-gray-400 text-sm leading-relaxed">
                  {description}
                </CardDescription>
              </CardHeader>
            </Card>
          ))}
        </div>
        <p className="text-center text-gray-600 text-sm">
          More guided lessons coming soon — start exploring the tools to learn by doing.
        </p>
      </div>

    </div>
  );
}
