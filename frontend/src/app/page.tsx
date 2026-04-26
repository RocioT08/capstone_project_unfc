import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import {
  TrendingUp,
  BookOpen,
  ShieldCheck,
  BarChart3,
  PieChart,
} from "lucide-react";
import { MouseTrail } from "@/components/MouseTrail";

export default function Home() {
  return (
    <div className="relative flex flex-col gap-16 py-10 overflow-hidden">
      <MouseTrail />

      {/* Background effects */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
      >
        {/* Subtle grid */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.025)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.025)_1px,transparent_1px)] [background-size:44px_44px]" />
        {/* Cyan glow top-right */}
        <div className="blob-1 absolute -top-40 right-0 h-[560px] w-[560px] rounded-full bg-cyan-500 opacity-[0.06] blur-[110px]" />
        {/* Violet glow bottom-left */}
        <div className="blob-2 absolute bottom-0 -left-20 h-[500px] w-[500px] rounded-full bg-violet-600 opacity-[0.07] blur-[100px]" />
        {/* Cyan glow center */}
        <div className="blob-3 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-[400px] w-[400px] rounded-full bg-cyan-400 opacity-[0.04] blur-[90px]" />
      </div>

      {/* ── Hero ── */}
      <section className="flex flex-col md:flex-row items-center gap-12 min-h-[60vh] justify-center">
        {/* Mascot */}
        <div className="shrink-0">
          <Image
            src="/agent-avatar.png"
            alt="Foxy the investment guide"
            width={290}
            height={290}
            className="drop-shadow-[0_0_40px_rgba(0,212,255,0.15)]"
            priority
          />
        </div>

        {/* Text block */}
        <div className="space-y-6 text-left max-w-xl">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-4 py-1.5 text-sm font-semibold text-cyan-400">
            <ShieldCheck className="h-4 w-4" />
            Educational Platform
          </div>

          <h1 className="text-5xl sm:text-6xl font-extrabold tracking-tight text-white leading-[1.1]">
            Learn Investment{" "}
            <span className="bg-gradient-to-r from-cyan-400 to-violet-500 bg-clip-text text-transparent">
              Science
            </span>
            <br />
            With Real Data
          </h1>

          <p className="text-xl text-gray-400 leading-relaxed">
            Explore how forecasting models and portfolio optimization work using
            real market data. No prior finance experience needed.
          </p>

          <div className="flex flex-wrap gap-4 pt-2">
            <Link href="/stock">
              <Button className="rounded-full px-7 py-5 text-base bg-cyan-400 text-[#080D18] hover:bg-cyan-300 font-bold shadow-[0_0_24px_rgba(0,212,255,0.3)]">
                <TrendingUp className="mr-2 h-5 w-5" />
                Start Exploring
              </Button>
            </Link>
            <Link href="/learn">
              <Button
                variant="ghost"
                className="rounded-full px-7 py-5 text-base border border-white/20 text-white hover:bg-white/5 font-semibold"
              >
                <BookOpen className="mr-2 h-5 w-5" />
                Learn First
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* ── Foxy guide card ── */}
      <section className="rounded-2xl border border-white/8 bg-white/[0.03] p-6 flex flex-col sm:flex-row items-center gap-6 w-full max-w-3xl mx-auto backdrop-blur-sm">
        <Image
          src="/agent-avatar.png"
          alt="Foxy"
          width={88}
          height={88}
          className="shrink-0 drop-shadow-lg"
        />
        <div className="space-y-2">
          <p className="text-xs font-bold uppercase tracking-widest text-gray-500">
            Your Guide
          </p>
          <h3 className="text-2xl font-bold text-white">
            Hey! I&apos;m Spark{" "}
            <span role="img" aria-label="fox">
              🦊
            </span>
          </h3>
          <p className="text-gray-400 text-base leading-relaxed">
            I&apos;ll walk you through forecasting models, portfolio theory, and
            key investment concepts — step by step. No jargon, no pressure.
            Let&apos;s learn together!
          </p>
          <Link href="/learn">
            <Button
              variant="ghost"
              size="sm"
              className="mt-1 rounded-full border border-cyan-400/30 bg-cyan-400/10 text-cyan-400 hover:bg-cyan-400/20 font-semibold"
            >
              <BookOpen className="mr-2 h-4 w-4" />
              Start Learning with Spark
            </Button>
          </Link>
        </div>
      </section>

      {/* ── Feature cards ── */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-5 w-full">
        <Link
          href="/stock"
          className="group rounded-2xl border border-white/8 bg-white/[0.03] hover:border-cyan-400/40 hover:bg-cyan-400/5 transition-all p-6 flex flex-col gap-4 backdrop-blur-sm"
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-cyan-400/10">
            <BarChart3 className="h-6 w-6 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Forecasting</h3>
            <p className="text-gray-400 mt-1 leading-relaxed">
              View price history and generate forecasts using EWM, Prophet, or
              Prophet+XGBoost models for any tracked asset.
            </p>
          </div>
          <span className="text-cyan-400 text-sm font-semibold flex items-center gap-1 group-hover:gap-2 transition-all">
            Open Forecasting <TrendingUp className="h-4 w-4" />
          </span>
        </Link>

        <Link
          href="/portfolio"
          className="group rounded-2xl border border-white/8 bg-white/[0.03] hover:border-violet-400/40 hover:bg-violet-400/5 transition-all p-6 flex flex-col gap-4 backdrop-blur-sm"
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-violet-500/10">
            <PieChart className="h-6 w-6 text-violet-400" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Portfolio Builder</h3>
            <p className="text-gray-400 mt-1 leading-relaxed">
              Construct multi-asset portfolios and run PyPortfolioOpt to find
              the optimal weights for your goals.
            </p>
          </div>
          <span className="text-violet-400 text-sm font-semibold flex items-center gap-1 group-hover:gap-2 transition-all">
            Build Portfolio <PieChart className="h-4 w-4" />
          </span>
        </Link>
      </section>
    </div>
  );
}
