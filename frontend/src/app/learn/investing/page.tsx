import ModulePageLayout, { ModulePageConfig } from "@/components/learn/ModulePageLayout";

export const dynamic = "force-dynamic";

const config: ModulePageConfig = {
  id: "investing",
  number: "04",
  timing: "1+ Year in Canada",
  title: "Investing Fundamentals",
  subtitle:
    "ETFs, diversification, risk tolerance — the concepts behind your live UNF Investor portfolio, explained from scratch.",
  stat: "→ Spark",
  statDescription: "this module connects directly to your live portfolio",
  accent: {
    circle: "bg-violet-700",
    text: "text-violet-400",
    badge: "border border-violet-500/30 bg-violet-500/10 text-violet-400",
    sectionHover: "hover:border-violet-500/20",
    exampleCard: "border-violet-500/20 bg-violet-500/5",
    exampleTitle: "text-violet-400",
    linkBadge: "bg-violet-500/15 text-violet-400",
    checkIcon: "text-violet-500",
    submitBtn: "bg-violet-700 hover:bg-violet-600",
    quizCorrect: "bg-violet-500/20 text-violet-400 border border-violet-500/30",
    nextCircle: "bg-cyan-600/30 border border-cyan-500/30 text-cyan-400",
  },
};

export default function InvestingModule() {
  return <ModulePageLayout config={config} />;
}
