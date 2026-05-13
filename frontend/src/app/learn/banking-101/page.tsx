import ModulePageLayout, { ModulePageConfig } from "@/components/learn/ModulePageLayout";

export const dynamic = "force-dynamic";

const config: ModulePageConfig = {
  id: "banking-101",
  number: "01",
  timing: "Newcomers · Day 1",
  title: "Canadian Banking 101",
  subtitle:
    "Everything you need to know about banking in Canada — from day one. No prior experience required.",
  stat: "38%",
  statDescription: "of newcomers don't understand the Canadian banking system",
  accent: {
    circle: "bg-emerald-600",
    text: "text-emerald-400",
    badge: "border border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
    sectionHover: "hover:border-emerald-500/20",
    exampleCard: "border-emerald-500/20 bg-emerald-500/5",
    exampleTitle: "text-emerald-400",
    linkBadge: "bg-emerald-500/15 text-emerald-400",
    checkIcon: "text-emerald-500",
    submitBtn: "bg-emerald-600 hover:bg-emerald-500",
    quizCorrect: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
    nextCircle: "bg-amber-600/30 border border-amber-500/30 text-amber-400",
  },
  next: { number: "02", title: "Credit & Building History", href: "/learn/credit" },
};

export default function BankingModule() {
  return <ModulePageLayout config={config} />;
}
