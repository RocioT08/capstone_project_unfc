import ModulePageLayout, { ModulePageConfig } from "@/components/learn/ModulePageLayout";

const config: ModulePageConfig = {
  id: "credit",
  number: "02",
  timing: "First 6 Months",
  title: "Credit & Building History",
  subtitle:
    "Why credit score matters in Canada, how to build it from zero, and how to avoid the traps newcomers fall into.",
  stat: "80%",
  statDescription: "of newcomers face barriers when applying for credit",
  accent: {
    circle: "bg-amber-600",
    text: "text-amber-400",
    badge: "border border-amber-500/30 bg-amber-500/10 text-amber-400",
    sectionHover: "hover:border-amber-500/20",
    exampleCard: "border-amber-500/20 bg-amber-500/5",
    exampleTitle: "text-amber-400",
    linkBadge: "bg-amber-500/15 text-amber-400",
    checkIcon: "text-amber-500",
    submitBtn: "bg-amber-600 hover:bg-amber-500",
    quizCorrect: "bg-amber-500/20 text-amber-400 border border-amber-500/30",
    nextCircle: "bg-orange-700/30 border border-orange-500/30 text-orange-400",
  },
  next: { number: "03", title: "Tax-Advantaged Accounts", href: "/learn/tax-accounts" },
};

export default function CreditModule() {
  return <ModulePageLayout config={config} />;
}
