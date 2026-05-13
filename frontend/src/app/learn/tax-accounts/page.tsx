import ModulePageLayout, { ModulePageConfig } from "@/components/learn/ModulePageLayout";

const config: ModulePageConfig = {
  id: "tax-accounts",
  number: "03",
  timing: "After First Paycheck",
  title: "Tax-Advantaged Accounts",
  subtitle:
    "TFSA, FHSA, RRSP — what they are, who qualifies, contribution rules, and which one to open first as a newcomer.",
  stat: "$40K",
  statDescription: "FHSA lifetime limit — most underused account by newcomers",
  accent: {
    circle: "bg-orange-700",
    text: "text-orange-400",
    badge: "border border-orange-500/30 bg-orange-500/10 text-orange-400",
    sectionHover: "hover:border-orange-500/20",
    exampleCard: "border-orange-500/20 bg-orange-500/5",
    exampleTitle: "text-orange-400",
    linkBadge: "bg-orange-500/15 text-orange-400",
    checkIcon: "text-orange-500",
    submitBtn: "bg-orange-700 hover:bg-orange-600",
    quizCorrect: "bg-orange-500/20 text-orange-400 border border-orange-500/30",
    nextCircle: "bg-violet-700/30 border border-violet-500/30 text-violet-400",
  },
  next: { number: "04", title: "Investing Fundamentals", href: "/learn/investing" },
};

export default function TaxAccountsModule() {
  return <ModulePageLayout config={config} />;
}
