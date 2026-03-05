"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { OptimizeResponse, AssetOut } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Loader2, X, DollarSign, TrendingUp, FileDown } from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { usePDF } from "react-to-pdf";
import { TourButton } from "@/components/TourButton";
import type { TourStep } from "@/hooks/use-shepherd-tour";
import { SparkChat } from "@/components/SparkChat";

const COLORS = [
  "#0088FE",
  "#00C49F",
  "#FFBB28",
  "#FF8042",
  "#8884D8",
  "#82CA9D",
  "#A4DE6C",
  "#D0ED57",
  "#F2C80F",
  "#FF6666",
];

const PORTFOLIO_TOUR_STEPS: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to Portfolio Builder 💼",
    text: "Build and optimise a multi-asset portfolio using Modern Portfolio Theory and Hierarchical Risk Parity. This tour walks you through every control.",
  },
  {
    id: "asset-select",
    title: "Select Assets",
    text: "Pick stocks already in our database from this dropdown. You can add up to 10 symbols to your portfolio at once.",
    attachTo: { element: "#tour-portfolio-assets", on: "right" },
  },
  {
    id: "fetch-new",
    title: "Fetch a New Stock",
    text: "Type any valid ticker and hit Sync to pull full daily history from Yahoo Finance. The new stock will be added instantly.",
    attachTo: { element: "#tour-portfolio-fetch", on: "right" },
  },
  {
    id: "objective",
    title: "Optimisation Objective",
    text: "Choose your goal: maximise Sharpe Ratio, minimise Volatility, hit a Target Return or Volatility, or use Hierarchical Risk Parity (HRP) — a cluster-based method that needs no covariance inversion.",
    attachTo: { element: "#tour-portfolio-objective", on: "right" },
  },
  {
    id: "run-btn",
    title: "Run Optimisation",
    text: "Click this to compute the efficient weights, expected return, and Sharpe ratio using PyPortfolioOpt on the backend.",
    attachTo: { element: "#tour-portfolio-run", on: "top" },
  },
  {
    id: "results",
    title: "Results Panel",
    text: "Once optimised, you'll see the weight donut chart, investment allocation calculator, individual asset stats, correlation matrix, covariance matrix, and beta rankings — everything you need for your report!",
    attachTo: { element: "#tour-portfolio-results", on: "left" },
  },
];

interface PortfolioBuilderProps {
  assets: AssetOut[];
}

export function PortfolioBuilder({ assets }: PortfolioBuilderProps) {
  const router = useRouter();
  const [symbols, setSymbols] = useState<string[]>(["GOOG", "AMZN"]);
  const [syncSymbol, setSyncSymbol] = useState("");
  const [isSyncing, setIsSyncing] = useState(false);
  
  // Default dates: last 3 years to today
  const today = new Date();
  const threeYearsAgo = new Date();
  threeYearsAgo.setFullYear(today.getFullYear() - 3);
  
  const [fromDate, setFromDate] = useState(threeYearsAgo.toISOString().split('T')[0]);
  const [toDate, setToDate] = useState(today.toISOString().split('T')[0]);

  const [target, setTarget] = useState<
    "max_sharpe" | "min_volatility" | "efficient_return" | "efficient_risk" | "hrp"
  >("max_sharpe");
  const [targetValue, setTargetValue] = useState<string>("");
  const [results, setResults] = useState<OptimizeResponse | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [investAmount, setInvestAmount] = useState<string>("10000");
  const { toast } = useToast();

  const { toPDF, targetRef } = usePDF({
    filename: `portfolio-${symbols.join("-")}-${new Date().toISOString().split("T")[0]}.pdf`,
    page: { format: "a4", orientation: "portrait", margin: 10 },
  });

  const handleAddSymbol = (sym: string) => {
    if (sym && !symbols.includes(sym)) {
      if (symbols.length >= 10) {
        toast({
          title: "Limit Reached",
          description: "You can only add up to 10 symbols.",
          variant: "destructive",
        });
        return;
      }
      setSymbols([...symbols, sym]);
    }
  };

  const handleSync = async () => {
    if (!syncSymbol) return;
    setIsSyncing(true);
    try {
      await api.syncAsset(syncSymbol);
      toast({ title: "Sync Successful", description: `${syncSymbol} has been synced.` });
      handleAddSymbol(syncSymbol.toUpperCase());
      setSyncSymbol("");
      router.refresh(); // Refresh to update the assets list
    } catch (error: any) {
      toast({ title: "Sync Failed", description: error.message, variant: "destructive" });
    } finally {
      setIsSyncing(false);
    }
  };

  const handleRemoveSymbol = (sym: string) => {
    setSymbols(symbols.filter((s) => s !== sym));
  };

  const handleOptimize = async () => {
    if (symbols.length < 2) {
      toast({
        title: "Not Enough Assets",
        description: "Please add at least 2 assets to optimize.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const reqData: any = {
        symbols,
        target,
        from_date: fromDate,
        to_date: toDate,
      };

      if (target === "efficient_return") {
        reqData.target_return = parseFloat(targetValue);
      } else if (target === "efficient_risk") {
        reqData.target_volatility = parseFloat(targetValue);
      }

      const [optRes, statsRes] = await Promise.all([
        api.portfolioOptimize(reqData),
        api.portfolioStats({
          symbols,
          from_date: fromDate,
          to_date: toDate,
        })
      ]);

      setResults(optRes);
      setStats(statsRes);
      toast({
        title: "Optimization Complete",
        description: "Portfolio weights and stats have been calculated.",
      });
    } catch (error: any) {
      toast({
        title: "Optimization Failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const pieData = results
    ? Object.entries(results.weights).map(([name, value]) => ({
        name,
        value: value * 100,
      }))
    : [];

  const handleExportPdf = async () => {
    await toPDF();
  };

  return (
    <div className="space-y-6">
      {/* ── Page header + action buttons ── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Portfolio Builder</h1>
          <p className="text-muted-foreground">
            Construct and optimize your investment portfolio.
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {results && (
            <Button
              variant="outline"
              onClick={handleExportPdf}
              className="flex items-center gap-2 h-10 px-4 border-emerald-400/40 text-emerald-400 hover:bg-emerald-400/10 hover:border-emerald-400 hover:text-emerald-300 transition-all font-medium"
            >
              <FileDown className="h-4 w-4 shrink-0" />
              Download PDF
            </Button>
          )}
          <TourButton tourKey="tour-portfolio" steps={PORTFOLIO_TOUR_STEPS} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Assets</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div id="tour-portfolio-assets" className="space-y-2">
                <label className="text-sm font-medium">Select from Database</label>
                <Select onValueChange={(val) => handleAddSymbol(val)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose a stock..." />
                  </SelectTrigger>
                  <SelectContent>
                    {assets.map((a) => (
                      <SelectItem key={a.symbol} value={a.symbol} disabled={symbols.includes(a.symbol)}>
                        {a.symbol} - {a.name || "Unknown"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div id="tour-portfolio-fetch" className="space-y-2">
                <label className="text-sm font-medium">Fetch New Stock</label>
                <div className="flex gap-2">
                  <Input
                    placeholder="Ticker (e.g. TSLA)"
                    value={syncSymbol}
                    onChange={(e) => setSyncSymbol(e.target.value.toUpperCase())}
                    onKeyDown={(e) => e.key === "Enter" && handleSync()}
                  />
                  <Button onClick={handleSync} disabled={isSyncing || !syncSymbol}>
                    {isSyncing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    Sync
                  </Button>
                </div>
              </div>

              <div className="pt-4 space-y-2">
                <label className="text-sm font-medium">Selected Assets ({symbols.length}/10)</label>
                {symbols.length === 0 && (
                  <div className="text-sm text-muted-foreground">No assets selected.</div>
                )}
                {symbols.map((sym) => (
                  <div
                    key={sym}
                    className="flex items-center justify-between bg-muted/50 p-2 rounded-md"
                  >
                    <span className="font-medium">{sym}</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      onClick={() => handleRemoveSymbol(sym)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Optimization Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Date Range</label>
                <div className="flex items-center gap-2">
                  <Input 
                    type="date" 
                    value={fromDate} 
                    onChange={(e) => setFromDate(e.target.value)} 
                    className="h-9 text-sm"
                  />
                  <span className="text-muted-foreground text-sm">to</span>
                  <Input 
                    type="date" 
                    value={toDate} 
                    onChange={(e) => setToDate(e.target.value)} 
                    className="h-9 text-sm"
                  />
                </div>
              </div>

              <div id="tour-portfolio-objective" className="space-y-2">
                <label className="text-sm font-medium">Objective</label>
                <Select
                  value={target}
                  onValueChange={(val: any) => setTarget(val)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select Objective" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="max_sharpe">Maximize Sharpe Ratio</SelectItem>
                    <SelectItem value="min_volatility">Minimize Volatility</SelectItem>
                    <SelectItem value="efficient_return">Target Return</SelectItem>
                    <SelectItem value="efficient_risk">Target Volatility</SelectItem>
                    <SelectItem value="hrp">Hierarchical Risk Parity</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {(target === "efficient_return" || target === "efficient_risk") && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    {target === "efficient_return" ? "Target Return (Decimal, e.g. 0.15 for 15%)" : "Target Volatility (Decimal, e.g. 0.15 for 15%)"}
                  </label>
                  <Input
                    type="number"
                    step="0.01"
                    placeholder="e.g. 0.15"
                    value={targetValue}
                    onChange={(e) => setTargetValue(e.target.value)}
                  />
                  {stats && stats.individual && (
                    <div className="text-xs text-muted-foreground mt-1">
                      {target === "efficient_return" 
                        ? `Valid range: ~${(Math.min(...Object.values(stats.individual).map((a: any) => a.avg_return * 252))).toFixed(4)} to ${(Math.max(...Object.values(stats.individual).map((a: any) => a.avg_return * 252))).toFixed(4)}`
                        : `Max volatility: ~${(Math.max(...Object.values(stats.individual).map((a: any) => a.annualized_volatility))).toFixed(4)}`
                      }
                    </div>
                  )}
                  {(!stats || !stats.individual) && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Run 'Maximize Sharpe Ratio' first to see valid ranges.
                    </div>
                  )}
                </div>
              )}

              <div id="tour-portfolio-run">
                <Button
                  className="w-full"
                  onClick={handleOptimize}
                  disabled={isLoading || symbols.length < 2}
                >
                  {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Run Optimization
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        <div id="tour-portfolio-results" className="lg:col-span-2 space-y-6">
          {results ? (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Optimal Weights</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={5}
                          dataKey="value"
                          label={({ name, value }) => `${name} ${(value).toFixed(1)}%`}
                        >
                          {pieData.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={COLORS[index % COLORS.length]}
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(value: number) => [`${value.toFixed(2)}%`, "Weight"]}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* ── Investment Allocation Calculator ── */}
              <Card className="border border-cyan-500/20 bg-cyan-500/5">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <DollarSign className="h-5 w-5 text-cyan-400" />
                    Investment Allocation
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-5">
                  <div className="flex items-center gap-3">
                    <div className="relative flex-1">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground font-medium">$</span>
                      <Input
                        type="number"
                        min="0"
                        step="100"
                        placeholder="Enter total investment amount"
                        value={investAmount}
                        onChange={(e) => setInvestAmount(e.target.value)}
                        className="pl-7 text-base h-11"
                      />
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="shrink-0 text-muted-foreground"
                      onClick={() => setInvestAmount("")}
                    >
                      Clear
                    </Button>
                  </div>

                  {investAmount && parseFloat(investAmount) > 0 ? (
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">
                        Based on the optimal weights, here is how to allocate{" "}
                        <span className="font-bold text-white">
                          ${parseFloat(investAmount).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </span>:
                      </p>
                      <div className="space-y-4">
                        {pieData.map((entry, index) => {
                          const alloc = (entry.value / 100) * parseFloat(investAmount);
                          return (
                            <div key={entry.name} className="space-y-1.5">
                              <div className="flex items-center justify-between text-sm">
                                <div className="flex items-center gap-2">
                                  <span
                                    className="inline-block h-3 w-3 rounded-full shrink-0"
                                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                                  />
                                  <span className="font-semibold">{entry.name}</span>
                                  <span className="text-muted-foreground">({entry.value.toFixed(1)}%)</span>
                                </div>
                                <div className="flex items-center gap-1.5 font-bold text-base">
                                  <TrendingUp className="h-4 w-4 text-cyan-400" />
                                  <span>
                                    ${alloc.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                  </span>
                                </div>
                              </div>
                              <div className="h-2 w-full rounded-full bg-muted/40 overflow-hidden">
                                <div
                                  className="h-full rounded-full transition-all duration-500"
                                  style={{
                                    width: `${entry.value}%`,
                                    backgroundColor: COLORS[index % COLORS.length],
                                  }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="flex justify-between pt-3 border-t border-border text-sm">
                        <span className="text-muted-foreground font-medium">Total</span>
                        <span className="font-bold text-base">
                          ${parseFloat(investAmount).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      Enter an amount above to see the exact dollar allocation per asset.
                    </p>
                  )}
                </CardContent>
              </Card>

              {stats && stats.individual && (
                <Card>
                  <CardHeader>
                    <CardTitle>Individual Asset Stats</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {Object.entries(stats.individual).map(([sym, data]: [string, any]) => (
                        <div key={sym} className="space-y-3 text-sm border p-4 rounded-lg">
                          <div className="font-bold text-lg border-b pb-2 mb-2">{sym}</div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Avg Return</span>
                            <span className="font-medium">{(data.avg_return * 100).toFixed(2)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Ann. Volatility</span>
                            <span className="font-medium">{(data.annualized_volatility * 100).toFixed(2)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Sharpe Ratio</span>
                            <span className="font-medium">{data.sharpe_score?.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Max Drawdown</span>
                            <span className="font-medium text-red-500">{(data.max_drawdown * 100).toFixed(2)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Cumulative Return</span>
                            <span className={`font-medium ${data.cumulative_return >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                              {(data.cumulative_return * 100).toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {stats && stats.advanced && stats.advanced.correlation_matrix && (
                <Card>
                  <CardHeader>
                    <CardTitle>Correlation Matrix</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left">
                        <thead className="text-xs text-muted-foreground uppercase bg-muted/50">
                          <tr>
                            <th className="px-4 py-2"></th>
                            {Object.keys(stats.advanced.correlation_matrix).map((sym) => (
                              <th key={sym} className="px-4 py-2 font-medium">{sym}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(stats.advanced.correlation_matrix).map(([rowSym, cols]: [string, any]) => (
                            <tr key={rowSym} className="border-b">
                              <td className="px-4 py-2 font-medium">{rowSym}</td>
                              {Object.values(cols).map((val: any, i) => (
                                <td key={i} className="px-4 py-2">
                                  {val.toFixed(2)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {stats && stats.advanced && stats.advanced.covariance_matrix && (
                <Card>
                  <CardHeader>
                    <CardTitle>Covariance Matrix</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left">
                        <thead className="text-xs text-muted-foreground uppercase bg-muted/50">
                          <tr>
                            <th className="px-4 py-2"></th>
                            {Object.keys(stats.advanced.covariance_matrix).map((sym) => (
                              <th key={sym} className="px-4 py-2 font-medium">{sym}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(stats.advanced.covariance_matrix).map(([rowSym, cols]: [string, any]) => (
                            <tr key={rowSym} className="border-b">
                              <td className="px-4 py-2 font-medium">{rowSym}</td>
                              {Object.values(cols).map((val: any, i) => (
                                <td key={i} className="px-4 py-2">
                                  {val.toFixed(6)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {stats && stats.advanced && stats.advanced.beta_vs_equal_weighted && (
                <Card>
                  <CardHeader>
                    <CardTitle>Beta vs Equal-Weighted Portfolio</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(stats.advanced.beta_vs_equal_weighted).map(([sym, beta]: [string, any]) => (
                        <div key={sym} className="p-4 border rounded-lg text-center bg-muted/20">
                          <div className="text-sm text-muted-foreground mb-1">{sym}</div>
                          <div className="text-xl font-bold">{beta.toFixed(4)}</div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            <Card className="h-full flex items-center justify-center min-h-[400px]">
              <CardContent className="text-center text-muted-foreground">
                Add assets and run optimization to see results.
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════
          DEDICATED OFF-SCREEN PDF REPORT — captured by react-to-pdf
          Completely separate white A4 layout, no dark CSS vars.
      ═══════════════════════════════════════════════════════════ */}
      {results && (
        <div
          ref={targetRef}
          style={{
            position: "absolute",
            left: "-9999px",
            top: 0,
            width: "794px",
            backgroundColor: "#ffffff",
            fontFamily: "'Segoe UI', Arial, sans-serif",
            color: "#0f172a",
            padding: "40px 48px",
            fontSize: "12px",
            lineHeight: "1.6",
          }}
        >
          {/* ── PAGE 1: Cover / Header ── */}
          {/* Brand bar */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px", borderBottom: "3px solid #1e3a5f", paddingBottom: "20px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/unf-logo.svg" alt="UNFC Logo" width={60} height={60} style={{ objectFit: "contain" }} />
              <div>
                <div style={{ fontSize: "13px", fontWeight: 800, color: "#1e3a5f", letterSpacing: "0.05em", textTransform: "uppercase" }}>
                  University of Niagara Falls Canada
                </div>
                <div style={{ fontSize: "10px", color: "#475569", marginTop: "2px" }}>School of Business &amp; Technology</div>
                <div style={{ fontSize: "10px", color: "#64748b" }}>UNFC Investor Analytics Platform</div>
              </div>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/agent-avatar.png" alt="Spark mascot" width={68} height={68}
              style={{ borderRadius: "50%", border: "3px solid #1e3a5f", objectFit: "cover" }} />
          </div>

          {/* Main title block */}
          <div style={{ marginBottom: "22px", borderLeft: "5px solid #1e3a5f", paddingLeft: "16px" }}>
            <div style={{ fontSize: "24px", fontWeight: 800, color: "#0f172a", lineHeight: 1.2 }}>
              Portfolio Optimization Report
            </div>
            <div style={{ fontSize: "13px", color: "#475569", marginTop: "6px" }}>
              Quantitative multi-asset analysis · Powered by PyPortfolioOpt &amp; Yahoo Finance
            </div>
          </div>

          {/* Metadata summary grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px", backgroundColor: "#f8fafc", borderRadius: "8px", padding: "16px 20px", marginBottom: "28px", border: "1px solid #e2e8f0" }}>
            <div>
              <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "3px" }}>Generated</div>
              <div style={{ fontSize: "12px", fontWeight: 600 }}>{new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}</div>
            </div>
            <div>
              <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "3px" }}>Portfolio Assets</div>
              <div style={{ fontSize: "12px", fontWeight: 600 }}>{symbols.join(" · ")}</div>
            </div>
            <div>
              <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "3px" }}>Optimisation Objective</div>
              <div style={{ fontSize: "12px", fontWeight: 600 }}>
                {target === "max_sharpe" ? "Maximize Sharpe Ratio"
                  : target === "min_volatility" ? "Minimize Volatility"
                  : target === "efficient_return" ? "Target Return"
                  : target === "efficient_risk" ? "Target Volatility"
                  : "Hierarchical Risk Parity"}
              </div>
            </div>
            <div>
              <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "3px" }}>Analysis Period</div>
              <div style={{ fontSize: "12px", fontWeight: 600 }}>{fromDate} → {toDate}</div>
            </div>
            <div>
              <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "3px" }}>Number of Assets</div>
              <div style={{ fontSize: "12px", fontWeight: 600 }}>{symbols.length} securities</div>
            </div>
            <div>
              <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "3px" }}>Data Source</div>
              <div style={{ fontSize: "12px", fontWeight: 600 }}>Yahoo Finance (yfinance)</div>
            </div>
          </div>

          {/* ── SECTION 1: Portfolio Performance KPIs ── */}
          <div style={{ marginBottom: "8px" }}>
            <div style={{ fontSize: "15px", fontWeight: 800, color: "#1e3a5f", borderBottom: "2px solid #1e3a5f", paddingBottom: "4px", marginBottom: "8px" }}>
              1. Portfolio Performance Indicators
            </div>
            <div style={{ fontSize: "11px", color: "#475569", marginBottom: "14px" }}>
              Key performance metrics for the optimised portfolio. These figures represent the expected annualised values derived from
              historical price data using Modern Portfolio Theory (MPT). The Sharpe Ratio measures risk-adjusted return;
              a value above 1.0 is considered good, above 2.0 is excellent.
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginBottom: "18px" }}>
              {results.performance?.expected_return != null && (
                <div style={{ backgroundColor: "#f0fdf4", borderRadius: "8px", padding: "14px 16px", border: "1px solid #bbf7d0" }}>
                  <div style={{ fontSize: "9px", fontWeight: 700, color: "#15803d", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "4px" }}>Expected Annual Return</div>
                  <div style={{ fontSize: "22px", fontWeight: 800, color: "#15803d" }}>{(results.performance.expected_return * 100).toFixed(2)}%</div>
                  <div style={{ fontSize: "9px", color: "#16a34a", marginTop: "4px" }}>The projected annual gain based on historical mean returns weighted by optimal allocation.</div>
                </div>
              )}
              {results.performance?.volatility != null && (
                <div style={{ backgroundColor: "#fff1f2", borderRadius: "8px", padding: "14px 16px", border: "1px solid #fecdd3" }}>
                  <div style={{ fontSize: "9px", fontWeight: 700, color: "#be123c", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "4px" }}>Portfolio Volatility</div>
                  <div style={{ fontSize: "22px", fontWeight: 800, color: "#be123c" }}>{(results.performance.volatility * 100).toFixed(2)}%</div>
                  <div style={{ fontSize: "9px", color: "#dc2626", marginTop: "4px" }}>Annualised standard deviation of portfolio returns. Lower = more stable.</div>
                </div>
              )}
              {results.performance?.sharpe_ratio != null && (
                <div style={{ backgroundColor: "#eff6ff", borderRadius: "8px", padding: "14px 16px", border: "1px solid #bfdbfe" }}>
                  <div style={{ fontSize: "9px", fontWeight: 700, color: "#1d4ed8", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "4px" }}>Sharpe Ratio</div>
                  <div style={{ fontSize: "22px", fontWeight: 800, color: "#1d4ed8" }}>{results.performance.sharpe_ratio.toFixed(4)}</div>
                  <div style={{ fontSize: "9px", color: "#2563eb", marginTop: "4px" }}>Return per unit of risk (excess return / volatility). Benchmark: {">"} 1.0 = good.</div>
                </div>
              )}
            </div>
            {/* Risk metrics row */}
            {results.risk_metrics && (
              <>
                <div style={{ fontSize: "12px", fontWeight: 700, color: "#334155", marginBottom: "8px", marginTop: "4px" }}>Risk Metrics</div>
                <div style={{ fontSize: "11px", color: "#475569", marginBottom: "10px" }}>
                  Value at Risk (VaR) estimates the maximum expected loss at a given confidence level over one trading day.
                  Conditional VaR (CVaR / Expected Shortfall) measures the average loss in the worst-case tail scenarios beyond VaR.
                  Max Drawdown is the largest peak-to-trough decline observed in the historical period.
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px", marginBottom: "20px" }}>
                  <div style={{ backgroundColor: "#fafafa", borderRadius: "6px", padding: "12px 14px", border: "1px solid #e2e8f0" }}>
                    <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", marginBottom: "3px" }}>VaR (95%)</div>
                    <div style={{ fontSize: "18px", fontWeight: 700, color: "#dc2626" }}>{(results.risk_metrics.var_95 * 100).toFixed(2)}%</div>
                    <div style={{ fontSize: "9px", color: "#64748b", marginTop: "2px" }}>Max daily loss with 95% confidence</div>
                  </div>
                  <div style={{ backgroundColor: "#fafafa", borderRadius: "6px", padding: "12px 14px", border: "1px solid #e2e8f0" }}>
                    <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", marginBottom: "3px" }}>CVaR (95%)</div>
                    <div style={{ fontSize: "18px", fontWeight: 700, color: "#dc2626" }}>{(results.risk_metrics.cvar_95 * 100).toFixed(2)}%</div>
                    <div style={{ fontSize: "9px", color: "#64748b", marginTop: "2px" }}>Average loss beyond VaR threshold</div>
                  </div>
                  <div style={{ backgroundColor: "#fafafa", borderRadius: "6px", padding: "12px 14px", border: "1px solid #e2e8f0" }}>
                    <div style={{ fontSize: "9px", fontWeight: 700, color: "#64748b", textTransform: "uppercase", marginBottom: "3px" }}>Max Drawdown</div>
                    <div style={{ fontSize: "18px", fontWeight: 700, color: "#dc2626" }}>{(results.risk_metrics.max_drawdown * 100).toFixed(2)}%</div>
                    <div style={{ fontSize: "9px", color: "#64748b", marginTop: "2px" }}>Worst peak-to-trough decline</div>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* ── SECTION 2: Optimal Weights & Allocation ── */}
          <div style={{ marginBottom: "20px" }}>
            <div style={{ fontSize: "15px", fontWeight: 800, color: "#1e3a5f", borderBottom: "2px solid #1e3a5f", paddingBottom: "4px", marginBottom: "8px" }}>
              2. Optimal Portfolio Weights
            </div>
            <div style={{ fontSize: "11px", color: "#475569", marginBottom: "12px" }}>
              The weights below represent the proportion of total capital to allocate to each asset in order to achieve the selected
              objective. These are derived by solving the optimisation problem defined by Modern Portfolio Theory or, in the case
              of HRP, by hierarchical clustering of the covariance matrix.
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", marginBottom: "14px" }}>
              <thead>
                <tr style={{ backgroundColor: "#1e3a5f", color: "#ffffff" }}>
                  <th style={{ padding: "10px 14px", textAlign: "left", fontWeight: 700 }}>Symbol</th>
                  <th style={{ padding: "10px 14px", textAlign: "right", fontWeight: 700 }}>Weight (%)</th>
                  {investAmount && parseFloat(investAmount) > 0 && (
                    <th style={{ padding: "10px 14px", textAlign: "right", fontWeight: 700 }}>
                      Allocation (${parseFloat(investAmount).toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })})
                    </th>
                  )}
                  <th style={{ padding: "10px 14px", textAlign: "center", fontWeight: 700 }}>Bar</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results.weights)
                  .sort(([, a], [, b]) => b - a)
                  .map(([sym, w], i) => {
                    const alloc = w * (parseFloat(investAmount) || 0);
                    return (
                      <tr key={sym} style={{ backgroundColor: i % 2 === 0 ? "#f8fafc" : "#ffffff", borderBottom: "1px solid #e2e8f0" }}>
                        <td style={{ padding: "10px 14px", fontWeight: 700 }}>{sym}</td>
                        <td style={{ padding: "10px 14px", textAlign: "right", fontWeight: 600 }}>{(w * 100).toFixed(2)}%</td>
                        {investAmount && parseFloat(investAmount) > 0 && (
                          <td style={{ padding: "10px 14px", textAlign: "right", color: "#15803d", fontWeight: 700 }}>
                            ${alloc.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                          </td>
                        )}
                        <td style={{ padding: "10px 14px" }}>
                          <div style={{ height: "8px", backgroundColor: "#e2e8f0", borderRadius: "4px", overflow: "hidden" }}>
                            <div style={{ height: "100%", borderRadius: "4px", backgroundColor: COLORS[i % COLORS.length], width: `${w * 100}%` }} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>

          {/* ── SECTION 3: Individual Asset Stats ── */}
          {stats?.individual && (
            <div style={{ marginBottom: "20px" }}>
              <div style={{ fontSize: "15px", fontWeight: 800, color: "#1e3a5f", borderBottom: "2px solid #1e3a5f", paddingBottom: "4px", marginBottom: "8px" }}>
                3. Individual Asset Performance
              </div>
              <div style={{ fontSize: "11px", color: "#475569", marginBottom: "12px" }}>
                Per-asset statistics computed over the selected date range. <strong>Avg Return</strong> is the mean daily return.
                <strong> Ann. Volatility</strong> is the annualised standard deviation (daily σ × √252).
                <strong> Sharpe Score</strong> is each asset&apos;s individual risk-adjusted return.
                <strong> Max Drawdown</strong> is the worst peak-to-trough loss. <strong>Cumulative Return</strong> is total return over the period.
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
                <thead>
                  <tr style={{ backgroundColor: "#1e3a5f", color: "#ffffff" }}>
                    <th style={{ padding: "8px 12px", textAlign: "left", fontWeight: 700 }}>Symbol</th>
                    <th style={{ padding: "8px 12px", textAlign: "right", fontWeight: 700 }}>Avg Return</th>
                    <th style={{ padding: "8px 12px", textAlign: "right", fontWeight: 700 }}>Ann. Volatility</th>
                    <th style={{ padding: "8px 12px", textAlign: "right", fontWeight: 700 }}>Sharpe Score</th>
                    <th style={{ padding: "8px 12px", textAlign: "right", fontWeight: 700 }}>Max Drawdown</th>
                    <th style={{ padding: "8px 12px", textAlign: "right", fontWeight: 700 }}>Cumulative Return</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.individual).map(([sym, d]: [string, any], i) => (
                    <tr key={sym} style={{ backgroundColor: i % 2 === 0 ? "#f8fafc" : "#ffffff", borderBottom: "1px solid #e2e8f0" }}>
                      <td style={{ padding: "8px 12px", fontWeight: 700 }}>{sym}</td>
                      <td style={{ padding: "8px 12px", textAlign: "right" }}>{(d.avg_return * 100).toFixed(3)}%</td>
                      <td style={{ padding: "8px 12px", textAlign: "right" }}>{(d.annualized_volatility * 100).toFixed(2)}%</td>
                      <td style={{ padding: "8px 12px", textAlign: "right" }}>{d.sharpe_score?.toFixed(3)}</td>
                      <td style={{ padding: "8px 12px", textAlign: "right", color: "#dc2626", fontWeight: 600 }}>{(d.max_drawdown * 100).toFixed(2)}%</td>
                      <td style={{ padding: "8px 12px", textAlign: "right", color: d.cumulative_return >= 0 ? "#15803d" : "#dc2626", fontWeight: 700 }}>
                        {(d.cumulative_return * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* ── SECTION 4: Correlation Matrix ── */}
          {stats?.advanced?.correlation_matrix && (
            <div style={{ marginBottom: "20px" }}>
              <div style={{ fontSize: "15px", fontWeight: 800, color: "#1e3a5f", borderBottom: "2px solid #1e3a5f", paddingBottom: "4px", marginBottom: "8px" }}>
                4. Correlation Matrix
              </div>
              <div style={{ fontSize: "11px", color: "#475569", marginBottom: "12px" }}>
                Pearson correlation of daily returns between each pair of assets (range: −1 to +1). A value of <strong>+1</strong> means the
                assets move perfectly in sync (no diversification benefit). A value near <strong>0</strong> means the assets are uncorrelated,
                providing maximum diversification. <strong>Negative values</strong> indicate inverse movement — ideal for hedges.
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
                <thead>
                  <tr style={{ backgroundColor: "#334155", color: "#ffffff" }}>
                    <th style={{ padding: "8px 12px", textAlign: "left" }}></th>
                    {Object.keys(stats.advanced.correlation_matrix).map((s) => (
                      <th key={s} style={{ padding: "8px 12px", textAlign: "right", fontWeight: 700 }}>{s}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.advanced.correlation_matrix).map(([row, cols]: [string, any], i) => (
                    <tr key={row} style={{ backgroundColor: i % 2 === 0 ? "#f8fafc" : "#ffffff", borderBottom: "1px solid #e2e8f0" }}>
                      <td style={{ padding: "8px 12px", fontWeight: 700 }}>{row}</td>
                      {Object.values(cols).map((v: any, j) => (
                        <td key={j} style={{ padding: "8px 12px", textAlign: "right", fontWeight: v === 1 ? 700 : 400,
                          color: v === 1 ? "#0f172a" : v > 0.7 ? "#dc2626" : v < 0 ? "#15803d" : "#0f172a" }}>
                          {v.toFixed(4)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* ── SECTION 5: Covariance Matrix ── */}
          {stats?.advanced?.covariance_matrix && (
            <div style={{ marginBottom: "20px" }}>
              <div style={{ fontSize: "15px", fontWeight: 800, color: "#1e3a5f", borderBottom: "2px solid #1e3a5f", paddingBottom: "4px", marginBottom: "8px" }}>
                5. Covariance Matrix
              </div>
              <div style={{ fontSize: "11px", color: "#475569", marginBottom: "12px" }}>
                The covariance matrix quantifies how asset returns move together in absolute terms (units: return²/day).
                Diagonal entries show each asset&apos;s own variance. Off-diagonal entries show the joint variability between pairs —
                used directly by PyPortfolioOpt to compute portfolio variance and solve the efficient frontier.
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
                <thead>
                  <tr style={{ backgroundColor: "#334155", color: "#ffffff" }}>
                    <th style={{ padding: "8px 12px", textAlign: "left" }}></th>
                    {Object.keys(stats.advanced.covariance_matrix).map((s) => (
                      <th key={s} style={{ padding: "8px 12px", textAlign: "right", fontWeight: 700 }}>{s}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.advanced.covariance_matrix).map(([row, cols]: [string, any], i) => (
                    <tr key={row} style={{ backgroundColor: i % 2 === 0 ? "#f8fafc" : "#ffffff", borderBottom: "1px solid #e2e8f0" }}>
                      <td style={{ padding: "8px 12px", fontWeight: 700 }}>{row}</td>
                      {Object.values(cols).map((v: any, j) => (
                        <td key={j} style={{ padding: "8px 12px", textAlign: "right" }}>{(v as number).toFixed(6)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* ── SECTION 6: Beta vs Equal-Weighted ── */}
          {stats?.advanced?.beta_vs_equal_weighted && (
            <div style={{ marginBottom: "20px" }}>
              <div style={{ fontSize: "15px", fontWeight: 800, color: "#1e3a5f", borderBottom: "2px solid #1e3a5f", paddingBottom: "4px", marginBottom: "8px" }}>
                6. Beta vs Equal-Weighted Portfolio
              </div>
              <div style={{ fontSize: "11px", color: "#475569", marginBottom: "12px" }}>
                Beta measures each asset&apos;s sensitivity relative to an equal-weighted benchmark portfolio of the same assets.
                A beta of <strong>1.0</strong> means the asset moves in line with the benchmark.
                Beta <strong>&gt; 1</strong> indicates the asset amplifies benchmark moves (higher risk/reward).
                Beta <strong>&lt; 1</strong> (or negative) indicates the asset dampens moves, acting as a stabiliser.
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
                <thead>
                  <tr style={{ backgroundColor: "#334155", color: "#ffffff" }}>
                    <th style={{ padding: "10px 14px", textAlign: "left", fontWeight: 700 }}>Symbol</th>
                    <th style={{ padding: "10px 14px", textAlign: "right", fontWeight: 700 }}>Beta</th>
                    <th style={{ padding: "10px 14px", textAlign: "left", fontWeight: 700 }}>Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(stats.advanced.beta_vs_equal_weighted).map(([sym, beta]: [string, any], i) => (
                    <tr key={sym} style={{ backgroundColor: i % 2 === 0 ? "#f8fafc" : "#ffffff", borderBottom: "1px solid #e2e8f0" }}>
                      <td style={{ padding: "10px 14px", fontWeight: 700 }}>{sym}</td>
                      <td style={{ padding: "10px 14px", textAlign: "right", fontWeight: 700,
                        color: beta > 1.2 ? "#dc2626" : beta < 0.8 ? "#15803d" : "#0f172a" }}>
                        {beta.toFixed(4)}
                      </td>
                      <td style={{ padding: "10px 14px", fontSize: "11px", color: "#475569" }}>
                        {beta > 1.2 ? "High sensitivity — amplifies market moves"
                          : beta > 1.0 ? "Slightly above benchmark — modest amplification"
                          : beta < 0 ? "Inverse relationship — acts as a hedge"
                          : beta < 0.8 ? "Low sensitivity — stabilising effect"
                          : "In line with equal-weighted benchmark"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* ── Footer ── */}
          <div style={{ borderTop: "2px solid #e2e8f0", paddingTop: "16px", marginTop: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div style={{ fontSize: "9px", color: "#94a3b8", maxWidth: "70%", fontStyle: "italic" }}>
              ⚠ This report is automatically generated by UNFC Investor Analytics for academic and informational purposes only.
              Past performance does not guarantee future results. This document does not constitute financial advice.
              Data sourced from Yahoo Finance via yfinance. All calculations performed by PyPortfolioOpt (open-source).
            </div>
            <div style={{ fontSize: "9px", color: "#94a3b8", textAlign: "right" }}>
              <div>University of Niagara Falls Canada</div>
              <div>Generated: {new Date().toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })}</div>
            </div>
          </div>
        </div>
      )}
            {/* ── SparkChat with portfolio context ── */}
            <SparkChat
              context={
                      results
                        ? {
                            type: "portfolio_optimize",
                            data: {
                              ...results,
                              // también incluimos stats si están disponibles
                              ...(stats ? { individual_stats: stats.individual, advanced: stats.advanced } : {}),
                            },
                          }
                        : undefined
        }
      />
    </div>
  );
}
