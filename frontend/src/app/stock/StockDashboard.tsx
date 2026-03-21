"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  AssetOut,
  PriceOut,
  StatsResponse,
  ForecastMetricsResponse,
  CryptoMetricsResponse,
  CryptoForecastResponse,
  type ForecastModelKey,
} from "@/types/api";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Loader2 } from "lucide-react";
import { StockChart } from "./StockChart";
import { TourButton } from "@/components/TourButton";
import type { TourStep } from "@/hooks/use-shepherd-tour";

const METRICS_MODEL_ORDER: ForecastModelKey[] = ["chronos"];

const STOCK_TOUR_STEPS: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to Stock Analysis 📈",
    text: "This page lets you explore price history, generate forecasts with AI models, and view risk metrics. Let's take a quick tour!",
  },
  {
    id: "asset-selector",
    title: "Pick a Stock",
    text: "Use this dropdown to select any asset already in our database. All stocks have historical daily price data synced automatically.",
    attachTo: { element: "#tour-stock-selector", on: "bottom" },
  },
  {
    id: "fetch-new",
    title: "Fetch a New Ticker",
    text: "Don't see your stock? Type any valid ticker symbol here and hit Sync — we'll pull its full price history from Yahoo Finance and add it to the database.",
    attachTo: { element: "#tour-stock-fetch", on: "bottom" },
  },
  {
    id: "model-select",
    title: "Forecast Model",
    text: "Choose Base (fast exponential smoothing) for forecasts.",
    attachTo: { element: "#tour-stock-model", on: "bottom" },
  },
  {
    id: "interval-select",
    title: "Chart Interval",
    text: "Switch between Daily, Weekly, and Monthly views. All data is stored at daily granularity and aggregated on-the-fly — no extra API calls needed!",
    attachTo: { element: "#tour-stock-interval", on: "bottom" },
  },
  {
    id: "forecast-btn",
    title: "Generate Forecast",
    text: "Click here to run the selected model and overlay a forecast (with confidence bands) directly on the price chart. Horizons are calculated automatically.",
    attachTo: { element: "#tour-stock-forecast-btn", on: "top" },
  },
  {
    id: "stats-panel",
    title: "Risk & Return Metrics",
    text: "On the right you'll find key statistics: Sharpe Ratio, Max Drawdown, Annualised Volatility, Skewness and more — all calculated from real historical returns.",
    attachTo: { element: "#tour-stock-stats", on: "left" },
  },
];

interface StockDashboardProps {
  assets: AssetOut[];
  initialSymbol?: string;
  initialPrices?: PriceOut[] | null;
  initialStats?: StatsResponse | null;
  initialFromDate?: string;
  initialToDate?: string;
}

export function StockDashboard({ assets, initialSymbol, initialPrices, initialStats, initialFromDate, initialToDate }: StockDashboardProps) {
  // Deduplicate: if both "BTC" and "BTC-USD" exist, keep only "BTC-USD"
  const cryptoSymbols = new Set(assets.filter(a => a.asset_type === "crypto").map(a => a.symbol));
  const dedupedAssets = assets.filter(a => !cryptoSymbols.has(`${a.symbol}-USD`));
  const router = useRouter();
  const { toast } = useToast();
  const [syncSymbol, setSyncSymbol] = useState("");
  const [isSyncing, setIsSyncing] = useState(false);

  const [fromDate, setFromDate] = useState(initialFromDate || "");
  const [toDate, setToDate] = useState(initialToDate || "");
  const [metrics, setMetrics] = useState<ForecastMetricsResponse | null>(null);
  const [cryptoMetrics, setCryptoMetrics] = useState<CryptoMetricsResponse | null>(null);
  const [assemblyForecast, setAssemblyForecast] = useState<CryptoForecastResponse | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [chartForecast, setChartForecast] = useState<{ data: import("@/types/api").ForecastResponse; model: "chronos" | "assembly" } | null>(null);
  const [news, setNews] = useState<{ title: string; summary: string; sentiment: string; source: string; url: string }[] | null>(null);
  const [newsSentiment, setNewsSentiment] = useState<string | null>(null);
  const newsSentimentRef = useRef<string | null>(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [inProgressModels, setInProgressModels] = useState<ForecastModelKey[]>([]);
  const [metricsInterval, setMetricsInterval] = useState<"1wk" | "1mo">("1wk");
  const [forecastDays, setForecastDays] = useState<7 | 14 | 21>(7);
  const [selectedModel, setSelectedModel] = useState<ForecastModelKey>("chronos");
  const [novaInsight, setNovaInsight] = useState<string | null>(null);
  const [novaInsightLoading, setNovaInsightLoading] = useState(false);

  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Auto-load news when a symbol with prices is available
  useEffect(() => {
    if (initialSymbol && initialPrices && initialPrices.length > 0) {
      handleLoadNews(initialSymbol.toUpperCase());
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialSymbol]);

  const fetchNovaInsight = async (symbol: string, forecast: import("@/types/api").ForecastResponse, sentiment?: string) => {
    setNovaInsightLoading(true);
    try {
      const res = await api.getNovaInsight({
        symbol,
        point_forecast: forecast.point_forecast,
        lower_bound: forecast.lower_bound,
        upper_bound: forecast.upper_bound,
        dates: forecast.dates,
        sentiment,
      });
      setNovaInsight(res.insight || null);
    } catch {
      setNovaInsight(null);
    } finally {
      setNovaInsightLoading(false);
    }
  };

  const handleSelect = (symbol: string) => {
    setMetrics(null);
    setNews(null);
    setNewsSentiment(null);
    newsSentimentRef.current = null;
    setChartForecast(null);
    setNovaInsight(null);
    router.push(`/stock?symbol=${symbol}&from=${fromDate}&to=${toDate}`);
  };

  const handleLoadNews = async (symbol: string) => {
    setNewsLoading(true);
    try {
      const res = await api.getNews(symbol);
      setNews(res.news);
      const s = res.news?.[0]?.sentiment ?? null;
      setNewsSentiment(s);
      newsSentimentRef.current = s;
    } catch {
      setNews([]);
      setNewsSentiment(null);
      newsSentimentRef.current = null;
    } finally {
      setNewsLoading(false);
    }
  };

  const handleDateUpdate = () => {
    if (initialSymbol) {
      router.push(`/stock?symbol=${initialSymbol}&from=${fromDate}&to=${toDate}`);
    }
  };

  const handleSync = async () => {
    if (!syncSymbol) return;
    setIsSyncing(true);
    try {
      const assetType = syncSymbol.toUpperCase().endsWith("-USD") ? "crypto" : "stock";
      await api.syncAsset(syncSymbol, assetType);
      toast({ title: "Sync Successful", description: `${syncSymbol} has been synced.` });
      router.push(`/stock?symbol=${syncSymbol.toUpperCase()}&from=${fromDate}&to=${toDate}`);
      router.refresh();
      setSyncSymbol("");
    } catch (error: any) {
      toast({ title: "Sync Failed", description: error.message, variant: "destructive" });
    } finally {
      setIsSyncing(false);
    }
  };

  const stats = initialSymbol && initialStats?.individual?.[initialSymbol.toUpperCase()];
  const isCrypto = assets.find((a) => a.symbol === initialSymbol?.toUpperCase())?.asset_type === "crypto";

  const handleLoadMetrics = async (modelFromChart?: ForecastModelKey) => {
    if (!initialSymbol) return;
    const symbol = initialSymbol.toUpperCase();
    const boundsPeriods = metricsInterval === "1wk" ? forecastDays / 7 : 1;
    const reqBase = {
      symbol,
      interval: metricsInterval,
      last_n_weeks: 20,
      bounds_horizon_periods: boundsPeriods,
    };
    const modelToLoad = modelFromChart ?? selectedModel;
    if (modelFromChart != null) setSelectedModel(modelFromChart);

    setMetricsLoading(true);
    setInProgressModels([modelToLoad]);
    setMetrics({ ...reqBase, last_n_weeks: 20, bounds_horizon_weeks: boundsPeriods, metrics: [], bounds: [], error: null });
    try {
      const res = await api.getForecastMetrics({ ...reqBase, models: [modelToLoad] });
      setMetrics(res);
      setInProgressModels([]);
      toast({ title: "Metrics loaded", description: `${modelToLoad.replace("_", "+")} — walk-forward and bounds.` });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load metrics";
      toast({ title: "Metrics failed", description: msg, variant: "destructive" });
      setInProgressModels([]);
    } finally {
      setMetricsLoading(false);
    }
  };

  const handleLoadCryptoMetrics = async (chartModel: "chronos" | "assembly" = "assembly") => {
    if (!initialSymbol) return;

    if (chartModel === "assembly") {
      try {
        const [metricsRes, forecastRes] = await Promise.all([
          api.getCryptoMetrics(initialSymbol.toUpperCase()),
          api.cryptoForecast(initialSymbol.toUpperCase(), { periods: 7, nova_sentiment: newsSentimentRef.current ?? undefined }),
        ]);
        setCryptoMetrics(metricsRes);
        setAssemblyForecast(forecastRes);
        return;
      } catch {
        // Assembly model not trained — fall through to Chronos-only walk-forward
        setCryptoMetrics(null);
        setAssemblyForecast(null);
      }
    }

    // Chronos-only: load standard walk-forward metrics
    handleLoadMetrics("chronos");
  };

  return (
    <div className="space-y-6">
      {/* ── Page header + Tour button (tour only shown once a stock is selected) ── */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Stock Analysis</h1>
          <p className="text-muted-foreground">
            Explore price history, generate AI forecasts, and review risk metrics.
          </p>
        </div>
        {initialSymbol && (
          <TourButton tourKey="tour-stock" steps={STOCK_TOUR_STEPS} />
        )}
      </div>

      {/* ── Stock selector / fetch toolbar ── */}
      <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center bg-muted/50 p-4 rounded-lg border">
        <div id="tour-stock-selector" className="flex items-center gap-4 w-full md:w-auto">
          <span className="font-medium whitespace-nowrap">Select Stock:</span>
          {mounted && (
            <Select value={initialSymbol} onValueChange={handleSelect}>
              <SelectTrigger className="w-[200px] bg-background">
                <SelectValue placeholder="Choose a stock..." />
              </SelectTrigger>
              <SelectContent>
                {dedupedAssets.map((a) => (
                  <SelectItem key={a.symbol} value={a.symbol}>
                    {a.symbol} - {a.name || "Unknown"}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        <div id="tour-stock-fetch" className="flex items-center gap-2 w-full md:w-auto">
          <span className="font-medium whitespace-nowrap">Fetch New:</span>
          <Input
            placeholder="Ticker (e.g. TSLA)"
            value={syncSymbol}
            onChange={(e) => setSyncSymbol(e.target.value.toUpperCase())}
            className="w-[150px] bg-background"
          />
          <Button onClick={handleSync} disabled={isSyncing || !syncSymbol}>
            {isSyncing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Sync
          </Button>
        </div>
      </div>

      {!initialSymbol && (
        <div className="text-center py-20 text-muted-foreground">
          Please select a stock from the dropdown or fetch a new one to view its details.
        </div>
      )}

      {initialSymbol && initialPrices && initialPrices.length > 0 && (
        <>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card id="tour-stock-chart">
              <CardHeader className="pb-2">
                <CardTitle>{initialSymbol.toUpperCase()} Price History & Forecast</CardTitle>
              </CardHeader>
              <CardContent>
                <StockChart
                  symbol={initialSymbol.toUpperCase()}
                  initialPrices={initialPrices}
                  forecastDays={forecastDays}
                  setForecastDays={setForecastDays}
                  onForecastComplete={(chartModel) => {
                    if (isCrypto) handleLoadCryptoMetrics(chartModel);
                    else handleLoadMetrics(chartModel === "chronos" ? chartModel : undefined);
                  }}
                  onForecastData={(data, model) => {
                    setChartForecast({ data, model });
                    // For crypto the assembly endpoint provides nova_insight;
                    // for stocks (chronos) we fetch it separately here.
                    if (!isCrypto) {
                      fetchNovaInsight(initialSymbol!.toUpperCase(), data);
                    }
                  }}
                  metricsLoading={metricsLoading}
                  isCrypto={isCrypto}
                />
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <Card id="tour-stock-stats">
              <CardHeader className="pb-3">
                <CardTitle>Asset Overview</CardTitle>
              </CardHeader>
              <CardContent>
                {(() => {
                  const prices = initialPrices ?? [];
                  const current = prices[0]?.close_price ?? null;
                  const prev = prices[1]?.close_price ?? null;
                  const price30d = prices[29]?.close_price ?? prices[prices.length - 1]?.close_price ?? null;
                  const change24h = current && prev ? current - prev : null;
                  const change24hPct = current && prev ? ((current - prev) / prev) * 100 : null;
                  const return30d = current && price30d ? ((current - price30d) / price30d) * 100 : null;
                  const indiv = stats?.individual?.[initialSymbol?.toUpperCase() ?? ""];

                  const fc = chartForecast?.data;
                  const fcLastPt = fc ? fc.point_forecast[fc.point_forecast.length - 1] : null;
                  const fcLow = fc ? fc.lower_bound[fc.lower_bound.length - 1] : null;
                  const fcHigh = fc ? fc.upper_bound[fc.upper_bound.length - 1] : null;
                  const fcChange = fcLastPt && current ? fcLastPt - current : null;
                  const fcChangePct = fcLastPt && current ? ((fcLastPt - current) / current) * 100 : null;

                  const fmt = (v: number) => v >= 1000
                    ? `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
                    : `$${v.toFixed(2)}`;

                  return (
                    <div className="space-y-3 text-sm">
                      {current !== null && (
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Current Price</span>
                          <span className="font-semibold text-base text-right">{fmt(current)}</span>
                        </div>
                      )}
                      {change24h !== null && (
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">24h Change</span>
                          <span className={`font-medium text-right ${change24h >= 0 ? "text-green-400" : "text-red-400"}`}>
                            {change24h >= 0 ? "▲" : "▼"} {fmt(Math.abs(change24h))} ({Math.abs(change24hPct!).toFixed(2)}%)
                          </span>
                        </div>
                      )}
                      {return30d !== null && (
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">30-Day Return</span>
                          <span className={`font-medium text-right ${return30d >= 0 ? "text-green-400" : "text-red-400"}`}>
                            {return30d >= 0 ? "▲" : "▼"} {Math.abs(return30d).toFixed(2)}%
                          </span>
                        </div>
                      )}
                      {indiv?.annualized_volatility != null && (
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Ann. Volatility</span>
                          <span className="font-medium text-right">{(indiv.annualized_volatility * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {indiv?.max_drawdown != null && (
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Max Drawdown</span>
                          <span className="font-medium text-right text-red-400">{(indiv.max_drawdown * 100).toFixed(1)}%</span>
                        </div>
                      )}

                      {fc && (
                        <>
                          <div className="border-t pt-3 mt-1">
                            <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wide font-medium">
                              7-Day Forecast · {chartForecast?.model === "assembly" ? "Assembly" : "Chronos"}
                            </p>
                            {fcLastPt !== null && (
                              <div className="flex items-center justify-between">
                                <span className="text-muted-foreground">Forecast Price</span>
                                <span className="font-semibold text-base text-cyan-400 text-right">{fmt(fcLastPt)}</span>
                              </div>
                            )}
                            {fcChange !== null && (
                              <div className="flex items-center justify-between mt-2">
                                <span className="text-muted-foreground">Expected Change</span>
                                <span className={`font-medium text-right ${fcChange >= 0 ? "text-green-400" : "text-red-400"}`}>
                                  {fcChange >= 0 ? "▲" : "▼"} {fmt(Math.abs(fcChange))} ({Math.abs(fcChangePct!).toFixed(2)}%)
                                </span>
                              </div>
                            )}
                            {fcLow !== null && fcHigh !== null && (
                              <div className="flex items-center justify-between mt-2">
                                <span className="text-muted-foreground">95% Range</span>
                                <span className="font-medium text-right text-muted-foreground">
                                  {fmt(fcLow)} – {fmt(fcHigh)}
                                </span>
                              </div>
                            )}
                            {assemblyForecast?.nova_sentiment && chartForecast?.model === "assembly" && (() => {
                              const s = assemblyForecast.nova_sentiment;
                              const { icon, label, cls } =
                                s === "bullish"
                                  ? { icon: "▲ ", label: "Bullish", cls: "text-green-400" }
                                  : s === "bearish"
                                  ? { icon: "▼ ", label: "Bearish", cls: "text-red-400" }
                                  : { icon: "", label: "Neutral", cls: "text-cyan-400" };
                              return (
                                <div className="flex items-center justify-between mt-2">
                                  <span className="text-muted-foreground">Market Sentiment</span>
                                  <span className={`font-medium ${cls}`}>{icon}{label}</span>
                                </div>
                              );
                            })()}
                          </div>
                        </>
                      )}

                      {!fc && (
                        <p className="text-xs text-muted-foreground pt-2 border-t">
                          Run a forecast to see price prediction.
                        </p>
                      )}
                    </div>
                  );
                })()}
              </CardContent>
            </Card>

            {/* ── Top News (Amazon Bedrock Nova) ── */}
            {(newsLoading || (news && news[0])) && (
              <div>
                <h2 className="text-sm font-semibold">Latest News</h2>
                <p className="text-xs text-muted-foreground">Powered by Amazon Bedrock Nova</p>
              </div>
            )}
            {newsLoading && (
              <Card className="border-dashed">
                <CardContent className="pt-4 pb-4 flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin shrink-0" />
                  Fetching latest news via Nova…
                </CardContent>
              </Card>
            )}
            {!newsLoading && news && news[0] && (() => {
              const item = news[0];
              const borderAccent =
                item.sentiment === "bullish" ? "border-l-green-500"
                : item.sentiment === "bearish" ? "border-l-red-500"
                : "border-l-border";
              const sentimentColor =
                item.sentiment === "bullish" ? "text-green-400 bg-green-500/10 border-green-500/30"
                : item.sentiment === "bearish" ? "text-red-400 bg-red-500/10 border-red-500/30"
                : "text-muted-foreground bg-muted/30 border-border";
              const sentimentLabel =
                item.sentiment === "bullish" ? "▲ Bullish"
                : item.sentiment === "bearish" ? "▼ Bearish"
                : "● Neutral";
              return (
                <Card className={`border-l-2 ${borderAccent}`}>
                  <CardContent className="pt-4 pb-4 flex flex-col gap-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${sentimentColor}`}>
                        {sentimentLabel}
                      </span>
                      <span className="text-xs text-muted-foreground truncate">{item.source}</span>
                    </div>
                    <p className="font-semibold text-sm leading-snug line-clamp-2">{item.title}</p>
                    <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3">{item.summary}</p>
                    <div className="flex items-center justify-between pt-1">
                      {item.url && (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-cyan-400 hover:underline"
                        >
                          Read more →
                        </a>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })()}
          </div>
        </div>

        {/* ── Nova Insight — full width, shown after any forecast is run ── */}
        {(() => {
          const insightText = isCrypto ? assemblyForecast?.nova_insight : novaInsight;
          const isLoading = isCrypto ? false : novaInsightLoading;
          if (!chartForecast && !assemblyForecast) return null;
          if (!insightText && !isLoading) return null;
          return (
            <div className="mt-2 rounded-lg border border-cyan-500/30 bg-cyan-500/5 p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded-full font-semibold tracking-wide">
                  Nova Insight
                </span>
                <span className="text-xs text-muted-foreground">Powered by Amazon Bedrock Nova</span>
              </div>
              {isLoading ? (
                <div className="flex items-center gap-2 text-sm text-cyan-400/70">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Generating insight…
                </div>
              ) : (
                <p className="text-sm text-cyan-100/80 leading-relaxed">{insightText}</p>
              )}
            </div>
          );
        })()}
        </>
      )}

      {/* ── Crypto metrics cards ── */}
      {isCrypto && initialSymbol && initialPrices && initialPrices.length > 0 && cryptoMetrics && (
        <>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Error Metrics Comparison</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-sm text-muted-foreground mb-3">
                Rolling window evaluation (-10 / -30 / -60 days). Average error across 3 holdout windows. Lower values = better accuracy.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 font-medium">Model</th>
                      <th className="text-right py-2 font-medium">MAE</th>
                      <th className="text-right py-2 font-medium">RMSE</th>
                      <th className="text-right py-2 font-medium">MAPE %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(() => {
                      const rollingRows = cryptoMetrics.metrics.filter(m => m.model === "assembly" || m.model === "chronos");
                      const bestMape = Math.min(...rollingRows.map(m => m.mape));
                      return rollingRows.map((row) => {
                        const isWinner = row.mape === bestMape;
                        return (
                          <tr key={row.model} className="border-b border-border/50">
                            <td className="py-2 font-medium flex items-center gap-2">
                              {row.model === "assembly" ? "Assembly (GRU+N-HiTS+LGB)" : "Chronos (benchmark)"}
                              {isWinner && <span className="text-xs bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded-full font-semibold tracking-wide">BEST</span>}
                            </td>
                            <td className={`text-right py-2 ${isWinner ? "font-semibold text-cyan-400" : ""}`}>{row.mae.toFixed(2)}</td>
                            <td className={`text-right py-2 ${isWinner ? "font-semibold text-cyan-400" : ""}`}>{row.rmse.toFixed(2)}</td>
                            <td className={`text-right py-2 ${isWinner ? "font-semibold text-cyan-400" : ""}`}>{row.mape.toFixed(2)}%</td>
                          </tr>
                        );
                      });
                    })()}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Model Training Info</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-sm text-muted-foreground mb-3">
                Assembly model components trained on {initialSymbol.toUpperCase()} daily OHLCV data.
              </p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Architecture</span><span className="font-medium">GRU + N-HiTS + LightGBM → Ridge</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Max horizon</span><span className="font-medium">7 days</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Confidence level</span><span className="font-medium">95%</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Last trained</span><span className="font-medium">{cryptoMetrics.metrics.find(m => m.model === "assembly")?.trained_at?.slice(0, 10) ?? "—"}</span></div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ── Regime robustness card ── */}
        {(() => {
          const regimeRows = cryptoMetrics.metrics.filter(m => m.model.endsWith("_regime"));
          if (regimeRows.length === 0) return null;
          const bestMape = Math.min(...regimeRows.map(m => m.mape));
          return (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Market Regime Robustness</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <p className="text-sm text-muted-foreground mb-3">
                  Avg MAPE across 3 regime cutoff dates (Apr 2024 halving, Jan 2025 bull run, Jan 2026 current). Tests model stability during market transitions.
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 font-medium">Model</th>
                        <th className="text-right py-2 font-medium">MAE</th>
                        <th className="text-right py-2 font-medium">RMSE</th>
                        <th className="text-right py-2 font-medium">MAPE %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {regimeRows.map(row => {
                        const isWinner = row.mape === bestMape;
                        const label = row.model === "assembly_regime" ? "Assembly (GRU+N-HiTS+LGB)" : "Chronos (benchmark)";
                        return (
                          <tr key={row.model} className="border-b border-border/50">
                            <td className="py-2 font-medium flex items-center gap-2">
                              {label}
                              {isWinner && <span className="text-xs bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded-full font-semibold tracking-wide">BEST</span>}
                            </td>
                            <td className={`text-right py-2 ${isWinner ? "font-semibold text-cyan-400" : ""}`}>{row.mae.toFixed(2)}</td>
                            <td className={`text-right py-2 ${isWinner ? "font-semibold text-cyan-400" : ""}`}>{row.rmse.toFixed(2)}</td>
                            <td className={`text-right py-2 ${isWinner ? "font-semibold text-cyan-400" : ""}`}>{row.mape.toFixed(2)}%</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          );
        })()}
        </>
      )}

      {/* ── Assembly 7-day forecast card ── */}
      {isCrypto && assemblyForecast && assemblyForecast.dates.length > 0 && (() => {
        const lastPrice = initialPrices && initialPrices.length > 0
          ? initialPrices[initialPrices.length - 1].close_price
          : null;
        return (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Assembly Model — 7-Day Price Forecast</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-sm text-muted-foreground mb-3">
                Predicted prices from GRU + N-HiTS + LightGBM → Ridge ensemble. 95% confidence interval shown.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 font-medium">Date</th>
                      <th className="text-right py-2 font-medium">Predicted Price</th>
                      <th className="text-right py-2 font-medium">Low</th>
                      <th className="text-right py-2 font-medium">High</th>
                      <th className="text-right py-2 font-medium">Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {assemblyForecast.dates.map((date, i) => {
                      const pt = assemblyForecast.point_forecast[i];
                      const lb = assemblyForecast.lower_bound[i];
                      const ub = assemblyForecast.upper_bound[i];
                      const base = i === 0 ? lastPrice : assemblyForecast.point_forecast[i - 1];
                      const change = base ? ((pt - base) / base) * 100 : null;
                      const isUp = change !== null && change >= 0;
                      return (
                        <tr key={date} className="border-b border-border/50">
                          <td className="py-2 text-muted-foreground">
                            {new Date(date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                          </td>
                          <td className="text-right py-2 font-semibold">
                            ${pt.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                          </td>
                          <td className="text-right py-2 text-muted-foreground">
                            ${lb.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                          </td>
                          <td className="text-right py-2 text-muted-foreground">
                            ${ub.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                          </td>
                          <td className={`text-right py-2 font-medium ${isUp ? "text-green-600 dark:text-green-400" : "text-red-500"}`}>
                            {change !== null ? `${isUp ? "+" : ""}${change.toFixed(2)}%` : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

            </CardContent>
          </Card>
        );
      })()}

      {/* Error Metrics & Forecast Bounds — below chart, progressive display */}
      {(!isCrypto || (isCrypto && !cryptoMetrics)) && initialSymbol && initialPrices && initialPrices.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Error Metrics Comparison</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-sm text-muted-foreground mb-3 min-h-[2.5rem]">
                Walk-forward 60-day Rolling Backtest: Lower values indicate better accuracy.
              </p>
              {metrics?.error && (
                <p className="text-sm text-amber-600 dark:text-amber-400">{metrics.error}</p>
              )}
              {metrics && !metrics.error && metrics.metrics.length === 0 && inProgressModels.length === 0 && (
                <p className="text-sm text-muted-foreground">No metrics computed (models may have failed).</p>
              )}
              {((metrics?.metrics.length ?? 0) > 0 || inProgressModels.length > 0) && (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 font-medium">Model</th>
                        <th className="text-right py-2 font-medium">MAE</th>
                        <th className="text-right py-2 font-medium">RMSE</th>
                        <th className="text-right py-2 font-medium">MAPE %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(inProgressModels.length > 0
                          ? inProgressModels
                          : (metrics?.metrics ?? []).map((r) => r.model)
                      ).map((modelKey) => {
                        const row = metrics?.metrics.find((r) => r.model === modelKey);
                        const loading = inProgressModels.includes(modelKey as ForecastModelKey);
                        return (
                          <tr key={modelKey} className="border-b border-border/50">
                            <td className="py-2 capitalize">{modelKey.replace("_", "+")}</td>
                            {loading ? (
                              <td className="text-right py-2" colSpan={3}>
                                <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                  Loading…
                                </span>
                              </td>
                            ) : row ? (
                              <>
                                <td className="text-right py-2">{row.mae.toFixed(2)}</td>
                                <td className="text-right py-2">{row.rmse.toFixed(2)}</td>
                                <td className="text-right py-2">{row.mape.toFixed(2)}%</td>
                              </>
                            ) : (
                              <td colSpan={3} className="text-right py-2 text-muted-foreground">—</td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>
                Forecast Bounds ({metrics ? `${metrics.bounds_horizon_weeks} ${metrics.interval === "1mo" ? "month(s)" : "week(s)"}` : `${forecastDays} days`} horizon)
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-sm text-muted-foreground mb-3 min-h-[2.5rem]">
                Lowest expected price, highest expected price, and average forecast per model over the selected horizon.
              </p>
              {metrics && metrics.bounds.length === 0 && !metrics.error && inProgressModels.length === 0 && (
                <p className="text-sm text-muted-foreground">Load metrics to see bounds (uses same horizon as chart).</p>
              )}
              {((metrics?.bounds.length ?? 0) > 0 || inProgressModels.length > 0) && (
                <div className="overflow-x-auto space-y-4">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 font-medium">Model</th>
                        <th className="text-right py-2 font-medium">Lowest expected</th>
                        <th className="text-right py-2 font-medium">Highest expected</th>
                        <th className="text-right py-2 font-medium">Average forecast</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(inProgressModels.length > 0
                          ? inProgressModels
                          : (metrics?.bounds ?? []).map((b) => b.model)
                      ).map((modelKey) => {
                        const b = metrics?.bounds.find((x) => x.model === modelKey);
                        const loading = inProgressModels.includes(modelKey as ForecastModelKey);
                        if (loading) {
                          return (
                            <tr key={modelKey} className="border-b border-border/50">
                              <td className="py-2 capitalize">{modelKey.replace("_", "+")}</td>
                              <td className="text-right py-2" colSpan={3}>
                                <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                  Loading…
                                </span>
                              </td>
                            </tr>
                          );
                        }
                        if (!b) return null;
                        const lowest = b.lower.length ? Math.min(...b.lower) : 0;
                        const highest = b.upper.length ? Math.max(...b.upper) : 0;
                        const avg =
                          b.forecast.length
                            ? b.forecast.reduce((s, v) => s + v, 0) / b.forecast.length
                            : 0;
                        return (
                          <tr key={b.model} className="border-b border-border/50">
                            <td className="py-2 capitalize">{b.model.replace("_", "+")}</td>
                            <td className="text-right py-2">${lowest.toFixed(2)}</td>
                            <td className="text-right py-2">${highest.toFixed(2)}</td>
                            <td className="text-right py-2 font-medium">${avg.toFixed(2)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}


      {initialSymbol && (!initialPrices || initialPrices.length === 0) && (
        <div className="text-center py-20 text-muted-foreground">
          No price data found for {initialSymbol.toUpperCase()}. Try syncing it again.
        </div>
      )}
    </div>
  );
}
