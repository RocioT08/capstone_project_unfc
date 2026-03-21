"use client";

import { useState } from "react";
import { PriceOut, ForecastResponse } from "@/types/api";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  ReferenceLine,
  Legend,
} from "recharts";
import { useToast } from "@/hooks/use-toast";

interface StockChartProps {
  symbol: string;
  initialPrices: PriceOut[];
  forecastDays?: 7 | 14 | 21;
  setForecastDays?: (days: 7 | 14 | 21) => void;
  onForecastComplete?: (model: "chronos" | "assembly") => void;
  onForecastData?: (data: ForecastResponse, model: "chronos" | "assembly") => void;
  metricsLoading?: boolean;
  isCrypto?: boolean;
}

export function StockChart({
  symbol,
  initialPrices,
  forecastDays = 7,
  setForecastDays,
  onForecastComplete,
  onForecastData,
  metricsLoading = false,
  isCrypto = false,
}: StockChartProps) {
  const [model, setModel] = useState<"chronos" | "assembly">("chronos");
  let interval: "1d" | "1wk" | "1mo" = "1d";
  const [viewDays, setViewDays] = useState<30 | 90 | 180 | 365 | 0>(30);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  // Reverse prices to be chronological for the chart
  let baseData = [...initialPrices].reverse();
  // Slice to view window (0 = all)
  if (viewDays > 0) baseData = baseData.slice(-viewDays);

  // Helper: group daily data by a key and return mean close per group.
  // Returned array is sorted chronologically by first-seen key.
  function aggregateByKey(
    data: PriceOut[],
    keyFn: (d: Date) => string
  ): PriceOut[] {
    const order: string[] = [];
    const groups: Record<
      string,
      { sum: number; count: number; first: PriceOut }
    > = {};
    data.forEach((p) => {
      const key = keyFn(new Date(p.timestamp));
      if (!groups[key]) {
        groups[key] = { sum: 0, count: 0, first: p };
        order.push(key);
      }
      groups[key].sum += p.close_price;
      groups[key].count += 1;
    });
    // Use insertion-order array to guarantee chronological output.
    return order.map((key) => ({
      ...groups[key].first,
      close_price: groups[key].sum / groups[key].count,
    }));
  }

  // Returns a stable Monday-anchored ISO week string "YYYY-Www".
  function isoWeekKey(d: Date): string {
    // Copy date and shift to the nearest Monday (start of ISO week).
    const day = new Date(d);
    const dow = day.getUTCDay(); // 0=Sun … 6=Sat
    const diff = dow === 0 ? -6 : 1 - dow; // shift to Monday
    day.setUTCDate(day.getUTCDate() + diff);
    const yyyy = day.getUTCFullYear();
    const mm = String(day.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(day.getUTCDate()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd}`; // one key per calendar week
  }

  // Aggregate daily data into weekly or monthly means.
  if (interval === "1wk") {
    baseData = aggregateByKey(baseData, isoWeekKey);
  } else if (interval === "1mo") {
    baseData = aggregateByKey(
      baseData,
      (d) =>
        `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`
    );
  }

  const chartData: Array<{
    date: string;
    price?: number;
    forecast?: number;
    lower?: number;
    upper?: number;
  }> = baseData.map((p) => ({
    date: p.timestamp,
    price: p.close_price,
  }));

  // Forecast overlays (only when backend returned non-empty forecast)
  let firstForecastDate: string | null = null;
  if (forecast && baseData.length > 0 && forecast.dates.length > 0) {
    const lastHist = baseData[baseData.length - 1];
    chartData.push({
      date: lastHist.timestamp,
      forecast: lastHist.close_price,
    });
    forecast.dates.forEach((dateStr, i) => {
      firstForecastDate = firstForecastDate ?? dateStr;
      chartData.push({
        date: dateStr,
        forecast: forecast.point_forecast[i],
        lower: forecast.lower_bound[i],
        upper: forecast.upper_bound[i],
      });
    });
  }

  // ── X-axis label format varies by interval ────────────────────────────
  const MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];

  function formatXAxis(iso: string): string {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    const mon = MONTHS[d.getUTCMonth()];
    const day = d.getUTCDate();
    const yr = String(d.getUTCFullYear()).slice(2); // "26"
    if (interval === "1mo") return `${mon} '${yr}`; // "Feb '26"
    if (interval === "1wk") return `${mon} ${day}`; // "Feb 23"
    return `${mon} ${day}`; // "Feb 23" (daily)
  }

  function formatTooltipLabel(iso: string): string {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    const mon = MONTHS[d.getUTCMonth()];
    const day = d.getUTCDate();
    const year = d.getUTCFullYear();
    if (interval === "1mo") return `${mon} ${year}`;
    return `${mon} ${day}, ${year}`;
  }

  // Widen tick spacing for denser intervals so labels don't overlap.
  const minTickGap = interval === "1d" ? 60 : interval === "1wk" ? 50 : 40;

  const handleAnalyze = async () => {
    setIsLoading(true);
    const periods = forecastDays ?? 7;
    try {
      let res;
      if (isCrypto && model === "assembly") {
        const cryptoRes = await api.cryptoForecast(symbol, { periods });
        res = {
          ...cryptoRes,
          interval: "1d",
          forecast_horizon_label: `${periods} day${
            periods > 1 ? "s" : ""
          } ahead (Assembly model)`,
          data_points_used: 0,
        };
      } else {
        res = await api.analyze(symbol, {
          model: "chronos",
          interval: "1d",
          periods,
        });
      }
      setForecast(res);
      onForecastData?.(res, model);
      toast({
        title: "Forecast Complete",
        description:
          isCrypto && model === "assembly"
            ? `Assembly model (GRU + N-HiTS + LightGBM) — ${periods}-day forecast.`
            : `Forecast generated using Chronos model.`,
      });
      onForecastComplete?.(model);
    } catch (error: any) {
      toast({
        title: "Forecast Failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Description row */}
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          Daily price data · 7-day forecast · 95% confidence interval
        </p>
        {/* View range buttons */}
        <div id="tour-stock-interval" className="flex items-center gap-1">
          {([30, 90, 180, 365, 0] as const).map((d) => (
            <button
              key={d}
              onClick={() => { setViewDays(d); setForecast(null); }}
              className={`px-2 py-1 text-xs rounded font-medium transition-colors ${
                viewDays === d
                  ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {d === 0 ? "All" : d === 30 ? "1M" : d === 90 ? "3M" : d === 180 ? "6M" : "1Y"}
            </button>
          ))}
        </div>
      </div>

      {/* Model + forecast controls row */}
      <div className="flex flex-wrap gap-3 items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">Forecast model:</span>
          <div id="tour-stock-model">
            <Select
              value={isCrypto ? model : "chronos"}
              onValueChange={(val: string) => setModel(val as "chronos" | "assembly")}
            >
              <SelectTrigger className="w-[210px] h-9">
                <SelectValue placeholder="Select Model" />
              </SelectTrigger>
              <SelectContent>
                {isCrypto && (
                  <SelectItem value="assembly">Assembly (GRU+N-HiTS+LGB)</SelectItem>
                )}
                <SelectItem value="chronos">
                  Chronos {isCrypto ? "(benchmark)" : ""}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <div id="tour-stock-forecast-btn">
          <Button onClick={handleAnalyze} disabled={isLoading || metricsLoading}>
            {(isLoading || metricsLoading) && (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            )}
            Generate Forecast
          </Button>
        </div>
      </div>

      <div className="h-[400px] w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={minTickGap}
              tickFormatter={formatXAxis}
            />
            <YAxis
              domain={["auto", "auto"]}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `$${value}`}
            />
            <Tooltip
              formatter={(value: number, name: string) => [
                `$${Number(value).toFixed(2)}`,
                name === "price"
                  ? "Historical"
                  : name === "forecast"
                  ? "Forecast"
                  : name,
              ]}
              labelFormatter={(label) => formatTooltipLabel(label)}
              contentStyle={{
                backgroundColor: "#0f172a",
                border: "1px solid rgba(0,212,255,0.25)",
                borderRadius: "8px",
                color: "#f1f5f9",
              }}
              labelStyle={{
                color: "#22d3ee",
                fontWeight: 700,
                marginBottom: 4,
              }}
              itemStyle={{ color: "#f1f5f9" }}
              cursor={{ stroke: "rgba(0,212,255,0.3)", strokeWidth: 1 }}
            />
            <Legend
              wrapperStyle={{ paddingTop: 8 }}
              formatter={(value) =>
                value === "price"
                  ? "Historical"
                  : value === "forecast"
                  ? "Forecast"
                  : value
              }
            />
            {firstForecastDate && (
              <ReferenceLine
                x={firstForecastDate}
                stroke="hsl(var(--muted-foreground))"
                strokeDasharray="4 4"
                strokeOpacity={0.7}
              />
            )}
            <Line
              type="monotone"
              dataKey="price"
              name="price"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              dot={false}
              connectNulls={false}
            />
            {forecast && (
              <Area
                type="monotone"
                dataKey="upper"
                stroke="none"
                fill="#a78bfa"
                fillOpacity={0.12}
                dot={false}
                connectNulls={true}
                legendType="none"
                name="upper_hidden"
              />
            )}
            {forecast && (
              <Area
                type="monotone"
                dataKey="lower"
                stroke="none"
                fill="#0f172a"
                fillOpacity={1}
                dot={false}
                connectNulls={true}
                legendType="none"
                name="lower_hidden"
              />
            )}
            {forecast && (
              <Line
                type="monotone"
                dataKey="forecast"
                name="Forecast"
                stroke="#a78bfa"
                strokeWidth={2}
                strokeDasharray="6 4"
                dot={false}
                connectNulls={true}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {forecast && (
        <p className="text-sm text-muted-foreground text-center">
          {forecast.forecast_horizon_label}
        </p>
      )}
    </div>
  );
}
