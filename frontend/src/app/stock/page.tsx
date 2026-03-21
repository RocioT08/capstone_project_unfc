import { api } from "@/lib/api";
import { StockDashboard } from "./StockDashboard";

export const dynamic = "force-dynamic";

export default async function StockPage({
  searchParams,
}: {
  searchParams: Promise<{ symbol?: string; from?: string; to?: string }>;
}) {
  const { symbol, from, to } = await searchParams;

  // Default dates: last year to today
  const today = new Date();
  const lastYear = new Date();
  lastYear.setFullYear(today.getFullYear() - 1);

  const fromDate = from || lastYear.toISOString().split('T')[0];
  const toDate = to || today.toISOString().split('T')[0];

  // Fetch all available assets for the dropdown
  const assets = await api.getAssets().catch(() => []);

  // Normalize symbol: if user typed "BTC" try to match "BTC-USD" from assets
  let upperSymbol = symbol?.toUpperCase();
  if (upperSymbol && !assets.find((a) => a.symbol === upperSymbol)) {
    const withSuffix = assets.find((a) => a.symbol === `${upperSymbol}-USD`);
    if (withSuffix) upperSymbol = withSuffix.symbol;
  }

  let prices = null;
  let stats = null;

  if (upperSymbol && assets.find((a) => a.symbol === upperSymbol)) {
    try {
      prices = await api.getPrices(upperSymbol, 2500);
    } catch (e) {
      console.error("Failed to fetch prices:", e);
    }

    try {
      // Pick a partner of the same asset_type to avoid cross-type alignment issues
      const assetType = assets.find((a) => a.symbol === upperSymbol)?.asset_type;
      const partner = assets.find((a) => a.symbol !== upperSymbol && a.asset_type === assetType)?.symbol;
      if (partner) {
        stats = await api.portfolioStats({
          symbols: [upperSymbol, partner],
          interval: "1d",
          from_date: fromDate,
          to_date: toDate,
        });
      }
    } catch {
      // Stats are optional — partner may not have enough data
    }
  }

  return (
    <StockDashboard
      assets={assets}
      initialSymbol={upperSymbol}
      initialPrices={prices}
      initialStats={stats}
      initialFromDate={fromDate}
      initialToDate={toDate}
    />
  );
}
