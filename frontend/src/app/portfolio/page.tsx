import { api } from "@/lib/api";
import { PortfolioBuilder } from "./PortfolioBuilder";

export const dynamic = "force-dynamic";

export default async function PortfolioPage() {
  // Fetch all available assets for the dropdown
  const assets = await api.getAssets().catch(() => []);

  return <PortfolioBuilder assets={assets} />;
}
