import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import "shepherd.js/dist/css/shepherd.css";
import { LocalEnvBanner } from "@/components/LocalEnvBanner";
import { Toaster } from "@/components/ui/toaster";
import { NavBar } from "@/components/NavBar";
import { SparkChat } from "@/components/SparkChat"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "UNF Investor — Learn Investment Science With Real Data",
  description: "Explore forecasting models and portfolio optimization with real market data. Powered by UNF.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen flex flex-col text-base`}
      >
        <LocalEnvBanner />
        <NavBar />
        <main className="flex-1 container mx-auto px-4 py-6">
          {children}
        </main>
        <Toaster />
      </body>
    </html>
  );
}
