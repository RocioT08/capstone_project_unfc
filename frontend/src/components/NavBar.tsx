"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Home, TrendingUp, PieChart, BookOpen } from "lucide-react";

const navItems = [
  { href: "/",          label: "Home",        icon: Home       },
  { href: "/learn",     label: "Learn",       icon: BookOpen   },
  { href: "/stock",     label: "Forecasting", icon: TrendingUp },
  { href: "/portfolio", label: "Portfolio",   icon: PieChart   },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/5 bg-[#080D18]/90 backdrop-blur-xl">
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-3">
          <Image
            src="/unf-logo.svg"
            alt="UNF Logo"
            width={122}
            height={37}
            className="brightness-0 invert opacity-90"
            priority
          />
          <span className="hidden sm:inline-block font-bold text-base text-white/60 border-l border-white/15 pl-3">
            
          </span>
        </Link>

        {/* Nav pills */}
        <nav className="hidden md:flex items-center gap-1 rounded-full border border-white/8 bg-white/4 px-2 py-1.5">
          {navItems.map(({ href, label, icon: Icon }) => {
            const active =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-1.5 rounded-full px-4 py-1.5 text-sm font-semibold transition-all duration-200 ${
                  active
                    ? "bg-cyan-400/15 text-cyan-400"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Mobile icon */}
        <div className="md:hidden flex items-center">
          <Image
            src="/unf-logo.svg"
            alt="UNF Logo"
            width={80}
            height={24}
            className="brightness-0 invert opacity-90"
          />
        </div>
      </div>
    </header>
  );
}
