"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  Plus,
  Compass,
  LayoutGrid,
  Play,
  Activity,
  Cpu,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { fetchMatches, API_BASE } from "@/lib/api";
import type { MatchCard } from "@/lib/types";
import { resolveMatchDisplay } from "@/lib/format";
import { BidiNum } from "@/components/ui/BidiNum";
import { LangToggle } from "./LangToggle";
import { HowItWorksModal } from "./HowItWorksModal";
import { UploadModal } from "@/components/hub/UploadModal";

/**
 * Custom Bespoke Animated Pitch-Radar Vector Emblem
 */
function BrandEmblem() {
  return (
    <div className="relative w-9 h-9 rounded-xl bg-gradient-to-br from-[#0F1B2E] to-[#070C16] border border-tv-cyan/30 flex items-center justify-center shadow-md shadow-tv-cyan/10 overflow-hidden shrink-0 group-hover:border-tv-cyan/60 transition-colors">
      <svg
        viewBox="0 0 36 36"
        className="w-7 h-7"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        {/* Outer mini FIFA pitch frame */}
        <rect
          x="4"
          y="7"
          width="28"
          height="22"
          rx="2.5"
          stroke="rgba(226,232,240,0.35)"
          strokeWidth="1.2"
        />
        {/* Halfway line & center circle */}
        <line
          x1="18"
          y1="7"
          x2="18"
          y2="29"
          stroke="rgba(226,232,240,0.28)"
          strokeWidth="1"
        />
        <circle
          cx="18"
          cy="18"
          r="4.2"
          stroke="rgba(30,200,255,0.55)"
          strokeWidth="1.1"
        />
        {/* Tactical curved pass arc */}
        <path
          d="M 10 22 Q 18 11 26 15"
          stroke="#34D399"
          strokeWidth="1.4"
          strokeLinecap="round"
          strokeDasharray="2 2"
        />
        {/* Team A & Team B nodes */}
        <circle cx="10" cy="22" r="2.2" fill="#38BDF8" />
        <circle cx="26" cy="15" r="2.2" fill="#34D399" />
        <circle cx="23" cy="23" r="1.9" fill="#FB7185" />
      </svg>
    </div>
  );
}

export function Navbar() {
  const { t, lang } = useLang();
  const pathname = usePathname();

  const [matches, setMatches] = useState<MatchCard[]>([]);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [switcherOpen, setSwitcherOpen] = useState(false);
  const [howItWorksOpen, setHowItWorksOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);

  const switcherRef = useRef<HTMLDivElement | null>(null);

  const loadNavData = useCallback(async () => {
    try {
      const healthRes = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
      setBackendOnline(healthRes.ok);
    } catch {
      setBackendOnline(false);
    }

    try {
      const list = await fetchMatches();
      setMatches(list);
    } catch {
      // Silent fallback if backend is offline
    }
  }, []);

  useEffect(() => {
    loadNavData();
  }, [loadNavData]);

  // Global custom events so Hero, MobileNav, and any page can open Upload or HowItWorks cleanly
  useEffect(() => {
    const handleOpenUpload = () => {
      setSwitcherOpen(false);
      setUploadOpen(true);
    };
    const handleOpenHiw = () => {
      setSwitcherOpen(false);
      setHowItWorksOpen(true);
    };
    const handleSwitchTab = () => {
      setSwitcherOpen(false);
    };
    const handleMatchesChanged = () => {
      loadNavData();
    };

    window.addEventListener("tv:open-upload", handleOpenUpload);
    window.addEventListener("tv:open-how-it-works", handleOpenHiw);
    window.addEventListener("tv:switch-tab", handleSwitchTab);
    window.addEventListener("tv:matches-changed", handleMatchesChanged);

    return () => {
      window.removeEventListener("tv:open-upload", handleOpenUpload);
      window.removeEventListener("tv:open-how-it-works", handleOpenHiw);
      window.removeEventListener("tv:switch-tab", handleSwitchTab);
      window.removeEventListener("tv:matches-changed", handleMatchesChanged);
    };
  }, [loadNavData]);

  // Close match switcher on route change, outside click, or Escape key
  useEffect(() => {
    setSwitcherOpen(false);
  }, [pathname]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (switcherRef.current && !switcherRef.current.contains(e.target as Node)) {
        setSwitcherOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSwitcherOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("click", handleClickOutside);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("click", handleClickOutside);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const handleMatchCreated = useCallback(() => {
    loadNavData();
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tv:matches-changed"));
    }
  }, [loadNavData]);

  const isHub = pathname === "/";
  const isSpotting = pathname.startsWith("/action-spotting");
  const isModels = pathname.startsWith("/models");
  const activeMatchId = pathname.startsWith("/match/")
    ? decodeURIComponent(pathname.replace("/match/", "").split("/")[0])
    : null;

  return (
    <>
      <header className="sticky top-0 z-40 h-14 w-full bg-[#04070F]/85 backdrop-blur-2xl border-b border-white/[0.08] shadow-[inset_0_-1px_0_rgba(255,255,255,0.03)] print:hidden">
        <div className="max-w-[1850px] mx-auto h-full px-3 sm:px-6 lg:px-8 flex items-center justify-between gap-2 sm:gap-4">
          {/* ── Start: Brand Identity (Strictly LTR so brand reads "TacticalVision AI" in EN & AR) ── */}
          <div className="flex items-center gap-3 min-w-0">
            <Link href="/" className="flex items-center gap-2.5 group min-w-0">
              <BrandEmblem />
              <div className="flex items-center gap-1.5 min-w-0">
                <span
                  dir="ltr"
                  className="inline-flex items-baseline gap-1 truncate font-sans"
                >
                  <span
                    data-brand-word="primary"
                    className="text-sm sm:text-base font-extrabold text-white tracking-tight"
                  >
                    {t("nav.title")}
                  </span>
                  <span
                    data-brand-word="ai"
                    className="text-xs sm:text-sm font-black text-tv-cyan"
                  >
                    AI
                  </span>
                </span>
                <span className="hidden xl:inline-flex px-2 py-0.5 rounded-md text-[10px] font-semibold bg-white/[0.05] border border-white/10 text-slate-400 ms-1 shrink-0">
                  {t("nav.studio_badge")}
                </span>
              </div>
            </Link>
          </div>

          {/* ── Center: Persistent Workspace Navigation ── */}
          <nav
            aria-label="Primary workspace navigation"
            className="hidden md:flex items-center gap-1 p-1 rounded-xl bg-white/[0.03] border border-white/[0.07]"
          >
            {/* 1. Match Library */}
            <Link
              href="/"
              className={`relative flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                isHub ? "text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              {isHub && (
                <motion.span
                  layoutId="navbar-workspace-pill"
                  transition={{ type: "spring", stiffness: 450, damping: 34 }}
                  className="absolute inset-0 rounded-lg bg-white/[0.09] border border-white/15 shadow-sm"
                />
              )}
              <LayoutGrid className="w-3.5 h-3.5 relative z-10 text-tv-cyan" />
              <span className="relative z-10">{t("nav.hub")}</span>
            </Link>

            {/* 2. Quick Match Switcher Dropdown */}
            <div ref={switcherRef} className="relative">
              <button
                type="button"
                onClick={() => setSwitcherOpen((prev) => !prev)}
                className={`relative flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                  activeMatchId || switcherOpen
                    ? "text-white"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {activeMatchId && (
                  <motion.span
                    layoutId="navbar-workspace-pill"
                    transition={{ type: "spring", stiffness: 450, damping: 34 }}
                    className="absolute inset-0 rounded-lg bg-white/[0.09] border border-white/15 shadow-sm"
                  />
                )}
                <span className="relative z-10">{t("nav.quick_switch")}</span>
                {matches.length > 0 && (
                  <BidiNum className="relative z-10 px-1.5 py-0.2 rounded bg-tv-cyan/15 text-tv-cyan text-[11px] font-bold">
                    {matches.length}
                  </BidiNum>
                )}
                <ChevronDown
                  className={`w-3.5 h-3.5 relative z-10 transition-transform duration-200 ${
                    switcherOpen ? "rotate-180 text-tv-cyan" : "text-slate-400"
                  }`}
                />
              </button>

              <AnimatePresence>
                {switcherOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 8, scale: 0.98 }}
                    transition={{ duration: 0.16 }}
                    className="absolute top-full mt-2 start-0 w-80 rounded-2xl obsidian-popover p-2 shadow-2xl z-50"
                  >
                    <div className="px-3 py-1.5 text-[11px] font-semibold text-slate-400 border-b border-white/[0.06] mb-1 flex items-center justify-between">
                      <span>{t("nav.quick_switch_title")}</span>
                      <BidiNum className="text-slate-500">{matches.length}</BidiNum>
                    </div>

                    <div className="max-h-72 overflow-y-auto space-y-1">
                      {matches.map((m) => {
                        const display = resolveMatchDisplay(m, t, lang);
                        const isCurrent = activeMatchId === m.id;
                        const posA = m.possession_pct?.team_0 ?? 50;
                        const posB = m.possession_pct?.team_1 ?? 50;

                        return (
                          <Link
                            key={m.id}
                            href={`/match/${m.id}`}
                            onClick={() => setSwitcherOpen(false)}
                            className={`block px-3 py-2.5 rounded-xl transition-all ${
                              isCurrent
                                ? "bg-tv-cyan/12 border border-tv-cyan/30"
                                : "hover:bg-white/[0.05] border border-transparent"
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2 mb-1.5">
                              <span
                                className={`text-xs font-bold truncate ${
                                  isCurrent ? "text-tv-cyan" : "text-white"
                                }`}
                              >
                                {display.title}
                              </span>
                              <span className="inline-flex items-baseline gap-0.5 text-[11px] text-slate-400 shrink-0">
                                <BidiNum>{m.duration_s?.toFixed(1)}</BidiNum>
                                <span>{t("seconds")}</span>
                              </span>
                            </div>

                            {/* Mini Possession Bar (RTL/LTR synchronized) */}
                            <div className="h-1 w-full rounded-full bg-slate-800 overflow-hidden flex">
                              <div
                                style={{ width: `${Math.max(4, Math.min(96, posA))}%` }}
                                className="h-full bg-tv-cyan rounded-s-full"
                              />
                              <div
                                style={{ width: `${Math.max(4, Math.min(96, posB))}%` }}
                                className="h-full bg-tv-coral rounded-e-full"
                              />
                            </div>

                            <div className="flex items-center justify-between mt-1 text-[11px] text-slate-400">
                              <span>{display.badge}</span>
                              <span className="inline-flex items-center gap-1 text-tv-cyan font-medium">
                                <Play className="w-2.5 h-2.5 fill-current" />
                                <span>{t("card.view")}</span>
                              </span>
                            </div>
                          </Link>
                        );
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* 3. Action Spotting Studio (Model 3) */}
            <Link
              href="/action-spotting"
              className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                isSpotting ? "text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              {isSpotting && (
                <motion.span
                  layoutId="navbar-workspace-pill"
                  transition={{ type: "spring", stiffness: 450, damping: 34 }}
                  className="absolute inset-0 rounded-lg bg-white/[0.09] border border-white/15 shadow-sm"
                />
              )}
              <Activity className="w-3.5 h-3.5 relative z-10 text-tv-mint" />
              <span className="relative z-10">{t("nav.action_spotting")}</span>
            </Link>

            {/* 4. AI Models & Benchmarks Lab */}
            <Link
              href="/models"
              className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                isModels ? "text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              {isModels && (
                <motion.span
                  layoutId="navbar-workspace-pill"
                  transition={{ type: "spring", stiffness: 450, damping: 34 }}
                  className="absolute inset-0 rounded-lg bg-white/[0.09] border border-white/15 shadow-sm"
                />
              )}
              <Cpu className="w-3.5 h-3.5 relative z-10 text-tv-amber" />
              <span className="relative z-10">{t("nav.models_lab")}</span>
            </Link>

            {/* 5. How It Works Workflow Drawer */}
            <button
              type="button"
              onClick={() => setHowItWorksOpen(true)}
              className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white hover:bg-white/[0.04] transition-colors cursor-pointer"
            >
              <Compass className="w-3.5 h-3.5 text-tv-mint" />
              <span>{t("nav.how_it_works")}</span>
            </button>
          </nav>

          {/* ── End: Engine Status, Primary Analyze CTA, & Always-Visible AR/EN Switcher ── */}
          <div className="flex items-center gap-2 sm:gap-2.5 shrink-0">
            {/* Live Backend Pulse Pill */}
            <button
              type="button"
              onClick={loadNavData}
              title={
                backendOnline
                  ? t("hub.status_connected")
                  : t("hub.status_offline")
              }
              className={`hidden lg:inline-flex items-center gap-2 px-2.5 py-1.5 rounded-xl text-xs font-medium border transition-colors cursor-pointer ${
                backendOnline === true
                  ? "bg-tv-mint/[0.08] border-tv-mint/25 text-tv-mint"
                  : backendOnline === false
                  ? "bg-tv-amber/[0.08] border-tv-amber/25 text-tv-amber"
                  : "bg-white/[0.03] border-white/10 text-slate-400"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  backendOnline === true
                    ? "bg-tv-mint animate-pulse"
                    : backendOnline === false
                    ? "bg-tv-amber"
                    : "bg-slate-500"
                }`}
              />
              <span>
                {backendOnline === false
                  ? t("nav.engine_offline")
                  : t("nav.engine_online")}
              </span>
            </button>

            {/* Mobile How It Works Icon Button */}
            <button
              type="button"
              onClick={() => setHowItWorksOpen(true)}
              aria-label={t("nav.how_it_works")}
              className="md:hidden p-2 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-slate-300 hover:text-white transition-colors cursor-pointer"
            >
              <Compass className="w-4 h-4 text-tv-mint" />
            </button>

            {/* Primary Analyze Match CTA */}
            <button
              type="button"
              onClick={() => setUploadOpen(true)}
              className="hidden sm:inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-bold bg-tv-cyan text-slate-950 hover:bg-tv-cyan/90 shadow-md shadow-tv-cyan/20 active:scale-[0.98] transition-all cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5 stroke-[2.75]" />
              <span>{t("nav.analyze_cta")}</span>
            </button>

            {/* Bilingual Segmented Switcher — Visible on Desktop & Mobile (375px+) */}
            <LangToggle />
          </div>
        </div>
      </header>

      {/* Global Product Workflow Modal */}
      <HowItWorksModal
        isOpen={howItWorksOpen}
        onClose={() => setHowItWorksOpen(false)}
        onOpenUpload={() => setUploadOpen(true)}
        featuredMatchId={matches[0]?.id || "match_01_counter"}
      />

      {/* Global Hybrid Upload Modal */}
      <UploadModal
        isOpen={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onMatchCreated={handleMatchCreated}
      />
    </>
  );
}
