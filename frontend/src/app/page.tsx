"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Plus,
  Play,
  Sparkles,
  Film,
  Compass,
  Activity,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { MatchCard as MatchCardType } from "@/lib/types";
import { fetchMatches, API_BASE } from "@/lib/api";
import { resolveMatchDisplay } from "@/lib/format";
import { BidiNum } from "@/components/ui/BidiNum";
import { HeroPitchVisual } from "@/components/hub/HeroPitchVisual";
import { KPIStrip } from "@/components/hub/KPIStrip";
import {
  FilterToolbar,
  type SortOption,
  type FilterType,
} from "@/components/hub/FilterToolbar";
import { MatchGrid } from "@/components/hub/MatchGrid";

export default function MatchHubPage() {
  const { t, lang, isRTL } = useLang();
  const [matches, setMatches] = useState<MatchCardType[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  // Filter & Search states
  const [searchQuery, setSearchQuery] = useState("");
  const [sortOption, setSortOption] = useState<SortOption>("default");
  const [filterType, setFilterType] = useState<FilterType>("all");

  const ArrowIcon = isRTL ? ArrowLeft : ArrowRight;

  const loadData = useCallback(async () => {
    try {
      const healthRes = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
      setBackendOnline(healthRes.ok);
    } catch {
      setBackendOnline(false);
    }

    try {
      const data = await fetchMatches();
      setMatches(data);
    } catch (err) {
      console.error("Failed to load matches:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Sync automatically when Navbar UploadModal finishes analyzing a new clip
  useEffect(() => {
    const handleChanged = () => {
      loadData();
    };
    window.addEventListener("tv:matches-changed", handleChanged);
    return () => window.removeEventListener("tv:matches-changed", handleChanged);
  }, [loadData]);

  const handleOpenUpload = () => {
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tv:open-upload"));
    }
  };

  const handleOpenWorkflow = () => {
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tv:open-how-it-works"));
    }
  };

  const handleDeleteMatch = useCallback((deletedId: string) => {
    setMatches((prev) => prev.filter((m) => m.id !== deletedId));
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tv:matches-changed"));
    }
  }, []);

  const filteredMatches = useMemo(() => {
    let result = [...matches];

    if (filterType === "demo") {
      result = result.filter((m) => !m.is_user_uploaded);
    } else if (filterType === "user") {
      result = result.filter((m) => m.is_user_uploaded);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter((m) => {
        const display = resolveMatchDisplay(m, t, lang);
        return (
          display.title.toLowerCase().includes(q) ||
          display.tag.toLowerCase().includes(q) ||
          m.title.toLowerCase().includes(q)
        );
      });
    }

    switch (sortOption) {
      case "speed":
        result.sort((a, b) => (b.top_speed_kmh || 0) - (a.top_speed_kmh || 0));
        break;
      case "passes":
        result.sort((a, b) => {
          const aPass = (a.passes?.team_0 || 0) + (a.passes?.team_1 || 0);
          const bPass = (b.passes?.team_0 || 0) + (b.passes?.team_1 || 0);
          return bPass - aPass;
        });
        break;
      case "events":
        result.sort((a, b) => (b.events_count || 0) - (a.events_count || 0));
        break;
      case "duration":
        result.sort((a, b) => (b.duration_s || 0) - (a.duration_s || 0));
        break;
      default:
        break;
    }

    return result;
  }, [matches, filterType, searchQuery, sortOption, t, lang]);

  const handleResetFilters = () => {
    setSearchQuery("");
    setSortOption("default");
    setFilterType("all");
  };

  const featuredMatch = matches[0] ?? null;
  const featuredId = featuredMatch?.id || "match_01_counter";

  const pipelineSteps = [
    {
      num: "01",
      icon: Film,
      title: t("hub.pipe_1_title"),
      sub: t("hub.pipe_1_sub"),
      accent: "text-tv-cyan",
    },
    {
      num: "02",
      icon: Compass,
      title: t("hub.pipe_2_title"),
      sub: t("hub.pipe_2_sub"),
      accent: "text-tv-mint",
    },
    {
      num: "03",
      icon: Activity,
      title: t("hub.pipe_3_title"),
      sub: t("hub.pipe_3_sub"),
      accent: "text-tv-coral",
    },
  ];

  return (
    <div className="relative min-h-screen pb-24">
      {/* ── Ambient Obsidian Studio Backdrop ── */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden -z-10">
        <div
          className="absolute inset-0 opacity-[0.025]"
          style={{
            backgroundImage: `
              linear-gradient(to right, rgba(255, 255, 255, 0.5) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(255, 255, 255, 0.5) 1px, transparent 1px)
            `,
            backgroundSize: "72px 72px",
          }}
        />
        <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-[900px] h-[420px] bg-tv-cyan/[0.06] blur-[140px] rounded-full" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 sm:pt-9">
        {/* ── 2-Column Editorial Studio Hero ── */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-7 lg:gap-9 items-center mb-9 pb-9 border-b border-white/[0.07]">
          {/* Narrative, CTAs & 3-Step Hairline Pipeline Strip */}
          <div className="lg:col-span-6 space-y-5">
            {/* Status Badges */}
            <div className="flex flex-wrap items-center gap-2.5">
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-tv-cyan/10 border border-tv-cyan/25 text-xs font-semibold text-tv-cyan shadow-[0_0_20px_-6px_rgba(56,189,248,0.45)]">
                <Sparkles className="w-3.5 h-3.5" />
                <span>{t("hub.hero_badge")}</span>
              </div>

              <div
                className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${
                  backendOnline === true
                    ? "bg-tv-mint/10 border-tv-mint/25 text-tv-mint"
                    : backendOnline === false
                    ? "bg-tv-amber/10 border-tv-amber/25 text-tv-amber"
                    : "bg-white/[0.04] border-white/10 text-slate-400"
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
                    ? t("hub.status_offline")
                    : t("hub.status_connected")}
                </span>
              </div>
            </div>

            {/* Display Headline & Value Proposition */}
            <div className="space-y-3">
              <h1 className="text-2xl sm:text-4xl lg:text-[42px] font-extrabold tracking-tight leading-[1.18] bg-gradient-to-b from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                {t("hub.hero_title")}
              </h1>
              <p className="text-sm sm:text-base text-slate-400 max-w-xl leading-relaxed">
                {t("hub.hero_subtitle")}
              </p>
            </div>

            {/* Action Cluster */}
            <div className="flex flex-wrap items-center gap-3 pt-1">
              <button
                type="button"
                onClick={handleOpenUpload}
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl font-bold text-sm bg-tv-cyan text-slate-950 hover:bg-tv-cyan/90 shadow-lg shadow-tv-cyan/25 active:scale-[0.98] transition-all cursor-pointer"
              >
                <Plus className="w-4 h-4 stroke-[2.75]" />
                <span>{t("hub.cta_upload")}</span>
              </button>

              <Link
                href={`/match/${featuredId}`}
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl font-semibold text-sm text-white bg-white/[0.06] hover:bg-white/[0.11] border border-white/15 transition-all"
              >
                <Play className="w-3.5 h-3.5 fill-current text-tv-cyan" />
                <span>{t("hub.cta_demo")}</span>
                <ArrowIcon className="w-3.5 h-3.5 text-slate-400" />
              </Link>

              <button
                type="button"
                onClick={handleOpenWorkflow}
                className="inline-flex items-center gap-1.5 px-3.5 py-3 rounded-xl font-semibold text-xs text-slate-300 hover:text-white hover:bg-white/[0.05] transition-colors cursor-pointer"
              >
                <Compass className="w-3.5 h-3.5 text-tv-mint" />
                <span>{t("hub.cta_workflow")}</span>
              </button>
            </div>

            {/* 3-Step Hairline-Divided Studio Pipeline Strip (Zero Heavy Box Soup) */}
            <div className="rounded-xl obsidian-well grid grid-cols-1 sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x rtl:sm:divide-x-reverse divide-white/[0.07] overflow-hidden mt-2">
              {pipelineSteps.map((step) => {
                const Icon = step.icon;
                return (
                  <div
                    key={step.num}
                    className="px-3.5 py-3 flex items-start gap-2.5"
                  >
                    <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${step.accent}`} />
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <BidiNum className="text-[11px] font-bold text-slate-500">
                          {step.num}
                        </BidiNum>
                        <span className="text-xs font-bold text-slate-200 truncate">
                          {step.title}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 truncate mt-0.5">
                        {step.sub}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Interactive 60FPS Real-Data Animated Broadcast Pitch Centerpiece */}
          <div className="lg:col-span-6">
            <HeroPitchVisual featuredMatch={featuredMatch} />
          </div>
        </section>

        {/* ── Dynamic D3 Executive KPI Strip ── */}
        <KPIStrip matches={matches} isLoading={isLoading} />

        {/* ── Interactive Filter & Sort Toolbar ── */}
        <FilterToolbar
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          sortOption={sortOption}
          onSortChange={setSortOption}
          filterType={filterType}
          onFilterChange={setFilterType}
          totalCount={filteredMatches.length}
        />

        {/* ── Broadcast-Grade Match Cards Grid ── */}
        <MatchGrid
          matches={filteredMatches}
          isLoading={isLoading}
          onResetFilters={handleResetFilters}
          onDeleteMatch={handleDeleteMatch}
        />
      </div>
    </div>
  );
}
