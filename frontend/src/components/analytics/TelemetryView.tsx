"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Layers,
  Activity,
  Shield,
  Users,
  LayoutGrid,
  Share2,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { BidiNum } from "@/components/ui/BidiNum";
import { D3PassNetworkPitch } from "@/components/d3/D3PassNetworkPitch";
import { D3MomentumStream } from "@/components/d3/D3MomentumStream";
import { D3ZonalControlMatrix } from "@/components/d3/D3ZonalControlMatrix";
import { TeamComparison } from "./TeamComparison";
import { SpeedChart } from "./SpeedChart";
import { DefLineChart } from "./DefLineChart";
import { PlayerLeaderboard } from "./PlayerLeaderboard";
import type { Telemetry, MatchCard } from "@/lib/types";

interface TelemetryViewProps {
  telemetry: Telemetry | null;
  match: MatchCard | null;
  currentTime?: number;
  onSeekToTime?: (timeSec: number) => void;
  onSeekToFrame?: (frame: number) => void;
  onSelectPlayerAndJump?: (playerId: number) => void;
}

type TelemetrySubTab =
  | "all"
  | "h2h"
  | "passnet"
  | "kinematics"
  | "defline"
  | "leaderboard";

export function TelemetryView({
  telemetry,
  match,
  currentTime = 0,
  onSeekToTime,
  onSeekToFrame,
  onSelectPlayerAndJump,
}: TelemetryViewProps) {
  const { t } = useLang();
  const [subTab, setSubTab] = useState<TelemetrySubTab>("all");

  const matchId = match?.id || telemetry?.match_info?.source_video || "match";
  const clipDuration =
    match?.duration_s || telemetry?.summary?.duration_s || 30.0;

  const subTabs: Array<{
    key: TelemetrySubTab;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
  }> = [
    { key: "all", label: t("tel.tab_all"), icon: LayoutGrid },
    { key: "h2h", label: t("tel.tab_overview"), icon: Layers },
    { key: "passnet", label: t("tel.tab_passnet"), icon: Share2 },
    { key: "kinematics", label: t("tel.tab_kinematics"), icon: Activity },
    { key: "defline", label: t("tel.tab_defline"), icon: Shield },
    { key: "leaderboard", label: t("tel.tab_leaderboard"), icon: Users },
  ];

  return (
    <div className="space-y-6">
      {/* ── Executive Telemetry KPI Strip ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {/* Tracked Players */}
        <div className="rounded-2xl p-3.5 obsidian-card">
          <span className="text-[11px] font-semibold text-slate-400 block mb-1 truncate">
            {t("tel.kpi_players")}
          </span>
          <BidiNum className="text-xl font-black text-white">
            {telemetry?.players?.length ?? "--"}
          </BidiNum>
        </div>

        {/* Tactical Events */}
        <div className="rounded-2xl p-3.5 obsidian-card">
          <span className="text-[11px] font-semibold text-slate-400 block mb-1 truncate">
            {t("tel.kpi_events")}
          </span>
          <BidiNum className="text-xl font-black text-tv-cyan">
            {telemetry?.events?.length ?? "--"}
          </BidiNum>
        </div>

        {/* Peak Sprint Speed */}
        <div className="rounded-2xl p-3.5 obsidian-card">
          <span className="text-[11px] font-semibold text-slate-400 block mb-1 truncate">
            {t("tel.kpi_top_speed")}
          </span>
          <div className="flex items-baseline gap-1.5">
            <BidiNum className="text-xl font-black text-tv-mint">
              {telemetry?.summary?.top_speed_kmh?.toFixed(1) ?? "--"}
            </BidiNum>
            <span className="text-[11px] text-slate-400 font-medium">
              {t("km_h")}
            </span>
          </div>
        </div>

        {/* Homography Stability */}
        <div className="rounded-2xl p-3.5 obsidian-card">
          <span className="text-[11px] font-semibold text-slate-400 block mb-1 truncate">
            {t("tel.kpi_homography")}
          </span>
          <BidiNum className="text-xl font-black text-tv-cyan">
            {`${
              telemetry?.engine_diagnostics?.homography?.stability_rate_pct?.toFixed(
                0
              ) ?? "100"
            }%`}
          </BidiNum>
        </div>

        {/* Ball Tracking Coverage */}
        <div className="rounded-2xl p-3.5 obsidian-card">
          <span className="text-[11px] font-semibold text-slate-400 block mb-1 truncate">
            {t("tel.kpi_ball")}
          </span>
          <BidiNum className="text-xl font-black text-white">
            {telemetry?.engine_diagnostics?.ball_coverage_pct !== undefined
              ? `${telemetry.engine_diagnostics.ball_coverage_pct.toFixed(1)}%`
              : "--"}
          </BidiNum>
        </div>

        {/* Capture Cadence */}
        <div className="rounded-2xl p-3.5 obsidian-card">
          <span className="text-[11px] font-semibold text-slate-400 block mb-1 truncate">
            {t("tel.kpi_fps")}
          </span>
          <div className="flex items-baseline gap-1.5">
            <BidiNum className="text-xl font-black text-slate-200">
              {Math.round(match?.fps ?? telemetry?.match_info?.fps ?? 25)}
            </BidiNum>
            <span className="text-[11px] text-slate-400 font-medium">
              {t("card.fps")}
            </span>
          </div>
        </div>
      </div>

      {/* ── Sub-Navigation Pill Strip (Hidden in Print) ── */}
      <div className="flex items-center justify-between flex-wrap gap-3 pb-2 border-b border-white/[0.08] print:hidden">
        <div className="inline-flex flex-wrap items-center gap-1 p-1 rounded-2xl obsidian-card max-w-full">
          {subTabs.map((item) => {
            const Icon = item.icon;
            const active = subTab === item.key;
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => setSubTab(item.key)}
                className={`relative flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-colors whitespace-nowrap cursor-pointer ${
                  active
                    ? "text-white font-bold"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="telemetry-subtab-pill"
                    transition={{
                      type: "spring",
                      stiffness: 460,
                      damping: 34,
                    }}
                    className="absolute inset-0 rounded-xl editorial-tab-active"
                  >
                    <span className="absolute bottom-0 inset-x-3 h-[2px] rounded-full bg-tv-cyan" />
                  </motion.span>
                )}
                <Icon
                  className={`w-3.5 h-3.5 relative z-10 ${
                    active ? "text-tv-cyan" : ""
                  }`}
                />
                <span className="relative z-10">{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Sub-Tab Views ── */}
      <div className="space-y-6">
        {/* 1. Head-to-Head Tactical Comparison & 12x8 Zonal Dominance Matrix */}
        {(subTab === "all" || subTab === "h2h") && (
          <div className="space-y-6">
            <TeamComparison telemetry={telemetry} />
            <D3ZonalControlMatrix telemetry={telemetry} />
          </div>
        )}

        {/* 2. Interactive D3 Curved Passing & xT Network */}
        {(subTab === "all" || subTab === "passnet") && (
          <div className="space-y-4">
            <D3PassNetworkPitch
              telemetry={telemetry}
              onSelectPlayer={onSelectPlayerAndJump}
            />
          </div>
        )}

        {/* 3. Territorial Voronoi Momentum Stream + Dual-Player Speed & Radar */}
        {(subTab === "all" || subTab === "kinematics") && (
          <div className="space-y-6">
            <D3MomentumStream
              telemetry={telemetry}
              currentTime={currentTime}
              duration_s={clipDuration}
              onSeekToTime={onSeekToTime}
              onSeekToFrame={onSeekToFrame}
            />
            <SpeedChart telemetry={telemetry} onSeekToFrame={onSeekToFrame} />
          </div>
        )}

        {/* 4. Defensive Block Height & Inter-Line Compression */}
        {(subTab === "all" || subTab === "defline") && (
          <div className="space-y-4">
            <DefLineChart telemetry={telemetry} onSeekToFrame={onSeekToFrame} />
          </div>
        )}

        {/* 5. Player Kinematic Leaderboard & CSV/JSON/PDF Export */}
        {(subTab === "all" || subTab === "leaderboard") && (
          <div className="space-y-4">
            <PlayerLeaderboard
              telemetry={telemetry}
              matchId={matchId}
              onSelectPlayer={onSelectPlayerAndJump}
            />
          </div>
        )}
      </div>
    </div>
  );
}
