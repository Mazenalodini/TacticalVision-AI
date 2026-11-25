"use client";

import React from "react";
import { Users, Grid, Shield, Compass, Flame, Share2 } from "lucide-react";
import { useLang } from "@/context/LangContext";
import { BidiNum } from "@/components/ui/BidiNum";
import type { HeatmapMode } from "./canvas/drawHeatmap";

export interface CanvasLayerConfig {
  players: boolean;
  voronoi: boolean;
  hulls: boolean;
  defLines: boolean;
  heatmap: boolean;
  heatmapMode: HeatmapMode;
  passNetwork: boolean;
}

export interface LiveTelemetryHud {
  voronoiSplit: [number, number];
  team0HullArea_m2: number;
  team1HullArea_m2: number;
  defLine0_x: number | null;
  defLine1_x: number | null;
  activePlayersCount: number;
  sprintingCount: number;
}

interface LayerTogglesProps {
  layers: CanvasLayerConfig;
  onToggleLayer: (layer: keyof Omit<CanvasLayerConfig, "heatmapMode">) => void;
  onChangeHeatmapMode: (mode: HeatmapMode) => void;
  hud?: LiveTelemetryHud;
  renderHudInline?: boolean;
}

/**
 * Unclipped 2-row responsive pill grid for all 6 tactical pitch layers.
 * Uses disciplined Obsidian glass hierarchy (no clashing rainbow borders).
 */
export function LayerToggles({
  layers,
  onToggleLayer,
  onChangeHeatmapMode,
  hud,
  renderHudInline = false,
}: LayerTogglesProps) {
  const { t } = useLang();

  const toggleItems = [
    {
      key: "players" as const,
      label: t("wr.layer_players"),
      icon: Users,
      active: layers.players,
      dotColor: "bg-tv-mint",
    },
    {
      key: "voronoi" as const,
      label: t("wr.layer_voronoi"),
      icon: Grid,
      active: layers.voronoi,
      dotColor: "bg-tv-cyan",
    },
    {
      key: "hulls" as const,
      label: t("wr.layer_hull"),
      icon: Shield,
      active: layers.hulls,
      dotColor: "bg-tv-coral",
    },
    {
      key: "defLines" as const,
      label: t("wr.layer_defline"),
      icon: Compass,
      active: layers.defLines,
      dotColor: "bg-tv-amber",
    },
    {
      key: "heatmap" as const,
      label: t("wr.layer_heatmap"),
      icon: Flame,
      active: layers.heatmap,
      dotColor: "bg-tv-amber",
    },
    {
      key: "passNetwork" as const,
      label: t("wr.layer_passnet"),
      icon: Share2,
      active: layers.passNetwork,
      dotColor: "bg-tv-cyan",
    },
  ];

  return (
    <div className="flex flex-col gap-2">
      {/* Unclipped 3x2 / 2x3 Responsive Grid — All 6 Layers 100% Visible */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
        {toggleItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onToggleLayer(item.key)}
              className={`flex items-center justify-between gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition-all cursor-pointer active:scale-[0.98] ${
                item.active
                  ? "bg-white/[0.08] border-tv-cyan/40 text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]"
                  : "obsidian-well text-slate-400 hover:text-slate-200 hover:border-white/15"
              }`}
            >
              <span className="inline-flex items-center gap-1.5 min-w-0">
                <Icon
                  className={`w-3.5 h-3.5 shrink-0 ${
                    item.active ? "text-tv-cyan" : "text-slate-500"
                  }`}
                />
                <span className="truncate">{item.label}</span>
              </span>
              <span
                className={`w-1.5 h-1.5 rounded-full shrink-0 transition-opacity ${
                  item.active ? `${item.dotColor} opacity-100` : "bg-slate-700 opacity-50"
                }`}
              />
            </button>
          );
        })}
      </div>

      {/* Heatmap Team Mode Selector (compact segmented bar when Heatmap is active) */}
      {layers.heatmap && (
        <div className="flex items-center justify-between gap-2 px-2.5 py-1 rounded-lg obsidian-well">
          <span className="text-[11px] font-medium text-slate-400">
            {t("wr.layer_heatmap")}:
          </span>
          <div className="inline-flex items-center gap-1">
            {(
              [
                ["team_0", t("team_a"), "text-tv-cyan bg-tv-cyan/15 border-tv-cyan/30"],
                ["team_1", t("team_b"), "text-tv-coral bg-tv-coral/15 border-tv-coral/30"],
                ["both", t("wr.hm_both"), "text-white bg-white/15 border-white/25"],
              ] as const
            ).map(([mode, label, activeClass]) => (
              <button
                key={mode}
                type="button"
                onClick={() => onChangeHeatmapMode(mode as HeatmapMode)}
                className={`px-2 py-0.5 rounded text-[11px] font-semibold border transition-colors cursor-pointer ${
                  layers.heatmapMode === mode
                    ? activeClass
                    : "border-transparent text-slate-400 hover:text-white"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {renderHudInline && hud && <LiveTacticalHud hud={hud} />}
    </div>
  );
}

/**
 * Compact 100% Localized & Bidi-Isolated Live Tactical HUD Strip
 */
export function LiveTacticalHud({ hud }: { hud: LiveTelemetryHud }) {
  const { t } = useLang();
  const pctA = Math.round(hud.voronoiSplit[0] ?? 50);
  const pctB = 100 - pctA;

  return (
    <div className="grid grid-cols-3 rounded-xl obsidian-well divide-x rtl:divide-x-reverse divide-white/[0.07] text-center overflow-hidden">
      {/* 1. Voronoi Control Split (Always Sums to 100% and Aligns Under Labels in EN & AR) */}
      <div className="px-2.5 py-2 flex flex-col justify-between">
        <span className="text-[11px] font-medium text-slate-400 block mb-1 truncate">
          {t("wr.hud_voronoi")}
        </span>
        <div className="flex items-center justify-between text-xs mb-1">
          <BidiNum className="text-tv-cyan font-bold">{pctA}%</BidiNum>
          <BidiNum className="text-tv-coral font-bold">{pctB}%</BidiNum>
        </div>
        <div className="h-1.5 w-full rounded-full bg-slate-800/90 overflow-hidden flex gap-0.5">
          <div
            style={{ width: `${Math.max(3, Math.min(97, pctA))}%` }}
            className="h-full bg-tv-cyan rounded-s-full transition-all duration-150"
          />
          <div
            style={{ width: `${Math.max(3, Math.min(97, pctB))}%` }}
            className="h-full bg-tv-coral rounded-e-full transition-all duration-150"
          />
        </div>
      </div>

      {/* 2. Outfield Convex Hull Area (m²) */}
      <div className="px-2.5 py-2 flex flex-col justify-center">
        <span className="text-[11px] font-medium text-slate-400 block mb-1 truncate">
          {t("wr.hud_hull")}
        </span>
        <div className="inline-flex items-baseline justify-center gap-1 text-xs font-bold">
          <BidiNum>
            <span className="text-tv-cyan">{hud.team0HullArea_m2 || 0}</span>
            <span className="text-slate-600 mx-1">/</span>
            <span className="text-tv-coral">{hud.team1HullArea_m2 || 0}</span>
          </BidiNum>
          <span className="text-[10px] text-slate-400">{t("meters")}²</span>
        </div>
      </div>

      {/* 3. Virtual Defensive Line X (m) */}
      <div className="px-2.5 py-2 flex flex-col justify-center">
        <span className="text-[11px] font-medium text-slate-400 block mb-1 truncate">
          {t("wr.hud_defline")}
        </span>
        <div className="inline-flex items-baseline justify-center gap-1 text-xs font-bold">
          <BidiNum>
            <span className="text-tv-cyan">
              {hud.defLine0_x !== null ? hud.defLine0_x.toFixed(1) : "--"}
            </span>
            <span className="text-slate-600 mx-1">/</span>
            <span className="text-tv-coral">
              {hud.defLine1_x !== null ? hud.defLine1_x.toFixed(1) : "--"}
            </span>
          </BidiNum>
          <span className="text-[10px] text-slate-400">{t("meters")}</span>
        </div>
      </div>
    </div>
  );
}
