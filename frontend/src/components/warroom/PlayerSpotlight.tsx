"use client";

import React, { useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Play } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { Telemetry, PlayerStat } from "@/lib/types";
import { formatRole, formatTeamName } from "@/lib/format";
import { BidiNum } from "@/components/ui/BidiNum";
import { D3PlayerRadar } from "./D3PlayerRadar";

interface PlayerSpotlightProps {
  selectedPlayerId: number | null;
  telemetry: Telemetry | null;
  currentFrameIndex: number;
  onDeselect: () => void;
  onSeekToFrame: (frame: number) => void;
}

/**
 * Inline Docked Tactical Player Inspector Panel.
 * Opens smoothly directly BELOW the War Room Video + 2D Pitch cockpit so
 * 0% of PitchCanvas or VideoPlayer is ever obstructed in desktop or mobile viewports.
 */
export function PlayerSpotlight({
  selectedPlayerId,
  telemetry,
  currentFrameIndex,
  onDeselect,
  onSeekToFrame,
}: PlayerSpotlightProps) {
  const { t, lang } = useLang();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && selectedPlayerId !== null) {
        onDeselect();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedPlayerId, onDeselect]);

  const playerStat: PlayerStat | null = useMemo(() => {
    if (selectedPlayerId === null || !telemetry?.players) return null;
    return (
      telemetry.players.find((p) => p.track_id === selectedPlayerId) ?? null
    );
  }, [selectedPlayerId, telemetry]);

  const currentFramePlayer = useMemo(() => {
    if (selectedPlayerId === null || !telemetry?.frames) return null;
    const f = telemetry.frames[currentFrameIndex];
    if (!f || !f.players) return null;
    return f.players.find((p) => p.id === selectedPlayerId) ?? null;
  }, [selectedPlayerId, telemetry, currentFrameIndex]);

  // Fallback metrics directly computed from real frames[] for <0.8s track IDs
  const fallbackStats: PlayerStat | null = useMemo(() => {
    if (playerStat || selectedPlayerId === null || !telemetry?.frames)
      return null;

    let framesCount = 0;
    let maxSpd = 0;
    let sumSpd = 0;
    let sumX = 0;
    let sumY = 0;
    let team: 0 | 1 = 0;
    let role: "player" | "goalkeeper" = "player";

    for (const f of telemetry.frames) {
      const p = f.players?.find((pl) => pl.id === selectedPlayerId);
      if (p) {
        framesCount++;
        team = p.team;
        role = p.role === "gk" ? "goalkeeper" : "player";
        if (p.spd > maxSpd) maxSpd = p.spd;
        sumSpd += p.spd;
        sumX += p.x;
        sumY += p.y;
      }
    }

    const fps = telemetry.summary?.fps || 25;
    return {
      track_id: selectedPlayerId,
      team,
      role,
      distance_m: Number(((sumSpd / 3.6 / fps) * framesCount).toFixed(1)),
      max_speed_kmh: Number(maxSpd.toFixed(1)),
      avg_speed_kmh: Number(
        (framesCount > 0 ? sumSpd / framesCount : 0).toFixed(1)
      ),
      sprints: 0,
      passes_completed: 0,
      xt_generated: 0,
      avg_pitch_xy:
        framesCount > 0 ? [sumX / framesCount, sumY / framesCount] : [52.5, 34.0],
      frames_visible: framesCount,
    };
  }, [playerStat, selectedPlayerId, telemetry]);

  const activeStat = playerStat ?? fallbackStats;

  const playerEvents = useMemo(() => {
    if (selectedPlayerId === null || !telemetry?.events) return [];
    return telemetry.events.filter((e) => {
      if (e.type === "pass")
        return e.from_id === selectedPlayerId || e.to_id === selectedPlayerId;
      if (e.type === "sprint") return e.track_id === selectedPlayerId;
      if (e.type === "turnover") return e.by_id === selectedPlayerId;
      return false;
    });
  }, [selectedPlayerId, telemetry]);

  const isOpen = selectedPlayerId !== null && activeStat !== null;
  const isTeam0 = activeStat?.team === 0;
  const teamBadgeClass = isTeam0
    ? "bg-tv-cyan/15 border-tv-cyan/35 text-tv-cyan"
    : "bg-tv-coral/15 border-tv-coral/35 text-tv-coral";
  const currentSpeed = currentFramePlayer?.spd?.toFixed(1) ?? "0.0";
  const isCurrentlySprinting = (currentFramePlayer?.spd ?? 0) >= 24.0;

  return (
    <AnimatePresence>
      {isOpen && activeStat && (
        <motion.section
          key="docked-player-inspector"
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ type: "spring", stiffness: 380, damping: 32 }}
          aria-label="Player Inspector Panel"
          className="w-full rounded-2xl obsidian-card border border-white/[0.14] p-4 sm:p-5 shadow-2xl"
        >
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
            {/* ── Start (7 cols): Identity, Live Velocity, Clean Hairline Metrics & Event Jumps ── */}
            <div className="lg:col-span-7 flex flex-col gap-4">
              {/* Top Header */}
              <div className="flex items-start justify-between gap-3 pb-3.5 border-b border-white/[0.08]">
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className={`w-11 h-11 rounded-xl flex items-center justify-center font-mono font-extrabold text-sm shadow-lg shrink-0 ${
                      isTeam0
                        ? "bg-tv-cyan text-slate-950 shadow-tv-cyan/25"
                        : "bg-tv-coral text-white shadow-tv-coral/25"
                    }`}
                  >
                    <BidiNum>#{activeStat.track_id}</BidiNum>
                  </div>

                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-sm sm:text-base font-bold text-white">
                        {formatRole(activeStat.role, lang)}{" "}
                        <BidiNum>#{activeStat.track_id}</BidiNum>
                      </h3>
                      <span
                        className={`px-2 py-0.5 rounded-md text-[11px] font-bold border ${teamBadgeClass}`}
                      >
                        {formatTeamName(activeStat.team, lang)}
                      </span>
                    </div>

                    {/* Live Frame Velocity */}
                    <div className="flex flex-wrap items-center gap-2 mt-1">
                      <span className="text-xs text-slate-400">
                        {t("spot.live_speed")}:
                      </span>
                      <span
                        className={`inline-flex items-baseline gap-1 text-xs font-bold px-2 py-0.5 rounded-md ${
                          isCurrentlySprinting
                            ? "bg-tv-mint/20 text-tv-mint border border-tv-mint/40 animate-pulse"
                            : "obsidian-well text-slate-200"
                        }`}
                      >
                        <BidiNum>{currentSpeed}</BidiNum>
                        <span>{t("km_h")}</span>
                      </span>
                      {isCurrentlySprinting && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-tv-mint/15 text-tv-mint">
                          {t("spot.sprinting_now")}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={onDeselect}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-xs font-semibold text-slate-300 hover:text-white bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.08] transition-colors cursor-pointer shrink-0"
                  title={t("spot.close")}
                  aria-label={t("spot.close")}
                >
                  <X className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Esc</span>
                </button>
              </div>

              {/* Clean Hairline-Divided 6-Metric Strip (Zero Box-Soup, Zero Truncated Labels) */}
              <div className="grid grid-cols-2 sm:grid-cols-3 rounded-xl obsidian-well divide-y sm:divide-y-0 sm:divide-x rtl:sm:divide-x-reverse divide-white/[0.07] overflow-hidden">
                <div className="p-3 text-center">
                  <span className="block text-[11px] text-slate-400 font-medium mb-1">
                    {t("spot.total_dist")}
                  </span>
                  <div className="inline-flex items-baseline gap-1">
                    <BidiNum className="text-base font-bold text-white">
                      {activeStat.distance_m.toFixed(1)}
                    </BidiNum>
                    <span className="text-xs text-slate-400">{t("meters")}</span>
                  </div>
                </div>

                <div className="p-3 text-center">
                  <span className="block text-[11px] text-slate-400 font-medium mb-1">
                    {t("spot.peak_speed")}
                  </span>
                  <div className="inline-flex items-baseline gap-1">
                    <BidiNum className="text-base font-bold text-tv-mint">
                      {activeStat.max_speed_kmh.toFixed(1)}
                    </BidiNum>
                    <span className="text-xs text-slate-400">{t("km_h")}</span>
                  </div>
                </div>

                <div className="p-3 text-center">
                  <span className="block text-[11px] text-slate-400 font-medium mb-1">
                    {t("spot.avg_velocity")}
                  </span>
                  <div className="inline-flex items-baseline gap-1">
                    <BidiNum className="text-base font-bold text-slate-200">
                      {activeStat.avg_speed_kmh.toFixed(1)}
                    </BidiNum>
                    <span className="text-xs text-slate-400">{t("km_h")}</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 rounded-xl obsidian-well divide-x rtl:divide-x-reverse divide-white/[0.07] overflow-hidden">
                <div className="p-3 text-center">
                  <span className="block text-[11px] text-slate-400 font-medium mb-1">
                    {t("spot.sprints")}
                  </span>
                  <BidiNum className="text-base font-bold text-tv-amber">
                    {activeStat.sprints}
                  </BidiNum>
                </div>

                <div className="p-3 text-center">
                  <span className="block text-[11px] text-slate-400 font-medium mb-1">
                    {t("spot.passes_done")}
                  </span>
                  <BidiNum className="text-base font-bold text-tv-cyan">
                    {activeStat.passes_completed}
                  </BidiNum>
                </div>

                <div className="p-3 text-center">
                  <span className="block text-[11px] text-slate-400 font-medium mb-1">
                    {t("spot.xt_created")}
                  </span>
                  <BidiNum className="text-base font-bold text-tv-mint">
                    +{activeStat.xt_generated.toFixed(3)}
                  </BidiNum>
                </div>
              </div>

              {/* Localized Quick Event Jump Chips */}
              {playerEvents.length > 0 && (
                <div className="pt-2 border-t border-white/[0.07]">
                  <div className="flex items-center justify-between mb-2 text-xs">
                    <span className="font-bold text-slate-300">
                      {t("spot.key_events")}
                    </span>
                    <BidiNum className="px-2 py-0.5 rounded-md obsidian-well text-[11px] text-tv-cyan font-bold">
                      {playerEvents.length}
                    </BidiNum>
                  </div>

                  <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pe-1">
                    {playerEvents.map((evt, idx) => (
                      <button
                        key={`p-evt-${evt.frame}-${idx}`}
                        type="button"
                        onClick={() => onSeekToFrame(evt.frame)}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg obsidian-well hover:border-tv-cyan/40 text-xs transition-colors text-slate-200 cursor-pointer"
                      >
                        <Play className="w-2.5 h-2.5 fill-current text-tv-cyan shrink-0" />
                        <span className="font-semibold">
                          {t(`evt.${evt.type}`)}
                        </span>
                        <span className="inline-flex items-baseline gap-0.5 text-[11px] text-slate-400">
                          <BidiNum>{evt.time_s.toFixed(1)}</BidiNum>
                          <span>{t("seconds")}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ── End (5 cols): 6-Axis D3 Tactical Spider Chart ── */}
            <div className="lg:col-span-5">
              <D3PlayerRadar
                player={activeStat}
                telemetry={telemetry}
                allPlayers={telemetry?.players || []}
              />
            </div>
          </div>
        </motion.section>
      )}
    </AnimatePresence>
  );
}
