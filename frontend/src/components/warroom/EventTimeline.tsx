"use client";

import React, { useMemo, useState } from "react";
import { useLang } from "@/context/LangContext";
import type { ActionSpottingEvent, MatchEvent, PlayerStat } from "@/lib/types";
import {
  getActionCategoryLabel,
  getActionClassLabel,
  getCategoryMeta,
} from "@/lib/actionSpottingHelpers";
import { resolvePlayerIdentity, usePlayerOverrides } from "@/lib/jerseyOverrides";
import { formatTeamName } from "@/lib/format";
import { BidiNum } from "@/components/ui/BidiNum";

export type EventFilterType = "all" | "broadcast" | "pass" | "turnover" | "sprint";

interface EventTimelineProps {
  matchId?: string;
  events: MatchEvent[];
  broadcastEvents?: ActionSpottingEvent[];
  players?: PlayerStat[];
  currentTime: number;
  duration_s: number;
  onSeekToFrame: (frame: number) => void;
  onSeekToTime: (timeSec: number) => void;
  onReplayEvent?: (timeSec: number) => void;
  onSelectPlayer?: (playerId: number | null) => void;
  docked?: boolean;
}

export function EventTimeline({
  matchId,
  events,
  broadcastEvents = [],
  players = [],
  currentTime,
  duration_s = 30.0,
  onSeekToFrame,
  onSeekToTime,
  onReplayEvent,
  onSelectPlayer,
  docked = false,
}: EventTimelineProps) {
  const { t, lang, dir } = useLang();
  const { overrides } = usePlayerOverrides(matchId);
  const statMap = useMemo(() => {
    const m = new Map<number, PlayerStat>();
    for (const p of players) m.set(p.track_id, p);
    return m;
  }, [players]);
  const playerLabel = (tid: number) =>
    resolvePlayerIdentity(tid, statMap.get(tid) ?? null, overrides).badgeText;
  const [filterType, setFilterType] = useState<EventFilterType>("all");
  const [hoveredEvent, setHoveredEvent] = useState<{
    kind: "spatial";
    event: MatchEvent;
    xPercent: number;
  } | {
    kind: "broadcast";
    event: ActionSpottingEvent;
    xPercent: number;
  } | null>(null);

  const counts = useMemo(() => {
    let passes = 0;
    let turnovers = 0;
    let sprints = 0;
    for (const e of events) {
      if (e.type === "pass") passes++;
      else if (e.type === "turnover") turnovers++;
      else if (e.type === "sprint") sprints++;
    }
    return {
      all: events.length + broadcastEvents.length,
      broadcast: broadcastEvents.length,
      pass: passes,
      turnover: turnovers,
      sprint: sprints,
    };
  }, [events, broadcastEvents]);

  const filteredEvents = useMemo(() => {
    if (filterType === "broadcast") return [];
    if (filterType === "all") return events;
    return events.filter((e) => e.type === filterType);
  }, [events, filterType]);

  const visibleBroadcastEvents = useMemo(() => {
    if (filterType === "all" || filterType === "broadcast") return broadcastEvents;
    return [];
  }, [broadcastEvents, filterType]);

  // Render sprints first so high-salience passes & turnovers layer cleanly on top
  const sortedRenderEvents = useMemo(() => {
    return [...filteredEvents].sort((a, b) => {
      const priority = (type: string) =>
        type === "turnover" ? 3 : type === "pass" ? 2 : 1;
      return priority(a.type) - priority(b.type);
    });
  }, [filteredEvents]);

  const progressPercent =
    duration_s > 0
      ? Math.min(100, Math.max(0, (currentTime / duration_s) * 100))
      : 0;

  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    onSeekToTime(ratio * duration_s);
  };

  return (
    <div
      className={`select-none px-3 py-2.5 ${
        docked
          ? "bg-[#080D17] border-t border-white/[0.07] rounded-b-2xl"
          : "obsidian-card rounded-xl"
      }`}
    >
      {/* ── Filter Chips & Localized Hint ── */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="inline-flex flex-wrap items-center gap-1 p-0.5 rounded-xl obsidian-well">
          <button
            type="button"
            onClick={() => setFilterType("all")}
            className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              filterType === "all"
                ? "bg-white/15 text-white font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <span>{t("evt.all")}</span>
            <BidiNum className="text-[11px] opacity-80">({counts.all})</BidiNum>
          </button>

          <button
            type="button"
            onClick={() => setFilterType("pass")}
            className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              filterType === "pass"
                ? "bg-tv-cyan/20 text-tv-cyan border border-tv-cyan/30 font-bold"
                : "text-slate-400 hover:text-tv-cyan"
            }`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-tv-cyan" />
            <span>{t("evt.pass")}</span>
            <BidiNum className="text-[11px]">({counts.pass})</BidiNum>
          </button>

          <button
            type="button"
            onClick={() => setFilterType("turnover")}
            className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              filterType === "turnover"
                ? "bg-tv-amber/20 text-tv-amber border border-tv-amber/30 font-bold"
                : "text-slate-400 hover:text-tv-amber"
            }`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-tv-amber" />
            <span>{t("evt.turnover")}</span>
            <BidiNum className="text-[11px]">({counts.turnover})</BidiNum>
          </button>

          {broadcastEvents.length > 0 && (
            <button
              type="button"
              onClick={() => setFilterType("broadcast")}
              className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                filterType === "broadcast"
                  ? "bg-violet-400/20 text-violet-300 border border-violet-400/40 font-bold"
                  : "text-slate-400 hover:text-violet-300"
              }`}
            >
              <span className="w-1.5 h-1.5 rotate-45 bg-violet-400" />
              <span>{t("evt.scope_broadcast")}</span>
              <BidiNum className="text-[11px]">({counts.broadcast})</BidiNum>
            </button>
          )}

          <button
            type="button"
            onClick={() => setFilterType("sprint")}
            className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              filterType === "sprint"
                ? "bg-tv-mint/20 text-tv-mint border border-tv-mint/30 font-bold"
                : "text-slate-400 hover:text-tv-mint"
            }`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-tv-mint" />
            <span>{t("evt.sprint")}</span>
            <BidiNum className="text-[11px]">({counts.sprint})</BidiNum>
          </button>
        </div>

        <span className="text-[11px] text-slate-500 hidden md:inline-block">
          {t("evt.click_hint")}
        </span>
      </div>

      {/* ── Proportional Timeline Track (Strictly LTR so 0s->end aligns with Video Scrubber) ── */}
      <div
        dir="ltr"
        onClick={handleTrackClick}
        onMouseLeave={() => setHoveredEvent(null)}
        className="relative h-8 w-full rounded-lg obsidian-well cursor-pointer overflow-visible flex items-center px-1"
      >
        {/* Subtle Time Grid Ticks */}
        <div className="absolute inset-0 flex justify-between pointer-events-none px-2 opacity-15">
          {[0, 1, 2, 3, 4, 5, 6].map((tick) => (
            <div key={tick} className="h-full w-px bg-white" />
          ))}
        </div>

        {/* Event Markers */}
        {sortedRenderEvents.map((evt, idx) => {
          const xPct = duration_s > 0 ? (evt.time_s / duration_s) * 100 : 0;
          const isPass = evt.type === "pass";
          const isTurnover = evt.type === "turnover";
          const isSprint = evt.type === "sprint";

          let color = "#34D399";
          let heightClass = "h-3";
          let widthClass = "w-0.5";
          let zIndex = 20;

          if (isPass) {
            color = evt.team === 0 ? "#38BDF8" : "#FB7185";
            heightClass = "h-5";
            widthClass = "w-1.5";
            zIndex = 30;
          } else if (isTurnover) {
            color = "#F59E0B";
            heightClass = "h-6";
            widthClass = "w-2";
            zIndex = 40;
          }

          const isActive = Math.abs(currentTime - evt.time_s) <= 0.3;

          return (
            <div
              key={`${evt.type}-${evt.frame}-${idx}`}
              style={{
                left: `${Math.min(99.5, Math.max(0.5, xPct))}%`,
                backgroundColor: color,
                zIndex: isActive ? 50 : zIndex,
              }}
              onClick={(e) => {
                e.stopPropagation();
                if (onReplayEvent) {
                  onReplayEvent(evt.time_s);
                } else {
                  onSeekToFrame(evt.frame);
                }
                if (isSprint && onSelectPlayer) {
                  onSelectPlayer(evt.track_id);
                } else if (isPass && onSelectPlayer) {
                  onSelectPlayer(evt.from_id);
                } else if (isTurnover && onSelectPlayer) {
                  onSelectPlayer(evt.by_id);
                }
              }}
              onMouseEnter={(e) => {
                e.stopPropagation();
                setHoveredEvent({ kind: "spatial", event: evt, xPercent: xPct });
              }}
              className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full cursor-pointer transition-transform hover:scale-150 hover:brightness-125 ${heightClass} ${widthClass} ${
                isActive ? "ring-2 ring-white scale-125 brightness-150" : ""
              }`}
            />
          );
        })}

        {/* Model 3 Broadcast Event Diamond Pins */}
        {visibleBroadcastEvents.map((bev, idx) => {
          const xPct = duration_s > 0 ? (bev.time_s / duration_s) * 100 : 0;
          const meta = getCategoryMeta(bev.category);
          const isActive = Math.abs(currentTime - bev.time_s) <= 0.4;

          return (
            <div
              key={`bcast-${bev.class_id}-${bev.frame_idx}-${idx}`}
              style={{
                left: `${Math.min(99.2, Math.max(0.8, xPct))}%`,
                backgroundColor: meta.color,
                zIndex: isActive ? 55 : 45,
              }}
              onClick={(e) => {
                e.stopPropagation();
                if (onReplayEvent) {
                  onReplayEvent(bev.time_s);
                } else {
                  onSeekToTime(bev.time_s);
                }
              }}
              onMouseEnter={(e) => {
                e.stopPropagation();
                setHoveredEvent({ kind: "broadcast", event: bev, xPercent: xPct });
              }}
              className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rotate-45 rounded-[2px] border border-[#060B10] cursor-pointer transition-transform hover:scale-150 shadow-sm ${
                isActive ? "ring-2 ring-white scale-125" : ""
              }`}
            />
          );
        })}

        {/* Live Synced Playhead Cursor */}
        <div
          style={{ left: `${progressPercent}%` }}
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.85)] pointer-events-none z-50 transition-all duration-75"
        >
          <div className="w-2.5 h-2.5 -top-1 left-1/2 -translate-x-1/2 absolute rounded-full bg-white shadow-md" />
        </div>

        {/* Localized Bidi-Safe Hover Tooltip */}
        {hoveredEvent && (
          <div
            dir={dir}
            style={{
              left: `${Math.min(82, Math.max(18, hoveredEvent.xPercent))}%`,
              bottom: "100%",
            }}
            className="absolute mb-2 -translate-x-1/2 z-50 pointer-events-none p-2.5 rounded-xl obsidian-popover shadow-2xl text-xs whitespace-nowrap"
          >
            {hoveredEvent.kind === "broadcast" ? (
              <>
                <div className="flex items-center justify-between gap-3 mb-1">
                  <span className="inline-flex items-center gap-1.5 font-bold text-white">
                    <span
                      className="w-2 h-2 rotate-45 rounded-[1px]"
                      style={{ backgroundColor: getCategoryMeta(hoveredEvent.event.category).color }}
                    />
                    <span>{getActionClassLabel(hoveredEvent.event.label, t)}</span>
                  </span>
                  <BidiNum className="text-[11px] text-tv-mint font-bold">
                    {`${Math.round(hoveredEvent.event.confidence * 100)}%`}
                  </BidiNum>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-slate-300">
                  <span>{getActionCategoryLabel(hoveredEvent.event.category, t)}</span>
                  <span>·</span>
                  <BidiNum className="text-tv-cyan font-semibold">
                    {`${hoveredEvent.event.time_s.toFixed(1)}s`}
                  </BidiNum>
                  <span>·</span>
                  <span className="text-tv-cyan font-bold">{t("evt.replay_3s")}</span>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center justify-between gap-3 mb-1">
                  <span className="inline-flex items-center gap-1.5 font-bold text-white">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor:
                          hoveredEvent.event.type === "turnover"
                            ? "#F59E0B"
                            : hoveredEvent.event.type === "sprint"
                            ? "#34D399"
                            : hoveredEvent.event.team === 0
                            ? "#38BDF8"
                            : "#FB7185",
                      }}
                    />
                    <span>{t(`evt.${hoveredEvent.event.type}`)}</span>
                  </span>
                  <BidiNum className="text-[11px] text-slate-400">
                    {hoveredEvent.event.time_s.toFixed(2)}s (F{hoveredEvent.event.frame})
                  </BidiNum>
                </div>

                {hoveredEvent.event.type === "pass" && (
                  <BidiNum className="text-[11px] text-slate-200">
                    {`${playerLabel(hoveredEvent.event.from_id)} → ${playerLabel(
                      hoveredEvent.event.to_id
                    )} · ${hoveredEvent.event.distance_m}m · xT `}
                    <span
                      className={
                        hoveredEvent.event.xt_delta >= 0
                          ? "text-tv-mint font-bold"
                          : "text-tv-coral font-bold"
                      }
                    >
                      {hoveredEvent.event.xt_delta >= 0
                        ? `+${hoveredEvent.event.xt_delta.toFixed(4)}`
                        : hoveredEvent.event.xt_delta.toFixed(4)}
                    </span>
                  </BidiNum>
                )}

                {hoveredEvent.event.type === "sprint" && (
                  <div className="text-[11px] text-tv-mint font-semibold flex items-center gap-1.5">
                    <span>
                      {t("evt.player_num")}{" "}
                      <BidiNum>{playerLabel(hoveredEvent.event.track_id)}</BidiNum>
                    </span>
                    <span>·</span>
                    <span className="inline-flex items-baseline gap-1">
                      <BidiNum>{hoveredEvent.event.speed_kmh.toFixed(1)}</BidiNum>
                      <span>{t("km_h")}</span>
                    </span>
                  </div>
                )}

                {hoveredEvent.event.type === "turnover" && (
                  <div className="text-[11px] text-tv-amber font-semibold flex items-center gap-1.5">
                    <span>
                      {t("evt.recovery_by")}{" "}
                      <BidiNum>{playerLabel(hoveredEvent.event.by_id)}</BidiNum>
                    </span>
                    <span className="text-slate-400">
                      ({formatTeamName(hoveredEvent.event.from_team, lang)} →{" "}
                      {formatTeamName(hoveredEvent.event.to_team, lang)})
                    </span>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
