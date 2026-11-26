"use client";

import React, { useMemo, useState } from "react";
import { Zap, Search, X, ArrowRight, ArrowLeft } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { ActionSpottingEvent, MatchEvent } from "@/lib/types";
import {
  getActionCategoryLabel,
  getActionClassLabel,
  getCategoryMeta,
} from "@/lib/actionSpottingHelpers";
import { formatEventCount, formatTeamName } from "@/lib/format";
import { BidiNum } from "@/components/ui/BidiNum";

interface EventFeedProps {
  events: MatchEvent[];
  broadcastEvents?: ActionSpottingEvent[];
  currentTime: number;
  onSeekToFrame: (frame: number) => void;
  onSeekToTime?: (timeSec: number) => void;
  onSelectPlayer?: (playerId: number | null) => void;
}

export function EventFeed({
  events,
  broadcastEvents = [],
  currentTime,
  onSeekToFrame,
  onSeekToTime,
  onSelectPlayer,
}: EventFeedProps) {
  const { t, lang, isRTL } = useLang();
  const [typeFilter, setTypeFilter] = useState<
    "all" | "broadcast" | "pass" | "turnover" | "sprint"
  >("all");
  const [teamFilter, setTeamFilter] = useState<"all" | "0" | "1">("all");
  const [searchPlayer, setSearchPlayer] = useState<string>("");

  const DirArrow = isRTL ? ArrowLeft : ArrowRight;

  type UnifiedFeedItem =
    | { kind: "spatial"; time_s: number; event: MatchEvent }
    | { kind: "broadcast"; time_s: number; event: ActionSpottingEvent };

  const unifiedFeed = useMemo<UnifiedFeedItem[]>(() => {
    const items: UnifiedFeedItem[] = [];

    if (typeFilter !== "broadcast") {
      for (const e of events) {
        if (typeFilter !== "all" && e.type !== typeFilter) continue;

        if (teamFilter !== "all") {
          const teamNum = parseInt(teamFilter, 10);
          if (e.type === "pass" && e.team !== teamNum) continue;
          if (e.type === "sprint" && e.team !== teamNum) continue;
          if (
            e.type === "turnover" &&
            e.to_team !== teamNum &&
            e.from_team !== teamNum
          ) {
            continue;
          }
        }

        if (searchPlayer.trim()) {
          const clean = searchPlayer.replace("#", "").trim();
          const targetId = parseInt(clean, 10);
          if (!isNaN(targetId)) {
            if (e.type === "pass" && e.from_id !== targetId && e.to_id !== targetId)
              continue;
            if (e.type === "sprint" && e.track_id !== targetId) continue;
            if (e.type === "turnover" && e.by_id !== targetId) continue;
          }
        }

        items.push({ kind: "spatial", time_s: e.time_s, event: e });
      }
    }

    if (
      (typeFilter === "all" || typeFilter === "broadcast") &&
      teamFilter === "all" &&
      !searchPlayer.trim()
    ) {
      for (const bev of broadcastEvents) {
        items.push({ kind: "broadcast", time_s: bev.time_s, event: bev });
      }
    }

    items.sort((a, b) => a.time_s - b.time_s);
    return items;
  }, [events, broadcastEvents, typeFilter, teamFilter, searchPlayer]);

  const activeEventIndex = useMemo(() => {
    let closestIdx = -1;
    let minDiff = 0.5;
    for (let i = 0; i < unifiedFeed.length; i++) {
      const diff = Math.abs(currentTime - unifiedFeed[i].time_s);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }
    return closestIdx;
  }, [unifiedFeed, currentTime]);

  return (
    <div className="rounded-2xl obsidian-card p-5 flex flex-col h-[460px]">
      {/* ── Feed Header & Controls (100% Localized, Fixed Event Count) ── */}
      <div className="mb-3.5 pb-3.5 border-b border-white/[0.08] flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-tv-amber/10 border border-tv-amber/25 text-tv-amber">
              <Zap className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm sm:text-base font-bold text-white">
                {t("evt.feed_title")}
              </h3>
              <p className="text-xs text-slate-400">
                {formatEventCount(unifiedFeed.length, lang)}
              </p>
            </div>
          </div>

          {/* Player #ID Filter Input */}
          <div className="relative w-44 sm:w-52">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute top-1/2 -translate-y-1/2 start-3 pointer-events-none" />
            <input
              type="text"
              value={searchPlayer}
              onChange={(e) => setSearchPlayer(e.target.value)}
              placeholder={t("evt.search_placeholder_full")}
              className="w-full ps-8 pe-7 py-1.5 text-xs rounded-xl obsidian-well text-white placeholder-slate-500 focus:outline-none focus:border-tv-cyan/50"
            />
            {searchPlayer && (
              <button
                type="button"
                onClick={() => setSearchPlayer("")}
                className="absolute top-1/2 -translate-y-1/2 end-2.5 text-slate-400 hover:text-white cursor-pointer"
                aria-label="Clear filter"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Type & Team Filter Strips */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="inline-flex flex-wrap items-center p-0.5 rounded-xl obsidian-well text-xs">
            {(
              [
                { id: "all", lbl: t("evt.all") },
                ...(broadcastEvents.length > 0
                  ? [{ id: "broadcast" as const, lbl: t("evt.scope_broadcast") }]
                  : []),
                { id: "pass", lbl: t("evt.pass") },
                { id: "turnover", lbl: t("evt.turnover") },
                { id: "sprint", lbl: t("evt.sprint") },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setTypeFilter(tab.id)}
                className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                  typeFilter === tab.id
                    ? "bg-white/15 text-white font-bold"
                    : "text-slate-400 hover:text-white font-medium"
                }`}
              >
                {tab.lbl}
              </button>
            ))}
          </div>

          <div className="inline-flex items-center p-0.5 rounded-xl obsidian-well text-xs">
            <button
              type="button"
              onClick={() => setTeamFilter("all")}
              className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                teamFilter === "all"
                  ? "bg-white/15 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {t("evt.all_teams")}
            </button>
            <button
              type="button"
              onClick={() => setTeamFilter("0")}
              className={`px-2.5 py-1 rounded-lg text-tv-cyan transition-colors cursor-pointer ${
                teamFilter === "0" ? "bg-tv-cyan/20 font-bold" : "opacity-80 hover:opacity-100"
              }`}
            >
              {t("team_a")}
            </button>
            <button
              type="button"
              onClick={() => setTeamFilter("1")}
              className={`px-2.5 py-1 rounded-lg text-tv-coral transition-colors cursor-pointer ${
                teamFilter === "1" ? "bg-tv-coral/20 font-bold" : "opacity-80 hover:opacity-100"
              }`}
            >
              {t("team_b")}
            </button>
          </div>
        </div>
      </div>

      {/* ── Scrollable Events Stream ── */}
      <div className="flex-1 overflow-y-auto space-y-2 pe-1">
        {unifiedFeed.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500 text-xs">
            {t("no_data")}
          </div>
        ) : (
          unifiedFeed.map((item, idx) => {
            const isActive = idx === activeEventIndex;

            if (item.kind === "broadcast") {
              const bev = item.event;
              const meta = getCategoryMeta(bev.category);
              return (
                <div
                  key={`bcast-${bev.class_id}-${bev.frame_idx}-${idx}`}
                  onClick={() => {
                    if (onSeekToTime) onSeekToTime(bev.time_s);
                    else onSeekToFrame(Math.round(bev.time_s * 25));
                  }}
                  className={`p-2.5 rounded-xl border cursor-pointer transition-all flex items-center justify-between gap-2 text-xs ${
                    isActive
                      ? "bg-violet-500/15 border-violet-400/50 shadow-sm"
                      : "obsidian-well hover:border-white/20"
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span
                      className="w-2 h-2 rotate-45 rounded-[1px] shrink-0"
                      style={{ backgroundColor: meta.color }}
                    />
                    <BidiNum className="text-slate-400 text-xs">
                      {bev.time_s.toFixed(1)}s
                    </BidiNum>
                    <span
                      className={`px-2 py-0.5 rounded-md text-[11px] font-bold border ${meta.bgClass} ${meta.borderClass} ${meta.textClass}`}
                    >
                      {getActionClassLabel(bev.label, t)}
                    </span>
                    <span className="text-[11px] text-slate-400 truncate">
                      {getActionCategoryLabel(bev.category, t)}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-[10px] font-semibold text-slate-400">
                      {t("evt.broadcast_badge")}
                    </span>
                    <BidiNum className="font-bold text-tv-mint">
                      {`${Math.round(bev.confidence * 100)}%`}
                    </BidiNum>
                  </div>
                </div>
              );
            }

            const evt = item.event;

            if (evt.type === "pass") {
              const isTeam0 = evt.team === 0;
              const teamColor = isTeam0 ? "text-tv-cyan" : "text-tv-coral";
              const dotBg = isTeam0 ? "bg-tv-cyan" : "bg-tv-coral";
              const isPositiveXt = evt.xt_delta >= 0;

              return (
                <div
                  key={`pass-${evt.frame}-${idx}`}
                  onClick={() => {
                    onSeekToFrame(evt.frame);
                    onSelectPlayer?.(evt.from_id);
                  }}
                  className={`p-2.5 rounded-xl border cursor-pointer transition-all flex items-center justify-between gap-2 text-xs ${
                    isActive
                      ? "bg-tv-cyan/12 border-tv-cyan/45 shadow-sm"
                      : "obsidian-well hover:border-white/20"
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className={`w-2 h-2 rounded-full ${dotBg} shrink-0`} />
                    <BidiNum className="text-slate-400 text-xs">
                      {evt.time_s.toFixed(1)}s
                    </BidiNum>
                    <span className="px-2 py-0.5 rounded-md text-[11px] font-bold bg-tv-cyan/10 text-tv-cyan border border-tv-cyan/20">
                      {t("evt.pass")}
                    </span>
                    <BidiNum className={`font-bold ${teamColor}`}>
                      #{evt.from_id} → #{evt.to_id}
                    </BidiNum>
                  </div>

                  <BidiNum className="flex items-center gap-2 text-xs shrink-0">
                    <span className="text-slate-400">{evt.distance_m}m</span>
                    <span
                      className={`font-bold ${
                        isPositiveXt ? "text-tv-mint" : "text-tv-coral"
                      }`}
                    >
                      {isPositiveXt
                        ? `+${evt.xt_delta.toFixed(4)}`
                        : evt.xt_delta.toFixed(4)}{" "}
                      xT
                    </span>
                  </BidiNum>
                </div>
              );
            }

            if (evt.type === "turnover") {
              return (
                <div
                  key={`turnover-${evt.frame}-${idx}`}
                  onClick={() => {
                    onSeekToFrame(evt.frame);
                    onSelectPlayer?.(evt.by_id);
                  }}
                  className={`p-2.5 rounded-xl border cursor-pointer transition-all flex items-center justify-between gap-2 text-xs ${
                    isActive
                      ? "bg-tv-amber/12 border-tv-amber/45 shadow-sm"
                      : "obsidian-well hover:border-white/20"
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="w-2 h-2 rounded-full bg-tv-amber shrink-0" />
                    <BidiNum className="text-slate-400 text-xs">
                      {evt.time_s.toFixed(1)}s
                    </BidiNum>
                    <span className="px-2 py-0.5 rounded-md text-[11px] font-bold bg-tv-amber/10 text-tv-amber border border-tv-amber/20">
                      {t("evt.turnover")}
                    </span>
                    <span className="font-semibold text-white truncate">
                      {t("evt.recovery_by")}
                      <BidiNum className="font-bold">{evt.by_id}</BidiNum>
                    </span>
                  </div>

                  <div className="inline-flex items-center gap-1.5 text-xs font-semibold shrink-0">
                    <span
                      className={
                        evt.from_team === 0 ? "text-tv-cyan" : "text-tv-coral"
                      }
                    >
                      {formatTeamName(evt.from_team, lang)}
                    </span>
                    <DirArrow className="w-3 h-3 text-slate-500" />
                    <span
                      className={
                        evt.to_team === 0 ? "text-tv-cyan" : "text-tv-coral"
                      }
                    >
                      {formatTeamName(evt.to_team, lang)}
                    </span>
                  </div>
                </div>
              );
            }

            if (evt.type === "sprint") {
              return (
                <div
                  key={`sprint-${evt.frame}-${idx}`}
                  onClick={() => {
                    onSeekToFrame(evt.frame);
                    onSelectPlayer?.(evt.track_id);
                  }}
                  className={`p-2.5 rounded-xl border cursor-pointer transition-all flex items-center justify-between gap-2 text-xs ${
                    isActive
                      ? "bg-tv-mint/12 border-tv-mint/45 shadow-sm"
                      : "obsidian-well hover:border-white/20"
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="w-2 h-2 rounded-full bg-tv-mint shrink-0" />
                    <BidiNum className="text-slate-400 text-xs">
                      {evt.time_s.toFixed(1)}s
                    </BidiNum>
                    <span className="px-2 py-0.5 rounded-md text-[11px] font-bold bg-tv-mint/10 text-tv-mint border border-tv-mint/20">
                      {t("evt.sprint")}
                    </span>
                    <span className="font-semibold text-white truncate">
                      {t("evt.player_num")}
                      <BidiNum className="font-bold">{evt.track_id}</BidiNum>
                    </span>
                  </div>

                  <div className="inline-flex items-baseline gap-1 text-tv-mint font-bold shrink-0">
                    <BidiNum>{evt.speed_kmh.toFixed(1)}</BidiNum>
                    <span className="text-[11px]">{t("km_h")}</span>
                  </div>
                </div>
              );
            }

            return null;
          })
        )}
      </div>
    </div>
  );
}
