"use client";

import React, { useMemo, useState } from "react";
import {
  Users,
  Search,
  ArrowUp,
  ArrowDown,
  FileSpreadsheet,
  FileJson,
  Printer,
  Crosshair,
  BadgeCheck,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { BidiNum } from "@/components/ui/BidiNum";
import { resolvePlayerIdentity, usePlayerOverrides } from "@/lib/jerseyOverrides";
import { getExportUrl } from "@/lib/api";
import { formatPlayerCount, formatRole } from "@/lib/format";
import type { Telemetry } from "@/lib/types";

interface PlayerLeaderboardProps {
  telemetry: Telemetry | null;
  matchId: string;
  onSelectPlayer?: (playerId: number) => void;
}

type SortField =
  | "track_id"
  | "team"
  | "role"
  | "distance_m"
  | "max_speed_kmh"
  | "avg_speed_kmh"
  | "sprints"
  | "passes_completed"
  | "xt_generated"
  | "frames_visible";

export function PlayerLeaderboard({
  telemetry,
  matchId,
  onSelectPlayer,
}: PlayerLeaderboardProps) {
  const { t, lang } = useLang();
  const { overrides } = usePlayerOverrides(matchId);

  const [searchQuery, setSearchQuery] = useState<string>("");
  const [teamFilter, setTeamFilter] = useState<"all" | "0" | "1">("all");
  const [roleFilter, setRoleFilter] = useState<
    "all" | "player" | "goalkeeper"
  >("all");

  const [sortField, setSortField] = useState<SortField>("max_speed_kmh");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");

  const totalFrames = Math.max(1, telemetry?.summary?.total_frames || 750);

  const displayedPlayers = useMemo(() => {
    if (!telemetry?.players) return [];

    let list = [...telemetry.players];

    if (teamFilter !== "all") {
      const tNum = Number(teamFilter);
      list = list.filter((p) => p.team === tNum);
    }

    if (roleFilter !== "all") {
      list = list.filter((p) => p.role === roleFilter);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.trim().replace(/^#/, "").toLowerCase();
      list = list.filter((p) => {
        const id = resolvePlayerIdentity(p.track_id, p, overrides);
        return (
          p.track_id.toString().includes(q) ||
          (id.jerseyNumber != null && id.jerseyNumber.toString().includes(q)) ||
          (id.customName && id.customName.toLowerCase().includes(q)) ||
          p.role.toLowerCase().includes(q)
        );
      });
    }

    list.sort((a, b) => {
      const valA: number | string = a[sortField] ?? 0;
      const valB: number | string = b[sortField] ?? 0;

      if (typeof valA === "string") {
        return sortDirection === "asc"
          ? valA.localeCompare(String(valB))
          : String(valB).localeCompare(valA);
      }

      return sortDirection === "asc"
        ? (valA as number) - (valB as number)
        : (valB as number) - (valA as number);
    });

    return list;
  }, [telemetry, teamFilter, roleFilter, searchQuery, sortField, sortDirection, overrides]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const handlePrint = () => {
    if (typeof window !== "undefined") {
      window.print();
    }
  };

  if (!telemetry || !telemetry.players) {
    return (
      <div className="p-8 text-center text-slate-500 text-xs obsidian-card rounded-2xl">
        {t("no_data")}
      </div>
    );
  }

  return (
    <div className="rounded-2xl obsidian-card p-5 space-y-5 print:bg-white print:text-black print:border-none print:p-0">
      {/* ── Top Header & Export Action Bar (Hidden in Print) ── */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-white/[0.08] print:hidden">
        <div>
          <h3 className="text-base font-bold text-white flex flex-wrap items-center gap-2">
            <Users className="w-4 h-4 text-tv-cyan shrink-0" />
            <span>{t("tel.leaderboard")}</span>
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold obsidian-well border border-white/[0.08] text-slate-300">
              {formatPlayerCount(displayedPlayers.length, lang)}
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            {t("tel.leaderboard_sub")}
          </p>
        </div>

        {/* Export Buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <a
            href={getExportUrl(matchId, "csv", "players")}
            download
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl obsidian-well hover:bg-white/[0.08] border border-white/[0.08] text-xs font-semibold text-slate-200 transition-colors"
            title={t("tel.export_csv")}
          >
            <FileSpreadsheet className="w-3.5 h-3.5 text-tv-mint shrink-0" />
            <span>{t("tel.export_csv")}</span>
          </a>

          <a
            href={getExportUrl(matchId, "json", "players")}
            download
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl obsidian-well hover:bg-white/[0.08] border border-white/[0.08] text-xs font-semibold text-slate-200 transition-colors"
            title={t("tel.export_json")}
          >
            <FileJson className="w-3.5 h-3.5 text-tv-cyan shrink-0" />
            <span>{t("tel.export_json")}</span>
          </a>

          <button
            type="button"
            onClick={handlePrint}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl obsidian-well hover:bg-white/[0.08] border border-white/[0.08] text-xs font-semibold text-slate-200 transition-colors cursor-pointer"
            title={t("tel.print")}
          >
            <Printer className="w-3.5 h-3.5 text-tv-coral shrink-0" />
            <span>{t("tel.print")}</span>
          </button>
        </div>
      </div>

      {/* ── Filters Toolbar (Hidden in Print) ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 print:hidden">
        {/* Search Input */}
        <div className="relative flex-1 max-w-xs">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute start-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t("tel.search_players")}
            className="w-full ps-9 pe-4 py-1.5 rounded-xl obsidian-well border border-white/[0.08] text-xs text-white placeholder-slate-500 focus:outline-none focus:border-tv-cyan"
          />
        </div>

        {/* Team & Role Filter Chips */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Team Filter */}
          <div className="flex items-center p-0.5 rounded-xl obsidian-well border border-white/[0.06] text-[11px]">
            <button
              type="button"
              onClick={() => setTeamFilter("all")}
              className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                teamFilter === "all"
                  ? "bg-white/15 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {t("tel.filter_all_teams")}
            </button>
            <button
              type="button"
              onClick={() => setTeamFilter("0")}
              className={`px-2.5 py-1 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer ${
                teamFilter === "0"
                  ? "bg-tv-cyan/20 text-tv-cyan font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-tv-cyan" />
              <span>{t("team_a")}</span>
            </button>
            <button
              type="button"
              onClick={() => setTeamFilter("1")}
              className={`px-2.5 py-1 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer ${
                teamFilter === "1"
                  ? "bg-tv-coral/20 text-tv-coral font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-tv-coral" />
              <span>{t("team_b")}</span>
            </button>
          </div>

          {/* Role Filter */}
          <div className="flex items-center p-0.5 rounded-xl obsidian-well border border-white/[0.06] text-[11px]">
            <button
              type="button"
              onClick={() => setRoleFilter("all")}
              className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                roleFilter === "all"
                  ? "bg-white/15 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {t("tel.filter_all_roles")}
            </button>
            <button
              type="button"
              onClick={() => setRoleFilter("player")}
              className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                roleFilter === "player"
                  ? "bg-white/15 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {t("tel.role_player")}
            </button>
            <button
              type="button"
              onClick={() => setRoleFilter("goalkeeper")}
              className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                roleFilter === "goalkeeper"
                  ? "bg-white/15 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {t("tel.role_gk")}
            </button>
          </div>
        </div>
      </div>

      {/* ── High-Density RTL-Safe Data Table ── */}
      <div className="overflow-x-auto rounded-xl border border-white/[0.08] print:border-black">
        <table className="w-full text-start text-xs border-collapse print:text-black">
          <thead>
            <tr className="bg-white/[0.03] border-b border-white/[0.08] text-slate-300 text-[11px] font-semibold select-none print:bg-slate-100 print:text-black">
              {/* ID */}
              <th
                onClick={() => handleSort("track_id")}
                className="py-3 px-3 cursor-pointer hover:text-white text-start"
              >
                <div className="flex items-center gap-1">
                  <span>{t("tel.col_id")}</span>
                  {sortField === "track_id" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Team */}
              <th
                onClick={() => handleSort("team")}
                className="py-3 px-3 cursor-pointer hover:text-white text-start"
              >
                <div className="flex items-center gap-1">
                  <span>{t("tel.col_team")}</span>
                  {sortField === "team" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Role */}
              <th
                onClick={() => handleSort("role")}
                className="py-3 px-3 cursor-pointer hover:text-white text-start"
              >
                <div className="flex items-center gap-1">
                  <span>{t("tel.col_role")}</span>
                  {sortField === "role" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Distance */}
              <th
                onClick={() => handleSort("distance_m")}
                className="py-3 px-3 cursor-pointer hover:text-white text-end"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>{t("tel.col_dist")}</span>
                  {sortField === "distance_m" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Max Speed */}
              <th
                onClick={() => handleSort("max_speed_kmh")}
                className="py-3 px-3 cursor-pointer hover:text-white text-end"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>{t("tel.col_top_speed")}</span>
                  {sortField === "max_speed_kmh" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Avg Speed */}
              <th
                onClick={() => handleSort("avg_speed_kmh")}
                className="py-3 px-3 cursor-pointer hover:text-white text-end"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>{t("tel.col_avg_speed")}</span>
                  {sortField === "avg_speed_kmh" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Sprints */}
              <th
                onClick={() => handleSort("sprints")}
                className="py-3 px-3 cursor-pointer hover:text-white text-end"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>{t("tel.col_sprints")}</span>
                  {sortField === "sprints" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Passes */}
              <th
                onClick={() => handleSort("passes_completed")}
                className="py-3 px-3 cursor-pointer hover:text-white text-end"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>{t("tel.col_passes")}</span>
                  {sortField === "passes_completed" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* xT */}
              <th
                onClick={() => handleSort("xt_generated")}
                className="py-3 px-3 cursor-pointer hover:text-white text-end"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>{t("tel.col_xt")}</span>
                  {sortField === "xt_generated" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Visibility % */}
              <th
                onClick={() => handleSort("frames_visible")}
                className="py-3 px-3 cursor-pointer hover:text-white text-end"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>{t("tel.col_vis")}</span>
                  {sortField === "frames_visible" &&
                    (sortDirection === "asc" ? (
                      <ArrowUp className="w-3 h-3 text-tv-cyan" />
                    ) : (
                      <ArrowDown className="w-3 h-3 text-tv-cyan" />
                    ))}
                </div>
              </th>

              {/* Action */}
              <th className="py-3 px-3 text-center print:hidden">
                <span>{t("tel.col_action")}</span>
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-white/[0.06] print:divide-slate-300">
            {displayedPlayers.length === 0 ? (
              <tr>
                <td
                  colSpan={11}
                  className="py-8 text-center text-slate-500 text-xs"
                >
                  {t("no_data")}
                </td>
              </tr>
            ) : (
              displayedPlayers.map((p) => {
                const isCyan = p.team === 0;
                const isGK = p.role === "goalkeeper";
                const isHighSpeed = p.max_speed_kmh >= 24.0;
                const visPct = Math.min(
                  100,
                  Math.round((p.frames_visible / totalFrames) * 100)
                );

                return (
                  <tr
                    key={`player-row-${p.track_id}`}
                    className="hover:bg-white/[0.03] transition-colors group print:hover:bg-transparent"
                  >
                    {/* ID: Honest `#jn ✓` (verified/override) vs muted `T{id}` fallback */}
                    <td className="py-2.5 px-3 font-bold text-white print:text-black text-start">
                      {(() => {
                        const identity = resolvePlayerIdentity(p.track_id, p, overrides);
                        return (
                          <span
                            className="inline-flex items-center gap-1.5"
                            title={
                              identity.isVerified
                                ? `${
                                    identity.isCoachOverride
                                      ? t("jersey.coach_override")
                                      : t("jersey.verified")
                                  } · ${t("jersey.track_fallback")} #${p.track_id}${
                                    p.jersey_confidence != null
                                      ? ` · ${Math.round(p.jersey_confidence * 100)}%`
                                      : ""
                                  }`
                                : t("jersey.unverified_tip")
                            }
                          >
                            <span
                              className={`w-2 h-2 rounded-full shrink-0 ${
                                isCyan ? "bg-tv-cyan" : "bg-tv-coral"
                              }`}
                            />
                            {identity.isVerified ? (
                              <span className="inline-flex items-center gap-1 flex-wrap">
                                <BidiNum className="text-white font-extrabold">
                                  {identity.badgeText}
                                </BidiNum>
                                <BadgeCheck
                                  className={`w-3.5 h-3.5 shrink-0 ${
                                    identity.isCoachOverride ? "text-tv-amber" : "text-tv-mint"
                                  }`}
                                />
                                {identity.customName && (
                                  <span className="text-[11px] font-semibold text-slate-200">
                                    {identity.customName}
                                  </span>
                                )}
                                <BidiNum className="text-[10px] text-slate-500 font-normal">
                                  {`#${p.track_id}`}
                                </BidiNum>
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-slate-400 font-semibold">
                                <span>{t("jersey.track_short")}</span>{" "}
                                <BidiNum>{`#${p.track_id}`}</BidiNum>
                                {identity.customName && (
                                  <span className="text-[11px] text-slate-200 font-semibold">
                                    {identity.customName}
                                  </span>
                                )}
                              </span>
                            )}
                          </span>
                        );
                      })()}
                    </td>

                    {/* Team */}
                    <td className="py-2.5 px-3 text-start">
                      <span
                        className={`inline-block px-2 py-0.5 rounded-md text-[10px] font-bold ${
                          isCyan
                            ? "bg-tv-cyan/15 text-tv-cyan border border-tv-cyan/30"
                            : "bg-tv-coral/15 text-tv-coral border border-tv-coral/30"
                        }`}
                      >
                        {isCyan ? t("team_a") : t("team_b")}
                      </span>
                    </td>

                    {/* Role */}
                    <td className="py-2.5 px-3 text-slate-300 print:text-black text-start">
                      <span
                        className={
                          isGK
                            ? "text-tv-mint font-semibold"
                            : "text-slate-400"
                        }
                      >
                        {formatRole(p.role, lang, true)}
                      </span>
                    </td>

                    {/* Distance */}
                    <td className="py-2.5 px-3 text-end text-slate-200 print:text-black">
                      <BidiNum>{p.distance_m.toFixed(1)}</BidiNum>
                      <span className="text-[10px] text-slate-500 ms-1">
                        {t("meters")}
                      </span>
                    </td>

                    {/* Max Speed */}
                    <td className="py-2.5 px-3 text-end">
                      <BidiNum
                        className={`font-bold ${
                          isHighSpeed
                            ? "text-tv-mint"
                            : "text-white print:text-black"
                        }`}
                      >
                        {p.max_speed_kmh.toFixed(1)}
                      </BidiNum>
                      <span className="text-[10px] text-slate-500 ms-1">
                        {t("km_h")}
                      </span>
                    </td>

                    {/* Avg Speed */}
                    <td className="py-2.5 px-3 text-end text-slate-400 print:text-black">
                      <BidiNum>{p.avg_speed_kmh.toFixed(1)}</BidiNum>
                    </td>

                    {/* Sprints */}
                    <td className="py-2.5 px-3 text-end">
                      {p.sprints > 0 ? (
                        <BidiNum className="text-tv-mint font-bold px-1.5 py-0.5 rounded bg-tv-mint/10">
                          {p.sprints}
                        </BidiNum>
                      ) : (
                        <BidiNum className="text-slate-600">0</BidiNum>
                      )}
                    </td>

                    {/* Passes */}
                    <td className="py-2.5 px-3 text-end text-slate-300 print:text-black">
                      <BidiNum>{p.passes_completed}</BidiNum>
                    </td>

                    {/* xT */}
                    <td className="py-2.5 px-3 text-end">
                      {p.xt_generated > 0 ? (
                        <BidiNum className="text-tv-cyan font-semibold">
                          {`+${p.xt_generated.toFixed(3)}`}
                        </BidiNum>
                      ) : (
                        <BidiNum className="text-slate-600">0.000</BidiNum>
                      )}
                    </td>

                    {/* Visibility % */}
                    <td className="py-2.5 px-3 text-end text-slate-400 print:text-black">
                      <BidiNum>{`${visPct}%`}</BidiNum>
                    </td>

                    {/* Spotlight Action */}
                    <td className="py-2.5 px-3 text-center print:hidden">
                      {onSelectPlayer && (
                        <button
                          type="button"
                          onClick={() => onSelectPlayer(p.track_id)}
                          className="p-1.5 rounded-lg obsidian-well hover:bg-tv-cyan/20 hover:text-tv-cyan text-slate-400 transition-colors cursor-pointer"
                          title={t("tel.jump_to_warroom")}
                        >
                          <Crosshair className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
