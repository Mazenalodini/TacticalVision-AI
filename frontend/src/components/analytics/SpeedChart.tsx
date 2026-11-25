"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
} from "recharts";
import { Activity, Sparkles } from "lucide-react";
import { useLang } from "@/context/LangContext";
import { BidiNum } from "@/components/ui/BidiNum";
import { D3PlayerRadar } from "@/components/warroom/D3PlayerRadar";
import {
  formatTeamName,
  formatRole,
} from "@/lib/format";
import type { Telemetry, PlayerStat } from "@/lib/types";

interface SpeedChartProps {
  telemetry: Telemetry | null;
  onSeekToFrame?: (frame: number) => void;
}

interface SpeedPoint {
  time: number;
  frame: number;
  speedA: number | null;
  speedB: number | null;
}

export function SpeedChart({ telemetry, onSeekToFrame }: SpeedChartProps) {
  const { t, lang, dir } = useLang();

  // All detected players sorted by peak speed descending
  const players: PlayerStat[] = useMemo(() => {
    if (!telemetry?.players) return [];
    return [...telemetry.players].sort(
      (a, b) => (b.max_speed_kmh || 0) - (a.max_speed_kmh || 0)
    );
  }, [telemetry]);

  const defaultPlayerA = useMemo(() => {
    return (
      players.find((p) => p.team === 0)?.track_id ??
      players[0]?.track_id ??
      null
    );
  }, [players]);

  const defaultPlayerB = useMemo(() => {
    return (
      players.find((p) => p.team === 1)?.track_id ??
      players[1]?.track_id ??
      players[0]?.track_id ??
      null
    );
  }, [players]);

  const [playerAId, setPlayerAId] = useState<number | null>(defaultPlayerA);
  const [playerBId, setPlayerBId] = useState<number | null>(defaultPlayerB);

  // Sync defaults cleanly when match/telemetry changes
  useEffect(() => {
    setPlayerAId(defaultPlayerA);
    setPlayerBId(defaultPlayerB);
  }, [defaultPlayerA, defaultPlayerB]);

  const playerA = useMemo(
    () => players.find((p) => p.track_id === playerAId) ?? null,
    [players, playerAId]
  );
  const playerB = useMemo(
    () => players.find((p) => p.track_id === playerBId) ?? null,
    [players, playerBId]
  );

  // Time series extraction
  const chartData: SpeedPoint[] = useMemo(() => {
    if (!telemetry?.frames) return [];

    const frames = telemetry.frames;
    const step = frames.length > 500 ? 2 : 1;
    const points: SpeedPoint[] = [];

    for (let i = 0; i < frames.length; i += step) {
      const fr = frames[i];
      let spdA: number | null = null;
      let spdB: number | null = null;

      for (const p of fr.players) {
        if (playerAId !== null && p.id === playerAId) {
          spdA = Number(p.spd.toFixed(1));
        }
        if (playerBId !== null && p.id === playerBId) {
          spdB = Number(p.spd.toFixed(1));
        }
      }

      points.push({
        time: Number(fr.t.toFixed(1)),
        frame: fr.f,
        speedA: spdA,
        speedB: spdB,
      });
    }

    return points;
  }, [telemetry, playerAId, playerBId]);

  if (!telemetry || players.length === 0) {
    return (
      <div className="p-8 text-center text-slate-500 text-xs obsidian-card rounded-2xl">
        {t("no_data")}
      </div>
    );
  }

  return (
    <div className="rounded-2xl obsidian-card p-5 space-y-6">
      {/* ── Top Header & Dual Player Selectors ── */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-white/[0.08]">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Activity className="w-4 h-4 text-tv-cyan shrink-0" />
            <span>{t("tel.speed_title")}</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            {t("tel.speed_desc")}
          </p>
        </div>

        {/* Dual Selectors */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Player A Select */}
          <div className="flex items-center gap-2 obsidian-well border border-white/[0.08] rounded-xl px-3 py-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-tv-cyan shrink-0" />
            <span className="text-[11px] font-semibold text-slate-400">
              {t("tel.player_a")}:
            </span>
            <select
              value={playerAId ?? ""}
              onChange={(e) => setPlayerAId(Number(e.target.value))}
              aria-label={t("tel.player_a")}
              className="bg-transparent text-xs font-bold text-tv-cyan outline-none cursor-pointer"
            >
              {players.map((p) => (
                <option
                  key={`a-${p.track_id}`}
                  value={p.track_id}
                  className="bg-slate-950 text-white"
                >
                  {`#${p.track_id} (${formatTeamName(
                    p.team,
                    lang
                  )} · ${formatRole(p.role, lang, true)} · ${
                    p.max_speed_kmh?.toFixed(1) ?? "0.0"
                  } ${t("km_h")})`}
                </option>
              ))}
            </select>
          </div>

          {/* Player B Select */}
          <div className="flex items-center gap-2 obsidian-well border border-white/[0.08] rounded-xl px-3 py-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-tv-coral shrink-0" />
            <span className="text-[11px] font-semibold text-slate-400">
              {t("tel.player_b")}:
            </span>
            <select
              value={playerBId ?? ""}
              onChange={(e) => setPlayerBId(Number(e.target.value))}
              aria-label={t("tel.player_b")}
              className="bg-transparent text-xs font-bold text-tv-coral outline-none cursor-pointer"
            >
              {players.map((p) => (
                <option
                  key={`b-${p.track_id}`}
                  value={p.track_id}
                  className="bg-slate-950 text-white"
                >
                  {`#${p.track_id} (${formatTeamName(
                    p.team,
                    lang
                  )} · ${formatRole(p.role, lang, true)} · ${
                    p.max_speed_kmh?.toFixed(1) ?? "0.0"
                  } ${t("km_h")})`}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* ── Mini Comparison KPI Chips ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {/* Player A Peak Speed */}
        <div className="p-3 rounded-xl bg-tv-cyan/[0.05] border border-tv-cyan/20">
          <span className="text-[11px] font-semibold text-slate-400 block mb-0.5 truncate">
            {t("evt.player_num")}
            <BidiNum>{playerA?.track_id ?? "--"}</BidiNum> ·{" "}
            {t("tel.peak_speed_lbl")}
          </span>
          <div className="flex items-baseline gap-1.5">
            <BidiNum className="text-lg font-black text-tv-cyan">
              {playerA?.max_speed_kmh?.toFixed(1) ?? "--"}
            </BidiNum>
            <span className="text-xs text-slate-400">{t("km_h")}</span>
          </div>
        </div>

        {/* Player A Sprints */}
        <div className="p-3 rounded-xl bg-tv-cyan/[0.05] border border-tv-cyan/20">
          <span className="text-[11px] font-semibold text-slate-400 block mb-0.5 truncate">
            {t("evt.player_num")}
            <BidiNum>{playerA?.track_id ?? "--"}</BidiNum> ·{" "}
            {t("tel.sprint_bursts_lbl")}
          </span>
          <div className="flex items-baseline gap-1.5">
            <BidiNum className="text-lg font-black text-white">
              {playerA?.sprints ?? 0}
            </BidiNum>
            <span className="text-xs text-slate-400">
              {t("hub.kpi_sprints_short")}
            </span>
          </div>
        </div>

        {/* Player B Peak Speed */}
        <div className="p-3 rounded-xl bg-tv-coral/[0.05] border border-tv-coral/20">
          <span className="text-[11px] font-semibold text-slate-400 block mb-0.5 truncate">
            {t("evt.player_num")}
            <BidiNum>{playerB?.track_id ?? "--"}</BidiNum> ·{" "}
            {t("tel.peak_speed_lbl")}
          </span>
          <div className="flex items-baseline gap-1.5">
            <BidiNum className="text-lg font-black text-tv-coral">
              {playerB?.max_speed_kmh?.toFixed(1) ?? "--"}
            </BidiNum>
            <span className="text-xs text-slate-400">{t("km_h")}</span>
          </div>
        </div>

        {/* Player B Sprints */}
        <div className="p-3 rounded-xl bg-tv-coral/[0.05] border border-tv-coral/20">
          <span className="text-[11px] font-semibold text-slate-400 block mb-0.5 truncate">
            {t("evt.player_num")}
            <BidiNum>{playerB?.track_id ?? "--"}</BidiNum> ·{" "}
            {t("tel.sprint_bursts_lbl")}
          </span>
          <div className="flex items-baseline gap-1.5">
            <BidiNum className="text-lg font-black text-white">
              {playerB?.sprints ?? 0}
            </BidiNum>
            <span className="text-xs text-slate-400">
              {t("hub.kpi_sprints_short")}
            </span>
          </div>
        </div>
      </div>

      {/* ── Combined Velocity Curve + Head-to-Head 6-Axis D3 Radar ── */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-stretch">
        {/* Velocity LineChart (7 cols, strict LTR geometric frame) */}
        <div className="xl:col-span-7 rounded-xl obsidian-well p-3 border border-white/[0.06] flex flex-col justify-center">
          <div dir="ltr" className="h-72 sm:h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 12, right: 20, left: -8, bottom: 0 }}
                onClick={(state) => {
                  if (
                    state &&
                    state.activePayload &&
                    state.activePayload.length > 0
                  ) {
                    const targetFrame = state.activePayload[0].payload.frame;
                    if (targetFrame !== undefined && onSeekToFrame) {
                      onSeekToFrame(targetFrame);
                    }
                  }
                }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(148, 163, 184, 0.08)"
                  vertical={false}
                />

                <XAxis
                  dataKey="time"
                  stroke="#64748B"
                  tick={{
                    fill: "#94A3B8",
                    fontSize: 11,
                    fontFamily: "monospace",
                  }}
                  tickFormatter={(val) => `${val}${t("seconds")}`}
                  minTickGap={30}
                />

                <YAxis
                  stroke="#64748B"
                  tick={{
                    fill: "#94A3B8",
                    fontSize: 11,
                    fontFamily: "monospace",
                  }}
                  tickFormatter={(v) => `${v}`}
                  domain={[0, 40]}
                  ticks={[0, 10, 20, 24, 30, 40]}
                />

                {/* Sprint threshold benchmark reference line */}
                <ReferenceLine
                  y={24}
                  stroke="#34D399"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  label={{
                    value: t("tel.sprint_threshold"),
                    fill: "#34D399",
                    fontSize: 10,
                    position: "insideTopRight",
                  }}
                />

                {/* Localized Obsidian Tooltip */}
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const spdA = payload.find((p) => p.dataKey === "speedA")
                        ?.value as number | null;
                      const spdB = payload.find((p) => p.dataKey === "speedB")
                        ?.value as number | null;
                      const frame = payload[0]?.payload?.frame;

                      return (
                        <div
                          dir={dir}
                          className="rounded-xl obsidian-popover p-3 shadow-xl text-xs space-y-2 min-w-[195px]"
                        >
                          <div className="flex items-center justify-between gap-3 text-slate-400 border-b border-white/[0.06] pb-1.5 text-[11px]">
                            <span className="inline-flex items-baseline gap-1">
                              <span>{t("tel.time_lbl")}:</span>
                              <BidiNum className="text-slate-200 font-semibold">
                                {label}
                              </BidiNum>
                              <span>{t("seconds")}</span>
                            </span>
                            <span className="inline-flex items-baseline gap-1">
                              <span>{t("tel.frame_lbl")}</span>
                              <BidiNum className="text-slate-300">
                                {`#${frame}`}
                              </BidiNum>
                            </span>
                          </div>

                          {/* Player A */}
                          <div className="flex items-center justify-between gap-4">
                            <div className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-tv-cyan shrink-0" />
                              <BidiNum className="text-slate-200 font-semibold">
                                {`#${playerA?.track_id ?? "--"}`}
                              </BidiNum>
                            </div>
                            <div className="flex items-center gap-1.5 font-bold">
                              {spdA !== null && spdA !== undefined ? (
                                <span className="inline-flex items-baseline gap-1 text-tv-cyan">
                                  <BidiNum>{spdA}</BidiNum>
                                  <span className="text-[11px]">{t("km_h")}</span>
                                </span>
                              ) : (
                                <span className="text-slate-500 font-normal">
                                  {t("tel.out_of_frame")}
                                </span>
                              )}
                              {spdA !== null && spdA !== undefined && spdA >= 24 && (
                                <span className="px-1.5 py-0.5 rounded text-[9px] bg-tv-mint/20 text-tv-mint">
                                  {t("evt.sprint")}
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Player B */}
                          <div className="flex items-center justify-between gap-4">
                            <div className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-tv-coral shrink-0" />
                              <BidiNum className="text-slate-200 font-semibold">
                                {`#${playerB?.track_id ?? "--"}`}
                              </BidiNum>
                            </div>
                            <div className="flex items-center gap-1.5 font-bold">
                              {spdB !== null && spdB !== undefined ? (
                                <span className="inline-flex items-baseline gap-1 text-tv-coral">
                                  <BidiNum>{spdB}</BidiNum>
                                  <span className="text-[11px]">{t("km_h")}</span>
                                </span>
                              ) : (
                                <span className="text-slate-500 font-normal">
                                  {t("tel.out_of_frame")}
                                </span>
                              )}
                              {spdB !== null && spdB !== undefined && spdB >= 24 && (
                                <span className="px-1.5 py-0.5 rounded text-[9px] bg-tv-mint/20 text-tv-mint">
                                  {t("evt.sprint")}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />

                <Line
                  type="monotone"
                  dataKey="speedA"
                  name={`${t("evt.player_num")}${playerA?.track_id ?? ""}`}
                  stroke="#38BDF8"
                  strokeWidth={2.4}
                  dot={false}
                  activeDot={{
                    r: 5,
                    fill: "#38BDF8",
                    stroke: "#060912",
                    strokeWidth: 2,
                  }}
                  connectNulls={false}
                />

                <Line
                  type="monotone"
                  dataKey="speedB"
                  name={`${t("evt.player_num")}${playerB?.track_id ?? ""}`}
                  stroke="#FB7185"
                  strokeWidth={2.4}
                  dot={false}
                  activeDot={{
                    r: 5,
                    fill: "#FB7185",
                    stroke: "#060912",
                    strokeWidth: 2,
                  }}
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Head-to-Head 6-Axis D3 Radar Card (5 cols) */}
        <div className="xl:col-span-5 flex flex-col justify-between gap-3">
          <div className="px-1">
            <h4 className="text-sm font-bold text-white flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-tv-mint shrink-0" />
              <span>{t("tel.d3_h2h_radar_title")}</span>
            </h4>
            <p className="text-[11px] text-slate-400 mt-0.5">
              {t("tel.d3_h2h_radar_desc")}
            </p>
          </div>

          {playerA && (
            <D3PlayerRadar
              player={playerA}
              telemetry={telemetry}
              comparePlayer={
                playerB && playerB.track_id !== playerA.track_id
                  ? playerB
                  : null
              }
              allPlayers={players}
            />
          )}
        </div>
      </div>
    </div>
  );
}
