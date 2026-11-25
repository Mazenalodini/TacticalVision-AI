"use client";

import React, { useMemo } from "react";
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
import { Shield } from "lucide-react";
import { useLang } from "@/context/LangContext";
import { BidiNum } from "@/components/ui/BidiNum";
import type { Telemetry } from "@/lib/types";

interface DefLineChartProps {
  telemetry: Telemetry | null;
  onSeekToFrame?: (frame: number) => void;
}

interface DefLinePoint {
  time: number;
  frame: number;
  def0: number | null;
  def1: number | null;
  compression: number | null;
}

export function DefLineChart({ telemetry, onSeekToFrame }: DefLineChartProps) {
  const { t, dir } = useLang();

  const { chartData, avg0, avg1, minCompression } = useMemo(() => {
    if (!telemetry?.frames) {
      return { chartData: [], avg0: 0, avg1: 0, minCompression: 0 };
    }

    const frames = telemetry.frames;
    const step = frames.length > 500 ? 2 : 1;
    const points: DefLinePoint[] = [];

    let sum0 = 0;
    let count0 = 0;
    let sum1 = 0;
    let count1 = 0;
    let minComp = 105;

    let last0: number | null = null;
    let last1: number | null = null;

    for (let i = 0; i < frames.length; i += step) {
      const fr = frames[i];
      const def = fr.def_x || {};

      const d0: number | null =
        def["0"] !== undefined ? Number(def["0"].toFixed(1)) : last0;
      const d1: number | null =
        def["1"] !== undefined ? Number(def["1"].toFixed(1)) : last1;

      if (def["0"] !== undefined) {
        last0 = d0;
        sum0 += d0!;
        count0++;
      }
      if (def["1"] !== undefined) {
        last1 = d1;
        sum1 += d1!;
        count1++;
      }

      const comp =
        d0 !== null && d1 !== null
          ? Number(Math.abs(d1 - d0).toFixed(1))
          : null;
      if (comp !== null && comp < minComp) {
        minComp = comp;
      }

      points.push({
        time: Number(fr.t.toFixed(1)),
        frame: fr.f,
        def0: d0,
        def1: d1,
        compression: comp,
      });
    }

    return {
      chartData: points,
      avg0: count0 > 0 ? Number((sum0 / count0).toFixed(1)) : 0,
      avg1: count1 > 0 ? Number((sum1 / count1).toFixed(1)) : 0,
      minCompression: minComp < 105 ? minComp : 0,
    };
  }, [telemetry]);

  if (!telemetry || chartData.length === 0) {
    return (
      <div className="p-8 text-center text-slate-500 text-xs obsidian-card rounded-2xl">
        {t("no_data")}
      </div>
    );
  }

  return (
    <div className="rounded-2xl obsidian-card p-5 space-y-6">
      {/* ── Top Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-white/[0.08]">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Shield className="w-4 h-4 text-tv-cyan shrink-0" />
            <span>{t("tel.defline_title")}</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            {t("tel.defline_desc")}
          </p>
        </div>

        {/* Localized Legend Pills */}
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-1 rounded-full bg-tv-cyan shrink-0" />
            <span className="text-slate-200 font-semibold">
              {t("team_a")} · {t("tel.def_line_legend")}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-1 rounded-full bg-tv-coral shrink-0" />
            <span className="text-slate-200 font-semibold">
              {t("team_b")} · {t("tel.def_line_legend")}
            </span>
          </div>
        </div>
      </div>

      {/* ── Tactical Block Insight Cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Team A Line Height */}
        <div className="p-3.5 rounded-xl bg-tv-cyan/[0.05] border border-tv-cyan/20">
          <span className="text-[11px] font-semibold text-slate-400 block mb-0.5">
            {t("team_a")} · {t("tel.avg_line_height")}
          </span>
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="inline-flex items-baseline gap-1 text-xl font-black text-tv-cyan">
              <BidiNum>{avg0}</BidiNum>
              <span className="text-xs font-semibold">{t("meters")}</span>
            </span>
            <span className="text-xs text-slate-400">
              {avg0 < 35 ? t("tel.low_mid_block") : t("tel.high_press_line")}
            </span>
          </div>
        </div>

        {/* Team B Line Height */}
        <div className="p-3.5 rounded-xl bg-tv-coral/[0.05] border border-tv-coral/20">
          <span className="text-[11px] font-semibold text-slate-400 block mb-0.5">
            {t("team_b")} · {t("tel.avg_line_height")}
          </span>
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="inline-flex items-baseline gap-1 text-xl font-black text-tv-coral">
              <BidiNum>{avg1}</BidiNum>
              <span className="text-xs font-semibold">{t("meters")}</span>
            </span>
            <span className="text-xs text-slate-400">
              {avg1 > 70 ? t("tel.low_block_def") : t("tel.agg_mid_push")}
            </span>
          </div>
        </div>

        {/* Tightest Inter-Line Compression */}
        <div className="p-3.5 rounded-xl obsidian-well border border-white/[0.06]">
          <span className="text-[11px] font-semibold text-slate-400 block mb-0.5">
            {t("tel.peak_compression")}
          </span>
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="inline-flex items-baseline gap-1 text-xl font-black text-tv-mint">
              <BidiNum>{minCompression}</BidiNum>
              <span className="text-xs font-semibold">{t("meters")}</span>
            </span>
            <span className="text-xs text-slate-400">
              {t("tel.tightest_space")}
            </span>
          </div>
        </div>
      </div>

      {/* ── Recharts Line Chart Container (Strict LTR Geometric Frame) ── */}
      <div className="rounded-xl obsidian-well p-3 border border-white/[0.06]">
        <div dir="ltr" className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 12, right: 20, left: -4, bottom: 0 }}
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
                tickFormatter={(v) => `${v}${t("meters")}`}
                domain={[0, 105]}
                ticks={[0, 35, 52.5, 70, 105]}
              />

              {/* Tactical Pitch Thirds Reference Lines */}
              <ReferenceLine
                y={35}
                stroke="#94A3B8"
                strokeDasharray="3 3"
                strokeOpacity={0.35}
                label={{
                  value: t("tel.def_third"),
                  fill: "#94A3B8",
                  fontSize: 10,
                  position: "insideTopLeft",
                }}
              />
              <ReferenceLine
                y={52.5}
                stroke="#34D399"
                strokeDasharray="2 2"
                strokeOpacity={0.45}
                label={{
                  value: t("tel.halfway"),
                  fill: "#34D399",
                  fontSize: 10,
                  position: "insideTopLeft",
                }}
              />
              <ReferenceLine
                y={70}
                stroke="#94A3B8"
                strokeDasharray="3 3"
                strokeOpacity={0.35}
                label={{
                  value: t("tel.att_third"),
                  fill: "#94A3B8",
                  fontSize: 10,
                  position: "insideTopLeft",
                }}
              />

              {/* Localized Tooltip */}
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    const d0 = payload.find((p) => p.dataKey === "def0")
                      ?.value as number | null;
                    const d1 = payload.find((p) => p.dataKey === "def1")
                      ?.value as number | null;
                    const comp = payload[0]?.payload?.compression;
                    const frame = payload[0]?.payload?.frame;

                    return (
                      <div
                        dir={dir}
                        className="rounded-xl obsidian-popover p-3 shadow-xl text-xs space-y-2 min-w-[210px]"
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

                        <div className="flex items-center justify-between gap-3">
                          <div className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-tv-cyan shrink-0" />
                            <span className="text-slate-200 font-semibold">
                              {t("team_a")} · {t("tel.def_line_legend")}
                            </span>
                          </div>
                          {d0 !== null && d0 !== undefined ? (
                            <span className="inline-flex items-baseline gap-1 font-bold text-tv-cyan">
                              <BidiNum>{d0}</BidiNum>
                              <span className="text-[11px]">{t("meters")}</span>
                            </span>
                          ) : (
                            <span className="text-slate-500">--</span>
                          )}
                        </div>

                        <div className="flex items-center justify-between gap-3">
                          <div className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-tv-coral shrink-0" />
                            <span className="text-slate-200 font-semibold">
                              {t("team_b")} · {t("tel.def_line_legend")}
                            </span>
                          </div>
                          {d1 !== null && d1 !== undefined ? (
                            <span className="inline-flex items-baseline gap-1 font-bold text-tv-coral">
                              <BidiNum>{d1}</BidiNum>
                              <span className="text-[11px]">{t("meters")}</span>
                            </span>
                          ) : (
                            <span className="text-slate-500">--</span>
                          )}
                        </div>

                        {comp !== null && comp !== undefined && (
                          <div className="pt-1.5 border-t border-white/[0.06] flex items-center justify-between gap-3 text-[11px]">
                            <span className="text-slate-400">
                              {t("tel.inter_line_gap")}:
                            </span>
                            <span className="inline-flex items-baseline gap-1 font-bold text-tv-mint">
                              <BidiNum>{comp}</BidiNum>
                              <span>{t("meters")}</span>
                            </span>
                          </div>
                        )}
                      </div>
                    );
                  }
                  return null;
                }}
              />

              <Line
                type="monotone"
                dataKey="def0"
                name={`${t("team_a")} · ${t("tel.def_line_legend")}`}
                stroke="#38BDF8"
                strokeWidth={2.4}
                dot={false}
                activeDot={{
                  r: 5,
                  fill: "#38BDF8",
                  stroke: "#060912",
                  strokeWidth: 2,
                }}
                connectNulls={true}
              />

              <Line
                type="monotone"
                dataKey="def1"
                name={`${t("team_b")} · ${t("tel.def_line_legend")}`}
                stroke="#FB7185"
                strokeWidth={2.4}
                dot={false}
                activeDot={{
                  r: 5,
                  fill: "#FB7185",
                  stroke: "#060912",
                  strokeWidth: 2,
                }}
                connectNulls={true}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
