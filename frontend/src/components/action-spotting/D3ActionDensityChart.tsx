"use client";

import React, { useMemo } from "react";
import * as d3 from "d3";
import { Flame } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { ActionSpottingDensityBin } from "@/lib/types";
import { BidiNum } from "@/components/ui/BidiNum";

interface D3ActionDensityChartProps {
  bins: ActionSpottingDensityBin[];
}

export function D3ActionDensityChart({ bins }: D3ActionDensityChartProps) {
  const { t } = useLang();

  const vbW = 540;
  const vbH = 220;
  const padL = 34;
  const padR = 14;
  const padT = 18;
  const padB = 34;

  const maxCount = useMemo(
    () => Math.max(4, d3.max(bins, (b) => b.count) ?? 4),
    [bins],
  );

  const xBand = useMemo(
    () =>
      d3
        .scaleBand<number>()
        .domain(bins.map((_, i) => i))
        .range([padL, vbW - padR])
        .padding(0.28),
    [bins],
  );

  const yScale = useMemo(
    () => d3.scaleLinear().domain([0, maxCount]).nice().range([vbH - padB, padT]),
    [maxCount],
  );

  const wavePath = useMemo(() => {
    if (bins.length < 2) return "";
    const area = d3
      .area<ActionSpottingDensityBin>()
      .x((_, i) => (xBand(i) ?? padL) + xBand.bandwidth() / 2)
      .y0(vbH - padB)
      .y1((b) => yScale(b.count))
      .curve(d3.curveMonotoneX);
    return area(bins) ?? "";
  }, [bins, xBand, yScale]);

  const yTicks = useMemo(() => yScale.ticks(4), [yScale]);

  return (
    <div className="rounded-2xl obsidian-card p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
            <Flame className="w-4 h-4 text-tv-coral shrink-0" />
            <span>{t("as.density_title")}</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">{t("as.density_desc")}</p>
        </div>

        <div className="flex items-center gap-3 text-[11px] font-semibold">
          <span className="inline-flex items-center gap-1.5 text-tv-cyan">
            <span className="w-2.5 h-2.5 rounded-sm bg-tv-cyan/65" />
            <span>{t("as.density_total")}</span>
          </span>
          <span className="inline-flex items-center gap-1.5 text-tv-mint">
            <span className="w-2.5 h-2.5 rounded-sm bg-tv-mint" />
            <span>{t("as.density_high")}</span>
          </span>
        </div>
      </div>

      {bins.length === 0 ? (
        <div className="h-44 flex items-center justify-center text-xs text-slate-500">
          {t("no_data")}
        </div>
      ) : (
        <div dir="ltr" className="w-full overflow-x-auto">
          <svg viewBox={`0 0 ${vbW} ${vbH}`} className="w-full min-w-[380px] h-auto block">
            <defs>
              <linearGradient id="as-density-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.24" />
                <stop offset="100%" stopColor="#38BDF8" stopOpacity="0.01" />
              </linearGradient>
            </defs>

            {/* Horizontal reference ticks */}
            {yTicks.map((tk) => {
              const y = yScale(tk);
              return (
                <g key={`yt-${tk}`}>
                  <line
                    x1={padL}
                    y1={y}
                    x2={vbW - padR}
                    y2={y}
                    stroke="rgba(148,163,184,0.12)"
                    strokeDasharray="3 3"
                  />
                  <text
                    x={padL - 6}
                    y={y + 3}
                    textAnchor="end"
                    fill="rgba(148,163,184,0.65)"
                    fontSize={10}
                    fontFamily="var(--font-jetbrains), monospace"
                  >
                    {tk}
                  </text>
                </g>
              );
            })}

            {/* Smooth background momentum curve */}
            {wavePath && <path d={wavePath} fill="url(#as-density-grad)" />}

            {/* Total + High-impact nested bars */}
            {bins.map((bin, idx) => {
              const bx = xBand(idx) ?? padL;
              const bw = xBand.bandwidth();
              const topY = yScale(bin.count);
              const barH = Math.max(0, vbH - padB - topY);

              const highY = yScale(bin.high_impact_count);
              const highH = Math.max(0, vbH - padB - highY);

              return (
                <g key={`bin-${bin.bin_index}-${bin.label}`}>
                  <rect
                    x={bx}
                    y={topY}
                    width={bw}
                    height={barH}
                    rx={4}
                    fill="#38BDF8"
                    fillOpacity={0.35}
                    stroke="#38BDF8"
                    strokeOpacity={0.7}
                    strokeWidth={1}
                  />
                  {bin.high_impact_count > 0 && (
                    <rect
                      x={bx + Math.max(2, bw * 0.16)}
                      y={highY}
                      width={Math.max(4, bw * 0.68)}
                      height={highH}
                      rx={3}
                      fill="#34D399"
                      fillOpacity={0.92}
                    />
                  )}
                  {bin.count > 0 && (
                    <text
                      x={bx + bw / 2}
                      y={topY - 4}
                      textAnchor="middle"
                      fill="#E2E8F0"
                      fontSize={9.5}
                      fontWeight={700}
                      fontFamily="var(--font-jetbrains), monospace"
                    >
                      {bin.count}
                    </text>
                  )}
                  <text
                    x={bx + bw / 2}
                    y={vbH - 12}
                    textAnchor="middle"
                    fill="rgba(148,163,184,0.8)"
                    fontSize={9.5}
                    fontFamily="var(--font-jetbrains), monospace"
                  >
                    {bin.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}

      {/* Compact peak bin highlight strip */}
      {bins.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-white/[0.06] text-xs text-slate-400">
          <span>
            {t("as.density_total")}:{" "}
            <BidiNum className="text-white font-bold">
              {bins.reduce((acc, b) => acc + b.count, 0)}
            </BidiNum>
          </span>
          <span>
            {t("as.density_high")}:{" "}
            <BidiNum className="text-tv-mint font-bold">
              {bins.reduce((acc, b) => acc + b.high_impact_count, 0)}
            </BidiNum>
          </span>
        </div>
      )}
    </div>
  );
}
