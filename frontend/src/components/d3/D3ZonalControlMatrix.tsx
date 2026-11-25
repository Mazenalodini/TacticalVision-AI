"use client";

import React, { useMemo, useState } from "react";
import * as d3 from "d3";
import { Compass } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { Telemetry } from "@/lib/types";
import { BidiNum } from "@/components/ui/BidiNum";

interface D3ZonalControlMatrixProps {
  telemetry: Telemetry | null;
}

interface ZoneCellData {
  row: number;
  col: number;
  c0: number;
  c1: number;
  total: number;
  pct0: number;
  pct1: number;
  fill: string;
  xMetersStart: number;
  xMetersEnd: number;
}

/**
 * 12x8 Pitch Zone Dominance Matrix powered by real `telemetry.heatmaps`.
 * Colors all 96 FIFA tactical zones by net territorial dominance (Team A vs Team B).
 */
export function D3ZonalControlMatrix({ telemetry }: D3ZonalControlMatrixProps) {
  const { t, dir } = useLang();
  const [hoveredCell, setHoveredCell] = useState<ZoneCellData | null>(null);

  const padX = 20;
  const padY = 20;
  const pitchW = 540;
  const pitchH = 348;
  const vbW = pitchW + padX * 2;
  const vbH = pitchH + padY * 2;

  const cells = useMemo(() => {
    const hm0 = telemetry?.heatmaps?.team_0;
    const hm1 = telemetry?.heatmaps?.team_1;
    if (!hm0 || !hm1) return [];

    let maxTotal = 1;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 12; c++) {
        const sum = (hm0[r]?.[c] || 0) + (hm1[r]?.[c] || 0);
        if (sum > maxTotal) maxTotal = sum;
      }
    }

    const intensityScale = d3
      .scaleSqrt()
      .domain([0, maxTotal])
      .range([0.08, 0.72])
      .clamp(true);

    const out: ZoneCellData[] = [];
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 12; c++) {
        const c0 = hm0[r]?.[c] || 0;
        const c1 = hm1[r]?.[c] || 0;
        const total = c0 + c1;
        const pct0 = total > 0 ? (c0 / total) * 100 : 50;
        const pct1 = total > 0 ? (c1 / total) * 100 : 50;

        let fill = "rgba(12, 27, 23, 0.45)";
        if (total > 0) {
          const alpha = intensityScale(total);
          const diff = (c0 - c1) / total; // -1..+1
          if (diff > 0.08) {
            fill = `rgba(56, 189, 248, ${(alpha * (0.45 + 0.55 * diff)).toFixed(3)})`;
          } else if (diff < -0.08) {
            fill = `rgba(251, 113, 133, ${(alpha * (0.45 + 0.55 * Math.abs(diff))).toFixed(3)})`;
          } else {
            fill = `rgba(148, 163, 184, ${(alpha * 0.4).toFixed(3)})`;
          }
        }

        out.push({
          row: r,
          col: c,
          c0,
          c1,
          total,
          pct0,
          pct1,
          fill,
          xMetersStart: Number(((c * 105) / 12).toFixed(1)),
          xMetersEnd: Number((((c + 1) * 105) / 12).toFixed(1)),
        });
      }
    }
    return out;
  }, [telemetry]);

  if (!telemetry || cells.length === 0) {
    return null;
  }

  const cellW = pitchW / 12;
  const cellH = pitchH / 8;

  return (
    <div className="rounded-2xl obsidian-card p-5 space-y-4 h-full flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-start justify-between gap-3 pb-3.5 mb-3.5 border-b border-white/[0.08]">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Compass className="w-4 h-4 text-tv-cyan" />
              <span>{t("d3.zone_title")}</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {t("d3.zone_desc")}
            </p>
          </div>
        </div>

        {/* Thirds Header Labels */}
        <div className="grid grid-cols-3 gap-2 mb-2 text-center text-[11px] font-semibold text-slate-400">
          <div className="py-1 rounded-lg obsidian-well truncate px-1">
            {t("tel.def_third")}
          </div>
          <div className="py-1 rounded-lg obsidian-well truncate px-1">
            {t("tel.mid_third")}
          </div>
          <div className="py-1 rounded-lg obsidian-well truncate px-1">
            {t("tel.att_third")}
          </div>
        </div>

        {/* 12x8 SVG Pitch Dominance Matrix */}
        <div
          dir="ltr"
          className="relative rounded-xl overflow-hidden border border-white/[0.09] bg-[#0C1B17]"
        >
          <svg
            viewBox={`0 0 ${vbW} ${vbH}`}
            onMouseLeave={() => setHoveredCell(null)}
            className="w-full h-auto block select-none"
            role="img"
            aria-label={t("d3.zone_title")}
          >
            {/* 96 Interactive Tactical Cells */}
            {cells.map((cell) => {
              const x = padX + cell.col * cellW;
              const y = padY + cell.row * cellH;
              const isHov =
                hoveredCell?.row === cell.row && hoveredCell?.col === cell.col;

              return (
                <rect
                  key={`z-${cell.row}-${cell.col}`}
                  x={x}
                  y={y}
                  width={cellW}
                  height={cellH}
                  fill={cell.fill}
                  stroke={
                    isHov
                      ? "#FFFFFF"
                      : "rgba(226, 232, 240, 0.08)"
                  }
                  strokeWidth={isHov ? 1.8 : 0.7}
                  onMouseEnter={() => setHoveredCell(cell)}
                  className="cursor-crosshair transition-colors"
                />
              );
            })}

            {/* Tactical Thirds Dividers (col 4 = 35m, col 8 = 70m) */}
            <line
              x1={padX + 4 * cellW}
              y1={padY}
              x2={padX + 4 * cellW}
              y2={padY + pitchH}
              stroke="rgba(226,232,240,0.28)"
              strokeDasharray="4 4"
              strokeWidth="1.2"
              pointerEvents="none"
            />
            <line
              x1={padX + 8 * cellW}
              y1={padY}
              x2={padX + 8 * cellW}
              y2={padY + pitchH}
              stroke="rgba(226,232,240,0.28)"
              strokeDasharray="4 4"
              strokeWidth="1.2"
              pointerEvents="none"
            />

            {/* Crisp FIFA Pitch Overlay */}
            <g
              stroke="rgba(226,232,240,0.52)"
              strokeWidth="1.4"
              fill="none"
              pointerEvents="none"
            >
              <rect x={padX} y={padY} width={pitchW} height={pitchH} />
              <line
                x1={padX + pitchW / 2}
                y1={padY}
                x2={padX + pitchW / 2}
                y2={padY + pitchH}
              />
              <circle
                cx={padX + pitchW / 2}
                cy={padY + pitchH / 2}
                r={9.15 * (pitchW / 105)}
              />
              <rect
                x={padX}
                y={padY + ((68 - 40.32) / 2) * (pitchH / 68)}
                width={16.5 * (pitchW / 105)}
                height={40.32 * (pitchH / 68)}
              />
              <rect
                x={padX + (105 - 16.5) * (pitchW / 105)}
                y={padY + ((68 - 40.32) / 2) * (pitchH / 68)}
                width={16.5 * (pitchW / 105)}
                height={40.32 * (pitchH / 68)}
              />
            </g>
          </svg>
        </div>
      </div>

      {/* Live Zone Readout Bar */}
      <div
        dir={dir}
        className="rounded-xl obsidian-well px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs"
      >
        {hoveredCell ? (
          <>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-300">
                {t("d3.zone_cell")}:
              </span>
              <BidiNum className="text-white font-bold">
                [{hoveredCell.col + 1}, {hoveredCell.row + 1}] · {hoveredCell.xMetersStart}–
                {hoveredCell.xMetersEnd}
              </BidiNum>
              <span className="text-slate-400">{t("meters")}</span>
            </div>

            {hoveredCell.total > 0 ? (
              <div className="flex items-center gap-3">
                <BidiNum className="text-tv-cyan font-bold">
                  {hoveredCell.pct0.toFixed(0)}% ({hoveredCell.c0})
                </BidiNum>
                <span className="text-slate-600">/</span>
                <BidiNum className="text-tv-coral font-bold">
                  {hoveredCell.pct1.toFixed(0)}% ({hoveredCell.c1})
                </BidiNum>
              </div>
            ) : (
              <span className="text-slate-500">{t("d3.zone_neutral")}</span>
            )}
          </>
        ) : (
          <div className="w-full flex items-center justify-between text-slate-400">
            <span className="inline-flex items-center gap-1.5 text-tv-cyan font-semibold">
              <span className="w-2.5 h-2.5 rounded-sm bg-tv-cyan/75" />
              <span>{t("d3.mom_a_dom")}</span>
            </span>
            <span className="inline-flex items-center gap-1.5 text-tv-coral font-semibold">
              <span className="w-2.5 h-2.5 rounded-sm bg-tv-coral/75" />
              <span>{t("d3.mom_b_dom")}</span>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
