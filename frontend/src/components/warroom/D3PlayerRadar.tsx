"use client";

import React, { useMemo, useState } from "react";
import * as d3 from "d3";
import { useLang } from "@/context/LangContext";
import type { Telemetry, PlayerStat } from "@/lib/types";
import {
  computePlayerRadarMetrics,
  type PlayerRadarMetrics,
} from "@/lib/telemetryHelpers";
import { BidiNum } from "@/components/ui/BidiNum";

interface D3PlayerRadarProps {
  player: PlayerStat;
  allPlayers: PlayerStat[];
  telemetry?: Telemetry | null;
  comparePlayer?: PlayerStat | null;
}

interface RadarAxisSpec {
  key: string;
  labelKey: string;
  getValue: (m: PlayerRadarMetrics) => number;
  minMaxFloor: number;
  formatVal: (m: PlayerRadarMetrics, unitKmh: string, unitM: string) => string;
}

const RADAR_AXES: RadarAxisSpec[] = [
  {
    key: "p95_speed",
    labelKey: "spot.radar_p95_speed",
    getValue: (m) => m.p95SpeedKmh,
    minMaxFloor: 28,
    formatVal: (m, kmh) => `${m.p95SpeedKmh.toFixed(1)} ${kmh}`,
  },
  {
    key: "avg_speed",
    labelKey: "tel.col_avg_speed",
    getValue: (m) => m.avgSpeedKmh,
    minMaxFloor: 15,
    formatVal: (m, kmh) => `${m.avgSpeedKmh.toFixed(1)} ${kmh}`,
  },
  {
    key: "distance",
    labelKey: "tel.col_dist",
    getValue: (m) => m.distanceM,
    minMaxFloor: 50,
    formatVal: (m, _kmh, mUnit) => `${m.distanceM.toFixed(1)} ${mUnit}`,
  },
  {
    key: "sprint_intensity",
    labelKey: "spot.radar_sprint_int",
    getValue: (m) => m.sprintIntensityScore,
    minMaxFloor: 15,
    formatVal: (m) => `${m.sprintsCount} (${m.highSpeedFrames}f)`,
  },
  {
    key: "activity",
    labelKey: "spot.radar_activity",
    getValue: (m) => m.activityPct,
    minMaxFloor: 50,
    formatVal: (m) => `${m.activityPct}%`,
  },
  {
    key: "ball_impact",
    labelKey: "spot.radar_ball_impact",
    getValue: (m) => m.ballImpactScore,
    minMaxFloor: 2,
    formatVal: (m) => `${m.passesInvolved}P · +${m.xtGenerated.toFixed(3)}`,
  },
];

/**
 * 6-Axis Declarative SVG + D3 Tactical Spider Chart.
 * Uses 6 differentiating real-data axes derived from frames[] + events[] + players[],
 * generous SVG padding to guarantee zero AR/EN label clipping, and Broadcast Sky / Rose colors.
 */
export function D3PlayerRadar({
  player,
  allPlayers,
  telemetry = null,
  comparePlayer = null,
}: D3PlayerRadarProps) {
  const { t, isRTL } = useLang();
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const cx = 200;
  const cy = 132;
  const maxR = 70;
  const numAxes = RADAR_AXES.length;

  const unitKmh = t("km_h");
  const unitM = t("meters");

  const { axisPoints, primaryPath, secondaryPath, primaryDots } = useMemo(() => {
    const pool = allPlayers.length > 0 ? allPlayers : [player];
    const teamPeers = pool.filter((p) => p.team === player.team);
    const baselinePool = teamPeers.length > 0 ? teamPeers : pool;

    const poolMetrics = pool.map((p) => computePlayerRadarMetrics(p, telemetry));
    const primaryMetric = computePlayerRadarMetrics(player, telemetry);

    let secondaryMetric: PlayerRadarMetrics;
    if (comparePlayer) {
      secondaryMetric = computePlayerRadarMetrics(comparePlayer, telemetry);
    } else {
      const baseMetrics = baselinePool.map((p) =>
        computePlayerRadarMetrics(p, telemetry)
      );
      const meanOf = (fn: (m: PlayerRadarMetrics) => number) =>
        d3.mean(baseMetrics, fn) ?? 0;
      secondaryMetric = {
        p95SpeedKmh: meanOf((m) => m.p95SpeedKmh),
        avgSpeedKmh: meanOf((m) => m.avgSpeedKmh),
        distanceM: meanOf((m) => m.distanceM),
        sprintIntensityScore: meanOf((m) => m.sprintIntensityScore),
        sprintsCount: Math.round(meanOf((m) => m.sprintsCount)),
        highSpeedFrames: Math.round(meanOf((m) => m.highSpeedFrames)),
        activityPct: Math.round(meanOf((m) => m.activityPct)),
        ballImpactScore: meanOf((m) => m.ballImpactScore),
        passesInvolved: Math.round(meanOf((m) => m.passesInvolved)),
        turnoversWon: Math.round(meanOf((m) => m.turnoversWon)),
        xtGenerated: meanOf((m) => m.xtGenerated),
      };
    }

    const primaryNorms: [number, number][] = [];
    const secondaryNorms: [number, number][] = [];
    const computedAxes: Array<{
      idx: number;
      label: string;
      angle: number;
      outerX: number;
      outerY: number;
      labelX: number;
      labelY: number;
      anchor: "start" | "middle" | "end";
      formattedPrimary: string;
      formattedSecondary: string;
    }> = [];
    const dots: Array<{ x: number; y: number; idx: number }> = [];

    RADAR_AXES.forEach((spec, idx) => {
      const maxObserved = d3.max(poolMetrics, spec.getValue) ?? 0;
      const safeMax = Math.max(spec.minMaxFloor, maxObserved, 0.0001);
      const scale = d3
        .scaleLinear()
        .domain([0, safeMax])
        .range([0.14, 1])
        .clamp(true);

      const rawP = spec.getValue(primaryMetric);
      const rawS = spec.getValue(secondaryMetric);

      const normP = scale(rawP);
      const normS = scale(rawS);

      const angle = (Math.PI * 2 * idx) / numAxes;
      const svgAngle = angle - Math.PI / 2;

      primaryNorms.push([angle, normP * maxR]);
      secondaryNorms.push([angle, normS * maxR]);

      const px = cx + Math.cos(svgAngle) * normP * maxR;
      const py = cy + Math.sin(svgAngle) * normP * maxR;
      dots.push({ x: px, y: py, idx });

      const outerX = cx + Math.cos(svgAngle) * maxR;
      const outerY = cy + Math.sin(svgAngle) * maxR;

      const labelR = maxR + 20;
      const labelX = cx + Math.cos(svgAngle) * labelR;
      const labelY = cy + Math.sin(svgAngle) * labelR;

      const cosVal = Math.cos(svgAngle);
      const anchor: "start" | "middle" | "end" =
        Math.abs(cosVal) < 0.25 ? "middle" : cosVal > 0 ? "start" : "end";

      computedAxes.push({
        idx,
        label: t(spec.labelKey),
        angle: svgAngle,
        outerX,
        outerY,
        labelX,
        labelY,
        anchor,
        formattedPrimary: spec.formatVal(primaryMetric, unitKmh, unitM),
        formattedSecondary: spec.formatVal(secondaryMetric, unitKmh, unitM),
      });
    });

    const radialGen = d3
      .lineRadial<[number, number]>()
      .angle((d) => d[0])
      .radius((d) => d[1])
      .curve(d3.curveLinearClosed);

    return {
      axisPoints: computedAxes,
      primaryPath: radialGen(primaryNorms) || "",
      secondaryPath: radialGen(secondaryNorms) || "",
      primaryDots: dots,
    };
  }, [player, allPlayers, telemetry, comparePlayer, maxR, numAxes, t, unitKmh, unitM]);

  const gridPolygons = useMemo(() => {
    return [0.25, 0.5, 0.75, 1.0].map((level) => {
      const pts = Array.from({ length: numAxes }, (_, i) => {
        const a = (Math.PI * 2 * i) / numAxes - Math.PI / 2;
        return `${(cx + Math.cos(a) * maxR * level).toFixed(1)},${(
          cy +
          Math.sin(a) * maxR * level
        ).toFixed(1)}`;
      });
      return pts.join(" ");
    });
  }, [numAxes, maxR]);

  const primaryColor = player.team === 0 ? "#38BDF8" : "#FB7185";
  const primaryFill =
    player.team === 0
      ? "rgba(56, 189, 248, 0.22)"
      : "rgba(251, 113, 133, 0.22)";
  const isSameTeamCompare =
    Boolean(comparePlayer) && comparePlayer?.team === player.team;
  const secondaryStroke = comparePlayer
    ? isSameTeamCompare
      ? player.team === 0
        ? "#FB7185"
        : "#38BDF8"
      : comparePlayer.team === 0
      ? "#38BDF8"
      : "#FB7185"
    : "rgba(148, 163, 184, 0.55)";
  const secondaryFill = comparePlayer
    ? isSameTeamCompare
      ? player.team === 0
        ? "rgba(251, 113, 133, 0.14)"
        : "rgba(56, 189, 248, 0.14)"
      : comparePlayer.team === 0
      ? "rgba(56, 189, 248, 0.14)"
      : "rgba(251, 113, 133, 0.14)"
    : "rgba(148, 163, 184, 0.11)";

  const activeAxis = hoveredIdx !== null ? axisPoints[hoveredIdx] : null;

  return (
    <div className="rounded-xl obsidian-well p-3 border border-white/[0.07] relative">
      {/* Header & Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1 text-xs">
        <span className="font-bold text-slate-200 truncate">
          {t("spot.radar_title")}
        </span>
        <div className="flex items-center gap-2.5 text-[11px] shrink-0">
          <span className="inline-flex items-center gap-1 text-slate-200 font-semibold">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: primaryColor }}
            />
            <BidiNum>#{player.track_id}</BidiNum>
          </span>
          <span className="inline-flex items-center gap-1 text-slate-400">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: secondaryStroke }}
            />
            {comparePlayer ? (
              <BidiNum>#{comparePlayer.track_id}</BidiNum>
            ) : (
              <span>{t("spot.vs_team_avg")}</span>
            )}
          </span>
        </div>
      </div>

      {/* SVG Spider Chart (Generous 400x264 frame so no EN/AR label ever clips) */}
      <div dir="ltr" className="w-full">
        <svg
          viewBox="0 0 400 264"
          className="w-full h-auto block overflow-visible"
          role="img"
          aria-label={t("spot.radar_title")}
        >
          {gridPolygons.map((pts, i) => (
            <polygon
              key={`ring-${i}`}
              points={pts}
              fill="none"
              stroke="rgba(148, 163, 184, 0.15)"
              strokeWidth="1"
              strokeDasharray={i < 3 ? "2 3" : undefined}
            />
          ))}

          {axisPoints.map((ax) => (
            <line
              key={`spoke-${ax.idx}`}
              x1={cx}
              y1={cy}
              x2={ax.outerX}
              y2={ax.outerY}
              stroke="rgba(148, 163, 184, 0.18)"
              strokeWidth="1"
            />
          ))}

          <g transform={`translate(${cx}, ${cy})`}>
            <path
              d={secondaryPath}
              fill={secondaryFill}
              stroke={secondaryStroke}
              strokeWidth="1.5"
              strokeDasharray={comparePlayer ? undefined : "4 3"}
            />
            <path
              d={primaryPath}
              fill={primaryFill}
              stroke={primaryColor}
              strokeWidth="2.2"
            />
          </g>

          {primaryDots.map((dot) => {
            const isHov = hoveredIdx === dot.idx;
            return (
              <circle
                key={`dot-${dot.idx}`}
                cx={dot.x}
                cy={dot.y}
                r={isHov ? 5.5 : 3.8}
                fill={primaryColor}
                stroke="#060912"
                strokeWidth="1.6"
                onMouseEnter={() => setHoveredIdx(dot.idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                className="cursor-pointer transition-all"
              />
            );
          })}

          {axisPoints.map((ax) => (
            <text
              key={`lbl-${ax.idx}`}
              x={ax.labelX}
              y={ax.labelY + 3.5}
              textAnchor={ax.anchor}
              direction={isRTL ? "rtl" : "ltr"}
              unicodeBidi="plaintext"
              fill={hoveredIdx === ax.idx ? "#FFFFFF" : "#94A3B8"}
              fontSize="10.5"
              fontWeight={hoveredIdx === ax.idx ? "700" : "600"}
              onMouseEnter={() => setHoveredIdx(ax.idx)}
              onMouseLeave={() => setHoveredIdx(null)}
              className="cursor-pointer select-none"
            >
              {ax.label}
            </text>
          ))}
        </svg>
      </div>

      {activeAxis && (
        <div className="mt-1 pt-2 border-t border-white/[0.06] flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-300">
            {activeAxis.label}:
          </span>
          <div className="flex items-center gap-2">
            <BidiNum className="font-bold text-white">
              {activeAxis.formattedPrimary}
            </BidiNum>
            <span className="text-slate-500">/</span>
            <BidiNum className="text-slate-400">
              {activeAxis.formattedSecondary}
            </BidiNum>
          </div>
        </div>
      )}
    </div>
  );
}
