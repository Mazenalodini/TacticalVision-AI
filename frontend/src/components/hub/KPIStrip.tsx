"use client";

import React, { useMemo } from "react";
import * as d3 from "d3";
import { Film, Zap, Gauge, Crosshair } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { MatchCard } from "@/lib/types";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";
import { BidiNum } from "@/components/ui/BidiNum";
import { KPIStripSkeleton } from "@/components/ui/Skeleton";

interface KPIStripProps {
  matches: MatchCard[];
  isLoading: boolean;
}

/**
 * D3 Radial Speedometer Gauge (0..40 km/h with 24 km/h Sprint Benchmark Tick)
 */
function D3SpeedRadialGauge({ speed }: { speed: number }) {
  const clamped = Math.max(0, Math.min(40, speed));
  const startAngle = -Math.PI * 0.72;
  const endAngle = Math.PI * 0.72;

  const angleScale = useMemo(
    () => d3.scaleLinear().domain([0, 40]).range([startAngle, endAngle]),
    [startAngle, endAngle]
  );

  const bgArc = useMemo(() => {
    const gen = d3
      .arc<unknown>()
      .innerRadius(21)
      .outerRadius(27)
      .startAngle(startAngle)
      .endAngle(endAngle)
      .cornerRadius(4);
    return gen({}) || "";
  }, [startAngle, endAngle]);

  const valArc = useMemo(() => {
    const gen = d3
      .arc<unknown>()
      .innerRadius(21)
      .outerRadius(27)
      .startAngle(startAngle)
      .endAngle(angleScale(clamped))
      .cornerRadius(4);
    return gen({}) || "";
  }, [startAngle, angleScale, clamped]);

  // 24 km/h sprint benchmark marker coordinates
  const sprintAngle = angleScale(24) - Math.PI / 2;
  const sx1 = Math.cos(sprintAngle) * 18;
  const sy1 = Math.sin(sprintAngle) * 18;
  const sx2 = Math.cos(sprintAngle) * 30;
  const sy2 = Math.sin(sprintAngle) * 30;

  return (
    <svg
      viewBox="-32 -32 64 54"
      className="w-16 h-14 shrink-0 overflow-visible"
      aria-hidden="true"
    >
      <path d={bgArc} fill="rgba(148, 163, 184, 0.14)" />
      <path d={valArc} fill="#34D399" />
      <line
        x1={sx1}
        y1={sy1}
        x2={sx2}
        y2={sy2}
        stroke="#F59E0B"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * D3 Dual Concentric Precision Rings (Homography Stability + Ball Coverage)
 */
function D3PrecisionRings({
  homographyPct,
  ballPct,
}: {
  homographyPct: number;
  ballPct: number;
}) {
  const full = Math.PI * 2;
  const hAngle = (Math.max(0, Math.min(100, homographyPct)) / 100) * full;
  const bAngle = (Math.max(0, Math.min(100, ballPct)) / 100) * full;

  const outerBg = d3.arc<unknown>().innerRadius(22).outerRadius(26).startAngle(0).endAngle(full)({}) || "";
  const outerVal =
    d3.arc<unknown>().innerRadius(22).outerRadius(26).startAngle(0).endAngle(hAngle).cornerRadius(3)({}) || "";

  const innerBg = d3.arc<unknown>().innerRadius(15).outerRadius(19).startAngle(0).endAngle(full)({}) || "";
  const innerVal =
    d3.arc<unknown>().innerRadius(15).outerRadius(19).startAngle(0).endAngle(bAngle).cornerRadius(3)({}) || "";

  return (
    <svg viewBox="-30 -30 60 60" className="w-14 h-14 shrink-0" aria-hidden="true">
      <path d={outerBg} fill="rgba(148, 163, 184, 0.12)" />
      <path d={outerVal} fill="#34D399" />
      <path d={innerBg} fill="rgba(148, 163, 184, 0.12)" />
      <path d={innerVal} fill="#38BDF8" />
    </svg>
  );
}

export function KPIStrip({ matches, isLoading }: KPIStripProps) {
  const { t } = useLang();

  if (isLoading) {
    return <KPIStripSkeleton />;
  }

  const totalMatches = matches.length;
  const totalDuration = matches.reduce((acc, m) => acc + (m.duration_s || 0), 0);
  const totalEvents = matches.reduce((acc, m) => acc + (m.events_count || 0), 0);
  const totalPasses = matches.reduce(
    (acc, m) => acc + (m.passes?.team_0 || 0) + (m.passes?.team_1 || 0),
    0
  );
  const totalTurnovers = matches.reduce((acc, m) => acc + (m.turnovers || 0), 0);
  const totalSprints = Math.max(0, totalEvents - totalPasses - totalTurnovers);

  const peakSpeed = matches.reduce((max, m) => Math.max(max, m.top_speed_kmh || 0), 0);

  const avgBallCoverage =
    totalMatches > 0
      ? matches.reduce((acc, m) => acc + (m.ball_coverage_pct || 0), 0) / totalMatches
      : 0;
  const avgHomography =
    totalMatches > 0
      ? matches.reduce((acc, m) => acc + (m.homography_stability_pct || 100), 0) /
        totalMatches
      : 100;

  const evtSum = Math.max(1, totalEvents);
  const passPct = (totalPasses / evtSum) * 100;
  const turnPct = (totalTurnovers / evtSum) * 100;
  const sprintPct = Math.max(0, 100 - passPct - turnPct);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-8">
      {/* 1. Analyzed Library */}
      <div className="obsidian-card obsidian-card-interactive rounded-2xl p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold text-slate-400">
              {t("hub.kpi_matches")}
            </span>
            <div className="p-2 rounded-xl bg-tv-cyan/10 text-tv-cyan border border-tv-cyan/20">
              <Film className="w-4 h-4" />
            </div>
          </div>

          <div className="flex items-baseline justify-between gap-2 mb-3">
            <span className="text-3xl font-extrabold text-white">
              <AnimatedNumber value={totalMatches} decimals={0} />
            </span>
            <span className="inline-flex items-baseline gap-0.5 text-xs text-tv-cyan font-semibold px-2 py-0.5 rounded-md bg-tv-cyan/10 border border-tv-cyan/20">
              <BidiNum>{totalDuration.toFixed(1)}</BidiNum>
              <span>{t("seconds")}</span>
            </span>
          </div>
        </div>

        <div>
          {/* Proportional Match Duration Segments */}
          <div className="h-1.5 w-full rounded-full bg-slate-800/90 overflow-hidden flex gap-0.5 mb-2">
            {matches.map((m, i) => {
              const share =
                totalDuration > 0 ? ((m.duration_s || 1) / totalDuration) * 100 : 33;
              return (
                <div
                  key={m.id}
                  style={{ width: `${Math.max(8, share)}%` }}
                  className={`h-full ${
                    i % 2 === 0 ? "bg-tv-cyan" : "bg-tv-cyan/55"
                  } first:rounded-s-full last:rounded-e-full`}
                />
              );
            })}
          </div>
          <p className="text-xs text-slate-400">{t("hub.kpi_matches_sub")}</p>
        </div>
      </div>

      {/* 2. Tactical Events Distribution (100% Localized Breakdown — zero 20P/181S/25T leaks) */}
      <div className="obsidian-card obsidian-card-interactive rounded-2xl p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold text-slate-400">
              {t("hub.kpi_events")}
            </span>
            <div className="p-2 rounded-xl bg-tv-amber/10 text-tv-amber border border-tv-amber/20">
              <Zap className="w-4 h-4" />
            </div>
          </div>

          <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
            <span className="text-3xl font-extrabold text-white">
              <AnimatedNumber value={totalEvents} decimals={0} />
            </span>
            <div className="flex flex-wrap items-center gap-2.5 text-[11px]">
              <span className="inline-flex items-baseline gap-1 text-tv-cyan font-semibold">
                <BidiNum>{totalPasses}</BidiNum>
                <span>{t("hub.kpi_passes_short")}</span>
              </span>
              <span className="inline-flex items-baseline gap-1 text-tv-mint font-semibold">
                <BidiNum>{totalSprints}</BidiNum>
                <span>{t("hub.kpi_sprints_short")}</span>
              </span>
              <span className="inline-flex items-baseline gap-1 text-tv-amber font-semibold">
                <BidiNum>{totalTurnovers}</BidiNum>
                <span>{t("hub.kpi_turnovers_short")}</span>
              </span>
            </div>
          </div>
        </div>

        <div>
          <div className="h-1.5 w-full rounded-full bg-slate-800/90 overflow-hidden flex gap-0.5 mb-2">
            <div style={{ width: `${passPct}%` }} className="h-full bg-tv-cyan" />
            <div style={{ width: `${sprintPct}%` }} className="h-full bg-tv-mint" />
            <div style={{ width: `${turnPct}%` }} className="h-full bg-tv-amber" />
          </div>
          <p className="text-xs text-slate-400">{t("hub.kpi_events_sub")}</p>
        </div>
      </div>

      {/* 3. Peak Velocity + D3 Radial Gauge */}
      <div className="obsidian-card obsidian-card-interactive rounded-2xl p-5 flex items-center justify-between gap-3">
        <div className="flex flex-col justify-between h-full">
          <div className="flex items-center gap-2 mb-2">
            <Gauge className="w-4 h-4 text-tv-mint" />
            <span className="text-xs font-semibold text-slate-400">
              {t("hub.kpi_speed")}
            </span>
          </div>

          <div className="mb-1.5 inline-flex items-baseline gap-1.5">
            <span className="text-3xl font-extrabold text-white">
              <AnimatedNumber value={peakSpeed} decimals={1} />
            </span>
            <span className="text-xs font-semibold text-slate-400">
              {t("km_h")}
            </span>
          </div>

          <p className="text-xs text-slate-400">{t("hub.kpi_speed_sub")}</p>
        </div>

        <D3SpeedRadialGauge speed={peakSpeed} />
      </div>

      {/* 4. Spatial Homography & Ball Precision + D3 Concentric Rings */}
      <div className="obsidian-card obsidian-card-interactive rounded-2xl p-5 flex items-center justify-between gap-3">
        <div className="flex flex-col justify-between h-full min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <Crosshair className="w-4 h-4 text-tv-cyan" />
            <span className="text-xs font-semibold text-slate-400 truncate">
              {t("hub.kpi_precision")}
            </span>
          </div>

          <div className="mb-1.5">
            <span className="text-3xl font-extrabold text-tv-mint">
              <AnimatedNumber value={avgHomography} decimals={0} suffix="%" />
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span>{t("hub.kpi_ball_short")}:</span>
            <BidiNum className="text-tv-cyan font-bold">
              {avgBallCoverage.toFixed(1)}%
            </BidiNum>
          </div>
        </div>

        <D3PrecisionRings
          homographyPct={avgHomography}
          ballPct={avgBallCoverage}
        />
      </div>
    </div>
  );
}
