"use client";

import React, { useMemo } from "react";
import { Compass, ArrowRightLeft } from "lucide-react";
import { useLang } from "@/context/LangContext";
import { BidiNum } from "@/components/ui/BidiNum";
import { formatTeamCode } from "@/lib/format";
import type { Telemetry } from "@/lib/types";

interface TeamComparisonProps {
  telemetry: Telemetry | null;
}

interface MetricRowProps {
  label: string;
  val0: number | string;
  val1: number | string;
  rawVal0: number;
  rawVal1: number;
  unit?: string;
  higherIsBetter?: boolean;
}

function MetricRow({
  label,
  val0,
  val1,
  rawVal0,
  rawVal1,
  unit = "",
  higherIsBetter = true,
}: MetricRowProps) {
  const sum = Math.max(rawVal0 + rawVal1, 0.001);
  const pct0 =
    rawVal0 === 0 && rawVal1 === 0
      ? 50
      : Math.min(Math.max((rawVal0 / sum) * 100, 5), 95);
  const pct1 = 100 - pct0;

  const isDraw = Math.abs(rawVal0 - rawVal1) < 0.001;
  const is0Leader = !isDraw && (higherIsBetter ? rawVal0 > rawVal1 : rawVal0 < rawVal1);
  const is1Leader = !isDraw && (higherIsBetter ? rawVal1 > rawVal0 : rawVal1 < rawVal0);

  return (
    <div className="group rounded-xl p-3.5 obsidian-well hover:border-white/[0.12] border border-white/[0.05] transition-all">
      <div className="flex items-center justify-between gap-2 text-xs mb-2">
        {/* Team A Value */}
        <div className="flex items-baseline gap-1 shrink-0">
          <BidiNum
            className={`font-bold text-sm ${
              is0Leader ? "text-tv-cyan" : "text-slate-300"
            }`}
          >
            {val0}
          </BidiNum>
          {unit && (
            <span className="text-[10px] text-slate-500 font-medium">
              {unit}
            </span>
          )}
        </div>

        {/* Metric Label */}
        <span className="text-xs font-semibold text-slate-300 text-center px-2 truncate">
          {label}
        </span>

        {/* Team B Value */}
        <div className="flex items-baseline gap-1 shrink-0">
          <BidiNum
            className={`font-bold text-sm ${
              is1Leader ? "text-tv-coral" : "text-slate-300"
            }`}
          >
            {val1}
          </BidiNum>
          {unit && (
            <span className="text-[10px] text-slate-500 font-medium">
              {unit}
            </span>
          )}
        </div>
      </div>

      {/* Mirrored Horizontal Comparison Bar (Aligned with Start=Team A, End=Team B) */}
      <div className="relative h-2 w-full bg-slate-950/90 rounded-full overflow-hidden flex items-center p-0.5 border border-white/[0.04]">
        <div
          className="h-full rounded-s-full bg-tv-cyan transition-all duration-700"
          style={{ width: `${pct0}%` }}
        />
        <div className="w-0.5 h-full bg-white/10 z-10 shrink-0" />
        <div
          className="h-full rounded-e-full bg-tv-coral transition-all duration-700"
          style={{ width: `${pct1}%` }}
        />
      </div>
    </div>
  );
}

export function TeamComparison({ telemetry }: TeamComparisonProps) {
  const { t, lang } = useLang();

  const metrics = useMemo(() => {
    if (!telemetry) return null;

    const sum = telemetry.summary;
    const events = telemetry.events || [];
    const players = telemetry.players || [];
    const passingNetwork = telemetry.passing_network || [];

    const sprints0 = events.filter(
      (e) => e.type === "sprint" && e.team === 0
    ).length;
    const sprints1 = events.filter(
      (e) => e.type === "sprint" && e.team === 1
    ).length;

    const turnovers0 = events.filter(
      (e) => e.type === "turnover" && e.from_team === 0
    ).length;
    const turnovers1 = events.filter(
      (e) => e.type === "turnover" && e.from_team === 1
    ).length;

    const team0Players = players.filter((p) => p.team === 0);
    const team1Players = players.filter((p) => p.team === 1);
    const topSpd0 =
      team0Players.length > 0
        ? Math.max(...team0Players.map((p) => p.max_speed_kmh || 0))
        : 0;
    const topSpd1 =
      team1Players.length > 0
        ? Math.max(...team1Players.map((p) => p.max_speed_kmh || 0))
        : 0;

    const dist0 = team0Players.reduce((acc, p) => acc + (p.distance_m || 0), 0);
    const dist1 = team1Players.reduce((acc, p) => acc + (p.distance_m || 0), 0);

    const netDensity0 = passingNetwork.filter((e) => e.team === 0).length;
    const netDensity1 = passingNetwork.filter((e) => e.team === 1).length;

    const zonal = sum.pitch_control_pct?.team_0_zonal || {
      defensive_third: 50,
      middle_third: 50,
      attacking_third: 50,
    };

    return {
      possession: {
        t0: sum.possession_pct?.team_0 ?? 50,
        t1: sum.possession_pct?.team_1 ?? 50,
      },
      pitchControl: {
        t0: sum.pitch_control_pct?.team_0 ?? 50,
        t1: sum.pitch_control_pct?.team_1 ?? 50,
      },
      passes: {
        t0: sum.passes?.team_0 ?? 0,
        t1: sum.passes?.team_1 ?? 0,
      },
      xt: {
        t0: sum.expected_threat_xt?.team_0 ?? 0,
        t1: sum.expected_threat_xt?.team_1 ?? 0,
      },
      hullArea: {
        t0: sum.avg_convex_hull_area_m2?.team_0 ?? 0,
        t1: sum.avg_convex_hull_area_m2?.team_1 ?? 0,
      },
      sprints: { t0: sprints0, t1: sprints1 },
      turnovers: { t0: turnovers0, t1: turnovers1 },
      topSpeed: { t0: topSpd0, t1: topSpd1 },
      totalDist: { t0: Math.round(dist0), t1: Math.round(dist1) },
      netDensity: { t0: netDensity0, t1: netDensity1 },
      zonal,
    };
  }, [telemetry]);

  if (!metrics) {
    return (
      <div className="p-8 text-center text-slate-500 text-xs obsidian-card rounded-2xl">
        {t("no_data")}
      </div>
    );
  }

  const areaUnit = `${t("meters")}²`;

  return (
    <div className="space-y-6">
      {/* ── Header Card with Symmetric Team Badges ── */}
      <div className="rounded-2xl obsidian-card p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-white/[0.08]">
          {/* Team A Badge */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-tv-cyan/15 border border-tv-cyan/30 flex items-center justify-center text-tv-cyan font-black text-sm shrink-0">
              {formatTeamCode(0, lang)}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-tv-cyan shadow-sm shadow-tv-cyan" />
                <h3 className="font-bold text-white text-base">{t("team_a")}</h3>
              </div>
              <p className="text-xs text-slate-400 flex items-center gap-1.5 mt-0.5">
                <BidiNum className="font-semibold text-tv-cyan">
                  {`${metrics.possession.t0.toFixed(1)}%`}
                </BidiNum>
                <span>{t("card.possession")}</span>
              </p>
            </div>
          </div>

          {/* Center Tactical Match Badge */}
          <div className="text-center px-4 py-2 rounded-xl obsidian-well border border-white/[0.06]">
            <span className="text-xs font-bold text-slate-200 flex items-center justify-center gap-1.5">
              <ArrowRightLeft className="w-3.5 h-3.5 text-tv-mint shrink-0" />
              <span>{t("tel.h2h_title")}</span>
            </span>
            <p className="text-[11px] text-slate-400 mt-0.5 hidden md:block">
              {t("tel.h2h_desc")}
            </p>
          </div>

          {/* Team B Badge */}
          <div className="flex items-center gap-3 sm:flex-row-reverse sm:text-end">
            <div className="w-10 h-10 rounded-xl bg-tv-coral/15 border border-tv-coral/30 flex items-center justify-center text-tv-coral font-black text-sm shrink-0">
              {formatTeamCode(1, lang)}
            </div>
            <div>
              <div className="flex items-center gap-2 sm:flex-row-reverse">
                <span className="w-2.5 h-2.5 rounded-full bg-tv-coral shadow-sm shadow-tv-coral" />
                <h3 className="font-bold text-white text-base">{t("team_b")}</h3>
              </div>
              <p className="text-xs text-slate-400 flex items-center sm:justify-end gap-1.5 mt-0.5">
                <BidiNum className="font-semibold text-tv-coral">
                  {`${metrics.possession.t1.toFixed(1)}%`}
                </BidiNum>
                <span>{t("card.possession")}</span>
              </p>
            </div>
          </div>
        </div>

        {/* ── Key Tactical Metrics Grid ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <MetricRow
            label={t("card.possession")}
            val0={`${metrics.possession.t0.toFixed(1)}%`}
            val1={`${metrics.possession.t1.toFixed(1)}%`}
            rawVal0={metrics.possession.t0}
            rawVal1={metrics.possession.t1}
          />
          <MetricRow
            label={t("card.pitch_control")}
            val0={`${metrics.pitchControl.t0.toFixed(1)}%`}
            val1={`${metrics.pitchControl.t1.toFixed(1)}%`}
            rawVal0={metrics.pitchControl.t0}
            rawVal1={metrics.pitchControl.t1}
          />
          <MetricRow
            label={t("card.passes")}
            val0={metrics.passes.t0}
            val1={metrics.passes.t1}
            rawVal0={metrics.passes.t0}
            rawVal1={metrics.passes.t1}
          />
          <MetricRow
            label={t("tel.xt_metric")}
            val0={`+${metrics.xt.t0.toFixed(3)}`}
            val1={`+${metrics.xt.t1.toFixed(3)}`}
            rawVal0={metrics.xt.t0}
            rawVal1={metrics.xt.t1}
          />
          <MetricRow
            label={t("tel.hull_area")}
            val0={metrics.hullArea.t0.toFixed(0)}
            val1={metrics.hullArea.t1.toFixed(0)}
            rawVal0={metrics.hullArea.t0}
            rawVal1={metrics.hullArea.t1}
            unit={areaUnit}
          />
          <MetricRow
            label={t("tel.col_sprints")}
            val0={metrics.sprints.t0}
            val1={metrics.sprints.t1}
            rawVal0={metrics.sprints.t0}
            rawVal1={metrics.sprints.t1}
          />
          <MetricRow
            label={t("card.turnovers")}
            val0={metrics.turnovers.t0}
            val1={metrics.turnovers.t1}
            rawVal0={metrics.turnovers.t0}
            rawVal1={metrics.turnovers.t1}
            higherIsBetter={false}
          />
          <MetricRow
            label={t("card.top_speed")}
            val0={metrics.topSpeed.t0.toFixed(1)}
            val1={metrics.topSpeed.t1.toFixed(1)}
            rawVal0={metrics.topSpeed.t0}
            rawVal1={metrics.topSpeed.t1}
            unit={t("km_h")}
          />
          <MetricRow
            label={t("tel.total_dist")}
            val0={metrics.totalDist.t0.toLocaleString("en-US")}
            val1={metrics.totalDist.t1.toLocaleString("en-US")}
            rawVal0={metrics.totalDist.t0}
            rawVal1={metrics.totalDist.t1}
            unit={t("meters")}
          />
          <MetricRow
            label={t("tel.pass_density")}
            val0={metrics.netDensity.t0}
            val1={metrics.netDensity.t1}
            rawVal0={metrics.netDensity.t0}
            rawVal1={metrics.netDensity.t1}
          />
        </div>
      </div>

      {/* ── Zonal Spatial Control Strip (Three Pitch Thirds) ── */}
      <div className="rounded-2xl obsidian-card p-5">
        <h4 className="text-xs font-bold text-slate-200 mb-4 flex items-center gap-2">
          <Compass className="w-4 h-4 text-tv-cyan shrink-0" />
          <span>{t("tel.zonal_title")}</span>
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Defensive Third (0 - 35m) */}
          <div className="rounded-xl p-3.5 obsidian-well border border-white/[0.05] flex flex-col justify-between">
            <div>
              <span className="text-[11px] font-semibold text-slate-400 block mb-1.5">
                {t("tel.def_third")}
              </span>
              <div className="flex items-baseline justify-between mb-2">
                <BidiNum className="text-base font-bold text-tv-cyan">
                  {`${metrics.zonal.defensive_third.toFixed(1)}%`}
                </BidiNum>
                <BidiNum className="text-base font-bold text-tv-coral">
                  {`${(100 - metrics.zonal.defensive_third).toFixed(1)}%`}
                </BidiNum>
              </div>
            </div>
            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden flex">
              <div
                className="h-full bg-tv-cyan rounded-s-full"
                style={{ width: `${metrics.zonal.defensive_third}%` }}
              />
              <div
                className="h-full bg-tv-coral rounded-e-full"
                style={{ width: `${100 - metrics.zonal.defensive_third}%` }}
              />
            </div>
          </div>

          {/* Middle Third (35 - 70m) */}
          <div className="rounded-xl p-3.5 obsidian-well border border-white/[0.05] flex flex-col justify-between">
            <div>
              <span className="text-[11px] font-semibold text-slate-400 block mb-1.5">
                {t("tel.mid_third")}
              </span>
              <div className="flex items-baseline justify-between mb-2">
                <BidiNum className="text-base font-bold text-tv-cyan">
                  {`${metrics.zonal.middle_third.toFixed(1)}%`}
                </BidiNum>
                <BidiNum className="text-base font-bold text-tv-coral">
                  {`${(100 - metrics.zonal.middle_third).toFixed(1)}%`}
                </BidiNum>
              </div>
            </div>
            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden flex">
              <div
                className="h-full bg-tv-cyan rounded-s-full"
                style={{ width: `${metrics.zonal.middle_third}%` }}
              />
              <div
                className="h-full bg-tv-coral rounded-e-full"
                style={{ width: `${100 - metrics.zonal.middle_third}%` }}
              />
            </div>
          </div>

          {/* Attacking Third (70 - 105m) */}
          <div className="rounded-xl p-3.5 obsidian-well border border-white/[0.05] flex flex-col justify-between">
            <div>
              <span className="text-[11px] font-semibold text-slate-400 block mb-1.5">
                {t("tel.att_third")}
              </span>
              <div className="flex items-baseline justify-between mb-2">
                <BidiNum className="text-base font-bold text-tv-cyan">
                  {`${metrics.zonal.attacking_third.toFixed(1)}%`}
                </BidiNum>
                <BidiNum className="text-base font-bold text-tv-coral">
                  {`${(100 - metrics.zonal.attacking_third).toFixed(1)}%`}
                </BidiNum>
              </div>
            </div>
            <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden flex">
              <div
                className="h-full bg-tv-cyan rounded-s-full"
                style={{ width: `${metrics.zonal.attacking_third}%` }}
              />
              <div
                className="h-full bg-tv-coral rounded-e-full"
                style={{ width: `${100 - metrics.zonal.attacking_third}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
