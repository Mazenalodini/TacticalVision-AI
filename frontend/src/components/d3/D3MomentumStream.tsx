"use client";

import React, { useMemo, useState } from "react";
import * as d3 from "d3";
import { Activity } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { Telemetry, MatchEvent } from "@/lib/types";
import { BidiNum } from "@/components/ui/BidiNum";

interface D3MomentumStreamProps {
  telemetry: Telemetry | null;
  currentTime: number;
  duration_s: number;
  onSeekToTime?: (timeSec: number) => void;
  onSeekToFrame?: (frame: number) => void;
}

interface MomentumSample {
  t: number;
  f: number;
  vor0: number;
  vor1: number;
  delta: number; // vor0 - 50 (-50..+50)
}

/**
 * Declarative SVG + D3 Territorial Voronoi Momentum Stream.
 * Diverges cleanly around 50% equilibrium: Cyan surge above for Team A,
 * Coral surge below for Team B, overlaid with clickable tactical event pins.
 */
export function D3MomentumStream({
  telemetry,
  currentTime,
  duration_s,
  onSeekToTime,
  onSeekToFrame,
}: D3MomentumStreamProps) {
  const { t, dir } = useLang();
  const [hoverSample, setHoverSample] = useState<MomentumSample | null>(null);

  const vbW = 760;
  const vbH = 195;
  const padL = 42;
  const padR = 18;
  const padT = 18;
  const padB = 26;

  const safeDur = Math.max(0.5, duration_s || telemetry?.summary?.duration_s || 30);

  const xScale = useMemo(
    () => d3.scaleLinear().domain([0, safeDur]).range([padL, vbW - padR]),
    [safeDur]
  );

  // Domain [-50, +50] so y=0 corresponds to 50% Voronoi equilibrium
  const yScale = useMemo(
    () => d3.scaleLinear().domain([-50, 50]).range([vbH - padB, padT]),
    []
  );
  const midY = yScale(0);

  const { samples, areaPath, linePath, keyEvents } = useMemo(() => {
    const frames = telemetry?.frames || [];
    if (frames.length === 0) {
      return { samples: [], areaPath: "", linePath: "", keyEvents: [] };
    }

    const step = frames.length > 400 ? 3 : 1;
    const pts: MomentumSample[] = [];
    for (let i = 0; i < frames.length; i += step) {
      const fr = frames[i];
      const v0 = fr.vor?.[0] ?? 50;
      const v1 = fr.vor?.[1] ?? 50;
      pts.push({
        t: fr.t,
        f: fr.f,
        vor0: v0,
        vor1: v1,
        delta: v0 - 50,
      });
    }

    const areaGen = d3
      .area<MomentumSample>()
      .x((d) => xScale(d.t))
      .y0(midY)
      .y1((d) => yScale(d.delta))
      .curve(d3.curveCatmullRom.alpha(0.5));

    const lineGen = d3
      .line<MomentumSample>()
      .x((d) => xScale(d.t))
      .y((d) => yScale(d.delta))
      .curve(d3.curveCatmullRom.alpha(0.5));

    // Select turnovers and progressive passes for stream markers
    const evts = (telemetry?.events || []).filter(
      (e) => e.type === "turnover" || (e.type === "pass" && e.xt_delta >= 0.004)
    );

    return {
      samples: pts,
      areaPath: areaGen(pts) || "",
      linePath: lineGen(pts) || "",
      keyEvents: evts,
    };
  }, [telemetry, xScale, yScale, midY]);

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * vbW;
    const clampedX = Math.max(padL, Math.min(vbW - padR, relX));
    const targetSec = xScale.invert(clampedX);
    onSeekToTime?.(targetSec);
  };

  const handleSvgMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (samples.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * vbW;
    const sec = xScale.invert(Math.max(padL, Math.min(vbW - padR, relX)));

    // Binary/nearest lookup
    const bisector = d3.bisector((d: MomentumSample) => d.t).center;
    const idx = bisector(samples, sec);
    setHoverSample(samples[idx] ?? null);
  };

  const playheadX = xScale(Math.max(0, Math.min(safeDur, currentTime)));
  const xTicks = useMemo(() => xScale.ticks(6), [xScale]);

  if (!telemetry || samples.length === 0) {
    return null;
  }

  return (
    <div className="rounded-2xl obsidian-card p-5 space-y-3">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-white/[0.08]">
        <div>
          <h3 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
            <Activity className="w-4 h-4 text-tv-cyan" />
            <span>{t("tel.d3_momentum_title")}</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            {t("tel.d3_momentum_desc")}
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs shrink-0">
          <span className="inline-flex items-center gap-1.5 text-tv-cyan font-semibold">
            <span className="w-2.5 h-2.5 rounded-sm bg-tv-cyan/80" />
            <span>{t("team_a")}</span>
          </span>
          <span className="inline-flex items-center gap-1.5 text-tv-coral font-semibold">
            <span className="w-2.5 h-2.5 rounded-sm bg-tv-coral/80" />
            <span>{t("team_b")}</span>
          </span>
        </div>
      </div>

      {/* Diverging Stream SVG (Strictly LTR Time Axis 0s -> end) */}
      <div dir="ltr" className="relative rounded-xl obsidian-well p-2 overflow-hidden">
        <svg
          viewBox={`0 0 ${vbW} ${vbH}`}
          onClick={handleSvgClick}
          onMouseMove={handleSvgMove}
          onMouseLeave={() => setHoverSample(null)}
          className="w-full h-auto block cursor-pointer select-none"
          role="img"
          aria-label={t("tel.d3_momentum_title")}
        >
          <defs>
            <clipPath id="mom-above-eq">
              <rect x={padL} y={0} width={vbW - padL - padR} height={midY} />
            </clipPath>
            <clipPath id="mom-below-eq">
              <rect
                x={padL}
                y={midY}
                width={vbW - padL - padR}
                height={vbH - midY}
              />
            </clipPath>
            <linearGradient id="mom-cyan-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#38BDF8" stopOpacity="0.06" />
            </linearGradient>
            <linearGradient id="mom-coral-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#FB7185" stopOpacity="0.06" />
              <stop offset="100%" stopColor="#FB7185" stopOpacity="0.6" />
            </linearGradient>
          </defs>

          {/* Horizontal Reference Guides (75% Sky, 50% Eq, 75% Coral) */}
          {[-25, 0, 25].map((dVal) => {
            const y = yScale(dVal);
            const label = dVal === 0 ? "50%" : "75%";
            const labelColor =
              dVal === 0 ? "#94A3B8" : dVal > 0 ? "#38BDF8" : "#FB7185";
            return (
              <g key={`grid-${dVal}`}>
                <line
                  x1={padL}
                  y1={y}
                  x2={vbW - padR}
                  y2={y}
                  stroke={
                    dVal === 0
                      ? "rgba(226,232,240,0.32)"
                      : "rgba(148,163,184,0.10)"
                  }
                  strokeDasharray={dVal === 0 ? undefined : "3 4"}
                  strokeWidth={dVal === 0 ? 1.2 : 1}
                />
                <text
                  x={padL - 6}
                  y={y + 3.5}
                  textAnchor="end"
                  fill={labelColor}
                  fontSize="9.5"
                  fontFamily="monospace"
                  fontWeight={dVal === 0 ? "500" : "700"}
                >
                  {label}
                </text>
              </g>
            );
          })}

          {/* Team A Positive Momentum Fill + Stroke */}
          <g clipPath="url(#mom-above-eq)">
            <path d={areaPath} fill="url(#mom-cyan-grad)" />
            <path
              d={linePath}
              fill="none"
              stroke="#38BDF8"
              strokeWidth="2"
            />
          </g>

          {/* Team B Negative Momentum Fill + Stroke */}
          <g clipPath="url(#mom-below-eq)">
            <path d={areaPath} fill="url(#mom-coral-grad)" />
            <path
              d={linePath}
              fill="none"
              stroke="#FB7185"
              strokeWidth="2"
            />
          </g>

          {/* Clickable Tactical Event Pins along Equilibrium Line */}
          {keyEvents.map((ev: MatchEvent, idx: number) => {
            const ex = xScale(ev.time_s);
            const isTurnover = ev.type === "turnover";
            const pinColor = isTurnover ? "#F59E0B" : "#34D399";
            return (
              <circle
                key={`pin-${ev.frame}-${idx}`}
                cx={ex}
                cy={midY}
                r={isTurnover ? 4.2 : 3.2}
                fill={pinColor}
                stroke="#050912"
                strokeWidth="1.3"
                onClick={(e) => {
                  e.stopPropagation();
                  onSeekToFrame?.(ev.frame);
                }}
                className="cursor-pointer hover:scale-125 transition-transform"
              />
            );
          })}

          {/* Hover Crosshair */}
          {hoverSample && (
            <line
              x1={xScale(hoverSample.t)}
              y1={padT}
              x2={xScale(hoverSample.t)}
              y2={vbH - padB}
              stroke="rgba(255,255,255,0.35)"
              strokeDasharray="2 2"
              strokeWidth="1"
            />
          )}

          {/* Live Synced Playhead */}
          <g transform={`translate(${playheadX}, 0)`}>
            <line
              x1={0}
              y1={padT - 4}
              x2={0}
              y2={vbH - padB}
              stroke="#FFFFFF"
              strokeWidth="1.8"
            />
            <circle cx={0} cy={padT - 4} r="3.5" fill="#FFFFFF" />
          </g>

          {/* X-Axis Time Ticks */}
          {xTicks.map((tick) => (
            <text
              key={`xtick-${tick}`}
              x={xScale(tick)}
              y={vbH - 7}
              textAnchor="middle"
              fill="#64748B"
              fontSize="10"
              fontFamily="monospace"
            >
              {tick}s
            </text>
          ))}
        </svg>

        {/* Localized Hover Readout Pill */}
        {hoverSample && (
          <div
            dir={dir}
            className="absolute top-2.5 end-2.5 px-3 py-1.5 rounded-lg obsidian-popover text-xs flex items-center gap-3 pointer-events-none shadow-lg"
          >
            <span className="inline-flex items-baseline gap-0.5 text-slate-400">
              <BidiNum>{hoverSample.t.toFixed(1)}</BidiNum>
              <span>{t("seconds")}</span>
            </span>
            <BidiNum className="font-bold text-tv-cyan">
              {hoverSample.vor0.toFixed(1)}%
            </BidiNum>
            <span className="text-slate-600">/</span>
            <BidiNum className="font-bold text-tv-coral">
              {hoverSample.vor1.toFixed(1)}%
            </BidiNum>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-500">
        <span>{t("d3.mom_click_seek")}</span>
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-tv-amber" />
            <span>{t("evt.turnover")}</span>
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-tv-mint" />
            <BidiNum className="text-tv-mint font-semibold">+xT</BidiNum>
            <span>{t("evt.pass")}</span>
          </span>
        </div>
      </div>
    </div>
  );
}
