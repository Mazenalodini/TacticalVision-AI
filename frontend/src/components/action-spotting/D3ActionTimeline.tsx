"use client";

import React, { useMemo, useState } from "react";
import * as d3 from "d3";
import { Target, Layers } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type {
  ActionSpottingEvent,
  ActionSpottingGroundTruthEvent,
  ActionSpottingHalfSummary,
} from "@/lib/types";
import {
  ACTION_CATEGORY_ORDER,
  getActionCategoryLabel,
  getActionClassLabel,
  getCategoryMeta,
} from "@/lib/actionSpottingHelpers";
import { BidiNum } from "@/components/ui/BidiNum";

interface D3ActionTimelineProps {
  events: ActionSpottingEvent[];
  halves: ActionSpottingHalfSummary[];
  groundTruthEvents?: ActionSpottingGroundTruthEvent[] | null;
  nmsWindowSec: number;
  selectedEventKey: string | null;
  onSelectEvent: (key: string | null) => void;
  showGtOverlay: boolean;
  onToggleGtOverlay: () => void;
}

export function eventUniqueKey(e: ActionSpottingEvent, idx: number): string {
  return `${e.half}-${e.class_id}-${e.frame_idx}-${idx}`;
}

export function D3ActionTimeline({
  events,
  halves,
  groundTruthEvents,
  nmsWindowSec,
  selectedEventKey,
  onSelectEvent,
  showGtOverlay,
  onToggleGtOverlay,
}: D3ActionTimelineProps) {
  const { t } = useLang();
  const [hoveredInfo, setHoveredInfo] = useState<{
    xPct: number;
    yPct: number;
    label: string;
    time: string;
    confPct?: number;
    half: number;
    isGt?: boolean;
    categoryColor: string;
  } | null>(null);

  const hasGt = Boolean(groundTruthEvents && groundTruthEvents.length > 0);
  const isTwoHalves = halves.length > 1;
  const h1Dur = halves.find((h) => h.half === 1)?.duration_s ?? 2700;
  const totalDomainSec = useMemo(() => {
    if (halves.length === 0) return 2700;
    if (isTwoHalves) {
      const h2Dur = halves.find((h) => h.half === 2)?.duration_s ?? 2700;
      return Math.max(60, h1Dur + h2Dur);
    }
    return Math.max(60, halves[0].duration_s);
  }, [halves, h1Dur, isTwoHalves]);

  const toGlobalSec = (half: number, timeS: number) =>
    isTwoHalves && half === 2 ? h1Dur + timeS : timeS;

  const vbW = 860;
  const laneH = 34;
  const topPad = 18;
  const gtExtraH = hasGt && showGtOverlay ? 36 : 0;
  const bottomPad = 28;
  const vbH = topPad + ACTION_CATEGORY_ORDER.length * laneH + gtExtraH + bottomPad;

  const xScale = useMemo(
    () => d3.scaleLinear().domain([0, totalDomainSec]).range([16, vbW - 16]),
    [totalDomainSec],
  );

  const ticks = useMemo(() => {
    const mins = Math.ceil(totalDomainSec / 60);
    const stepMin = mins > 60 ? 10 : 5;
    const arr: { sec: number; label: string }[] = [];
    for (let m = 0; m <= mins; m += stepMin) {
      const sec = m * 60;
      if (sec <= totalDomainSec + 10) {
        arr.push({ sec: Math.min(sec, totalDomainSec), label: `${m}'` });
      }
    }
    return arr;
  }, [totalDomainSec]);

  return (
    <div className="rounded-2xl obsidian-card p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h3 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-tv-cyan shrink-0" />
            <span>{t("as.timeline_title")}</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">{t("as.timeline_desc")}</p>
        </div>

        {hasGt && (
          <button
            type="button"
            onClick={onToggleGtOverlay}
            className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-colors cursor-pointer self-start sm:self-auto ${
              showGtOverlay
                ? "bg-tv-amber/15 border-tv-amber/40 text-tv-amber"
                : "obsidian-well text-slate-300 hover:text-white"
            }`}
          >
            <Target className="w-3.5 h-3.5 shrink-0" />
            <span>{t("as.gt_toggle")}</span>
          </button>
        )}
      </div>

      {/* Hybrid React Swimlane Labels + Strictly LTR D3 SVG Timeline */}
      <div className="relative flex items-stretch gap-2 sm:gap-3">
        {/* Domain Swimlane Labels (Auto-flips cleanly in EN/AR) */}
        <div
          className="w-24 sm:w-32 shrink-0 flex flex-col justify-start"
          style={{ paddingTop: `${(topPad / vbH) * 100}%` }}
        >
          {ACTION_CATEGORY_ORDER.map((cat) => {
            const meta = getCategoryMeta(cat);
            return (
              <div
                key={`lane-lbl-${cat}`}
                className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-300 truncate"
                style={{ height: `${laneH}px` }}
                title={getActionCategoryLabel(cat, t)}
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: meta.color }}
                />
                <span className="truncate">{getActionCategoryLabel(cat, t)}</span>
              </div>
            );
          })}
          {hasGt && showGtOverlay && (
            <div
              className="flex items-center gap-1.5 text-[11px] font-bold text-tv-amber truncate border-t border-white/[0.08] mt-1 pt-1"
              style={{ height: "32px" }}
            >
              <span className="w-2 h-2 rotate-45 bg-tv-amber shrink-0" />
              <span className="truncate">{t("as.gt_ref")}</span>
            </div>
          )}
        </div>

        {/* Strictly LTR D3 Canvas */}
        <div dir="ltr" className="relative flex-1 min-w-0 overflow-x-auto">
          <svg
            viewBox={`0 0 ${vbW} ${vbH}`}
            className="w-full min-w-[540px] h-auto select-none block"
            style={{ minHeight: `${vbH}px` }}
          >
            {/* Background subtle lane bands */}
            {ACTION_CATEGORY_ORDER.map((cat, idx) => {
              const y = topPad + idx * laneH;
              return (
                <rect
                  key={`band-${cat}`}
                  x={12}
                  y={y + 2}
                  width={vbW - 24}
                  height={laneH - 4}
                  rx={6}
                  fill={idx % 2 === 0 ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0.012)"}
                />
              );
            })}

            {/* Vertical gridlines & minute ticks */}
            {ticks.map((tk) => {
              const x = xScale(tk.sec);
              return (
                <g key={`tick-${tk.sec}`}>
                  <line
                    x1={x}
                    y1={topPad}
                    x2={x}
                    y2={vbH - bottomPad}
                    stroke="rgba(148,163,184,0.12)"
                    strokeDasharray="3 4"
                  />
                  <text
                    x={x}
                    y={vbH - 8}
                    textAnchor="middle"
                    fill="rgba(148,163,184,0.68)"
                    fontSize={10}
                    fontFamily="var(--font-jetbrains), monospace"
                  >
                    {tk.label}
                  </text>
                </g>
              );
            })}

            {/* Half-time divider if showing Both Halves */}
            {isTwoHalves && (
              <g>
                <line
                  x1={xScale(h1Dur)}
                  y1={topPad - 6}
                  x2={xScale(h1Dur)}
                  y2={vbH - bottomPad + 4}
                  stroke="#38BDF8"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
                <text
                  x={xScale(h1Dur)}
                  y={11}
                  textAnchor="middle"
                  fill="#38BDF8"
                  fontSize={10}
                  fontWeight={700}
                  fontFamily="var(--font-jetbrains), monospace"
                >
                  HT
                </text>
              </g>
            )}

            {/* Ground Truth Alignment Rail (if available & enabled) */}
            {hasGt && showGtOverlay && groundTruthEvents && (
              <g>
                <line
                  x1={14}
                  y1={topPad + ACTION_CATEGORY_ORDER.length * laneH + 18}
                  x2={vbW - 14}
                  y2={topPad + ACTION_CATEGORY_ORDER.length * laneH + 18}
                  stroke="rgba(245,158,11,0.3)"
                  strokeWidth={1.2}
                />
                {groundTruthEvents.map((gt, gIdx) => {
                  const gHalf = gt.half ?? 1;
                  const gx = xScale(toGlobalSec(gHalf, gt.time_s));
                  const gy = topPad + ACTION_CATEGORY_ORDER.length * laneH + 18;
                  const matched = events.some(
                    (ev) =>
                      ev.half === gHalf &&
                      ev.label === gt.label &&
                      Math.abs(ev.time_s - gt.time_s) <= nmsWindowSec,
                  );
                  return (
                    <polygon
                      key={`gt-${gHalf}-${gIdx}`}
                      points={`${gx},${gy - 5} ${gx + 5},${gy} ${gx},${gy + 5} ${gx - 5},${gy}`}
                      fill={matched ? "#34D399" : "#F59E0B"}
                      fillOpacity={matched ? 0.9 : 0.6}
                      stroke="#060B10"
                      strokeWidth={1}
                      className="cursor-pointer"
                      onMouseEnter={() => {
                        const mins = Math.floor(gt.time_s / 60);
                        const secs = Math.floor(gt.time_s % 60);
                        setHoveredInfo({
                          xPct: (gx / vbW) * 100,
                          yPct: (gy / vbH) * 100,
                          label: getActionClassLabel(gt.label, t),
                          time: `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`,
                          half: gHalf,
                          isGt: true,
                          categoryColor: matched ? "#34D399" : "#F59E0B",
                        });
                      }}
                      onMouseLeave={() => setHoveredInfo(null)}
                    />
                  );
                })}
              </g>
            )}

            {/* Predicted Event Pins */}
            {events.map((ev, idx) => {
              const laneIdx = Math.max(
                0,
                ACTION_CATEGORY_ORDER.indexOf(ev.category as (typeof ACTION_CATEGORY_ORDER)[number]),
              );
              const meta = getCategoryMeta(ev.category);
              const cx = xScale(toGlobalSec(ev.half, ev.time_s));
              const cy = topPad + laneIdx * laneH + laneH / 2;
              const r = 4.2 + ev.confidence * 3.8;
              const key = eventUniqueKey(ev, idx);
              const isSelected = selectedEventKey === key;

              return (
                <g
                  key={key}
                  className="cursor-pointer"
                  onClick={() => onSelectEvent(isSelected ? null : key)}
                  onMouseEnter={() =>
                    setHoveredInfo({
                      xPct: (cx / vbW) * 100,
                      yPct: (cy / vbH) * 100,
                      label: getActionClassLabel(ev.label, t),
                      time: ev.minute_str,
                      confPct: Math.round(ev.confidence * 100),
                      half: ev.half,
                      categoryColor: meta.color,
                    })
                  }
                  onMouseLeave={() => setHoveredInfo(null)}
                >
                  {isSelected && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={r + 5}
                      fill="none"
                      stroke="#FFFFFF"
                      strokeWidth={1.6}
                      strokeDasharray="3 2"
                    />
                  )}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={r}
                    fill={meta.color}
                    fillOpacity={isSelected ? 1 : 0.84}
                    stroke={isSelected ? "#FFFFFF" : "#060B10"}
                    strokeWidth={isSelected ? 1.8 : 1.1}
                  />
                </g>
              );
            })}
          </svg>

          {/* Floating Bilingual Hover Tooltip */}
          {hoveredInfo && (
            <div
              className="pointer-events-none absolute z-20 px-3 py-2 rounded-xl obsidian-popover border border-white/15 shadow-xl text-xs whitespace-nowrap -translate-x-1/2 -translate-y-full"
              style={{
                left: `${Math.max(12, Math.min(88, hoveredInfo.xPct))}%`,
                top: `${Math.max(18, hoveredInfo.yPct - 4)}%`,
              }}
            >
              <div className="flex items-center gap-1.5 font-bold text-white">
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: hoveredInfo.categoryColor }}
                />
                <span>{hoveredInfo.label}</span>
                {hoveredInfo.isGt && (
                  <span className="px-1.5 py-0.5 rounded bg-tv-amber/20 text-tv-amber text-[10px]">
                    {t("as.gt_ref")}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 mt-1 text-[11px] text-slate-300">
                <BidiNum className="text-tv-cyan font-semibold">
                  {`H${hoveredInfo.half} · ${hoveredInfo.time}`}
                </BidiNum>
                {hoveredInfo.confPct != null && (
                  <BidiNum className="text-tv-mint font-bold">
                    {`${hoveredInfo.confPct}%`}
                  </BidiNum>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
