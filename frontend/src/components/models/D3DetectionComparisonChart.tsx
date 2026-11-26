"use client";

import React, { useMemo, useState } from "react";
import * as d3 from "d3";
import { Crosshair } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { ModelsSuiteResponse } from "@/lib/types";
import { BidiNum } from "@/components/ui/BidiNum";

interface D3DetectionComparisonChartProps {
  tuned: NonNullable<ModelsSuiteResponse["metrics"]["model_1_players_detector"]>;
  baseline: NonNullable<ModelsSuiteResponse["metrics"]["model_1_coco_baseline"]>;
}

const CLASSES = ["ball", "goalkeeper", "referee", "player"] as const;

export function D3DetectionComparisonChart({
  tuned,
  baseline,
}: D3DetectionComparisonChartProps) {
  const { t, lang } = useLang();
  const [metric, setMetric] = useState<"f1" | "precision" | "recall">("f1");

  const classLabel = (cls: (typeof CLASSES)[number]) => {
    if (lang === "ar") {
      if (cls === "ball") return "الكرة";
      if (cls === "goalkeeper") return "حارس المرمى";
      if (cls === "referee") return "الحكم";
      return "اللاعبون";
    }
    return cls.charAt(0).toUpperCase() + cls.slice(1);
  };

  const rows = useMemo(
    () =>
      CLASSES.map((cls) => ({
        cls,
        label: classLabel(cls),
        tunedVal: Number(((tuned.per_class_calibrated_iou50[cls]?.[metric] ?? 0) * 100).toFixed(1)),
        cocoVal: Number(((baseline.per_class_iou50[cls]?.[metric] ?? 0) * 100).toFixed(1)),
        map50: Number(((tuned.per_class_map[cls]?.mAP50 ?? 0) * 100).toFixed(1)),
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [baseline, lang, metric, tuned],
  );

  const vbW = 540;
  const vbH = 220;
  const padL = 38;
  const padR = 16;
  const padT = 18;
  const padB = 34;

  const xBand = useMemo(
    () =>
      d3
        .scaleBand<string>()
        .domain(rows.map((r) => r.cls))
        .range([padL, vbW - padR])
        .padding(0.28),
    [rows],
  );

  const yScale = useMemo(
    () => d3.scaleLinear().domain([0, 100]).range([vbH - padB, padT]),
    [],
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3 text-xs font-semibold">
          <span className="inline-flex items-center gap-1.5 text-tv-cyan">
            <span className="w-2.5 h-2.5 rounded-sm bg-tv-cyan" />
            <span>{t("models.metric_tuned")}</span>
          </span>
          <span className="inline-flex items-center gap-1.5 text-slate-400">
            <span className="w-2.5 h-2.5 rounded-sm bg-slate-600" />
            <span>{t("models.metric_coco")}</span>
          </span>
        </div>

        <div className="flex items-center p-0.5 rounded-lg obsidian-well border border-white/[0.08] text-[11px]">
          {(["f1", "precision", "recall"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMetric(m)}
              className={`px-2.5 py-1 rounded-md font-bold uppercase transition-colors cursor-pointer ${
                metric === m
                  ? "bg-tv-cyan text-slate-950"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div dir="ltr" className="w-full overflow-x-auto">
        <svg viewBox={`0 0 ${vbW} ${vbH}`} className="w-full min-w-[380px] h-auto block">
          {[0, 25, 50, 75, 100].map((tick) => {
            const y = yScale(tick);
            return (
              <g key={tick}>
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
                  fontSize={9.5}
                  fontFamily="var(--font-jetbrains), monospace"
                >
                  {tick}%
                </text>
              </g>
            );
          })}

          {rows.map((r) => {
            const gx = xBand(r.cls) ?? padL;
            const bw = xBand.bandwidth();
            const subW = bw * 0.44;
            const yTuned = yScale(r.tunedVal);
            const hTuned = vbH - padB - yTuned;
            const yCoco = yScale(r.cocoVal);
            const hCoco = vbH - padB - yCoco;

            return (
              <g key={r.cls}>
                {/* Tuned Model 1 Bar */}
                <rect
                  x={gx}
                  y={yTuned}
                  width={subW}
                  height={Math.max(2, hTuned)}
                  rx={4}
                  fill="#38BDF8"
                />
                <text
                  x={gx + subW / 2}
                  y={yTuned - 4}
                  textAnchor="middle"
                  fill="#38BDF8"
                  fontSize={9.5}
                  fontWeight={700}
                  fontFamily="var(--font-jetbrains), monospace"
                >
                  {r.tunedVal}%
                </text>

                {/* COCO Zero-Shot Bar */}
                <rect
                  x={gx + subW + 4}
                  y={r.cocoVal > 0 ? yCoco : vbH - padB - 3}
                  width={subW}
                  height={r.cocoVal > 0 ? hCoco : 3}
                  rx={4}
                  fill={r.cocoVal > 0 ? "rgba(148,163,184,0.55)" : "rgba(251,113,133,0.55)"}
                />
                <text
                  x={gx + subW + 4 + subW / 2}
                  y={(r.cocoVal > 0 ? yCoco : vbH - padB - 4) - 4}
                  textAnchor="middle"
                  fill={r.cocoVal > 0 ? "#94A3B8" : "#FB7185"}
                  fontSize={9.5}
                  fontWeight={700}
                  fontFamily="var(--font-jetbrains), monospace"
                >
                  {r.cocoVal}%
                </text>

                <text
                  x={gx + bw / 2}
                  y={vbH - 12}
                  textAnchor="middle"
                  fill="#CBD5E1"
                  fontSize={10.5}
                  fontWeight={600}
                >
                  {r.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Per-class mAP@50 callout row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-white/[0.06]">
        {rows.map((r) => (
          <div
            key={`map-${r.cls}`}
            className="p-2.5 rounded-xl obsidian-well flex items-center justify-between text-xs"
          >
            <span className="text-slate-300 font-semibold flex items-center gap-1.5">
              <Crosshair className="w-3 h-3 text-tv-cyan shrink-0" />
              <span>{r.label}</span>
            </span>
            <BidiNum className="font-bold text-white">{`${r.map50}%`}</BidiNum>
          </div>
        ))}
      </div>
    </div>
  );
}
