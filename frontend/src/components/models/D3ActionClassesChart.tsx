"use client";

import React, { useMemo } from "react";
import * as d3 from "d3";
import { useLang } from "@/context/LangContext";
import {
  getActionClassLabel,
  getCategoryMeta,
} from "@/lib/actionSpottingHelpers";
import { BidiNum } from "@/components/ui/BidiNum";

interface D3ActionClassesChartProps {
  perClassTightApPct: Record<string, number>;
}

const CLASS_TO_CATEGORY: Record<string, string> = {
  Goal: "scoring",
  Penalty: "scoring",
  "Shots on target": "shot",
  "Shots off target": "shot",
  Corner: "set_piece",
  "Direct free-kick": "set_piece",
  "Indirect free-kick": "set_piece",
  "Throw-in": "set_piece",
  "Kick-off": "set_piece",
  Foul: "foul",
  "Yellow card": "card",
  "Red card": "card",
  "Yellow->red card": "card",
  Offside: "tactical",
  Substitution: "tactical",
  Clearance: "tactical",
  "Ball out of play": "flow",
};

export function D3ActionClassesChart({
  perClassTightApPct,
}: D3ActionClassesChartProps) {
  const { t } = useLang();

  const sorted = useMemo(
    () =>
      Object.entries(perClassTightApPct)
        .map(([label, val]) => ({
          label,
          ap: Number(val),
          category: CLASS_TO_CATEGORY[label] || "tactical",
        }))
        .sort((a, b) => b.ap - a.ap),
    [perClassTightApPct],
  );

  const widthScale = useMemo(
    () => d3.scaleLinear().domain([0, 100]).range([0, 100]),
    [],
  );

  return (
    <div className="space-y-2.5">
      {sorted.map((row) => {
        const meta = getCategoryMeta(row.category);
        const wPct = Math.max(2, widthScale(row.ap));

        return (
          <div key={row.label} className="grid grid-cols-12 items-center gap-3 text-xs">
            <div className="col-span-5 sm:col-span-4 flex items-center gap-2 min-w-0">
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: meta.color }}
              />
              <span className="truncate font-semibold text-slate-200">
                {getActionClassLabel(row.label, t)}
              </span>
            </div>

            <div dir="ltr" className="col-span-5 sm:col-span-6 h-2.5 rounded-full bg-white/[0.06] overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${wPct}%`,
                  backgroundColor: meta.color,
                }}
              />
            </div>

            <div className="col-span-2 text-end">
              <BidiNum className="font-bold text-white">
                {`${row.ap.toFixed(1)}%`}
              </BidiNum>
            </div>
          </div>
        );
      })}
    </div>
  );
}
