"use client";

import React, { useMemo } from "react";
import { Compass, Zap, Shield, Flame, Lightbulb, Sparkles } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { Telemetry } from "@/lib/types";
import { generateCoachBrief } from "@/lib/coachBriefEngine";
import { BidiNum } from "@/components/ui/BidiNum";

interface CoachBriefProps {
  telemetry: Telemetry | null;
}

export function CoachBrief({ telemetry }: CoachBriefProps) {
  const { t, lang } = useLang();

  const insights = useMemo(() => {
    if (!telemetry) return [];
    return generateCoachBrief(telemetry, lang);
  }, [telemetry, lang]);

  if (!telemetry || insights.length === 0) return null;

  const iconMap = {
    territorial: Compass,
    progression: Zap,
    defensive: Shield,
    physical: Flame,
    broadcast: Sparkles,
  };

  const accentMap = {
    cyan: {
      badge: "bg-tv-cyan/10 border-tv-cyan/25 text-tv-cyan",
      bar: "border-s-tv-cyan",
      icon: "text-tv-cyan",
    },
    coral: {
      badge: "bg-tv-coral/10 border-tv-coral/25 text-tv-coral",
      bar: "border-s-tv-coral",
      icon: "text-tv-coral",
    },
    mint: {
      badge: "bg-tv-mint/10 border-tv-mint/25 text-tv-mint",
      bar: "border-s-tv-mint",
      icon: "text-tv-mint",
    },
    amber: {
      badge: "bg-tv-amber/10 border-tv-amber/25 text-tv-amber",
      bar: "border-s-tv-amber",
      icon: "text-tv-amber",
    },
  };

  return (
    <div className="rounded-2xl obsidian-card p-5 sm:p-6 space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-white/[0.08]">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-tv-cyan/10 border border-tv-cyan/20 text-tv-cyan">
            <Compass className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">
              {t("coach.title")}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {t("coach.subtitle")}
            </p>
          </div>
        </div>

        <span className="px-3 py-1 rounded-xl obsidian-well text-xs font-semibold text-slate-300">
          {insights.length > 4 ? t("coach.insights_count_5") : t("coach.insights_count")}
        </span>
      </div>

      {/* 4 Calm Editorial Insight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {insights.map((insight) => {
          const Icon = iconMap[insight.category] || Compass;
          const style = accentMap[insight.badgeColor] || accentMap.cyan;

          return (
            <div
              key={insight.id}
              className={`rounded-xl obsidian-well p-4 sm:p-5 border-s-2 ${style.bar} flex flex-col justify-between gap-4 hover:border-white/15 transition-colors`}
            >
              <div>
                {/* Top Title & Tactical Tag */}
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <Icon className={`w-4 h-4 shrink-0 ${style.icon}`} />
                    <h4 className="text-sm font-bold text-white truncate">
                      {insight.title}
                    </h4>
                  </div>
                  <span
                    className={`px-2.5 py-0.5 rounded-md text-[11px] font-semibold border shrink-0 ${style.badge}`}
                  >
                    {insight.badge}
                  </span>
                </div>

                {/* Strictly Separated Bidi Metric + Unit + Qualifier Chip */}
                <div className="flex flex-wrap items-baseline gap-2 mb-3">
                  <div className="inline-flex items-baseline gap-1.5">
                    <BidiNum className="text-xl font-extrabold text-white">
                      {insight.metric}
                    </BidiNum>
                    {insight.metricUnit && (
                      <span className="text-xs font-semibold text-slate-400">
                        {insight.metricUnit}
                      </span>
                    )}
                  </div>
                  {insight.metricSub && (
                    <span className="px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.07] text-xs text-slate-300 font-medium">
                      {insight.metricSub}
                    </span>
                  )}
                </div>

                {/* Editorial Analysis Prose */}
                <p className="text-xs sm:text-sm text-slate-300 leading-[1.7]">
                  {insight.analysis}
                </p>
              </div>

              {/* Coach Directive Callout */}
              <div className="pt-3 border-t border-white/[0.06] flex items-start gap-2 text-xs text-slate-400">
                <Lightbulb className="w-4 h-4 text-tv-amber shrink-0 mt-0.5" />
                <p className="leading-relaxed">
                  <strong className="text-slate-200 font-semibold me-1">
                    {t("coach.takeaway")}
                  </strong>
                  <span>{insight.coachTakeaway}</span>
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
