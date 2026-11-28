import type { ActionCategory, ActionSpottingDensityBin, ActionSpottingEvent } from "./types";

export const ACTION_CATEGORY_ORDER: ActionCategory[] = [
  "scoring",
  "shot",
  "set_piece",
  "foul",
  "card",
  "tactical",
  "flow",
];

export const ACTION_CATEGORY_META: Record<
  ActionCategory,
  { color: string; bgClass: string; borderClass: string; textClass: string }
> = {
  scoring: {
    color: "#34D399",
    bgClass: "bg-tv-mint/15",
    borderClass: "border-tv-mint/40",
    textClass: "text-tv-mint",
  },
  shot: {
    color: "#38BDF8",
    bgClass: "bg-tv-cyan/15",
    borderClass: "border-tv-cyan/40",
    textClass: "text-tv-cyan",
  },
  set_piece: {
    color: "#A78BFA",
    bgClass: "bg-violet-400/15",
    borderClass: "border-violet-400/40",
    textClass: "text-violet-300",
  },
  foul: {
    color: "#FB923C",
    bgClass: "bg-orange-400/15",
    borderClass: "border-orange-400/40",
    textClass: "text-orange-300",
  },
  card: {
    color: "#FB7185",
    bgClass: "bg-tv-coral/15",
    borderClass: "border-tv-coral/40",
    textClass: "text-tv-coral",
  },
  tactical: {
    color: "#FBBF24",
    bgClass: "bg-tv-amber/15",
    borderClass: "border-tv-amber/40",
    textClass: "text-tv-amber",
  },
  flow: {
    color: "#94A3B8",
    bgClass: "bg-slate-400/15",
    borderClass: "border-slate-400/35",
    textClass: "text-slate-300",
  },
};

export function getCategoryMeta(cat: string) {
  return (
    ACTION_CATEGORY_META[(cat as ActionCategory)] ?? ACTION_CATEGORY_META.flow
  );
}

export function getActionClassLabel(
  label: string,
  t: (key: string) => string,
): string {
  const key = `as.cls.${label}`;
  const translated = t(key);
  return translated === key ? label : translated;
}

export function getActionCategoryLabel(
  category: string,
  t: (key: string) => string,
): string {
  const key = `as.cat.${category}`;
  const translated = t(key);
  return translated === key ? category : translated;
}

/**
 * Recomputes 5-minute intensity bins from filtered events so sliding
 * the live confidence slider updates both the timeline and density wave smoothly.
 */
export function buildFilteredDensityBins(
  templateBins: ActionSpottingDensityBin[],
  events: ActionSpottingEvent[],
  halfFilter: "all" | "1" | "2",
): ActionSpottingDensityBin[] {
  if (!templateBins || templateBins.length === 0) return [];
  const hasMultipleHalves = templateBins.some((b) => b.label.startsWith("H"));

  return templateBins
    .filter((bin) => {
      if (halfFilter === "all" || !hasMultipleHalves) return true;
      return bin.label.startsWith(`H${halfFilter} `);
    })
    .map((bin) => {
      const binHalf = bin.label.startsWith("H2 ") ? 2 : 1;
      const inBin = events.filter((e) => {
        if (hasMultipleHalves && e.half !== binHalf) return false;
        return e.time_s >= bin.start_s && e.time_s < bin.end_s;
      });
      return {
        ...bin,
        count: inBin.length,
        high_impact_count: inBin.filter(
          (e) => e.category === "scoring" || e.category === "card" || e.category === "shot",
        ).length,
      };
    });
}
