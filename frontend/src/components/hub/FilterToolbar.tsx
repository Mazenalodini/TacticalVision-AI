"use client";

import React from "react";
import { motion } from "framer-motion";
import { Search, ArrowUpDown, X, Layers } from "lucide-react";
import { useLang } from "@/context/LangContext";
import { formatMatchCount } from "@/lib/format";

export type SortOption = "default" | "speed" | "passes" | "events" | "duration";
export type FilterType = "all" | "demo" | "user";

interface FilterToolbarProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  sortOption: SortOption;
  onSortChange: (opt: SortOption) => void;
  filterType: FilterType;
  onFilterChange: (type: FilterType) => void;
  totalCount: number;
}

export function FilterToolbar({
  searchQuery,
  onSearchChange,
  sortOption,
  onSortChange,
  filterType,
  onFilterChange,
  totalCount,
}: FilterToolbarProps) {
  const { t, lang } = useLang();

  return (
    <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3.5 mb-6 p-3.5 rounded-2xl obsidian-card">
      {/* Search Input */}
      <div className="relative flex-1">
        <Search className="w-4 h-4 text-slate-400 absolute top-1/2 -translate-y-1/2 start-3.5 pointer-events-none" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder={t("hub.search_placeholder")}
          className="w-full ps-10 pe-9 py-2 rounded-xl obsidian-well text-xs sm:text-sm text-white placeholder-slate-500 focus:outline-none focus:border-tv-cyan/50 transition-all"
        />
        {searchQuery && (
          <button
            type="button"
            onClick={() => onSearchChange("")}
            className="absolute top-1/2 -translate-y-1/2 end-3 text-slate-400 hover:text-white cursor-pointer"
            aria-label="Clear search"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Filter & Sort Controls */}
      <div className="flex flex-wrap items-center justify-between sm:justify-end gap-2.5">
        {/* Animated Segmented Filter Pills */}
        <div className="flex items-center p-1 rounded-xl obsidian-well">
          {(["all", "demo", "user"] as FilterType[]).map((type) => {
            const isActive = filterType === type;
            const labelKey =
              type === "all"
                ? "hub.filter_all"
                : type === "demo"
                ? "hub.filter_demo"
                : "hub.filter_user";
            return (
              <button
                key={type}
                type="button"
                onClick={() => onFilterChange(type)}
                className={`relative px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                  isActive ? "text-white font-bold" : "text-slate-400 hover:text-white"
                }`}
              >
                {isActive && (
                  <motion.span
                    layoutId="hub-filter-segmented"
                    transition={{ type: "spring", stiffness: 460, damping: 34 }}
                    className="absolute inset-0 rounded-lg editorial-tab-active"
                  >
                    <span className="absolute bottom-0 inset-x-2.5 h-[2px] rounded-full bg-tv-cyan" />
                  </motion.span>
                )}
                <span className="relative z-10">{t(labelKey)}</span>
              </button>
            );
          })}
        </div>

        {/* Sort Dropdown */}
        <div className="flex items-center gap-2 px-3 py-2 rounded-xl obsidian-well">
          <ArrowUpDown className="w-3.5 h-3.5 text-tv-cyan shrink-0" />
          <select
            value={sortOption}
            onChange={(e) => onSortChange(e.target.value as SortOption)}
            aria-label={t("hub.sort_by")}
            className="bg-transparent text-xs font-medium text-slate-200 focus:outline-none cursor-pointer"
          >
            <option value="default" className="bg-[#0A101C] text-slate-200">
              {t("hub.sort_default")}
            </option>
            <option value="speed" className="bg-[#0A101C] text-slate-200">
              {t("hub.sort_speed")}
            </option>
            <option value="passes" className="bg-[#0A101C] text-slate-200">
              {t("hub.sort_passes")}
            </option>
            <option value="events" className="bg-[#0A101C] text-slate-200">
              {t("hub.sort_events")}
            </option>
            <option value="duration" className="bg-[#0A101C] text-slate-200">
              {t("hub.sort_duration")}
            </option>
          </select>
        </div>

        {/* Grammatical Count Badge */}
        <div className="hidden sm:inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/[0.03] border border-white/[0.07] text-xs font-medium text-slate-300">
          <Layers className="w-3.5 h-3.5 text-tv-cyan" />
          <span>{formatMatchCount(totalCount, lang)}</span>
        </div>
      </div>
    </div>
  );
}
