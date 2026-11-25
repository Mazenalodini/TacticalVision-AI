"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertCircle } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { MatchCard as MatchCardType } from "@/lib/types";
import { MatchCard } from "./MatchCard";
import { MatchGridSkeleton } from "@/components/ui/Skeleton";

interface MatchGridProps {
  matches: MatchCardType[];
  isLoading: boolean;
  onResetFilters?: () => void;
  onDeleteMatch?: (matchId: string) => void;
}

const cardVariants = {
  hidden: { opacity: 0, y: 18 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.32,
      ease: [0.22, 1, 0.36, 1],
    },
  },
  exit: {
    opacity: 0,
    scale: 0.94,
    y: 8,
    transition: { duration: 0.22 },
  },
};

export function MatchGrid({
  matches,
  isLoading,
  onResetFilters,
  onDeleteMatch,
}: MatchGridProps) {
  const { t } = useLang();

  if (isLoading) {
    return <MatchGridSkeleton />;
  }

  if (matches.length === 0) {
    return (
      <div className="rounded-2xl obsidian-card p-12 text-center">
        <div className="w-12 h-12 rounded-2xl bg-tv-amber/10 border border-tv-amber/25 flex items-center justify-center mx-auto mb-4 text-tv-amber">
          <AlertCircle className="w-6 h-6" />
        </div>
        <h4 className="text-base font-bold text-white mb-1.5">
          {t("hub.no_matches")}
        </h4>
        <p className="text-xs sm:text-sm text-slate-400 mb-5 max-w-md mx-auto leading-relaxed">
          {t("hub.no_matches_sub")}
        </p>
        {onResetFilters && (
          <button
            type="button"
            onClick={onResetFilters}
            className="px-5 py-2.5 rounded-xl bg-white/[0.08] hover:bg-white/[0.14] border border-white/10 text-xs font-semibold text-white transition-colors cursor-pointer"
          >
            {t("hub.reset_filters")}
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
      <AnimatePresence mode="popLayout">
        {matches.map((match) => (
          <motion.div
            key={match.id}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            layout
            className="h-full"
          >
            <MatchCard match={match} onDelete={onDeleteMatch} />
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
