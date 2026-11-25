"use client";

import React, { useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Activity,
  Trash2,
  Loader2,
  ChevronRight,
  ChevronLeft,
  ShieldAlert,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { MatchCard as MatchCardType } from "@/lib/types";
import { getThumbnailUrl, deleteMatch } from "@/lib/api";
import { formatResolutionBadge, resolveMatchDisplay } from "@/lib/format";
import { BidiNum } from "@/components/ui/BidiNum";

interface MatchCardProps {
  match: MatchCardType;
  onDelete?: (matchId: string) => void;
}

export function MatchCard({ match, onDelete }: MatchCardProps) {
  const { t, lang, isRTL } = useLang();
  const [imgError, setImgError] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [spotlight, setSpotlight] = useState<{ x: number; y: number; active: boolean }>({
    x: 0,
    y: 0,
    active: false,
  });

  const ChevronIcon = isRTL ? ChevronLeft : ChevronRight;
  const display = resolveMatchDisplay(match, t, lang);

  // Metrics
  const posA = match.possession_pct?.team_0 ?? 50.0;
  const posB = match.possession_pct?.team_1 ?? 50.0;
  const ctrlA = match.pitch_control_pct?.team_0 ?? 50.0;
  const ctrlB = match.pitch_control_pct?.team_1 ?? 50.0;
  const passesA = match.passes?.team_0 ?? 0;
  const passesB = match.passes?.team_1 ?? 0;

  const thumbnailUrl = getThumbnailUrl(match.id);

  const handleMouseMove = (e: React.MouseEvent<HTMLAnchorElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setSpotlight({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      active: true,
    });
  };

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setShowConfirm(true);
  };

  const handleConfirmDelete = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDeleting(true);
    setDeleteError(null);

    try {
      await deleteMatch(match.id);
      setIsDeleting(false);
      setShowConfirm(false);
      onDelete?.(match.id);
    } catch (err: any) {
      setIsDeleting(false);
      setDeleteError(err?.message || "Delete failed");
    }
  };

  const handleCancelDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setShowConfirm(false);
    setDeleteError(null);
  };

  return (
    <>
      <Link
        href={`/match/${match.id}`}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setSpotlight((prev) => ({ ...prev, active: false }))}
        className="group relative rounded-2xl obsidian-card obsidian-card-interactive overflow-hidden flex flex-col h-full"
      >
        {/* Tactile Radial Cursor Spotlight */}
        {spotlight.active && (
          <div
            className="pointer-events-none absolute inset-0 z-0 transition-opacity duration-300"
            style={{
              background: `radial-gradient(380px circle at ${spotlight.x}px ${spotlight.y}px, rgba(30, 200, 255, 0.08), transparent 70%)`,
            }}
          />
        )}

        {/* ── 16:9 Broadcast Thumbnail Header ── */}
        <div className="relative aspect-video w-full overflow-hidden bg-[#050912] z-10">
          {!imgError ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={thumbnailUrl}
              alt={display.title}
              onError={() => setImgError(true)}
              className="w-full h-full object-cover object-center group-hover:scale-[1.04] transition-transform duration-500 ease-out"
              loading="lazy"
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-[#0B1526] to-[#050912] p-4 text-center">
              <Activity className="w-8 h-8 text-tv-cyan/50 mb-2" />
              <span className="text-xs font-semibold text-slate-400">
                {display.badge}
              </span>
            </div>
          )}

          {/* Subtle Studio Vignette */}
          <div className="absolute inset-0 bg-gradient-to-t from-[#0A101C] via-transparent to-black/55 pointer-events-none" />

          {/* Top Spec & Duration Badges (100% Bidi-Isolated Numbers) */}
          <div className="absolute top-3 inset-x-3 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 pointer-events-none">
              <span className="inline-flex items-baseline gap-1 px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-md border border-white/10 text-xs text-slate-200 font-semibold">
                <BidiNum>{formatResolutionBadge(match.resolution)}</BidiNum>
                <span className="text-slate-500">·</span>
                <BidiNum>{Math.round(match.fps || 25)}</BidiNum>
                <span className="text-[11px] text-slate-400">{t("card.fps")}</span>
              </span>

              {match.is_user_uploaded && (
                <span className="px-2.5 py-1 rounded-lg bg-tv-cyan/15 backdrop-blur-md border border-tv-cyan/30 text-xs font-semibold text-tv-cyan">
                  {t("card.badge_custom")}
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              <span className="inline-flex items-baseline gap-0.5 px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-md border border-white/10 text-xs text-tv-cyan font-bold pointer-events-none">
                <BidiNum>{(match.duration_s ?? 0).toFixed(1)}</BidiNum>
                <span>{t("seconds")}</span>
              </span>

              {match.is_user_uploaded && (
                <button
                  type="button"
                  onClick={handleDeleteClick}
                  className="p-1.5 rounded-lg bg-black/75 backdrop-blur-md border border-white/10 text-slate-300 hover:text-tv-coral hover:bg-tv-coral/20 hover:border-tv-coral/40 transition-all z-20 cursor-pointer"
                  title={t("card.delete_tooltip")}
                  aria-label={t("card.delete_tooltip")}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Hover Studio Play Reticle */}
          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-250 bg-black/35 backdrop-blur-[2px]">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-tv-cyan text-slate-950 font-bold text-xs shadow-lg shadow-tv-cyan/30 transform translate-y-1.5 group-hover:translate-y-0 transition-transform">
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{t("card.open_warroom")}</span>
            </div>
          </div>
        </div>

        {/* ── Card Content Body ── */}
        <div className="p-5 flex-1 flex flex-col justify-between relative z-10">
          <div>
            {/* Human Title & Directional Chevron */}
            <div className="flex items-start justify-between gap-2 mb-1">
              <h3 className="text-base font-bold text-white group-hover:text-tv-cyan transition-colors leading-snug">
                {display.title}
              </h3>
              <ChevronIcon className="w-4 h-4 text-slate-500 group-hover:text-tv-cyan group-hover:translate-x-0.5 rtl:group-hover:-translate-x-0.5 transition-all shrink-0 mt-1" />
            </div>

            {/* Scenario Description */}
            {display.tag && (
              <p className="text-xs text-slate-400 mb-4 line-clamp-1 leading-relaxed">
                {display.tag}
              </p>
            )}

            {/* ── Skew-Safe Possession & Pitch Control Strip ── */}
            <div className="mb-4 rounded-xl obsidian-well p-3 border border-white/[0.05]">
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="inline-flex items-center gap-1.5 text-tv-cyan font-semibold">
                  <span className="w-2 h-2 rounded-full bg-tv-cyan" />
                  <span>{t("team_a")}</span>
                  <BidiNum className="font-bold">{posA.toFixed(1)}%</BidiNum>
                </span>

                <span className="text-xs text-slate-400 font-semibold">
                  {t("card.possession")}
                </span>

                <span className="inline-flex items-center gap-1.5 text-tv-coral font-semibold">
                  <BidiNum className="font-bold">{posB.toFixed(1)}%</BidiNum>
                  <span>{t("team_b")}</span>
                  <span className="w-2 h-2 rounded-full bg-tv-coral" />
                </span>
              </div>

              {/* Dual Bar (Follows document flow so Team A aligns under Team A in EN & AR) */}
              <div className="h-2 w-full rounded-full bg-slate-900 overflow-hidden flex gap-0.5 p-0.5 border border-white/[0.06]">
                <div
                  style={{ width: `${Math.max(3, Math.min(97, posA))}%` }}
                  className="h-full bg-tv-cyan rounded-s-full transition-all duration-500"
                />
                <div
                  style={{ width: `${Math.max(3, Math.min(97, posB))}%` }}
                  className="h-full bg-tv-coral rounded-e-full transition-all duration-500"
                />
              </div>

              {/* Localized Pitch Control Secondary Readout */}
              <div className="flex items-center justify-between text-xs text-slate-400 mt-2">
                <span>{t("card.pitch_control")}</span>
                <div className="inline-flex items-center gap-1 font-semibold">
                  <BidiNum className="text-tv-cyan">{ctrlA.toFixed(0)}%</BidiNum>
                  <span className="text-slate-600">/</span>
                  <BidiNum className="text-tv-coral">{ctrlB.toFixed(0)}%</BidiNum>
                </div>
              </div>
            </div>
          </div>

          {/* ── Clean Hairline-Divided 3-Column Metrics Footer (Zero Box-Soup) ── */}
          <div className="pt-3 mt-1 border-t border-white/[0.07] grid grid-cols-3 divide-x rtl:divide-x-reverse divide-white/[0.07] text-center">
            <div className="px-2 py-1">
              <span className="block text-xs text-slate-400 font-medium mb-0.5 truncate">
                {t("card.passes")}
              </span>
              <BidiNum className="text-sm font-bold text-white">
                {passesA} – {passesB}
              </BidiNum>
            </div>

            <div className="px-2 py-1">
              <span className="block text-xs text-slate-400 font-medium mb-0.5 truncate">
                {t("card.turnovers")}
              </span>
              <BidiNum className="text-sm font-bold text-tv-amber">
                {match.turnovers ?? 0}
              </BidiNum>
            </div>

            <div className="px-2 py-1">
              <span className="block text-xs text-slate-400 font-medium mb-0.5 truncate">
                {t("card.top_speed")}
              </span>
              <div className="inline-flex items-baseline justify-center gap-1">
                <BidiNum className="text-sm font-bold text-tv-mint">
                  {(match.top_speed_kmh ?? 0).toFixed(1)}
                </BidiNum>
                <span className="text-[11px] text-slate-400">{t("km_h")}</span>
              </div>
            </div>
          </div>
        </div>
      </Link>

      {/* ── Crisp Obsidian Confirmation Dialog ── */}
      <AnimatePresence>
        {showConfirm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={handleCancelDelete}
              className="fixed inset-0 bg-black/80 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 10 }}
              className="relative w-full max-w-md rounded-2xl obsidian-card border-tv-coral/30 p-6 shadow-2xl z-10 space-y-4"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-tv-coral/15 border border-tv-coral/30 flex items-center justify-center text-tv-coral shrink-0">
                  <ShieldAlert className="w-5 h-5" />
                </div>
                <div className="min-w-0">
                  <h4 className="text-base font-bold text-white">
                    {t("delete.confirm_title")}
                  </h4>
                  <p className="text-xs text-slate-400 truncate mt-0.5">
                    {display.title}
                  </p>
                </div>
              </div>

              <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                {t("delete.confirm_desc")}
              </p>

              {deleteError && (
                <div className="p-3 rounded-xl bg-tv-coral/10 border border-tv-coral/20 text-tv-coral text-xs">
                  {deleteError}
                </div>
              )}

              <div className="flex items-center justify-end gap-2.5 pt-2">
                <button
                  type="button"
                  disabled={isDeleting}
                  onClick={handleCancelDelete}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-300 hover:text-white bg-white/[0.06] hover:bg-white/[0.12] transition-colors cursor-pointer"
                >
                  {t("delete.cancel_btn")}
                </button>
                <button
                  type="button"
                  disabled={isDeleting}
                  onClick={handleConfirmDelete}
                  className="px-4 py-2 rounded-xl text-xs font-bold text-white bg-tv-coral hover:bg-tv-coral/90 transition-all shadow-lg shadow-tv-coral/20 inline-flex items-center gap-1.5 cursor-pointer"
                >
                  {isDeleting ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>{t("delete.deleting")}</span>
                    </>
                  ) : (
                    <span>{t("delete.confirm_btn")}</span>
                  )}
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
