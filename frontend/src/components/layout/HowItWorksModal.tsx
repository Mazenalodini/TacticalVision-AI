"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Film,
  Compass,
  Activity,
  ArrowRight,
  ArrowLeft,
  Plus,
  Sparkles,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { BidiNum } from "@/components/ui/BidiNum";

interface HowItWorksModalProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenUpload: () => void;
  featuredMatchId?: string;
}

export function HowItWorksModal({
  isOpen,
  onClose,
  onOpenUpload,
  featuredMatchId = "match_01_counter",
}: HowItWorksModalProps) {
  const { t, isRTL } = useLang();
  const router = useRouter();
  const ArrowIcon = isRTL ? ArrowLeft : ArrowRight;

  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [isOpen, onClose]);

  const steps = [
    {
      num: t("hiw.step1_num"),
      title: t("hiw.step1_title"),
      desc: t("hiw.step1_desc"),
      icon: Film,
      accent: "text-tv-cyan",
      ring: "border-tv-cyan/30 bg-tv-cyan/10",
    },
    {
      num: t("hiw.step2_num"),
      title: t("hiw.step2_title"),
      desc: t("hiw.step2_desc"),
      icon: Compass,
      accent: "text-tv-mint",
      ring: "border-tv-mint/30 bg-tv-mint/10",
    },
    {
      num: t("hiw.step3_num"),
      title: t("hiw.step3_title"),
      desc: t("hiw.step3_desc"),
      icon: Activity,
      accent: "text-tv-coral",
      ring: "border-tv-coral/30 bg-tv-coral/10",
    },
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/80 backdrop-blur-md"
          />

          {/* Modal Panel */}
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 14 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 10, transition: { duration: 0.14 } }}
            transition={{ type: "spring", duration: 0.3, bounce: 0 }}
            className="relative w-full max-w-3xl rounded-2xl obsidian-popover p-6 sm:p-8 z-10 my-8"
          >
            {/* Header */}
            <div className="flex items-start justify-between gap-4 pb-5 border-b border-white/10">
              <div>
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-tv-cyan/10 border border-tv-cyan/25 text-xs font-semibold text-tv-cyan mb-2.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{t("hiw.badge")}</span>
                </div>
                <h2 className="text-xl sm:text-2xl font-bold text-white leading-snug">
                  {t("hiw.title")}
                </h2>
                <p className="text-sm text-slate-400 mt-1.5 leading-relaxed">
                  {t("hiw.subtitle")}
                </p>
              </div>

              <button
                type="button"
                onClick={onClose}
                className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors shrink-0 cursor-pointer"
                aria-label={t("hiw.close")}
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* 3-Step Product Pipeline Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-6">
              {steps.map((step) => {
                const Icon = step.icon;
                return (
                  <div
                    key={step.num}
                    className="rounded-xl obsidian-well p-4 sm:p-5 flex flex-col justify-between border border-white/[0.07] hover:border-white/15 transition-colors"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-4">
                        <div
                          className={`w-10 h-10 rounded-xl border flex items-center justify-center ${step.ring} ${step.accent}`}
                        >
                          <Icon className="w-5 h-5" />
                        </div>
                        <BidiNum className="text-xs font-bold text-slate-500 px-2 py-0.5 rounded-md bg-white/[0.03] border border-white/5">
                          {step.num}
                        </BidiNum>
                      </div>
                      <h3 className="text-sm font-bold text-white mb-2 leading-snug">
                        {step.title}
                      </h3>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        {step.desc}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Footer Actions */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-4 border-t border-white/10">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="w-2 h-2 rounded-full bg-tv-mint" />
                <BidiNum className="text-slate-300 font-semibold">
                  105m × 68m · 60FPS · H.264
                </BidiNum>
              </div>

              <div className="flex flex-wrap items-center gap-2.5">
                <button
                  type="button"
                  onClick={() => {
                    onClose();
                    router.push(`/match/${featuredMatchId}`);
                  }}
                  className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold text-white bg-white/[0.07] hover:bg-white/[0.12] border border-white/10 transition-colors cursor-pointer"
                >
                  <span>{t("hiw.cta_explore")}</span>
                  <ArrowIcon className="w-3.5 h-3.5" />
                </button>

                <button
                  type="button"
                  onClick={() => {
                    onClose();
                    onOpenUpload();
                  }}
                  className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-xl text-xs font-bold bg-tv-cyan text-slate-950 hover:bg-tv-cyan/90 shadow-lg shadow-tv-cyan/20 transition-all cursor-pointer"
                >
                  <Plus className="w-4 h-4 stroke-[2.5]" />
                  <span>{t("hiw.cta_upload")}</span>
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
