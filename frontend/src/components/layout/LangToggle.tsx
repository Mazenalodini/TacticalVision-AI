"use client";

import React from "react";
import { motion } from "framer-motion";
import { useLang } from "@/context/LangContext";

export function LangToggle() {
  const { lang, setLang } = useLang();

  return (
    <div
      dir="ltr"
      role="group"
      aria-label="Language switcher"
      className="inline-flex items-center p-0.5 rounded-xl bg-black/50 border border-white/10 shadow-inner select-none shrink-0"
    >
      <button
        type="button"
        onClick={() => setLang("en")}
        className={`relative px-2.5 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
          lang === "en" ? "text-white font-bold" : "text-slate-400 hover:text-white"
        }`}
      >
        {lang === "en" && (
          <motion.span
            layoutId="lang-segmented-pill"
            transition={{ type: "spring", stiffness: 480, damping: 34 }}
            className="absolute inset-0 rounded-lg editorial-tab-active"
          >
            <span className="absolute bottom-0 inset-x-2 h-[2px] rounded-full bg-tv-cyan" />
          </motion.span>
        )}
        <span className="relative z-10 font-sans">EN</span>
      </button>

      <button
        type="button"
        onClick={() => setLang("ar")}
        className={`relative px-2.5 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
          lang === "ar" ? "text-white font-bold" : "text-slate-400 hover:text-white"
        }`}
      >
        {lang === "ar" && (
          <motion.span
            layoutId="lang-segmented-pill"
            transition={{ type: "spring", stiffness: 480, damping: 34 }}
            className="absolute inset-0 rounded-lg editorial-tab-active"
          >
            <span className="absolute bottom-0 inset-x-2 h-[2px] rounded-full bg-tv-cyan" />
          </motion.span>
        )}
        <span className="relative z-10 font-arabic">عربي</span>
      </button>
    </div>
  );
}
