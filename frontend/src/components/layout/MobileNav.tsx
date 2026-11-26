"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutGrid,
  Crosshair,
  Calendar,
  BarChart3,
  PlusCircle,
  Home,
} from "lucide-react";
import { motion } from "framer-motion";
import { useLang } from "@/context/LangContext";

export function MobileNav() {
  const { t } = useLang();
  const pathname = usePathname();
  const [warRoomActiveSection, setWarRoomActiveSection] = useState<
    "pitch" | "events" | "stats"
  >("pitch");

  const isHub = pathname === "/";
  const isWarRoom = pathname.startsWith("/match/");

  const handleOpenUpload = (e: React.MouseEvent) => {
    e.preventDefault();
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tv:open-upload"));
    }
  };

  const handleScrollToSection =
    (id: string, sectionKey: "pitch" | "events") => (e: React.MouseEvent) => {
      e.preventDefault();
      setWarRoomActiveSection(sectionKey);
      if (typeof window !== "undefined") {
        window.dispatchEvent(
          new CustomEvent("tv:switch-tab", { detail: "warroom" })
        );
        setTimeout(() => {
          const el = document.getElementById(id);
          if (el) {
            el.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        }, 50);
      }
    };

  const handleSwitchToTelemetry = (e: React.MouseEvent) => {
    e.preventDefault();
    setWarRoomActiveSection("stats");
    if (typeof window !== "undefined") {
      window.dispatchEvent(
        new CustomEvent("tv:switch-tab", { detail: "telemetry" })
      );
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-[#050912]/95 backdrop-blur-xl border-t border-white/[0.08] px-2 py-1.5 safe-bottom print:hidden">
      {isHub ? (
        <div className="flex justify-around items-center">
          <Link
            href="/"
            className="relative flex flex-col items-center py-1 px-6 text-tv-cyan"
          >
            <motion.span
              layoutId="mobile-bottom-pill"
              transition={{ type: "spring", stiffness: 480, damping: 32 }}
              className="absolute -top-1.5 w-10 h-0.5 rounded-full bg-tv-cyan"
            />
            <LayoutGrid className="w-5 h-5" />
            <span className="text-xs mt-0.5 font-bold whitespace-nowrap">{t("mobile.matches")}</span>
          </Link>

          <button
            type="button"
            onClick={handleOpenUpload}
            className="relative flex flex-col items-center py-1 px-6 text-slate-300 hover:text-white transition-colors cursor-pointer"
          >
            <PlusCircle className="w-5 h-5 text-tv-mint" />
            <span className="text-xs mt-0.5 font-semibold whitespace-nowrap">
              {t("mobile.analyze")}
            </span>
          </button>
        </div>
      ) : isWarRoom ? (
        <div className="grid grid-cols-4 items-center">
          <Link
            href="/"
            className="relative flex flex-col items-center py-1 text-slate-400 hover:text-white transition-colors"
          >
            <Home className="w-4 h-4" />
            <span className="text-[11px] mt-0.5 font-medium">
              {t("mobile.hub")}
            </span>
          </Link>

          <button
            type="button"
            onClick={handleScrollToSection("pitch-canvas-section", "pitch")}
            className={`relative flex flex-col items-center py-1 transition-colors cursor-pointer ${
              warRoomActiveSection === "pitch"
                ? "text-tv-cyan font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            {warRoomActiveSection === "pitch" && (
              <motion.span
                layoutId="mobile-bottom-pill"
                transition={{ type: "spring", stiffness: 480, damping: 32 }}
                className="absolute -top-1.5 w-8 h-0.5 rounded-full bg-tv-cyan"
              />
            )}
            <Crosshair className="w-4 h-4" />
            <span className="text-[11px] mt-0.5">{t("mobile.pitch")}</span>
          </button>

          <button
            type="button"
            onClick={handleScrollToSection("events-section", "events")}
            className={`relative flex flex-col items-center py-1 transition-colors cursor-pointer ${
              warRoomActiveSection === "events"
                ? "text-tv-cyan font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            {warRoomActiveSection === "events" && (
              <motion.span
                layoutId="mobile-bottom-pill"
                transition={{ type: "spring", stiffness: 480, damping: 32 }}
                className="absolute -top-1.5 w-8 h-0.5 rounded-full bg-tv-cyan"
              />
            )}
            <Calendar className="w-4 h-4" />
            <span className="text-[11px] mt-0.5">{t("mobile.events")}</span>
          </button>

          <button
            type="button"
            onClick={handleSwitchToTelemetry}
            className={`relative flex flex-col items-center py-1 transition-colors cursor-pointer ${
              warRoomActiveSection === "stats"
                ? "text-tv-cyan font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            {warRoomActiveSection === "stats" && (
              <motion.span
                layoutId="mobile-bottom-pill"
                transition={{ type: "spring", stiffness: 480, damping: 32 }}
                className="absolute -top-1.5 w-8 h-0.5 rounded-full bg-tv-cyan"
              />
            )}
            <BarChart3 className="w-4 h-4" />
            <span className="text-[11px] mt-0.5">{t("mobile.stats")}</span>
          </button>
        </div>
      ) : null}
    </nav>
  );
}
