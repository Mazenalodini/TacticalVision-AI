"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { dictionary, type Lang } from "@/lib/i18n";

const STORAGE_KEY = "tv_lang";

interface LangContextType {
  lang: Lang;
  dir: "ltr" | "rtl";
  isRTL: boolean;
  t: (key: string) => string;
  toggle: () => void;
  setLang: (next: Lang) => void;
}

const LangContext = createContext<LangContextType | undefined>(undefined);

export function LangProvider({ children }: { children: ReactNode }) {
  // Always initialize to "en" on SSR & initial client render to guarantee zero hydration mismatch
  const [lang, setLangState] = useState<Lang>("en");
  const [hydrated, setHydrated] = useState(false);

  // Hydrate persisted language preference strictly inside useEffect
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved === "ar" || saved === "en") {
        setLangState(saved);
      }
    } catch {
      // Ignore storage access errors in restricted browsers
    }
    setHydrated(true);
  }, []);

  const dir = lang === "ar" ? "rtl" : "ltr";
  const isRTL = lang === "ar";

  // Synchronize <html lang="..." dir="..."> ONLY after hydration so <head> pre-paint script never gets overwritten
  useEffect(() => {
    if (!hydrated) return;

    const html = document.documentElement;
    html.lang = lang;
    html.dir = dir;

    if (lang === "ar") {
      html.classList.add("font-arabic");
      html.classList.remove("font-sans");
    } else {
      html.classList.add("font-sans");
      html.classList.remove("font-arabic");
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      // Ignore storage write errors
    }
  }, [lang, dir, hydrated]);

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
  }, []);

  const toggle = useCallback(() => {
    setLangState((prev) => (prev === "en" ? "ar" : "en"));
  }, []);

  const t = useCallback(
    (key: string): string => dictionary[lang]?.[key] ?? dictionary.en[key] ?? key,
    [lang]
  );

  return (
    <LangContext.Provider value={{ lang, dir, isRTL, t, toggle, setLang }}>
      {children}
    </LangContext.Provider>
  );
}

export function useLang(): LangContextType {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useLang must be used within <LangProvider>");
  return ctx;
}
