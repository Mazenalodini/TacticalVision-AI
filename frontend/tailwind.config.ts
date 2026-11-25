import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        "tv-base": "#060912",
        "tv-surface": "#0E1627",
        "tv-elevated": "#141F35",
        "tv-well": "#080D18",
        "tv-border": "rgba(148,163,184,0.12)",
        "tv-cyan": "#38BDF8",
        "tv-coral": "#FB7185",
        "tv-mint": "#34D399",
        "tv-amber": "#F59E0B",
      },
      fontFamily: {
        sans: ["var(--font-jakarta)", "sans-serif"],
        arabic: ["var(--font-plex-arabic)", "var(--font-jakarta)", "sans-serif"],
        mono: ["var(--font-jetbrains)", "monospace"],
      },
      boxShadow: {
        obsidian:
          "inset 0 1px 0 0 rgba(255,255,255,0.09), 0 18px 40px -14px rgba(0,0,0,0.7)",
        "obsidian-lg":
          "inset 0 1px 0 0 rgba(255,255,255,0.11), 0 24px 56px -12px rgba(0,0,0,0.85)",
      },
    },
  },
  plugins: [],
};

export default config;
