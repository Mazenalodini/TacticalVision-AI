import type { Metadata } from "next";
import {
  Plus_Jakarta_Sans,
  IBM_Plex_Sans_Arabic,
  JetBrains_Mono,
} from "next/font/google";
import "./globals.css";
import { LangProvider } from "@/context/LangContext";
import { Navbar } from "@/components/layout/Navbar";
import { MobileNav } from "@/components/layout/MobileNav";

/* ── Web Font Loading ─────────────────────────────────────────── */

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-jakarta",
  display: "swap",
});

const plexArabic = IBM_Plex_Sans_Arabic({
  subsets: ["arabic"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-plex-arabic",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

/* ── Metadata ─────────────────────────────────────────────────── */

export const metadata: Metadata = {
  title: "TacticalVision AI",
  description:
    "Broadcast-grade Football Tactical & Kinematic Analytics Platform",
};

const PRE_HYDRATION_LANG_SCRIPT = `
(function() {
  try {
    var saved = window.localStorage.getItem('tv_lang');
    var lang = saved === 'ar' ? 'ar' : 'en';
    var dir = lang === 'ar' ? 'rtl' : 'ltr';
    var el = document.documentElement;
    el.lang = lang;
    el.dir = dir;
    if (lang === 'ar') {
      el.classList.add('font-arabic');
      el.classList.remove('font-sans');
    } else {
      el.classList.add('font-sans');
      el.classList.remove('font-arabic');
    }
  } catch (e) {}
})();
`;

/* ── Root Layout ──────────────────────────────────────────────── */

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{ __html: PRE_HYDRATION_LANG_SCRIPT }}
        />
      </head>
      <body
        className={`${jakarta.variable} ${plexArabic.variable} ${jetbrains.variable} font-sans antialiased`}
      >
        {/* Persistent Multi-Layer Atmospheric Broadcast Studio Lighting */}
        <div
          aria-hidden="true"
          className="fixed inset-0 -z-10 pointer-events-none overflow-hidden bg-[#04070F]"
        >
          {/* Deep cobalt-navy top-start stadium floodlight */}
          <div className="absolute -top-44 -start-28 w-[820px] h-[560px] rounded-full bg-[#1D4ED8]/[0.15] blur-[155px]" />
          {/* Dark emerald tactical turf glow bottom-end */}
          <div className="absolute -bottom-44 -end-28 w-[760px] h-[520px] rounded-full bg-[#10B981]/[0.10] blur-[165px]" />
          {/* Broadcast sky & luminous rose subtle studio aura */}
          <div className="absolute top-[22%] left-1/2 -translate-x-1/2 w-[880px] h-[360px] rounded-full bg-[#38BDF8]/[0.06] blur-[150px]" />
          <div className="absolute top-[58%] start-[16%] w-[520px] h-[320px] rounded-full bg-[#FB7185]/[0.05] blur-[150px]" />
          {/* Radial-masked fine architectural grid */}
          <div
            className="absolute inset-0 opacity-[0.032]"
            style={{
              backgroundImage: `
                linear-gradient(to right, rgba(226, 232, 240, 0.5) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(226, 232, 240, 0.5) 1px, transparent 1px)
              `,
              backgroundSize: "72px 72px",
              maskImage:
                "radial-gradient(ellipse 85% 75% at 50% 25%, black 20%, transparent 100%)",
              WebkitMaskImage:
                "radial-gradient(ellipse 85% 75% at 50% 25%, black 20%, transparent 100%)",
            }}
          />
        </div>

        <LangProvider>
          <div className="min-h-screen flex flex-col">
            <Navbar />
            <main className="flex-1">{children}</main>
            <MobileNav />
          </div>
        </LangProvider>
      </body>
    </html>
  );
}
