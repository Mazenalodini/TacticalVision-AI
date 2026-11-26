"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  Activity,
  BarChart2,
  AlertTriangle,
  Crosshair,
  Compass,
  Zap,
  Share2,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { useMatchData } from "@/hooks/useMatchData";
import { formatResolution, resolveMatchDisplay } from "@/lib/format";
import { BidiNum } from "@/components/ui/BidiNum";
import {
  VideoPlayer,
  VideoPlayerHandle,
} from "@/components/warroom/VideoPlayer";
import { PitchCanvas } from "@/components/warroom/PitchCanvas";
import {
  LayerToggles,
  LiveTacticalHud,
  CanvasLayerConfig,
  LiveTelemetryHud,
} from "@/components/warroom/LayerToggles";
import { EventTimeline } from "@/components/warroom/EventTimeline";
import { EventFeed } from "@/components/warroom/EventFeed";
import { PlayerSpotlight } from "@/components/warroom/PlayerSpotlight";
import { CoachBrief } from "@/components/warroom/CoachBrief";
import { D3MomentumStream } from "@/components/d3/D3MomentumStream";
import { D3PassNetworkPitch } from "@/components/d3/D3PassNetworkPitch";
import { TelemetryView } from "@/components/analytics/TelemetryView";
import { Skeleton } from "@/components/ui/Skeleton";
import type { HeatmapMode } from "@/components/warroom/canvas/drawHeatmap";

type LowerDeckTab = "brief" | "momentum" | "passnet";

export default function WarRoomPage({
  params,
}: {
  params: { id: string };
}) {
  const { t, lang, isRTL } = useLang();
  const { match, telemetry, isLoading, error, refetch } = useMatchData(params.id);

  const videoPlayerRef = useRef<VideoPlayerHandle | null>(null);

  const [activeTab, setActiveTab] = useState<"warroom" | "telemetry">("warroom");
  const [lowerTab, setLowerTab] = useState<LowerDeckTab>("brief");
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);

  const [layers, setLayers] = useState<CanvasLayerConfig>({
    players: true,
    voronoi: true,
    hulls: true,
    defLines: true,
    heatmap: false,
    heatmapMode: "team_0",
    passNetwork: false,
  });

  const [hud, setHud] = useState<LiveTelemetryHud>({
    voronoiSplit: [50, 50],
    team0HullArea_m2: 0,
    team1HullArea_m2: 0,
    defLine0_x: null,
    defLine1_x: null,
    activePlayersCount: 0,
    sprintingCount: 0,
  });

  useEffect(() => {
    const handleSwitch = (e: Event) => {
      const detail = (e as CustomEvent<"warroom" | "telemetry">).detail;
      if (detail) setActiveTab(detail);
    };
    window.addEventListener("tv:switch-tab", handleSwitch);
    return () => window.removeEventListener("tv:switch-tab", handleSwitch);
  }, []);

  const handleToggleLayer = useCallback(
    (layerKey: keyof Omit<CanvasLayerConfig, "heatmapMode">) => {
      setLayers((prev) => ({
        ...prev,
        [layerKey]: !prev[layerKey],
      }));
    },
    []
  );

  const handleChangeHeatmapMode = useCallback((mode: HeatmapMode) => {
    setLayers((prev) => ({
      ...prev,
      heatmapMode: mode,
    }));
  }, []);

  const handleHudUpdate = useCallback((newHud: LiveTelemetryHud) => {
    setHud(newHud);
  }, []);

  const handleTimeUpdate = useCallback((time: number, frame: number) => {
    setCurrentTime(time);
    setCurrentFrame(frame);
  }, []);

  const getVideoElement = useCallback(() => {
    return videoPlayerRef.current?.getVideoElement() ?? null;
  }, []);

  const BackArrow = isRTL ? ArrowRight : ArrowLeft;

  if (isLoading) {
    return (
      <div className="min-h-screen text-white p-3 sm:p-4 lg:px-6 max-w-[1520px] mx-auto">
        <div className="flex items-center justify-between mb-3 pb-2.5 border-b border-white/[0.08]">
          <Skeleton className="h-7 w-56 rounded-xl" />
          <Skeleton className="h-8 w-64 rounded-xl" />
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
          <div className="xl:col-span-7">
            <Skeleton className="aspect-video w-full rounded-2xl" />
          </div>
          <div className="xl:col-span-5 space-y-3">
            <Skeleton className="h-16 w-full rounded-xl" />
            <Skeleton className="aspect-[105/68] w-full rounded-2xl" />
            <Skeleton className="h-14 w-full rounded-xl" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !match) {
    return (
      <div className="min-h-[80vh] text-white p-8 flex flex-col items-center justify-center text-center">
        <div className="w-14 h-14 rounded-2xl bg-tv-coral/10 border border-tv-coral/30 flex items-center justify-center mb-4 text-tv-coral">
          <AlertTriangle className="w-7 h-7" />
        </div>
        <h2 className="text-xl font-bold mb-2">{t("wr.err_load_title")}</h2>
        <p className="text-sm text-slate-400 mb-6 max-w-md leading-relaxed">
          {error || t("wr.err_load_desc")}
        </p>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => refetch()}
            className="px-5 py-2.5 rounded-xl font-bold text-xs bg-tv-cyan text-slate-950 hover:bg-tv-cyan/90 transition-colors cursor-pointer"
          >
            {t("wr.retry_btn")}
          </button>
          <Link
            href="/"
            className="px-5 py-2.5 rounded-xl font-semibold text-xs bg-white/10 hover:bg-white/15 text-slate-200 transition-colors"
          >
            {t("wr.back_to_hub")}
          </Link>
        </div>
      </div>
    );
  }

  const display = resolveMatchDisplay(match, t, lang);

  const deckTabs: Array<{
    key: LowerDeckTab;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
  }> = [
    { key: "brief", label: t("wr.deck_brief"), icon: Compass },
    { key: "momentum", label: t("wr.deck_momentum"), icon: Zap },
    { key: "passnet", label: t("wr.deck_passnet"), icon: Share2 },
  ];

  return (
    <div className="min-h-screen text-white pb-24">
      {/* ── Sticky Compact Match Studio Sub-Header (Never Clips Behind Sticky Navbar) ── */}
      <div
        id="warroom-subheader"
        className="md:sticky md:top-14 z-30 w-full bg-[#04070F]/88 backdrop-blur-xl border-b border-white/[0.08] print:hidden"
      >
        <div className="max-w-[1520px] mx-auto px-3 sm:px-5 lg:px-6 py-2.5 flex flex-wrap items-center justify-between gap-2.5">
          <div className="flex items-center gap-2 min-w-0">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl obsidian-well hover:border-white/20 text-xs font-semibold text-slate-300 hover:text-white transition-colors shrink-0"
            >
              <BackArrow className="w-3.5 h-3.5 text-tv-cyan" />
              <span>{t("wr.back_to_hub")}</span>
            </Link>

            <span className="text-slate-700 select-none">/</span>

            <h1 className="text-sm sm:text-lg font-extrabold text-white tracking-tight truncate">
              {display.title}
            </h1>

            <span
              className={`hidden sm:inline-flex px-2.5 py-0.5 rounded-lg text-[11px] font-semibold border shrink-0 ${
                match.is_user_uploaded
                  ? "bg-tv-cyan/10 border-tv-cyan/25 text-tv-cyan"
                  : "bg-white/[0.04] border-white/10 text-slate-300"
              }`}
            >
              {display.badge}
            </span>

            {selectedPlayerId !== null && (
              <button
                type="button"
                onClick={() => setSelectedPlayerId(null)}
                className="hidden md:inline-flex items-center gap-1 px-2 py-0.5 rounded-lg bg-tv-mint/15 border border-tv-mint/35 text-[11px] font-bold text-tv-mint cursor-pointer"
              >
                <Crosshair className="w-3 h-3" />
                <BidiNum>#{selectedPlayerId}</BidiNum>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2.5 ms-auto">
            {/* Calm Editorial Segmented Switcher */}
            <div className="inline-flex items-center p-0.5 rounded-xl obsidian-well">
              <button
                type="button"
                onClick={() => setActiveTab("warroom")}
                className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                  activeTab === "warroom"
                    ? "text-white font-bold"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {activeTab === "warroom" && (
                  <motion.span
                    layoutId="warroom-mode-pill"
                    transition={{ type: "spring", stiffness: 460, damping: 34 }}
                    className="absolute inset-0 rounded-lg editorial-tab-active"
                  >
                    <span className="absolute bottom-0 inset-x-3 h-[2px] rounded-full bg-tv-cyan shadow-[0_0_8px_rgba(56,189,248,0.8)]" />
                  </motion.span>
                )}
                <Activity className="w-3.5 h-3.5 relative z-10 text-tv-cyan" />
                <span className="relative z-10">{t("wr.tab_cockpit")}</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("telemetry")}
                className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                  activeTab === "telemetry"
                    ? "text-white font-bold"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {activeTab === "telemetry" && (
                  <motion.span
                    layoutId="warroom-mode-pill"
                    transition={{ type: "spring", stiffness: 460, damping: 34 }}
                    className="absolute inset-0 rounded-lg editorial-tab-active"
                  >
                    <span className="absolute bottom-0 inset-x-3 h-[2px] rounded-full bg-tv-cyan shadow-[0_0_8px_rgba(56,189,248,0.8)]" />
                  </motion.span>
                )}
                <BarChart2 className="w-3.5 h-3.5 relative z-10 text-tv-cyan" />
                <span className="relative z-10">{t("wr.tab_telemetry")}</span>
              </button>
            </div>

            <div className="hidden lg:inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl obsidian-well text-xs text-slate-300">
              <BidiNum className="font-semibold">
                {formatResolution(match.resolution)}
              </BidiNum>
              <span className="text-slate-600">·</span>
              <span className="inline-flex items-baseline gap-1 font-semibold">
                <BidiNum>{Math.round(match.fps || 25)}</BidiNum>
                <span className="text-[11px] text-slate-400">{t("card.fps")}</span>
              </span>
              <span className="text-slate-600">·</span>
              <span className="inline-flex items-baseline gap-0.5 text-tv-mint font-bold">
                <BidiNum>{(match.duration_s ?? 0).toFixed(1)}</BidiNum>
                <span>{t("seconds")}</span>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Main Studio Workspace ── */}
      <div className="max-w-[1520px] mx-auto px-3 sm:px-5 lg:px-6 pt-3.5">
        {activeTab === "warroom" ? (
          <div className="space-y-5">
            {/* ── Above-the-Fold Unified Studio Cockpit ── */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-start">
              {/* Solid Opaque Mobile Sticky Wrapper prevents bleed-through behind blurred card when scrolling */}
              <div className="xl:col-span-7 sticky top-14 z-20 md:static rounded-2xl bg-[#04070F] md:bg-transparent shadow-[0_14px_30px_rgba(0,0,0,0.9)] md:shadow-none">
                <div className="rounded-2xl obsidian-card border-b border-white/[0.12] md:border-b-0 overflow-visible">
                  <VideoPlayer
                    ref={videoPlayerRef}
                    matchId={match.id}
                    fps={match.fps || 25.0}
                    totalFrames={telemetry?.summary?.total_frames || 750}
                    duration_s={match.duration_s || 30.0}
                    onTimeUpdate={handleTimeUpdate}
                    dockedBottom={Boolean(telemetry?.events)}
                  />

                  {telemetry?.events && (
                    <EventTimeline
                      events={telemetry.events}
                      broadcastEvents={telemetry.broadcast_events}
                      currentTime={currentTime}
                      duration_s={match.duration_s || 30.0}
                      onSeekToFrame={(f) => videoPlayerRef.current?.seekToFrame(f)}
                      onSeekToTime={(tm) => videoPlayerRef.current?.seekToTime(tm)}
                      onSelectPlayer={(id) => setSelectedPlayerId(id)}
                      docked
                    />
                  )}
                </div>
              </div>

              <div
                id="pitch-canvas-section"
                className="xl:col-span-5 scroll-mt-28 rounded-2xl obsidian-card p-3 flex flex-col gap-2.5"
              >
                <LayerToggles
                  layers={layers}
                  onToggleLayer={handleToggleLayer}
                  onChangeHeatmapMode={handleChangeHeatmapMode}
                />

                <div className="rounded-xl overflow-hidden border border-white/[0.09] shadow-inner">
                  <PitchCanvas
                    telemetry={telemetry}
                    getVideoElement={getVideoElement}
                    layers={layers}
                    selectedPlayerId={selectedPlayerId}
                    onSelectPlayer={setSelectedPlayerId}
                    onHudUpdate={handleHudUpdate}
                  />
                </div>

                <LiveTacticalHud hud={hud} />
              </div>
            </div>

            {/* ── Inline Docked Player Inspector Panel (0% Radar Obstruction!) ── */}
            <PlayerSpotlight
              selectedPlayerId={selectedPlayerId}
              telemetry={telemetry}
              currentFrameIndex={currentFrame}
              onDeselect={() => setSelectedPlayerId(null)}
              onSeekToFrame={(f) => videoPlayerRef.current?.seekToFrame(f)}
            />

            {/* ── Calm 3-Tab Tactical Intelligence Deck ── */}
            <section
              id="events-section"
              className="scroll-mt-28 pt-4 border-t border-white/[0.08] space-y-4"
            >
              {/* Understated Editorial Segmented Tab Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="inline-flex flex-wrap items-center gap-1 p-1 rounded-2xl obsidian-card">
                  {deckTabs.map((tab) => {
                    const Icon = tab.icon;
                    const active = lowerTab === tab.key;
                    return (
                      <button
                        key={tab.key}
                        type="button"
                        onClick={() => setLowerTab(tab.key)}
                        className={`relative flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-colors cursor-pointer ${
                          active
                            ? "text-white font-bold"
                            : "text-slate-400 hover:text-white"
                        }`}
                      >
                        {active && (
                          <motion.span
                            layoutId="warroom-lower-deck-pill"
                            transition={{
                              type: "spring",
                              stiffness: 460,
                              damping: 34,
                            }}
                            className="absolute inset-0 rounded-xl editorial-tab-active"
                          >
                            <span className="absolute bottom-0 inset-x-4 h-[2px] rounded-full bg-tv-cyan shadow-[0_0_8px_rgba(56,189,248,0.8)]" />
                          </motion.span>
                        )}
                        <Icon
                          className={`w-3.5 h-3.5 relative z-10 ${
                            active ? "text-tv-cyan" : ""
                          }`}
                        />
                        <span className="relative z-10">{tab.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Active Intelligence Tab Content */}
              {lowerTab === "brief" && <CoachBrief telemetry={telemetry} />}

              {lowerTab === "momentum" && (
                <div className="space-y-4">
                  <D3MomentumStream
                    telemetry={telemetry}
                    currentTime={currentTime}
                    duration_s={match.duration_s || 30.0}
                    onSeekToTime={(tm) => videoPlayerRef.current?.seekToTime(tm)}
                    onSeekToFrame={(f) => videoPlayerRef.current?.seekToFrame(f)}
                  />
                  {telemetry?.events && (
                    <EventFeed
                      events={telemetry.events}
                      broadcastEvents={telemetry.broadcast_events}
                      currentTime={currentTime}
                      onSeekToFrame={(f) => videoPlayerRef.current?.seekToFrame(f)}
                      onSeekToTime={(tm) => videoPlayerRef.current?.seekToTime(tm)}
                      onSelectPlayer={(id) => setSelectedPlayerId(id)}
                    />
                  )}
                </div>
              )}

              {lowerTab === "passnet" && (
                <D3PassNetworkPitch
                  telemetry={telemetry}
                  onSelectPlayer={(id) => {
                    setSelectedPlayerId(id);
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  }}
                />
              )}
            </section>
          </div>
        ) : (
          <TelemetryView
            telemetry={telemetry}
            match={match}
            currentTime={currentTime}
            onSeekToTime={(tm) => {
              setActiveTab("warroom");
              setTimeout(() => {
                videoPlayerRef.current?.seekToTime(tm);
              }, 60);
            }}
            onSeekToFrame={(f) => {
              setActiveTab("warroom");
              setTimeout(() => {
                videoPlayerRef.current?.seekToFrame(f);
              }, 60);
            }}
            onSelectPlayerAndJump={(id) => {
              setSelectedPlayerId(id);
              setActiveTab("warroom");
              if (typeof window !== "undefined") {
                window.scrollTo({ top: 0, behavior: "smooth" });
              }
            }}
          />
        )}
      </div>
    </div>
  );
}
