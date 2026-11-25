"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Play,
  Pause,
  Grid,
  Shield,
  ArrowUpRight,
  ArrowUpLeft,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { fetchTelemetry } from "@/lib/api";
import { resolveMatchDisplay, formatTeamName } from "@/lib/format";
import type { MatchCard, Telemetry } from "@/lib/types";
import {
  stabilizeTelemetry,
  interpolateStabilizedFrame,
  findBestHeroWindow,
  type StabilizedFrame,
  type StabilizedPlayer,
} from "@/lib/radarStabilizer";
import {
  computePitchViewport,
  canvasToPitch,
  pitchToCanvas,
  type PitchViewport,
} from "@/components/warroom/canvas/pitchConfig";
import { drawPitch } from "@/components/warroom/canvas/drawPitch";
import { VoronoiBuffer } from "@/components/warroom/canvas/drawVoronoi";
import { drawHulls } from "@/components/warroom/canvas/drawHulls";
import { drawDefLines } from "@/components/warroom/canvas/drawDefLines";
import {
  drawPlayers,
  type BallHistoryPoint,
} from "@/components/warroom/canvas/drawPlayers";
import { BidiNum } from "@/components/ui/BidiNum";

interface HeroPitchVisualProps {
  featuredMatch?: MatchCard | null;
}

interface HeroHudState {
  timeSec: number;
  frameNum: number;
  vor0: number;
  vor1: number;
  hull0: number;
  hull1: number;
  progressPct: number;
}

/**
 * Alive 60FPS Real-Data Broadcast Centerpiece for the Match Hub Hero.
 * Loads real `telemetry.json` from `featuredMatch`, stabilizes the tracking/homography signal,
 * selects the cleanest high-tempo 7s tactical sequence, and animates a silky 60FPS loop
 * with live hover inspection, Voronoi/Hull toggles, and real-time HUD readouts.
 */
export function HeroPitchVisual({ featuredMatch }: HeroPitchVisualProps) {
  const { t, lang, isRTL } = useLang();
  const router = useRouter();

  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [showVoronoi, setShowVoronoi] = useState(true);
  const [showHulls, setShowHulls] = useState(true);
  const [hoveredInfo, setHoveredInfo] = useState<{
    player: StabilizedPlayer;
    leftPct: number;
    topPct: number;
  } | null>(null);
  const [hud, setHud] = useState<HeroHudState>({
    timeSec: 0,
    frameNum: 0,
    vor0: 50,
    vor1: 50,
    hull0: 0,
    hull1: 0,
    progressPct: 0,
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const voronoiRef = useRef<VoronoiBuffer | null>(null);
  const viewportRef = useRef<PitchViewport | null>(null);
  const lastInterpRef = useRef<StabilizedFrame | null>(null);

  const playheadRef = useRef<number>(0);
  const lastRafTsRef = useRef<number | null>(null);
  const lastHudTsRef = useRef<number>(0);
  const hoveredIdRef = useRef<number | null>(null);

  const targetMatchId = featuredMatch?.id || "match_01_counter";

  useEffect(() => {
    let cancelled = false;
    fetchTelemetry(targetMatchId)
      .then((data) => {
        if (!cancelled) setTelemetry(data);
      })
      .catch(() => {
        // Silent fallback if backend unavailable
      });
    return () => {
      cancelled = true;
    };
  }, [targetMatchId]);

  const stabilized = useMemo(() => stabilizeTelemetry(telemetry), [telemetry]);
  const heroWindow = useMemo(
    () => findBestHeroWindow(stabilized, 175),
    [stabilized]
  );

  // Reset playhead whenever hero window changes
  useEffect(() => {
    playheadRef.current = heroWindow.startFrame;
    lastRafTsRef.current = null;
  }, [heroWindow]);

  useEffect(() => {
    if (!voronoiRef.current) {
      voronoiRef.current = new VoronoiBuffer();
    }
  }, []);

  // ResizeObserver for razor-sharp HiDPI canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const handleResize = () => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };

    handleResize();
    const ro = new ResizeObserver(handleResize);
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // ── 60FPS Real-Telemetry Continuous Broadcast Loop ──
  useEffect(() => {
    let rafId: number;

    const loop = (ts: number) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");

      if (canvas && ctx && stabilized && stabilized.frames.length > 0) {
        const span = Math.max(1, heroWindow.endFrame - heroWindow.startFrame);

        if (isPlaying && lastRafTsRef.current !== null) {
          const dtSec = Math.min(0.1, (ts - lastRafTsRef.current) / 1000);
          // 0.85x broadcast tempo for calm, legible tactical motion
          const advanceFrames = dtSec * stabilized.fps * 0.85;
          playheadRef.current += advanceFrames;
          if (playheadRef.current > heroWindow.endFrame) {
            playheadRef.current = heroWindow.startFrame;
          }
        }
        lastRafTsRef.current = ts;

        const exactFrame = Math.max(
          heroWindow.startFrame,
          Math.min(heroWindow.endFrame, playheadRef.current)
        );
        const f0 = Math.floor(exactFrame);
        const interpFrame = interpolateStabilizedFrame(stabilized, exactFrame);

        if (interpFrame) {
          lastInterpRef.current = interpFrame;

          // Subtle 220ms loop wrap breath fade at edges
          const distFromStart = exactFrame - heroWindow.startFrame;
          const distToEnd = heroWindow.endFrame - exactFrame;
          let fadeAlpha = 1.0;
          if (distFromStart < 6) fadeAlpha = Math.max(0.35, distFromStart / 6);
          else if (distToEnd < 6) fadeAlpha = Math.max(0.35, distToEnd / 6);

          // Real ball tail slice
          const ballTrail: BallHistoryPoint[] = [];
          const tailStart = Math.max(heroWindow.startFrame, f0 - 16);
          for (let i = tailStart; i <= f0; i++) {
            const bf = stabilized.frames[i];
            if (bf?.ball) {
              ballTrail.push({ x: bf.ball[0], y: bf.ball[1], frame: bf.f });
            }
          }
          if (interpFrame.ball) {
            ballTrail.push({
              x: interpFrame.ball[0],
              y: interpFrame.ball[1],
              frame: interpFrame.f,
            });
          }

          const rect = canvas.getBoundingClientRect();
          const dpr = window.devicePixelRatio || 1;
          const vp = computePitchViewport(rect.width, rect.height, 0.042);
          viewportRef.current = vp;

          ctx.save();
          ctx.scale(dpr, dpr);

          drawPitch(ctx, vp);

          ctx.save();
          ctx.globalAlpha = fadeAlpha;

          if (showVoronoi && voronoiRef.current) {
            voronoiRef.current.render(ctx, interpFrame.players, vp);
          }

          if (showHulls) {
            drawHulls(ctx, interpFrame.players, vp);
          }

          drawDefLines(ctx, interpFrame.def_x, vp);

          drawPlayers(
            ctx,
            interpFrame.players,
            interpFrame.ball,
            ballTrail,
            null,
            undefined,
            interpFrame.f,
            vp,
            hoveredIdRef.current
          );

          ctx.restore();
          ctx.restore();

          // Update React HUD readouts smoothly at ~12Hz to avoid heavy re-renders
          if (ts - lastHudTsRef.current >= 80) {
            lastHudTsRef.current = ts;
            setHud({
              timeSec: interpFrame.t,
              frameNum: interpFrame.f,
              vor0: Math.round(interpFrame.vor[0]),
              vor1: 100 - Math.round(interpFrame.vor[0]),
              hull0: interpFrame.hullArea.team0_m2,
              hull1: interpFrame.hullArea.team1_m2,
              progressPct: Math.min(
                100,
                Math.max(0, ((exactFrame - heroWindow.startFrame) / span) * 100)
              ),
            });

            if (hoveredIdRef.current !== null) {
              const hp = interpFrame.players.find((p) => p.id === hoveredIdRef.current);
              if (hp && vp.width > 0 && vp.height > 0) {
                const [px, py] = pitchToCanvas(hp.x, hp.y, vp);
                setHoveredInfo({
                  player: hp,
                  leftPct: Math.max(12, Math.min(88, (px / vp.width) * 100)),
                  topPct: Math.max(14, Math.min(82, (py / vp.height) * 100)),
                });
              } else {
                hoveredIdRef.current = null;
                setHoveredInfo(null);
              }
            }
          }
        }
      }

      rafId = window.requestAnimationFrame(loop);
    };

    rafId = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(rafId);
  }, [stabilized, heroWindow, isPlaying, showVoronoi, showHulls]);

  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      const vp = viewportRef.current;
      const fr = lastInterpRef.current;
      if (!canvas || !vp || !fr) return;

      const rect = canvas.getBoundingClientRect();
      const [mx, my] = canvasToPitch(
        e.clientX - rect.left,
        e.clientY - rect.top,
        vp
      );

      let closest: StabilizedPlayer | null = null;
      let minDist = 4.0;
      for (const p of fr.players) {
        const d = Math.hypot(p.x - mx, p.y - my);
        if (d < minDist) {
          minDist = d;
          closest = p;
        }
      }

      if (closest) {
        hoveredIdRef.current = closest.id;
        const [px, py] = pitchToCanvas(closest.x, closest.y, vp);
        setHoveredInfo({
          player: closest,
          leftPct: Math.max(12, Math.min(88, (px / vp.width) * 100)),
          topPct: Math.max(14, Math.min(82, (py / vp.height) * 100)),
        });
      } else if (hoveredIdRef.current !== null) {
        hoveredIdRef.current = null;
        setHoveredInfo(null);
      }
    },
    []
  );

  const handleCanvasMouseLeave = useCallback(() => {
    hoveredIdRef.current = null;
    setHoveredInfo(null);
  }, []);

  const handleCanvasClick = useCallback(() => {
    router.push(`/match/${targetMatchId}`);
  }, [router, targetMatchId]);

  const matchDisplay = featuredMatch
    ? resolveMatchDisplay(featuredMatch, t, lang)
    : null;
  const CornerIcon = isRTL ? ArrowUpLeft : ArrowUpRight;

  const formatTimecode = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
  };

  return (
    <div className="obsidian-card studio-hero-frame rounded-2xl p-3.5 sm:p-4 relative overflow-hidden select-none">
      {/* ── Top Studio Control Bezel ── */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-lg obsidian-well min-w-0">
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${
              isPlaying ? "bg-tv-mint animate-pulse" : "bg-tv-amber"
            }`}
          />
          <span className="text-[11px] font-bold text-slate-200 truncate">
            {t("hub.schematic_live")}
          </span>
          <span className="text-slate-600">·</span>
          <BidiNum
            data-hero-timecode={hud.timeSec.toFixed(2)}
            data-hero-frame={hud.frameNum}
            className="text-[11px] font-bold text-tv-cyan"
          >
            {formatTimecode(hud.timeSec)}
          </BidiNum>
        </div>

        {/* Compact Interactive Telemetry Controls */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            type="button"
            onClick={() => setShowVoronoi((v) => !v)}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-semibold border transition-colors cursor-pointer ${
              showVoronoi
                ? "bg-tv-cyan/15 border-tv-cyan/35 text-tv-cyan"
                : "obsidian-well text-slate-400 hover:text-white"
            }`}
          >
            <Grid className="w-3 h-3" />
            <span>{t("hub.hero_voronoi_short")}</span>
          </button>

          <button
            type="button"
            onClick={() => setShowHulls((h) => !h)}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-semibold border transition-colors cursor-pointer ${
              showHulls
                ? "bg-tv-coral/15 border-tv-coral/35 text-tv-coral"
                : "obsidian-well text-slate-400 hover:text-white"
            }`}
          >
            <Shield className="w-3 h-3" />
            <span>{t("hub.hero_hull_short")}</span>
          </button>

          <button
            type="button"
            onClick={() => setIsPlaying((p) => !p)}
            aria-label={isPlaying ? t("wr.pause") : t("wr.play")}
            className="p-1.5 rounded-lg obsidian-well hover:border-tv-cyan/40 text-slate-200 hover:text-tv-cyan transition-colors cursor-pointer"
          >
            {isPlaying ? (
              <Pause className="w-3.5 h-3.5 fill-current" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )}
          </button>
        </div>
      </div>

      {/* ── Live 60FPS Real-Telemetry Broadcast Pitch Canvas ── */}
      <div
        ref={containerRef}
        dir="ltr"
        className="relative aspect-[105/68] w-full rounded-xl overflow-hidden border border-white/[0.1] bg-[#060B10] shadow-inner"
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleCanvasMouseMove}
          onMouseLeave={handleCanvasMouseLeave}
          onClick={handleCanvasClick}
          className="block w-full h-full cursor-pointer"
        />

        {/* Sequence Loop Timeline Hairline */}
        <div className="absolute bottom-0 inset-x-0 h-[2px] bg-white/[0.06] pointer-events-none">
          <div
            style={{ width: `${hud.progressPct}%` }}
            className="h-full bg-gradient-to-r from-tv-cyan to-tv-mint transition-[width] duration-75"
          />
        </div>

        {/* Live Floating Glass Player Telemetry Tooltip */}
        {hoveredInfo && (
          <div
            dir={isRTL ? "rtl" : "ltr"}
            style={{
              left: `${hoveredInfo.leftPct}%`,
              top: `${hoveredInfo.topPct}%`,
            }}
            className="absolute -translate-x-1/2 -translate-y-[135%] pointer-events-none z-20 px-2.5 py-1.5 rounded-lg obsidian-popover text-[11px] flex items-center gap-2 whitespace-nowrap shadow-xl"
          >
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                hoveredInfo.player.team === 0 ? "bg-tv-cyan" : "bg-tv-coral"
              }`}
            />
            <span className="font-bold text-white">
              {formatTeamName(hoveredInfo.player.team, lang)}{" "}
              <BidiNum>#{hoveredInfo.player.id}</BidiNum>
            </span>
            <span className="text-slate-600">·</span>
            <span className="inline-flex items-baseline gap-0.5 text-tv-mint font-bold">
              <BidiNum>{hoveredInfo.player.spd.toFixed(1)}</BidiNum>
              <span>{t("km_h")}</span>
            </span>
          </div>
        )}

        {/* Loading State Until Real Telemetry Arrives */}
        {!stabilized && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#060B10]/80">
            <span className="text-xs text-slate-400 animate-pulse">
              {matchDisplay?.title || t("hub.schematic_live")}
            </span>
          </div>
        )}
      </div>

      {/* ── Live Interpolated Metrics & Source Deep Link ── */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2.5 text-xs">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-slate-400">{t("hub.schematic_voronoi")}:</span>
            <span className="inline-flex items-center gap-1 font-bold">
              <BidiNum className="text-tv-cyan">{hud.vor0}%</BidiNum>
              <span className="text-slate-600">/</span>
              <BidiNum className="text-tv-coral">{hud.vor1}%</BidiNum>
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 ps-3 border-s border-white/[0.08] text-[11px] text-slate-400">
            <span>{t("hub.schematic_hull")}:</span>
            <BidiNum className="font-semibold text-slate-200">
              {hud.hull0} / {hud.hull1}
            </BidiNum>
            <span>{t("meters")}²</span>
          </div>
        </div>

        <Link
          href={`/match/${targetMatchId}`}
          className="inline-flex items-center gap-1 text-[11px] font-semibold text-tv-cyan hover:text-white transition-colors truncate max-w-[240px]"
        >
          <span className="truncate">
            {matchDisplay
              ? `${t("hub.hero_live_from")}: ${matchDisplay.title}`
              : t("hub.hero_open_match")}
          </span>
          <CornerIcon className="w-3.5 h-3.5 shrink-0" />
        </Link>
      </div>
    </div>
  );
}
