"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Crosshair, VideoOff } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { Telemetry } from "@/lib/types";
import {
  resolvePlayerIdentity,
  syncMatchSquadNumbers,
  usePlayerOverrides,
} from "@/lib/jerseyOverrides";
import { BidiNum } from "@/components/ui/BidiNum";
import {
  stabilizeTelemetry,
  interpolateStabilizedFrame,
  type StabilizedFrame,
} from "@/lib/radarStabilizer";
import {
  computePitchViewport,
  canvasToPitch,
  PitchViewport,
} from "./canvas/pitchConfig";
import { drawPitch } from "./canvas/drawPitch";
import { VoronoiBuffer } from "./canvas/drawVoronoi";
import { drawHulls } from "./canvas/drawHulls";
import { drawDefLines } from "./canvas/drawDefLines";
import { HeatmapRenderer } from "./canvas/drawHeatmap";
import { drawPassNetwork } from "./canvas/drawPassNetwork";
import { drawPlayers, BallHistoryPoint } from "./canvas/drawPlayers";
import type { CanvasLayerConfig, LiveTelemetryHud } from "./LayerToggles";

interface PitchCanvasProps {
  matchId?: string;
  telemetry: Telemetry | null;
  getVideoElement: () => HTMLVideoElement | null;
  layers: CanvasLayerConfig;
  selectedPlayerId: number | null;
  onSelectPlayer: (id: number | null) => void;
  onHudUpdate?: (hud: LiveTelemetryHud) => void;
}

export function PitchCanvas({
  matchId,
  telemetry,
  getVideoElement,
  layers,
  selectedPlayerId,
  onSelectPlayer,
  onHudUpdate,
}: PitchCanvasProps) {
  const { t } = useLang();
  const { overrides } = usePlayerOverrides(matchId);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const voronoiBufferRef = useRef<VoronoiBuffer | null>(null);
  const heatmapRendererRef = useRef<HeatmapRenderer | null>(null);

  const viewportRef = useRef<PitchViewport | null>(null);
  const lastInterpRef = useRef<StabilizedFrame | null>(null);
  const lastNotifiedKeyRef = useRef<string>("");
  const lastTrackingPausedRef = useRef<boolean>(false);
  const [isTrackingPaused, setIsTrackingPaused] = useState<boolean>(false);
  const hoveredIdRef = useRef<number | null>(null);
  const [hoveredInfo, setHoveredInfo] = useState<{
    id: number;
    team: 0 | 1;
    role: "p" | "gk" | "referee";
    spd: number;
    jn?: number | null;
    jv?: boolean | null;
    pxX: number;
    pxY: number;
  } | null>(null);

  // Keep match-wide collision-free 1..99 squad slot mapping synced for unverified tracks > 99
  useMemo(
    () => syncMatchSquadNumbers(matchId, telemetry, overrides),
    [matchId, telemetry, overrides]
  );

  // Stabilize raw tracker & homography signal once per telemetry load
  const stabilized = useMemo(() => stabilizeTelemetry(telemetry), [telemetry]);

  useEffect(() => {
    if (!voronoiBufferRef.current) {
      voronoiBufferRef.current = new VoronoiBuffer();
    }
    if (!heatmapRendererRef.current) {
      heatmapRendererRef.current = new HeatmapRenderer();
    }
  }, []);

  // ── 60FPS Sub-Frame Interpolated Broadcast Rendering Loop ──
  useEffect(() => {
    let rafId: number;

    const renderLoop = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      const videoEl = getVideoElement();

      if (canvas && ctx && stabilized && stabilized.frames.length > 0) {
        ctx.direction = "ltr";

        const currentTime = videoEl?.currentTime ?? 0;
        const fps = stabilized.fps || 25.0;
        const exactFrame = Math.max(
          0,
          Math.min(stabilized.frames.length - 1, currentTime * fps)
        );
        const frameIdx = Math.floor(exactFrame);
        const interpFrame = interpolateStabilizedFrame(stabilized, exactFrame);

        if (interpFrame) {
          lastInterpRef.current = interpFrame;

          // Build short ~0.55s ball comet trail strictly from valid past frames
          const ballTrail: BallHistoryPoint[] = [];
          const trailWindow = Math.max(5, Math.round(0.55 * fps));
          const trailStart = Math.max(0, frameIdx - trailWindow);
          for (let i = trailStart; i <= frameIdx; i++) {
            const bf = stabilized.frames[i];
            if (!bf?.homographyValid) {
              ballTrail.length = 0;
              continue;
            }
            if (bf?.ball) {
              ballTrail.push({
                x: bf.ball[0],
                y: bf.ball[1],
                frame: bf.f,
              });
            }
          }
          if (interpFrame.homographyValid && interpFrame.ball) {
            ballTrail.push({
              x: interpFrame.ball[0],
              y: interpFrame.ball[1],
              frame: interpFrame.f,
            });
          }

          const rect = canvas.getBoundingClientRect();
          const dpr = window.devicePixelRatio || 1;
          const vp = computePitchViewport(rect.width, rect.height, 0.045);
          viewportRef.current = vp;

          ctx.save();
          ctx.scale(dpr, dpr);

          drawPitch(ctx, vp);

          // When homography is paused (genuine close-up / celebration / replay),
          // render ONLY the clean empty pitch + centered pause badge — ZERO stale dots or layers!
          if (interpFrame.homographyValid) {
            if (layers.heatmap && heatmapRendererRef.current && telemetry?.heatmaps) {
              heatmapRendererRef.current.render(
                ctx,
                telemetry.heatmaps,
                layers.heatmapMode,
                vp
              );
            }

            if (layers.voronoi && voronoiBufferRef.current) {
              voronoiBufferRef.current.render(ctx, interpFrame.players, vp);
            }

            if (layers.hulls) {
              drawHulls(ctx, interpFrame.players, vp);
            }

            if (layers.defLines) {
              drawDefLines(ctx, interpFrame.def_x, vp);
            }

            if (layers.passNetwork && telemetry?.passing_network) {
              drawPassNetwork(ctx, telemetry, layers.players, vp, overrides);
            }

            if (layers.players) {
              const trajectory =
                selectedPlayerId !== null
                  ? stabilized.trajectories.get(selectedPlayerId)
                  : undefined;
              drawPlayers(
                ctx,
                interpFrame.players,
                interpFrame.ball,
                ballTrail,
                selectedPlayerId,
                trajectory,
                interpFrame.f,
                vp,
                hoveredIdRef.current,
                overrides,
                fps
              );
            }
          }

          ctx.restore();

          // Notify LiveTacticalHud smoothly (keyed at 20Hz quantized step so UI stays crisp)
          const pausedNow = !interpFrame.homographyValid;
          if (pausedNow !== lastTrackingPausedRef.current) {
            lastTrackingPausedRef.current = pausedNow;
            setIsTrackingPaused(pausedNow);
            if (pausedNow) {
              hoveredIdRef.current = null;
              setHoveredInfo(null);
            }
          }

          const notifyKey = `${Math.round(exactFrame * 2)}:${interpFrame.hullArea.team0_m2}:${interpFrame.hullArea.team1_m2}`;
          if (notifyKey !== lastNotifiedKeyRef.current) {
            lastNotifiedKeyRef.current = notifyKey;

            const def0 = interpFrame.def_x["0"] ?? null;
            const def1 = interpFrame.def_x["1"] ?? null;
            const activeFieldPlayers = interpFrame.players.filter((p) => p.role !== "referee");
            const sprinting = activeFieldPlayers.filter((p) => p.spd >= 24.0).length;

            onHudUpdate?.({
              voronoiSplit: interpFrame.vor,
              team0HullArea_m2: interpFrame.hullArea.team0_m2,
              team1HullArea_m2: interpFrame.hullArea.team1_m2,
              defLine0_x: def0,
              defLine1_x: def1,
              activePlayersCount: activeFieldPlayers.length,
              sprintingCount: sprinting,
            });
          }
        }
      }

      rafId = window.requestAnimationFrame(renderLoop);
    };

    rafId = window.requestAnimationFrame(renderLoop);
    return () => window.cancelAnimationFrame(rafId);
  }, [stabilized, telemetry, getVideoElement, layers, selectedPlayerId, onHudUpdate, overrides]);

  // ── ResizeObserver for HiDPI Crispness ──
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const handleResize = () => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };

    handleResize();

    const ro = new ResizeObserver(handleResize);
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // ── Hit-Testing Against Interpolated Stabilized Positions (~3.5m radius) ──
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      const vp = viewportRef.current;
      const currentFrame = lastInterpRef.current;

      if (!canvas || !vp || !currentFrame || !currentFrame.homographyValid || !currentFrame.players) return;

      const rect = canvas.getBoundingClientRect();
      const clickCanvasX = e.clientX - rect.left;
      const clickCanvasY = e.clientY - rect.top;

      const [clickMetersX, clickMetersY] = canvasToPitch(
        clickCanvasX,
        clickCanvasY,
        vp
      );

      let closestPlayerId: number | null = null;
      let minDistance = 3.5;

      for (const p of currentFrame.players) {
        const dist = Math.hypot(p.x - clickMetersX, p.y - clickMetersY);
        if (dist < minDistance) {
          minDistance = dist;
          closestPlayerId = p.id;
        }
      }

      if (closestPlayerId !== null) {
        onSelectPlayer(closestPlayerId === selectedPlayerId ? null : closestPlayerId);
      } else {
        onSelectPlayer(null);
      }
    },
    [selectedPlayerId, onSelectPlayer]
  );

  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      const vp = viewportRef.current;
      const currentFrame = lastInterpRef.current;
      if (!canvas || !vp || !currentFrame || !currentFrame.homographyValid || !currentFrame.players) {
        hoveredIdRef.current = null;
        setHoveredInfo(null);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const pxX = e.clientX - rect.left;
      const pxY = e.clientY - rect.top;
      const [mx, my] = canvasToPitch(pxX, pxY, vp);

      let closest = null as (typeof currentFrame.players)[number] | null;
      let minDistance = 3.2;
      for (const p of currentFrame.players) {
        if (p.role === "referee") continue;
        const dist = Math.hypot(p.x - mx, p.y - my);
        if (dist < minDistance) {
          minDistance = dist;
          closest = p;
        }
      }
      if (closest) {
        hoveredIdRef.current = closest.id;
        setHoveredInfo({
          id: closest.id,
          team: closest.team,
          role: closest.role,
          spd: closest.spd,
          jn: closest.jn,
          jv: closest.jv,
          pxX,
          pxY,
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

  return (
    <div
      ref={containerRef}
      className="relative aspect-[105/68] w-full bg-[#060B10] cursor-crosshair group select-none overflow-hidden"
    >
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={handleCanvasMouseLeave}
        className="block w-full h-full"
      />

      {hoveredInfo && (() => {
        const hid = resolvePlayerIdentity(hoveredInfo.id, hoveredInfo, overrides);
        return (
          <div
            style={{
              left: Math.min(Math.max(8, hoveredInfo.pxX + 12), 420),
              top: Math.max(8, hoveredInfo.pxY - 46),
            }}
            className="absolute z-20 pointer-events-none px-2.5 py-1.5 rounded-lg bg-slate-950/95 border border-white/15 shadow-xl text-[11px] whitespace-nowrap"
          >
            <div className="flex items-center gap-1.5 font-bold text-white">
              <span
                className={`w-2 h-2 rounded-full ${
                  hoveredInfo.team === 0 ? "bg-tv-cyan" : "bg-tv-coral"
                }`}
              />
              {hid.isVerified ? (
                <span>
                  {hid.customName ? `${hid.customName} ` : ""}
                  <BidiNum className="text-tv-mint">{hid.badgeText}</BidiNum>
                </span>
              ) : (
                <span className="text-slate-200">
                  {t("jersey.unidentified_player")}{" "}
                  <BidiNum className="text-slate-400">{`(#${hoveredInfo.id})`}</BidiNum>
                </span>
              )}
              <BidiNum className="text-slate-400 font-mono ms-1">
                {`${hoveredInfo.spd.toFixed(1)} km/h`}
              </BidiNum>
            </div>
          </div>
        );
      })()}

      {isTrackingPaused && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-4 z-10">
          <div className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-950/85 backdrop-blur-md border border-white/15 shadow-xl text-xs font-semibold text-slate-200">
            <VideoOff className="w-3.5 h-3.5 text-tv-amber shrink-0" />
            <span>{t("wr.closeup_paused")}</span>
          </div>
        </div>
      )}

      {selectedPlayerId === null && (
        <div className="absolute bottom-2 start-2.5 inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-[11px] text-slate-300 bg-black/70 backdrop-blur-sm border border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
          <Crosshair className="w-3 h-3 text-tv-cyan" />
          <span>{t("wr.inspect_hint")}</span>
        </div>
      )}

      <div className="absolute bottom-2 end-2.5 px-2 py-0.5 rounded-md text-[10px] font-medium text-slate-400 bg-black/70 backdrop-blur-sm border border-white/[0.08] pointer-events-none">
        {t("wr.pitch_spec")}
      </div>
    </div>
  );
}
