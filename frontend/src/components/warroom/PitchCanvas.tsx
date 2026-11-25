"use client";

import React, { useCallback, useEffect, useRef } from "react";
import { Crosshair } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { Telemetry } from "@/lib/types";
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
import {
  drawPlayers,
  BallHistoryPoint,
  PlayerTrajectoryPoint,
} from "./canvas/drawPlayers";
import type { CanvasLayerConfig, LiveTelemetryHud } from "./LayerToggles";

interface PitchCanvasProps {
  telemetry: Telemetry | null;
  getVideoElement: () => HTMLVideoElement | null;
  layers: CanvasLayerConfig;
  selectedPlayerId: number | null;
  onSelectPlayer: (id: number | null) => void;
  onHudUpdate?: (hud: LiveTelemetryHud) => void;
}

export function PitchCanvas({
  telemetry,
  getVideoElement,
  layers,
  selectedPlayerId,
  onSelectPlayer,
  onHudUpdate,
}: PitchCanvasProps) {
  const { t } = useLang();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Persistent offscreen canvas caches
  const voronoiBufferRef = useRef<VoronoiBuffer | null>(null);
  const heatmapRendererRef = useRef<HeatmapRenderer | null>(null);

  // Ball history for comet trail
  const ballHistoryRef = useRef<BallHistoryPoint[]>([]);
  const lastRecordedBallFrameRef = useRef<number>(-1);

  // Viewport tracking for click hit-testing
  const viewportRef = useRef<PitchViewport | null>(null);
  const lastNotifiedFrameRef = useRef<number>(-1);

  // Precompute memoized trajectory points for all players (zero per-frame searching)
  const trajectoriesMap = React.useMemo(() => {
    const map = new Map<number, PlayerTrajectoryPoint[]>();
    if (!telemetry?.frames) return map;
    for (const f of telemetry.frames) {
      if (!f.players) continue;
      for (const p of f.players) {
        let arr = map.get(p.id);
        if (!arr) {
          arr = [];
          map.set(p.id, arr);
        }
        arr.push({ x: p.x, y: p.y, frame: f.f });
      }
    }
    return map;
  }, [telemetry]);

  useEffect(() => {
    if (!voronoiBufferRef.current) {
      voronoiBufferRef.current = new VoronoiBuffer();
    }
    if (!heatmapRendererRef.current) {
      heatmapRendererRef.current = new HeatmapRenderer();
    }
  }, []);

  // ── 60FPS RAF Rendering Loop ──
  useEffect(() => {
    let rafId: number;

    const renderLoop = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      const frames = telemetry?.frames;
      const videoEl = getVideoElement();

      if (canvas && ctx && frames && frames.length > 0) {
        ctx.direction = "ltr";

        const currentTime = videoEl?.currentTime ?? 0;
        const fps = telemetry.summary?.fps || 25.0;
        const frameIdx = Math.min(
          frames.length - 1,
          Math.max(0, Math.floor(currentTime * fps))
        );
        const currentFrame = frames[frameIdx];

        if (currentFrame && currentFrame.f !== lastRecordedBallFrameRef.current) {
          lastRecordedBallFrameRef.current = currentFrame.f;
          if (currentFrame.ball) {
            ballHistoryRef.current.push({
              x: currentFrame.ball[0],
              y: currentFrame.ball[1],
              frame: currentFrame.f,
            });
            if (ballHistoryRef.current.length > 18) {
              ballHistoryRef.current.shift();
            }
          } else {
            ballHistoryRef.current = [];
          }
        }

        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const vp = computePitchViewport(rect.width, rect.height, 0.045);
        viewportRef.current = vp;

        ctx.save();
        ctx.scale(dpr, dpr);

        drawPitch(ctx, vp);

        if (layers.heatmap && heatmapRendererRef.current && telemetry.heatmaps) {
          heatmapRendererRef.current.render(
            ctx,
            telemetry.heatmaps,
            layers.heatmapMode,
            vp
          );
        }

        if (layers.voronoi && voronoiBufferRef.current && currentFrame?.players) {
          voronoiBufferRef.current.render(ctx, currentFrame.players, vp);
        }

        let hullAreas = { team0Area_m2: 0, team1Area_m2: 0 };
        if (layers.hulls && currentFrame?.players) {
          hullAreas = drawHulls(ctx, currentFrame.players, vp);
        }

        if (layers.defLines && currentFrame?.def_x) {
          drawDefLines(ctx, currentFrame.def_x, vp);
        }

        if (layers.passNetwork && telemetry.passing_network) {
          drawPassNetwork(
            ctx,
            telemetry,
            layers.players,
            vp
          );
        }

        if (layers.players && currentFrame) {
          const trajectory =
            selectedPlayerId !== null
              ? trajectoriesMap.get(selectedPlayerId)
              : undefined;
          drawPlayers(
            ctx,
            currentFrame.players,
            currentFrame.ball,
            ballHistoryRef.current,
            selectedPlayerId,
            trajectory,
            frameIdx,
            vp
          );
        }

        ctx.restore();

        if (currentFrame && currentFrame.f !== lastNotifiedFrameRef.current) {
          lastNotifiedFrameRef.current = currentFrame.f;

          const defX = currentFrame.def_x;
          const def0 = defX?.["0"] !== undefined ? defX["0"] : null;
          const def1 = defX?.["1"] !== undefined ? defX["1"] : null;

          const players = currentFrame.players || [];
          const sprinting = players.filter((p) => p.spd >= 24.0).length;

          onHudUpdate?.({
            voronoiSplit: currentFrame.vor || [50, 50],
            team0HullArea_m2: hullAreas.team0Area_m2,
            team1HullArea_m2: hullAreas.team1Area_m2,
            defLine0_x: def0,
            defLine1_x: def1,
            activePlayersCount: players.length,
            sprintingCount: sprinting,
          });
        }
      }

      rafId = window.requestAnimationFrame(renderLoop);
    };

    rafId = window.requestAnimationFrame(renderLoop);
    return () => window.cancelAnimationFrame(rafId);
  }, [telemetry, getVideoElement, layers, selectedPlayerId, trajectoriesMap, onHudUpdate]);

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

  // ── Click Hit-Testing for Spotlight Selection (~3.5m radius) ──
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      const vp = viewportRef.current;
      const frames = telemetry?.frames;
      const videoEl = getVideoElement();

      if (!canvas || !vp || !frames || frames.length === 0) return;

      const rect = canvas.getBoundingClientRect();
      const clickCanvasX = e.clientX - rect.left;
      const clickCanvasY = e.clientY - rect.top;

      const [clickMetersX, clickMetersY] = canvasToPitch(
        clickCanvasX,
        clickCanvasY,
        vp
      );

      const currentTime = videoEl?.currentTime ?? 0;
      const fps = telemetry?.summary?.fps || 25.0;
      const frameIdx = Math.min(
        frames.length - 1,
        Math.max(0, Math.floor(currentTime * fps))
      );
      const currentFrame = frames[frameIdx];

      if (!currentFrame || !currentFrame.players) return;

      let closestPlayerId: number | null = null;
      let minDistance = 3.5;

      for (const p of currentFrame.players) {
        const dx = p.x - clickMetersX;
        const dy = p.y - clickMetersY;
        const dist = Math.sqrt(dx * dx + dy * dy);
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
    [telemetry, getVideoElement, selectedPlayerId, onSelectPlayer]
  );

  return (
    <div
      ref={containerRef}
      className="relative aspect-[105/68] w-full bg-[#070E0B] cursor-crosshair group select-none overflow-hidden"
    >
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="block w-full h-full"
      />

      {/* Non-intrusive interactive spotlight hint on hover when no player selected */}
      {selectedPlayerId === null && (
        <div className="absolute bottom-2 start-2.5 inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-[11px] text-slate-300 bg-black/65 backdrop-blur-sm border border-white/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
          <Crosshair className="w-3 h-3 text-tv-cyan" />
          <span>{t("wr.inspect_hint")}</span>
        </div>
      )}

      {/* Localized Pitch Spec Pill */}
      <div className="absolute bottom-2 end-2.5 px-2 py-0.5 rounded-md text-[10px] font-medium text-slate-400 bg-black/65 backdrop-blur-sm border border-white/[0.08] pointer-events-none">
        {t("wr.pitch_spec")}
      </div>
    </div>
  );
}
