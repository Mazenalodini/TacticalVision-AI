"use client";

import React, { useCallback, useEffect, useMemo, useRef } from "react";
import { Crosshair } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { Telemetry } from "@/lib/types";
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

  const voronoiBufferRef = useRef<VoronoiBuffer | null>(null);
  const heatmapRendererRef = useRef<HeatmapRenderer | null>(null);

  const viewportRef = useRef<PitchViewport | null>(null);
  const lastInterpRef = useRef<StabilizedFrame | null>(null);
  const lastNotifiedKeyRef = useRef<string>("");

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

          // Build ball trail strictly from clean past window (works identically when seeking or scrubbing!)
          const ballTrail: BallHistoryPoint[] = [];
          const trailStart = Math.max(0, frameIdx - 16);
          for (let i = trailStart; i <= frameIdx; i++) {
            const bf = stabilized.frames[i];
            if (bf?.ball) {
              ballTrail.push({
                x: bf.ball[0],
                y: bf.ball[1],
                frame: bf.f,
              });
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
          const vp = computePitchViewport(rect.width, rect.height, 0.045);
          viewportRef.current = vp;

          ctx.save();
          ctx.scale(dpr, dpr);

          drawPitch(ctx, vp);

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
            drawPassNetwork(ctx, telemetry, layers.players, vp);
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
              vp
            );
          }

          ctx.restore();

          // Notify LiveTacticalHud smoothly (keyed at 20Hz quantized step so UI stays crisp)
          const notifyKey = `${Math.round(exactFrame * 2)}:${interpFrame.hullArea.team0_m2}:${interpFrame.hullArea.team1_m2}`;
          if (notifyKey !== lastNotifiedKeyRef.current) {
            lastNotifiedKeyRef.current = notifyKey;

            const def0 = interpFrame.def_x["0"] ?? null;
            const def1 = interpFrame.def_x["1"] ?? null;
            const sprinting = interpFrame.players.filter((p) => p.spd >= 24.0).length;

            onHudUpdate?.({
              voronoiSplit: interpFrame.vor,
              team0HullArea_m2: interpFrame.hullArea.team0_m2,
              team1HullArea_m2: interpFrame.hullArea.team1_m2,
              defLine0_x: def0,
              defLine1_x: def1,
              activePlayersCount: interpFrame.players.length,
              sprintingCount: sprinting,
            });
          }
        }
      }

      rafId = window.requestAnimationFrame(renderLoop);
    };

    rafId = window.requestAnimationFrame(renderLoop);
    return () => window.cancelAnimationFrame(rafId);
  }, [stabilized, telemetry, getVideoElement, layers, selectedPlayerId, onHudUpdate]);

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

      if (!canvas || !vp || !currentFrame || !currentFrame.players) return;

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

  return (
    <div
      ref={containerRef}
      className="relative aspect-[105/68] w-full bg-[#060B10] cursor-crosshair group select-none overflow-hidden"
    >
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="block w-full h-full"
      />

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
