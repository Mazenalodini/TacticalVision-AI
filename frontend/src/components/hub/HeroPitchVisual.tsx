"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import * as d3 from "d3";
import { ArrowUpRight, ArrowUpLeft } from "lucide-react";
import { useLang } from "@/context/LangContext";
import { fetchTelemetry } from "@/lib/api";
import { resolveMatchDisplay } from "@/lib/format";
import type { MatchCard, Telemetry, FramePlayer } from "@/lib/types";
import { BidiNum } from "@/components/ui/BidiNum";

interface HeroPitchVisualProps {
  featuredMatch?: MatchCard | null;
}

/**
 * 100% Real-Data Match Snapshot Preview for Match Hub Hero.
 * Fetches actual `telemetry.json` of `featuredMatch` (zero hardcoded coordinates or fake xT)
 * and renders a real synchronized tactical frame (players, ball, convex hull, def lines, Voronoi split).
 */
export function HeroPitchVisual({ featuredMatch }: HeroPitchVisualProps) {
  const { t, lang, isRTL } = useLang();
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [hoveredNode, setHoveredNode] = useState<FramePlayer | null>(null);

  useEffect(() => {
    const targetId = featuredMatch?.id;
    if (!targetId) return;

    let cancelled = false;
    fetchTelemetry(targetId)
      .then((data) => {
        if (!cancelled) {
          setTelemetry(data);
        }
      })
      .catch(() => {
        // Silent fallback when backend is offline
      });

    return () => {
      cancelled = true;
    };
  }, [featuredMatch?.id]);

  // Pick a genuine informative frame (around 15%-75% into clip, maximizing players + ball + def_x)
  const snapshotFrame = useMemo(() => {
    const frames = telemetry?.frames;
    if (!frames || frames.length === 0) return null;

    const startIdx = Math.floor(frames.length * 0.15);
    const endIdx = Math.max(startIdx + 1, Math.floor(frames.length * 0.8));

    let best = frames[startIdx] || frames[0];
    let bestScore = -1;

    for (let i = startIdx; i < endIdx; i++) {
      const fr = frames[i];
      const defCount = Object.keys(fr.def_x || {}).length;
      const score =
        (fr.players?.length || 0) + (fr.ball ? 5 : 0) + defCount * 3;
      if (score > bestScore) {
        bestScore = score;
        best = fr;
      }
    }
    return best;
  }, [telemetry]);

  const pad = 16;
  const pitchW = 420;
  const pitchH = 272;
  const vbW = pitchW + pad * 2;
  const vbH = pitchH + pad * 2;

  const scaleX = useMemo(
    () => d3.scaleLinear().domain([0, 105]).range([pad, pad + pitchW]),
    []
  );
  const scaleY = useMemo(
    () => d3.scaleLinear().domain([0, 68]).range([pad, pad + pitchH]),
    []
  );

  // Real outfield Convex Hulls computed directly from snapshotFrame.players
  const { hull0Path, hull1Path } = useMemo(() => {
    if (!snapshotFrame?.players) return { hull0Path: "", hull1Path: "" };

    const buildHull = (teamNum: 0 | 1) => {
      const pts: [number, number][] = snapshotFrame.players
        .filter(
          (p) =>
            p.team === teamNum &&
            p.role === "p" &&
            p.x >= -2 &&
            p.x <= 107 &&
            p.y >= -2 &&
            p.y <= 70
        )
        .map((p) => [
          scaleX(Math.max(0, Math.min(105, p.x))),
          scaleY(Math.max(0, Math.min(68, p.y))),
        ]);
      if (pts.length < 3) return "";
      const hull = d3.polygonHull(pts);
      if (!hull) return "";
      return `M ${hull
        .map((pt) => `${pt[0].toFixed(1)},${pt[1].toFixed(1)}`)
        .join(" L ")} Z`;
    };

    return { hull0Path: buildHull(0), hull1Path: buildHull(1) };
  }, [snapshotFrame, scaleX, scaleY]);

  // Exact matching Voronoi percentages (always sum to 100%)
  const vor0Pct = Math.round(
    snapshotFrame?.vor?.[0] ?? featuredMatch?.pitch_control_pct?.team_0 ?? 50
  );
  const vor1Pct = 100 - vor0Pct;

  const def0X = snapshotFrame?.def_x?.["0"];
  const def1X = snapshotFrame?.def_x?.["1"];
  const activeDefLine =
    typeof def1X === "number"
      ? { label: "B", x: def1X, color: "#FB7185" }
      : typeof def0X === "number"
      ? { label: "A", x: def0X, color: "#38BDF8" }
      : null;

  const matchDisplay = featuredMatch
    ? resolveMatchDisplay(featuredMatch, t, lang)
    : null;
  const CornerIcon = isRTL ? ArrowUpLeft : ArrowUpRight;

  return (
    <div className="obsidian-card rounded-2xl p-4 sm:p-5 relative overflow-hidden select-none">
      {/* Top Telemetry Bezel: Real Match Source Badge + Pitch Spec */}
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-lg obsidian-well min-w-0">
          <span className="w-2 h-2 rounded-full bg-tv-mint animate-pulse shrink-0" />
          <span className="text-[11px] font-bold text-slate-200 truncate">
            {t("hub.schematic_live")}
          </span>
        </div>

        <span className="inline-flex items-baseline gap-1 text-[11px] text-slate-400 px-2.5 py-1 rounded-lg obsidian-well shrink-0">
          <BidiNum>
            {telemetry?.match_info?.pitch_dimensions_m
              ? `${telemetry.match_info.pitch_dimensions_m[0]}×${telemetry.match_info.pitch_dimensions_m[1]}`
              : "105×68"}
          </BidiNum>
          <span>{t("meters")}</span>
          <span className="text-slate-600">·</span>
          <BidiNum>{Math.round(featuredMatch?.fps ?? 25)}</BidiNum>
          <span>{t("card.fps")}</span>
        </span>
      </div>

      {/* Real-Frame SVG FIFA Pitch */}
      <div
        dir="ltr"
        className="relative rounded-xl overflow-hidden border border-white/[0.09] bg-[#0C1B17]"
      >
        <svg
          viewBox={`0 0 ${vbW} ${vbH}`}
          className="w-full h-auto block"
          role="img"
          aria-label={t("hub.schematic_live")}
        >
          <defs>
            <linearGradient id="heroRealVoronoi" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.16" />
              <stop
                offset={`${Math.max(10, Math.min(90, vor0Pct))}%`}
                stopColor="#38BDF8"
                stopOpacity="0.06"
              />
              <stop
                offset={`${Math.max(10, Math.min(90, vor0Pct + 5))}%`}
                stopColor="#FB7185"
                stopOpacity="0.07"
              />
              <stop offset="100%" stopColor="#FB7185" stopOpacity="0.17" />
            </linearGradient>
          </defs>

          {/* Calm Stadium Slate-Emerald Turf Stripes */}
          {Array.from({ length: 10 }).map((_, idx) => (
            <rect
              key={`stripe-${idx}`}
              x={pad + idx * (pitchW / 10)}
              y={pad}
              width={pitchW / 10}
              height={pitchH}
              fill={idx % 2 === 0 ? "#0C1B17" : "#10231E"}
            />
          ))}

          {/* Dynamic Voronoi Split Matched 1:1 to Real Frame Percentage */}
          {snapshotFrame && (
            <rect
              x={pad}
              y={pad}
              width={pitchW}
              height={pitchH}
              fill="url(#heroRealVoronoi)"
            />
          )}

          {/* Standard Crisp FIFA Markings */}
          <g stroke="rgba(255,255,255,0.24)" strokeWidth="1.2" fill="none">
            <rect x={pad} y={pad} width={pitchW} height={pitchH} />
            <line
              x1={scaleX(52.5)}
              y1={pad}
              x2={scaleX(52.5)}
              y2={pad + pitchH}
            />
            <circle cx={scaleX(52.5)} cy={scaleY(34)} r={9.15 * (pitchW / 105)} />
            <rect
              x={pad}
              y={scaleY((68 - 40.32) / 2)}
              width={16.5 * (pitchW / 105)}
              height={40.32 * (pitchH / 68)}
            />
            <rect
              x={scaleX(105 - 16.5)}
              y={scaleY((68 - 40.32) / 2)}
              width={16.5 * (pitchW / 105)}
              height={40.32 * (pitchH / 68)}
            />
          </g>

          {/* Real Outfield Convex Hulls */}
          {hull0Path && (
            <path
              d={hull0Path}
              fill="rgba(56, 189, 248, 0.11)"
              stroke="#38BDF8"
              strokeWidth="1.4"
              strokeDasharray="5 4"
            />
          )}
          {hull1Path && (
            <path
              d={hull1Path}
              fill="rgba(251, 113, 133, 0.11)"
              stroke="#FB7185"
              strokeWidth="1.4"
              strokeDasharray="5 4"
            />
          )}

          {/* Real Virtual Defensive Line from Frame */}
          {activeDefLine && (
            <g>
              <line
                x1={scaleX(Math.max(0, Math.min(105, activeDefLine.x)))}
                y1={pad}
                x2={scaleX(Math.max(0, Math.min(105, activeDefLine.x)))}
                y2={pad + pitchH}
                stroke={activeDefLine.color}
                strokeWidth="1.4"
                strokeDasharray="4 4"
              />
              <g
                transform={`translate(${Math.max(
                  pad + 4,
                  Math.min(
                    pad + pitchW - 56,
                    scaleX(activeDefLine.x) - 26
                  )
                )}, ${pad + pitchH - 20})`}
              >
                <rect
                  width="52"
                  height="16"
                  rx="4"
                  fill="rgba(8, 13, 24, 0.92)"
                  stroke={activeDefLine.color}
                  strokeOpacity="0.65"
                  strokeWidth="1"
                />
                <text
                  x="26"
                  y="11"
                  textAnchor="middle"
                  fill={activeDefLine.color}
                  fontSize="9.2"
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  {activeDefLine.label} {activeDefLine.x.toFixed(1)}m
                </text>
              </g>
            </g>
          )}

          {/* Real Tracked Ball */}
          {snapshotFrame?.ball && (
            <circle
              cx={scaleX(Math.max(0, Math.min(105, snapshotFrame.ball[0])))}
              cy={scaleY(Math.max(0, Math.min(68, snapshotFrame.ball[1])))}
              r="4.5"
              fill="#FFFFFF"
              stroke="#34D399"
              strokeWidth="2"
            />
          )}

          {/* Real Tracked Players from `telemetry.frames` */}
          {(snapshotFrame?.players || []).map((p) => {
            if (p.x < -3 || p.x > 108 || p.y < -3 || p.y > 71) return null;
            const cx = scaleX(Math.max(0, Math.min(105, p.x)));
            const cy = scaleY(Math.max(0, Math.min(68, p.y)));
            const isTeam0 = p.team === 0;
            const isSprint = p.spd >= 24.0;
            const isHovered = hoveredNode?.id === p.id;
            const fill = isTeam0 ? "#38BDF8" : "#FB7185";

            return (
              <g
                key={p.id}
                onMouseEnter={() => setHoveredNode(p)}
                onMouseLeave={() => setHoveredNode(null)}
                className="cursor-pointer"
              >
                {isSprint && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 12 : 9.8}
                    fill="none"
                    stroke="#34D399"
                    strokeWidth="1.3"
                    strokeOpacity="0.85"
                  />
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 8.2 : 6.6}
                  fill={fill}
                  stroke="#060912"
                  strokeWidth="1.6"
                />
                <text
                  x={cx}
                  y={cy + 2.8}
                  textAnchor="middle"
                  fill={isTeam0 ? "#041019" : "#FFFFFF"}
                  fontSize="7.4"
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  {p.id}
                </text>
              </g>
            );
          })}

          {/* Interactive Real Speed Tooltip */}
          {hoveredNode && (
            <g
              transform={`translate(${Math.min(
                vbW - 112,
                Math.max(
                  12,
                  scaleX(Math.max(0, Math.min(105, hoveredNode.x))) - 52
                )
              )}, ${Math.max(
                10,
                scaleY(Math.max(0, Math.min(68, hoveredNode.y))) - 30
              )})`}
            >
              <rect
                width="104"
                height="20"
                rx="6"
                fill="rgba(8, 13, 24, 0.96)"
                stroke="rgba(255,255,255,0.22)"
                strokeWidth="1"
              />
              <text
                x="52"
                y="13.5"
                textAnchor="middle"
                fill="#F8FAFC"
                fontSize="9.5"
                fontFamily="monospace"
                fontWeight="bold"
              >
                #{hoveredNode.id} · {hoveredNode.spd.toFixed(1)} km/h
              </text>
            </g>
          )}
        </svg>

        {/* Subtle Shimmer Overlay Only While Telemetry Loads */}
        {!snapshotFrame && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/25">
            <span className="text-xs text-slate-400 animate-pulse">
              {matchDisplay?.title || t("hub.schematic_live")}
            </span>
          </div>
        )}
      </div>

      {/* Real-Data Legend & Deep Link Footer */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2 text-slate-400">
          <span>{t("hub.schematic_voronoi")}:</span>
          <span className="inline-flex items-center gap-1 font-bold">
            <BidiNum className="text-tv-cyan">{vor0Pct}%</BidiNum>
            <span className="text-slate-600">/</span>
            <BidiNum className="text-tv-coral">{vor1Pct}%</BidiNum>
          </span>
        </div>

        {featuredMatch && matchDisplay && (
          <Link
            href={`/match/${featuredMatch.id}`}
            className="inline-flex items-center gap-1 text-[11px] font-semibold text-tv-cyan hover:text-white transition-colors truncate max-w-[230px]"
          >
            <span className="truncate">
              {t("hub.hero_live_from")}: {matchDisplay.title}
            </span>
            <CornerIcon className="w-3.5 h-3.5 shrink-0" />
          </Link>
        )}
      </div>
    </div>
  );
}
