"use client";

import React, { useEffect, useMemo, useState } from "react";
import * as d3 from "d3";
import { motion } from "framer-motion";
import { Share2, Grid, Sparkles, Play, Pause, RotateCcw, ArrowUpRight } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { Telemetry, PassingEdge, PassEvent } from "@/lib/types";
import {
  selectPassNetworkNodes,
  type PassNetworkNode,
} from "@/lib/telemetryHelpers";
import { resolvePlayerIdentity, usePlayerOverrides } from "@/lib/jerseyOverrides";
import { formatTeamName, formatRole } from "@/lib/format";
import { BidiNum } from "@/components/ui/BidiNum";

// Standard Karun Singh 12x8 Expected Threat (xT) Grid matching engine/config.py
const XT_GRID: number[][] = [
  [0.00638, 0.0078, 0.00845, 0.00978, 0.01126, 0.01248, 0.01474, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792],
  [0.0075, 0.00879, 0.00942, 0.01059, 0.01215, 0.01385, 0.01612, 0.0187, 0.02402, 0.02953, 0.04067, 0.04648],
  [0.00888, 0.00978, 0.01001, 0.0111, 0.01269, 0.01429, 0.0169, 0.01997, 0.02485, 0.03249, 0.04806, 0.06801],
  [0.00941, 0.01083, 0.01017, 0.01132, 0.01263, 0.01485, 0.0169, 0.01997, 0.02685, 0.0389, 0.065, 0.108],
  [0.00941, 0.01083, 0.01017, 0.01132, 0.01263, 0.01485, 0.0169, 0.01997, 0.02685, 0.0389, 0.065, 0.108],
  [0.00888, 0.00978, 0.01001, 0.0111, 0.01269, 0.01429, 0.0169, 0.01997, 0.02485, 0.03249, 0.04806, 0.06801],
  [0.0075, 0.00879, 0.00942, 0.01059, 0.01215, 0.01385, 0.01612, 0.0187, 0.02402, 0.02953, 0.04067, 0.04648],
  [0.00638, 0.0078, 0.00845, 0.00978, 0.01126, 0.01248, 0.01474, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792],
];

interface D3PassNetworkPitchProps {
  matchId?: string;
  telemetry: Telemetry | null;
  onSelectPlayer?: (playerId: number) => void;
  onJumpToPass?: (timeSec: number, fromId: number) => void;
}

interface ComputedArc {
  key: string;
  edge: PassingEdge;
  pathD: string;
  strokeW: number;
  xtDelta: number;
  isProgressive: boolean;
  firstPassTime: number | null;
  midX: number;
  midY: number;
}

export function D3PassNetworkPitch({
  matchId,
  telemetry,
  onSelectPlayer,
  onJumpToPass,
}: D3PassNetworkPitchProps) {
  const { t, lang, dir } = useLang();
  const { overrides } = usePlayerOverrides(matchId);
  const [teamFilter, setTeamFilter] = useState<"all" | "0" | "1">("all");
  const [showXtGrid, setShowXtGrid] = useState<boolean>(true);
  const [highlightProg, setHighlightProg] = useState<boolean>(false);
  const [hoveredPlayerId, setHoveredPlayerId] = useState<number | null>(null);
  const [isReplaying, setIsReplaying] = useState<boolean>(false);
  const [replayIdx, setReplayIdx] = useState<number>(-1);

  const padX = 28;
  const padY = 24;
  const pitchW = 630;
  const pitchH = 408;
  const vbW = pitchW + padX * 2;
  const vbH = pitchH + padY * 2;

  const scaleX = useMemo(
    () => d3.scaleLinear().domain([0, 105]).range([padX, padX + pitchW]),
    []
  );
  const scaleY = useMemo(
    () => d3.scaleLinear().domain([0, 68]).range([padY, padY + pitchH]),
    []
  );

  // Pairwise xT, progressive flags, first timestamp & chronological build-up sequence
  const { pairMetaMap, chronologicalPasses } = useMemo(() => {
    const map = new Map<string, { xt: number; prog: boolean; firstTime: number }>();
    const list: PassEvent[] = [];
    if (!telemetry?.events) return { pairMetaMap: map, chronologicalPasses: list };

    for (const ev of telemetry.events) {
      if (ev.type === "pass") {
        if (teamFilter === "all" || String(ev.team) === teamFilter) {
          list.push(ev);
        }
        const k = `${ev.from_id}-${ev.to_id}`;
        const prev = map.get(k);
        if (prev) {
          prev.xt += ev.xt_delta || 0;
          prev.prog = prev.prog || Boolean(ev.is_progressive);
          prev.firstTime = Math.min(prev.firstTime, ev.time_s);
        } else {
          map.set(k, {
            xt: ev.xt_delta || 0,
            prog: Boolean(ev.is_progressive),
            firstTime: ev.time_s,
          });
        }
      }
    }
    return { pairMetaMap: map, chronologicalPasses: list };
  }, [telemetry, teamFilter]);

  useEffect(() => {
    setIsReplaying(false);
    setReplayIdx(-1);
  }, [teamFilter, telemetry]);

  useEffect(() => {
    if (!isReplaying || chronologicalPasses.length === 0) return;
    if (replayIdx < 0) setReplayIdx(0);
    const id = window.setInterval(() => {
      setReplayIdx((prev) => {
        if (prev + 1 >= chronologicalPasses.length) {
          setIsReplaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1150);
    return () => window.clearInterval(id);
  }, [isReplaying, chronologicalPasses.length, replayIdx]);

  const activeReplayPass =
    replayIdx >= 0 && replayIdx < chronologicalPasses.length
      ? chronologicalPasses[replayIdx]
      : null;

  const { playersList, computedArcs, connectionCountMap } = useMemo(() => {
    const { nodes, nodeMap, filteredEdges } = selectPassNetworkNodes(
      telemetry,
      teamFilter
    );

    const connCount = new Map<number, number>();
    for (const e of filteredEdges) {
      connCount.set(e.from_id, (connCount.get(e.from_id) || 0) + e.count);
      connCount.set(e.to_id, (connCount.get(e.to_id) || 0) + e.count);
    }

    const maxCount = Math.max(1, d3.max(filteredEdges, (e) => e.count) ?? 1);
    const widthScale = d3.scaleSqrt().domain([1, maxCount]).range([2.0, 5.8]);

    const maxInv = Math.max(
      1,
      d3.max(
        nodes,
        (n) => connCount.get(n.track_id) || n.stat?.passes_completed || 0
      ) ?? 1
    );
    // Max radius 14.5px guarantees 2*maxR = 29px < min separation 32.4px -> 0 overlap
    const radiusScale = d3.scaleSqrt().domain([0, maxInv]).range([10, 14.5]);

    const arcs: ComputedArc[] = [];
    for (const edge of filteredEdges) {
      const pFrom = nodeMap.get(edge.from_id);
      const pTo = nodeMap.get(edge.to_id);
      if (!pFrom || !pTo) continue;

      const x1 = scaleX(pFrom.x);
      const y1 = scaleY(pFrom.y);
      const x2 = scaleX(pTo.x);
      const y2 = scaleY(pTo.y);

      const dx = x2 - x1;
      const dy = y2 - y1;
      const dist = Math.hypot(dx, dy);
      if (dist < 4) continue;

      const nx = -dy / dist;
      const ny = dx / dist;
      const bow = Math.min(28, Math.max(12, dist * 0.16));

      const ctrlX = (x1 + x2) / 2 + nx * bow;
      const ctrlY = (y1 + y2) / 2 + ny * bow;

      const targetR =
        radiusScale(
          connCount.get(edge.to_id) || pTo.stat?.passes_completed || 0
        ) + 5;
      const tDx = x2 - ctrlX;
      const tDy = y2 - ctrlY;
      const tLen = Math.max(1, Math.hypot(tDx, tDy));
      const endX = x2 - (tDx / tLen) * targetR;
      const endY = y2 - (tDy / tLen) * targetR;

      const startR =
        radiusScale(
          connCount.get(edge.from_id) || pFrom.stat?.passes_completed || 0
        ) + 2;
      const sDx = ctrlX - x1;
      const sDy = ctrlY - y1;
      const sLen = Math.max(1, Math.hypot(sDx, sDy));
      const startX = x1 + (sDx / sLen) * startR;
      const startY = y1 + (sDy / sLen) * startR;

      const pairMeta = pairMetaMap.get(`${edge.from_id}-${edge.to_id}`);
      const xtDelta = pairMeta?.xt || 0;
      const isProgressive = Boolean(pairMeta?.prog || xtDelta >= 0.012);

      arcs.push({
        key: `${edge.team}-${edge.from_id}-${edge.to_id}`,
        edge,
        pathD: `M ${startX.toFixed(1)} ${startY.toFixed(
          1
        )} Q ${ctrlX.toFixed(1)} ${ctrlY.toFixed(1)} ${endX.toFixed(
          1
        )} ${endY.toFixed(1)}`,
        strokeW: widthScale(edge.count),
        xtDelta,
        isProgressive,
        firstPassTime: pairMeta?.firstTime ?? null,
        midX: 0.25 * startX + 0.5 * ctrlX + 0.25 * endX,
        midY: 0.25 * startY + 0.5 * ctrlY + 0.25 * endY,
      });
    }

    const enrichedPlayers = nodes.map((node: PassNetworkNode) => ({
      node,
      identity: resolvePlayerIdentity(node.track_id, node.stat, overrides),
      cx: scaleX(node.x),
      cy: scaleY(node.y),
      r: radiusScale(
        connCount.get(node.track_id) || node.stat?.passes_completed || 0
      ),
    }));

    return {
      playersList: enrichedPlayers,
      computedArcs: arcs,
      connectionCountMap: connCount,
    };
  }, [telemetry, teamFilter, scaleX, scaleY, pairMetaMap, overrides]);

  const hoveredPlayer = useMemo(
    () =>
      playersList.find((p) => p.node.track_id === hoveredPlayerId) ?? null,
    [playersList, hoveredPlayerId]
  );

  const xtOpacityScale = useMemo(
    () => d3.scaleLinear().domain([0.006, 0.108]).range([0.02, 0.19]).clamp(true),
    []
  );

  if (!telemetry) {
    return (
      <div className="rounded-2xl obsidian-card p-8 text-center text-xs text-slate-400">
        {t("no_data")}
      </div>
    );
  }

  return (
    <div className="rounded-2xl obsidian-card p-5 space-y-4">
      {/* ── Header & Calm Editorial Controls ── */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-3.5 border-b border-white/[0.08]">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Share2 className="w-4 h-4 text-tv-cyan" />
            <span>{t("tel.d3_passnet_title")}</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            {t("tel.d3_passnet_desc")}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Team Filter Segmented Pills */}
          <div className="inline-flex items-center p-0.5 rounded-xl obsidian-well">
            {(
              [
                ["all", t("evt.all_teams")],
                ["0", t("team_a")],
                ["1", t("team_b")],
              ] as const
            ).map(([val, label]) => {
              const active = teamFilter === val;
              return (
                <button
                  key={val}
                  type="button"
                  onClick={() => setTeamFilter(val)}
                  className={`relative px-3 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                    active ? "text-white font-bold" : "text-slate-400 hover:text-white"
                  }`}
                >
                  {active && (
                    <motion.span
                      layoutId="d3-passnet-team-pill"
                      transition={{ type: "spring", stiffness: 460, damping: 34 }}
                      className="absolute inset-0 rounded-lg editorial-tab-active"
                    />
                  )}
                  <span className="relative z-10">{label}</span>
                </button>
              );
            })}
          </div>

          {/* Progressive / High-xT Highlight Toggle */}
          <button
            type="button"
            onClick={() => setHighlightProg((prev) => !prev)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-colors cursor-pointer ${
              highlightProg
                ? "bg-tv-amber/15 border-tv-amber/40 text-tv-amber"
                : "obsidian-well text-slate-400 hover:text-white"
            }`}
          >
            <ArrowUpRight className="w-3.5 h-3.5" />
            <span>{t("d3.progressive_toggle")}</span>
          </button>

          {/* Animated Chronological Build-Up Replay Button */}
          {chronologicalPasses.length > 0 && (
            <button
              type="button"
              onClick={() => {
                if (isReplaying) {
                  setIsReplaying(false);
                } else {
                  setReplayIdx(0);
                  setIsReplaying(true);
                }
              }}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold border transition-colors cursor-pointer ${
                isReplaying
                  ? "bg-tv-coral/15 border-tv-coral/40 text-tv-coral"
                  : "bg-tv-cyan/15 border-tv-cyan/40 text-tv-cyan hover:bg-tv-cyan/25"
              }`}
            >
              {isReplaying ? (
                <>
                  <Pause className="w-3.5 h-3.5" />
                  <span>{t("d3.replay_stop")}</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{t("d3.replay_buildup")}</span>
                </>
              )}
            </button>
          )}

          {/* 12x8 xT Threat Grid Toggle */}
          <button
            type="button"
            onClick={() => setShowXtGrid((prev) => !prev)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-colors cursor-pointer ${
              showXtGrid
                ? "bg-tv-mint/12 border-tv-mint/35 text-tv-mint"
                : "obsidian-well text-slate-400 hover:text-white"
            }`}
          >
            <Grid className="w-3.5 h-3.5" />
            <span>{t("tel.d3_xt_grid_toggle")}</span>
          </button>
        </div>
      </div>

      {/* ── Declarative SVG + D3 Pitch Pass Graph ── */}
      <div
        dir="ltr"
        className="relative rounded-xl overflow-hidden border border-white/[0.09] bg-[#0A141A]"
      >
        <svg
          viewBox={`0 0 ${vbW} ${vbH}`}
          className="w-full h-auto block select-none"
          role="img"
          aria-label={t("tel.d3_passnet_title")}
        >
          <defs>
            <marker
              id="d3-arrow-cyan"
              viewBox="0 0 10 10"
              refX="7"
              refY="5"
              markerWidth="4.5"
              markerHeight="4.5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#38BDF8" />
            </marker>
            <marker
              id="d3-arrow-coral"
              viewBox="0 0 10 10"
              refX="7"
              refY="5"
              markerWidth="4.5"
              markerHeight="4.5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#FB7185" />
            </marker>
            <marker
              id="d3-arrow-mint"
              viewBox="0 0 10 10"
              refX="7"
              refY="5"
              markerWidth="4.8"
              markerHeight="4.8"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#34D399" />
            </marker>
          </defs>

          {/* Alternating Night-Emerald Slate Turf Stripes */}
          {Array.from({ length: 10 }).map((_, i) => (
            <rect
              key={`turf-${i}`}
              x={padX + i * (pitchW / 10)}
              y={padY}
              width={pitchW / 10}
              height={pitchH}
              fill={i % 2 === 0 ? "#0A141A" : "#0D1920"}
            />
          ))}

          {/* Optional Karun Singh 12x8 Expected Threat (xT) Zone Overlay */}
          {showXtGrid &&
            XT_GRID.map((rowVals, rIdx) =>
              rowVals.map((xtVal, cIdx) => {
                const cellW = pitchW / 12;
                const cellH = pitchH / 8;
                return (
                  <rect
                    key={`xt-${rIdx}-${cIdx}`}
                    x={padX + cIdx * cellW}
                    y={padY + rIdx * cellH}
                    width={cellW}
                    height={cellH}
                    fill="#34D399"
                    fillOpacity={xtOpacityScale(xtVal)}
                    stroke="rgba(148,163,184,0.06)"
                    strokeWidth="0.6"
                  />
                );
              })
            )}

          {/* Crisp FIFA Pitch Markings */}
          <g stroke="rgba(255,255,255,0.22)" strokeWidth="1.3" fill="none">
            <rect x={padX} y={padY} width={pitchW} height={pitchH} />
            <line
              x1={scaleX(52.5)}
              y1={padY}
              x2={scaleX(52.5)}
              y2={padY + pitchH}
            />
            <circle cx={scaleX(52.5)} cy={scaleY(34)} r={9.15 * (pitchW / 105)} />
            <rect
              x={padX}
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
            <rect
              x={padX}
              y={scaleY((68 - 18.32) / 2)}
              width={5.5 * (pitchW / 105)}
              height={18.32 * (pitchH / 68)}
            />
            <rect
              x={scaleX(105 - 5.5)}
              y={scaleY((68 - 18.32) / 2)}
              width={5.5 * (pitchW / 105)}
              height={18.32 * (pitchH / 68)}
            />
          </g>

          {/* Calm, Static Curved Quadratic Bezier Pass Corridors */}
          {computedArcs.map((arc) => {
            const isConnected =
              hoveredPlayerId === null ||
              arc.edge.from_id === hoveredPlayerId ||
              arc.edge.to_id === hoveredPlayerId;
            const isReplayMatch =
              activeReplayPass != null &&
              activeReplayPass.from_id === arc.edge.from_id &&
              activeReplayPass.to_id === arc.edge.to_id;
            const dimForProg = highlightProg && !arc.isProgressive;
            const dimForReplay = activeReplayPass != null && !isReplayMatch;

            const isHighXt = arc.isProgressive;
            const isTeam0 = arc.edge.team === 0;
            const strokeColor =
              isReplayMatch || isHighXt
                ? "#34D399"
                : isTeam0
                ? "#38BDF8"
                : "#FB7185";
            const markerId =
              isReplayMatch || isHighXt
                ? "url(#d3-arrow-mint)"
                : isTeam0
                ? "url(#d3-arrow-cyan)"
                : "url(#d3-arrow-coral)";

            const computedOpacity = dimForReplay
              ? 0.14
              : dimForProg
              ? 0.18
              : hoveredPlayerId === null
              ? 0.88
              : isConnected
              ? 1
              : 0.14;

            const xtBoost = Math.min(1.1, Math.max(0, arc.xtDelta * 22));
            const finalStrokeW =
              (isReplayMatch ? arc.strokeW + 1.6 : arc.strokeW) + xtBoost;

            return (
              <g
                key={arc.key}
                opacity={computedOpacity}
                onClick={() => {
                  if (arc.firstPassTime != null && onJumpToPass) {
                    onJumpToPass(arc.firstPassTime, arc.edge.from_id);
                  }
                }}
                className="transition-opacity duration-200 cursor-pointer"
              >
                <path
                  d={arc.pathD}
                  fill="none"
                  stroke={strokeColor}
                  strokeWidth={finalStrokeW}
                  strokeOpacity={isReplayMatch || (hoveredPlayerId !== null && isConnected) ? 0.96 : 0.78}
                  strokeLinecap="round"
                  markerEnd={markerId}
                />
                {(isReplayMatch || (hoveredPlayerId !== null && isConnected)) &&
                  arc.edge.count >= 1 && (
                    <g transform={`translate(${arc.midX - 13}, ${arc.midY - 8})`}>
                      <rect
                        width="26"
                        height="16"
                        rx="4"
                        fill="rgba(6, 9, 18, 0.94)"
                        stroke={strokeColor}
                        strokeWidth="1"
                      />
                      <text
                        x="13"
                        y="11"
                        textAnchor="middle"
                        fill="#F8FAFC"
                        fontSize="9.5"
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        ×{arc.edge.count}
                      </text>
                    </g>
                  )}
              </g>
            );
          })}

          {/* De-Overlapped Player Nodes with Centered Numeric Labels */}
          {playersList.map(({ node, identity, cx, cy, r }) => {
            const isTeam0 = node.team === 0;
            const color = isTeam0 ? "#38BDF8" : "#FB7185";
            const isHov = hoveredPlayerId === node.track_id;

            const isNeighbor =
              hoveredPlayerId === null ||
              isHov ||
              computedArcs.some(
                (a) =>
                  (a.edge.from_id === hoveredPlayerId &&
                    a.edge.to_id === node.track_id) ||
                  (a.edge.to_id === hoveredPlayerId &&
                    a.edge.from_id === node.track_id)
              );

            const label = identity.canvasLabel;
            const labelFontSize = label.length >= 3 ? "8.5" : "10";

            return (
              <g
                key={`pnode-${node.track_id}`}
                opacity={isNeighbor ? (node.isParticipant ? 1 : 0.72) : 0.22}
                onMouseEnter={() => setHoveredPlayerId(node.track_id)}
                onMouseLeave={() => setHoveredPlayerId(null)}
                onClick={() => onSelectPlayer?.(node.track_id)}
                className="cursor-pointer transition-opacity duration-200"
              >
                {(node.stat?.xt_generated ?? 0) > 0.005 && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={r + 4}
                    fill="none"
                    stroke="#34D399"
                    strokeWidth="1.4"
                    strokeDasharray="3 2"
                  />
                )}
                {identity.isVerified && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHov ? r + 4.2 : r + 2.2}
                    fill="none"
                    stroke={identity.isCoachOverride ? "#FBBF24" : "rgba(52, 211, 153, 0.82)"}
                    strokeWidth="1.2"
                  />
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHov ? r + 2.5 : r}
                  fill="#060912"
                  stroke={color}
                  strokeWidth={isHov ? 3 : node.isParticipant ? 2.4 : 1.8}
                />
                {identity.isVerified && (
                  <circle
                    cx={cx + r * 0.72}
                    cy={cy - r * 0.72}
                    r={3.2}
                    fill={identity.isCoachOverride ? "#FBBF24" : "#34D399"}
                    stroke="#060912"
                    strokeWidth={1}
                  />
                )}
                {label && (
                  <text
                    x={cx}
                    y={cy}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill={identity.isVerified ? "#FFFFFF" : "#E2E8F0"}
                    fontSize={labelFontSize}
                    fontFamily="var(--font-jetbrains), monospace"
                    fontWeight={identity.isVerified ? "800" : "700"}
                  >
                    {label}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {computedArcs.length === 0 && (
          <div
            dir={dir}
            className="absolute inset-0 flex items-center justify-center pointer-events-none p-4"
          >
            <div className="px-4 py-2.5 rounded-xl obsidian-popover text-xs font-medium text-slate-300 shadow-xl">
              {t("d3.pass_empty_team")}
            </div>
          </div>
        )}

        {hoveredPlayer && (
          <div
            dir={dir}
            className="absolute top-3 end-3 rounded-xl obsidian-popover p-3 shadow-2xl text-xs space-y-1.5 pointer-events-none min-w-[195px]"
          >
            <div className="flex items-center justify-between gap-3 pb-1 border-b border-white/[0.08]">
              <span className="font-bold text-white">
                {hoveredPlayer.identity.customName ||
                  (hoveredPlayer.identity.isVerified
                    ? formatRole(hoveredPlayer.node.role, lang, true)
                    : t("jersey.unidentified_player"))}{" "}
                <BidiNum>
                  {hoveredPlayer.identity.isVerified
                    ? hoveredPlayer.identity.badgeText
                    : `(#${hoveredPlayer.node.track_id})`}
                </BidiNum>
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  hoveredPlayer.node.team === 0
                    ? "bg-tv-cyan/15 text-tv-cyan"
                    : "bg-tv-coral/15 text-tv-coral"
                }`}
              >
                {formatTeamName(hoveredPlayer.node.team, lang)}
              </span>
            </div>

            <div className="flex items-center justify-between text-slate-300">
              <span>{t("card.passes")}:</span>
              <BidiNum className="font-bold text-white">
                {hoveredPlayer.node.stat?.passes_completed ?? 0}
              </BidiNum>
            </div>

            <div className="flex items-center justify-between text-slate-300">
              <span>{t("d3.connections")}:</span>
              <BidiNum className="font-bold text-tv-cyan">
                {connectionCountMap.get(hoveredPlayer.node.track_id) || 0}
              </BidiNum>
            </div>

            <div className="flex items-center justify-between text-slate-300">
              <span>{t("tel.col_xt")}:</span>
              <BidiNum className="font-bold text-tv-mint">
                +{(hoveredPlayer.node.stat?.xt_generated ?? 0).toFixed(3)}
              </BidiNum>
            </div>
          </div>
        )}
      </div>

      {/* Live Animated Build-Up Step Bar */}
      {activeReplayPass && (
        <div className="p-3 rounded-xl obsidian-well border border-tv-mint/30 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2 py-0.5 rounded-md bg-tv-mint/15 text-tv-mint font-bold">
              {t("d3.pass_step")}{" "}
              <BidiNum>{`${replayIdx + 1}/${chronologicalPasses.length}`}</BidiNum>
            </span>
            <BidiNum className="text-slate-300 font-semibold">
              {`${activeReplayPass.time_s.toFixed(1)}s`}
            </BidiNum>
            <BidiNum className="font-bold text-white">
              {(() => {
                const fId = resolvePlayerIdentity(activeReplayPass.from_id, null, overrides);
                const tId = resolvePlayerIdentity(activeReplayPass.to_id, null, overrides);
                const fLbl = fId.isVerified ? fId.badgeText : `#${activeReplayPass.from_id}`;
                const tLbl = tId.isVerified ? tId.badgeText : `#${activeReplayPass.to_id}`;
                return `${fLbl} → ${tLbl}`;
              })()}
            </BidiNum>
            <BidiNum className="text-tv-cyan font-semibold">
              {`${activeReplayPass.distance_m}m`}
            </BidiNum>
            {activeReplayPass.is_progressive && (
              <span className="px-2 py-0.5 rounded bg-tv-mint/15 text-tv-mint font-bold text-[11px]">
                {t("evt.progressive")}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {onJumpToPass && (
              <button
                type="button"
                onClick={() =>
                  onJumpToPass(activeReplayPass.time_s, activeReplayPass.from_id)
                }
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-tv-cyan text-slate-950 font-bold text-[11px] cursor-pointer hover:bg-tv-cyan/90"
              >
                <RotateCcw className="w-3 h-3" />
                <span>{t("evt.replay_3s")}</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Footer Hint & Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
        <span>{t("d3.node_hint")}</span>
        <div className="flex items-center gap-3 text-[11px]">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-1 rounded-full bg-tv-cyan" />
            <span>{t("team_a")}</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-1 rounded-full bg-tv-coral" />
            <span>{t("team_b")}</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <Sparkles className="w-3 h-3 text-tv-mint" />
            <BidiNum>+xT</BidiNum>
          </span>
        </div>
      </div>
    </div>
  );
}
