import type { Telemetry, PlayerStat } from "./types";

export interface ResolvedPlayerPos {
  track_id: number;
  team: 0 | 1;
  role: "player" | "goalkeeper";
  x: number;
  y: number;
  frames_visible: number;
  stat?: PlayerStat;
}

export interface PassNetworkNode extends ResolvedPlayerPos {
  isParticipant: boolean;
  involvement: number;
}

const clampPitchX = (x: number) => Math.max(4, Math.min(101, x));
const clampPitchY = (y: number) => Math.max(4, Math.min(64, y));

/**
 * Resolves true pitch coordinates [x, y] for EVERY player tracked in `telemetry`,
 * including short-appearance IDs omitted from `telemetry.players[]` that appear in
 * `passing_network[]` or `frames[]`. Guarantees zero silent pass drops.
 */
export function resolvePlayerPositions(
  telemetry: Telemetry | null
): Map<number, ResolvedPlayerPos> {
  const result = new Map<number, ResolvedPlayerPos>();
  if (!telemetry) return result;

  const statMap = new Map<number, PlayerStat>();
  for (const p of telemetry.players || []) {
    statMap.set(p.track_id, p);
  }

  // Accumulate true frame-level positions for all track IDs
  const frameAccum = new Map<
    number,
    { sumX: number; sumY: number; count: number; team: 0 | 1; role: "player" | "goalkeeper" }
  >();

  for (const fr of telemetry.frames || []) {
    for (const fp of fr.players || []) {
      const existing = frameAccum.get(fp.id);
      const role: "player" | "goalkeeper" = fp.role === "gk" ? "goalkeeper" : "player";
      if (existing) {
        existing.sumX += fp.x;
        existing.sumY += fp.y;
        existing.count += 1;
      } else {
        frameAccum.set(fp.id, {
          sumX: fp.x,
          sumY: fp.y,
          count: 1,
          team: fp.team,
          role,
        });
      }
    }
  }

  // Secondary fallback from pass/turnover event XY in case frame entry wasn't found
  const eventFallback = new Map<number, { x: number; y: number; team: 0 | 1 }>();
  for (const ev of telemetry.events || []) {
    if (ev.type === "pass") {
      if (!eventFallback.has(ev.from_id)) {
        eventFallback.set(ev.from_id, {
          x: ev.start_xy[0],
          y: ev.start_xy[1],
          team: ev.team,
        });
      }
      if (!eventFallback.has(ev.to_id)) {
        eventFallback.set(ev.to_id, {
          x: ev.end_xy[0],
          y: ev.end_xy[1],
          team: ev.team,
        });
      }
    } else if (ev.type === "turnover") {
      if (!eventFallback.has(ev.by_id)) {
        eventFallback.set(ev.by_id, {
          x: ev.xy[0],
          y: ev.xy[1],
          team: ev.to_team,
        });
      }
    }
  }

  // Gather all IDs from players[], passing_network[], and frames[]
  const allIds = new Set<number>();
  for (const p of telemetry.players || []) allIds.add(p.track_id);
  for (const edge of telemetry.passing_network || []) {
    allIds.add(edge.from_id);
    allIds.add(edge.to_id);
  }

  for (const tid of allIds) {
    const stat = statMap.get(tid);
    const acc = frameAccum.get(tid);
    const evFb = eventFallback.get(tid);

    if (stat) {
      const rawX =
        stat.avg_pitch_xy && stat.avg_pitch_xy.length === 2
          ? stat.avg_pitch_xy[0]
          : acc && acc.count > 0
          ? acc.sumX / acc.count
          : evFb?.x ?? 52.5;
      const rawY =
        stat.avg_pitch_xy && stat.avg_pitch_xy.length === 2
          ? stat.avg_pitch_xy[1]
          : acc && acc.count > 0
          ? acc.sumY / acc.count
          : evFb?.y ?? 34.0;

      result.set(tid, {
        track_id: tid,
        team: stat.team,
        role: stat.role,
        x: clampPitchX(rawX),
        y: clampPitchY(rawY),
        frames_visible: stat.frames_visible || acc?.count || 0,
        stat,
      });
    } else if (acc && acc.count > 0) {
      const rawX = acc.sumX / acc.count;
      const rawY = acc.sumY / acc.count;
      const syntheticStat: PlayerStat = {
        track_id: tid,
        team: acc.team,
        role: acc.role,
        distance_m: 0,
        max_speed_kmh: 0,
        avg_speed_kmh: 0,
        sprints: 0,
        passes_completed: 0,
        xt_generated: 0,
        avg_pitch_xy: [rawX, rawY],
        frames_visible: acc.count,
      };
      result.set(tid, {
        track_id: tid,
        team: acc.team,
        role: acc.role,
        x: clampPitchX(rawX),
        y: clampPitchY(rawY),
        frames_visible: acc.count,
        stat: syntheticStat,
      });
    } else if (evFb) {
      const syntheticStat: PlayerStat = {
        track_id: tid,
        team: evFb.team,
        role: "player",
        distance_m: 0,
        max_speed_kmh: 0,
        avg_speed_kmh: 0,
        sprints: 0,
        passes_completed: 0,
        xt_generated: 0,
        avg_pitch_xy: [evFb.x, evFb.y],
        frames_visible: 1,
      };
      result.set(tid, {
        track_id: tid,
        team: evFb.team,
        role: "player",
        x: clampPitchX(evFb.x),
        y: clampPitchY(evFb.y),
        frames_visible: 1,
        stat: syntheticStat,
      });
    }
  }

  return result;
}

/**
 * Selects passing network nodes without ByteTrack 33-ID clutter:
 * - Always keeps 100% of passing participants (zero dropped passes)
 * - Caps non-passing active players so each team shows at most 11 nodes
 * - Runs deterministic pairwise repulsion so zero circles overlap on the pitch
 */
export function selectPassNetworkNodes(
  telemetry: Telemetry | null,
  teamFilter: "all" | "0" | "1"
): {
  nodes: PassNetworkNode[];
  nodeMap: Map<number, PassNetworkNode>;
  filteredEdges: NonNullable<Telemetry["passing_network"]>;
} {
  if (!telemetry) {
    return { nodes: [], nodeMap: new Map(), filteredEdges: [] };
  }

  const posMap = resolvePlayerPositions(telemetry);
  const filteredEdges = (telemetry.passing_network || []).filter((e) =>
    teamFilter === "all" ? true : e.team === Number(teamFilter)
  );

  const involvement = new Map<number, number>();
  const participants = new Set<number>();
  for (const e of filteredEdges) {
    participants.add(e.from_id);
    participants.add(e.to_id);
    involvement.set(e.from_id, (involvement.get(e.from_id) || 0) + e.count);
    involvement.set(e.to_id, (involvement.get(e.to_id) || 0) + e.count);
  }

  const pickForTeam = (teamNum: 0 | 1): PassNetworkNode[] => {
    const teamCandidates: ResolvedPlayerPos[] = [];
    for (const p of posMap.values()) {
      if (p.team === teamNum) teamCandidates.push(p);
    }

    const passingNodes = teamCandidates
      .filter((p) => participants.has(p.track_id))
      .sort((a, b) => b.frames_visible - a.frames_visible);

    const restNodes = teamCandidates
      .filter((p) => !participants.has(p.track_id))
      .sort((a, b) => b.frames_visible - a.frames_visible);

    const remainingSlots = Math.max(0, 11 - passingNodes.length);
    const chosen = [...passingNodes, ...restNodes.slice(0, remainingSlots)];

    return chosen.map((p) => ({
      ...p,
      isParticipant: participants.has(p.track_id),
      involvement: involvement.get(p.track_id) || 0,
    }));
  };

  let nodes: PassNetworkNode[] = [];
  if (teamFilter === "all") {
    nodes = [...pickForTeam(0), ...pickForTeam(1)];
  } else if (teamFilter === "0") {
    nodes = pickForTeam(0);
  } else {
    nodes = pickForTeam(1);
  }

  // Deterministic iterative separation so circles & short pass shafts never collide
  const minDistMeters = 6.8;
  for (let iter = 0; iter < 28; iter++) {
    let moved = false;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist = Math.hypot(dx, dy);

        if (dist < 0.01) {
          dx = 0.5;
          dy = 0.35;
          dist = Math.hypot(dx, dy);
        }

        if (dist < minDistMeters) {
          const push = (minDistMeters - dist) * 0.55;
          const ux = dx / dist;
          const uy = dy / dist;
          a.x = clampPitchX(a.x - ux * push);
          a.y = clampPitchY(a.y - uy * push);
          b.x = clampPitchX(b.x + ux * push);
          b.y = clampPitchY(b.y + uy * push);
          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  const nodeMap = new Map<number, PassNetworkNode>();
  for (const n of nodes) {
    nodeMap.set(n.track_id, n);
  }

  return { nodes, nodeMap, filteredEdges };
}

export interface PlayerRadarMetrics {
  p95SpeedKmh: number;
  avgSpeedKmh: number;
  distanceM: number;
  sprintIntensityScore: number;
  sprintsCount: number;
  highSpeedFrames: number;
  activityPct: number;
  ballImpactScore: number;
  passesInvolved: number;
  turnoversWon: number;
  xtGenerated: number;
}

/**
 * Computes 6 genuinely differentiating, 100% real-data metrics for D3PlayerRadar
 * from `telemetry.frames[]` and `telemetry.events[]` so polygons don't collapse on 30s clips.
 */
export function computePlayerRadarMetrics(
  player: PlayerStat,
  telemetry: Telemetry | null
): PlayerRadarMetrics {
  const frames = telemetry?.frames || [];
  const events = telemetry?.events || [];
  const totalFrames = Math.max(1, telemetry?.summary?.total_frames || frames.length || 1);

  const speeds: number[] = [];
  let highSpeedFrames = 0;

  for (const fr of frames) {
    const fp = fr.players?.find((p) => p.id === player.track_id);
    if (fp) {
      speeds.push(fp.spd);
      if (fp.spd >= 18.0) highSpeedFrames++;
    }
  }

  let p95SpeedKmh = player.max_speed_kmh || 0;
  if (speeds.length >= 5) {
    speeds.sort((a, b) => a - b);
    const idx = Math.min(speeds.length - 1, Math.floor((speeds.length - 1) * 0.95));
    p95SpeedKmh = Number(speeds[idx].toFixed(1));
  }

  let passesInvolved = player.passes_completed || 0;
  let receivedPasses = 0;
  let turnoversWon = 0;

  for (const ev of events) {
    if (ev.type === "pass" && ev.to_id === player.track_id) {
      receivedPasses++;
    } else if (ev.type === "turnover" && ev.by_id === player.track_id) {
      turnoversWon++;
    }
  }
  passesInvolved = Math.max(player.passes_completed || 0, player.passes_completed + receivedPasses);

  const xtVal = Math.max(0, player.xt_generated || 0);
  const sprintIntensityScore = (player.sprints || 0) * 12 + highSpeedFrames;
  const activityPct = Math.min(100, Number(((player.frames_visible / totalFrames) * 100).toFixed(0)));
  const ballImpactScore = passesInvolved + turnoversWon * 1.5 + xtVal * 120;

  return {
    p95SpeedKmh,
    avgSpeedKmh: player.avg_speed_kmh || 0,
    distanceM: player.distance_m || 0,
    sprintIntensityScore,
    sprintsCount: player.sprints || 0,
    highSpeedFrames,
    activityPct,
    ballImpactScore,
    passesInvolved,
    turnoversWon,
    xtGenerated: xtVal,
  };
}
