import type { Telemetry, FramePlayer } from "./types";
import { computeConvexHull, computePolygonArea } from "@/components/warroom/canvas/drawHulls";

export interface StabilizedPlayer extends FramePlayer {
  vx?: number;
  vy?: number;
}

export interface StabilizedFrame {
  f: number;
  t: number;
  players: StabilizedPlayer[];
  ball: [number, number] | null;
  def_x: Record<string, number>;
  vor: [number, number];
  hullArea: {
    team0_m2: number;
    team1_m2: number;
  };
  isGlitchBridged: boolean;
  homographyValid: boolean;
}

export interface CleanTrajectoryPoint {
  x: number;
  y: number;
  frame: number;
  spd: number;
  breakBefore: boolean;
}

export interface HeroSequenceWindow {
  startFrame: number;
  endFrame: number;
  score: number;
}

export interface StabilizedTelemetry {
  fps: number;
  duration_s: number;
  frames: StabilizedFrame[];
  trajectories: Map<number, CleanTrajectoryPoint[]>;
  ballTimeline: Array<[number, number] | null>;
  glitchFrameCount: number;
}

const PITCH_L = 105.0;
const PITCH_W = 68.0;

/**
 * Detects engine homography sentinel clamp coordinates (e.g. x=-2/107, y=-2/70)
 * or NaN/Infinity corruption so we never let them draw corner laser spikes.
 */
function isSentinelOob(x: number, y: number): boolean {
  return (
    !Number.isFinite(x) ||
    !Number.isFinite(y) ||
    x <= -1.4 ||
    x >= 106.4 ||
    y <= -1.4 ||
    y >= 69.4
  );
}

function clampSoftX(x: number): number {
  return Math.max(0.4, Math.min(PITCH_L - 0.4, x));
}

function clampSoftY(y: number): number {
  return Math.max(0.4, Math.min(PITCH_W - 0.4, y));
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) * 0.5;
}

/**
 * Filters, de-glitches, bridges, and smooths raw YOLO + RANSAC homography telemetry
 * into a fluid broadcast-grade signal suitable for 60FPS interpolation.
 */
export function stabilizeTelemetry(
  telemetry: Telemetry | null
): StabilizedTelemetry | null {
  const rawFrames = telemetry?.frames;
  if (!telemetry || !rawFrames || rawFrames.length === 0) {
    return null;
  }

  const fps = telemetry.summary?.fps || telemetry.match_info?.fps || 25.0;
  const nFrames = rawFrames.length;

  // Max physically plausible single-player displacement over `dtFrames` at `fps`:
  // Exact 1.80m for 1 frame at 25fps (0.04s), scaling smoothly with dt = dtFrames / fps.
  const maxJumpForDt = (dtFrames: number): number => {
    const dtSec = Math.max(1, dtFrames) / fps;
    return 1.26 + dtSec * 13.5;
  };

  // Pre-scan sentinel prevalence so zoomed/corner-kick clips where edge players routinely
  // clip outside pitch bounds never get wiped as whole-frame glitches.
  let framesWithSentinelBurst = 0;
  for (let i = 0; i < nFrames; i++) {
    const pls = rawFrames[i].players || [];
    let sc = 0;
    for (const p of pls) {
      if (isSentinelOob(p.x, p.y)) sc++;
    }
    if (pls.length > 0 && sc >= 3 && sc >= pls.length * 0.18) {
      framesWithSentinelBurst++;
    }
  }
  const isPersistentZoomedClip =
    nFrames > 0 && framesWithSentinelBurst >= Math.ceil(nFrames * 0.45);

  // ── 1. Whole-Frame Homography Glitch & Sentinel Spike Detection ──
  const isGlitchFrame = new Uint8Array(nFrames);
  let lastGoodMap = new Map<number, { x: number; y: number }>();
  let lastGoodFrameIdx = -1;
  let prevRawMap = new Map<number, { x: number; y: number }>();
  let stableCutStreak = 0;
  let glitchFrameCount = 0;

  for (let i = 0; i < nFrames; i++) {
    const fr = rawFrames[i];
    const players = fr.players || [];
    let sentinelCount = 0;
    const dispsFromGood: number[] = [];
    const dispsFromPrev: number[] = [];
    const curValidMap = new Map<number, { x: number; y: number }>();

    for (const p of players) {
      if (isSentinelOob(p.x, p.y)) {
        sentinelCount++;
      } else {
        curValidMap.set(p.id, { x: p.x, y: p.y });
        if (lastGoodFrameIdx >= 0) {
          const prevGood = lastGoodMap.get(p.id);
          if (prevGood) {
            dispsFromGood.push(Math.hypot(p.x - prevGood.x, p.y - prevGood.y));
          }
        }
        const prevImm = prevRawMap.get(p.id);
        if (prevImm) {
          dispsFromPrev.push(Math.hypot(p.x - prevImm.x, p.y - prevImm.y));
        }
      }
    }

    const validInFrame = curValidMap.size;
    const dtFrames = lastGoodFrameIdx >= 0 ? Math.max(1, i - lastGoodFrameIdx) : 1;
    const dtSec = dtFrames / fps;
    const medDispGood = medianOf(dispsFromGood);
    const medDispPrev = medianOf(dispsFromPrev);

    // Whole-pitch median jump across >= 3 players cannot physically exceed ~1.45m in 1 frame
    const massTeleport =
      lastGoodFrameIdx >= 0 &&
      dispsFromGood.length >= 3 &&
      medDispGood > Math.max(1.45, dtSec * 14.5);

    // Sudden homography sentinel OOB burst (on wide matches) OR almost-empty OOB collapse
    const severeSentinelGlitch =
      players.length > 0 &&
      ((validInFrame < 2 && sentinelCount >= 2) ||
        (!isPersistentZoomedClip &&
          sentinelCount >= 3 &&
          sentinelCount >= Math.ceil(players.length * 0.18)));

    // Distinguish transient homography warps (chaotic frame-to-frame) from genuine camera cuts
    // (internally smooth new framing lasting >= 6 consecutive frames)
    if (massTeleport && !severeSentinelGlitch) {
      if (dispsFromPrev.length >= 3 && medDispPrev <= 1.25) {
        stableCutStreak++;
      } else {
        stableCutStreak = 0;
      }
    } else {
      stableCutStreak = 0;
    }

    const transientTeleport = massTeleport && stableCutStreak < 6;

    if (severeSentinelGlitch || transientTeleport) {
      isGlitchFrame[i] = 1;
      glitchFrameCount++;
    } else if (validInFrame > 0) {
      lastGoodMap = curValidMap;
      lastGoodFrameIdx = i;
    }

    if (validInFrame > 0) {
      prevRawMap = curValidMap;
    }
  }

  // Fallback safeguard: if frame-level glitch filter would leave >= 75% of clip blank,
  // retain frames and rely on per-player kinematic filtering so valid tracks always render.
  if (nFrames > 0 && glitchFrameCount >= Math.ceil(nFrames * 0.75)) {
    isGlitchFrame.fill(0);
    glitchFrameCount = 0;
  }

  // ── 2. Per-Track Kinematic Outlier Rejection, Dropout Bridging & Zero-Phase Smoothing ──
  interface RawObs {
    idx: number;
    f: number;
    x: number;
    y: number;
    spd: number;
    team: 0 | 1;
    role: "p" | "gk" | "referee";
    jn?: number | null;
    jv?: boolean | null;
  }

  const rawTracks = new Map<number, RawObs[]>();
  for (let i = 0; i < nFrames; i++) {
    if (isGlitchFrame[i]) continue;
    const fr = rawFrames[i];
    for (const p of fr.players || []) {
      if (isSentinelOob(p.x, p.y)) continue;
      let arr = rawTracks.get(p.id);
      if (!arr) {
        arr = [];
        rawTracks.set(p.id, arr);
      }
      arr.push({
        idx: i,
        f: fr.f,
        x: p.x,
        y: p.y,
        spd: Number.isFinite(p.spd) ? Math.max(0, Math.min(42, p.spd)) : 0,
        team: p.team === 1 ? 1 : 0,
        role: p.role === "referee" ? "referee" : p.role === "gk" ? "gk" : "p",
        jn: p.jn ?? null,
        jv: p.jv ?? null,
      });
    }
  }

  // Reconstructed player list per frame index
  const perFramePlayers: StabilizedPlayer[][] = Array.from(
    { length: nFrames },
    () => []
  );
  const trajectories = new Map<number, CleanTrajectoryPoint[]>();

  // 7-tap zero-phase Gaussian kernel (sum = 1.00) + light 3-tap polish for silky, zero-lag tracking
  const GAUSS_7 = [0.045, 0.11, 0.2, 0.29, 0.2, 0.11, 0.045];
  const POLISH_3 = [0.2, 0.6, 0.2];

  rawTracks.forEach((obsList, trackId) => {
    if (obsList.length === 0) return;

    // 2A. Iterative kinematic outlier rejection (> 1.8m/frame @ 25fps scaled by dt)
    let cleaned = [...obsList];

    // Guard against a corrupted very first observation poisoning the anchor
    if (cleaned.length >= 3) {
      const d01 = Math.hypot(cleaned[1].x - cleaned[0].x, cleaned[1].y - cleaned[0].y);
      const d12 = Math.hypot(cleaned[2].x - cleaned[1].x, cleaned[2].y - cleaned[1].y);
      if (
        d01 > maxJumpForDt(cleaned[1].idx - cleaned[0].idx) &&
        d12 <= maxJumpForDt(cleaned[2].idx - cleaned[1].idx)
      ) {
        cleaned = cleaned.slice(1);
      }
    }

    for (let pass = 0; pass < 3; pass++) {
      if (cleaned.length <= 2) break;
      const nextCleaned: RawObs[] = [cleaned[0]];
      for (let k = 1; k < cleaned.length; k++) {
        const prev = nextCleaned[nextCleaned.length - 1];
        const cur = cleaned[k];
        const dtPrev = Math.max(1, cur.idx - prev.idx);
        const dPrev = Math.hypot(cur.x - prev.x, cur.y - prev.y);
        const maxAllowPrev = maxJumpForDt(dtPrev);

        let drop = false;
        if (dPrev > maxAllowPrev) {
          if (k === cleaned.length - 1) {
            drop = true;
          } else {
            // Check lookahead (1..3 steps) to see if track returns near `prev` (multi-frame spike)
            let returnsNearPrev = false;
            const maxLook = Math.min(cleaned.length - 1, k + 3);
            for (let m = k + 1; m <= maxLook; m++) {
              const fut = cleaned[m];
              const dFutPrev = Math.hypot(fut.x - prev.x, fut.y - prev.y);
              if (dFutPrev <= maxJumpForDt(fut.idx - prev.idx) * 1.15) {
                returnsNearPrev = true;
                break;
              }
            }
            const nxt = cleaned[k + 1];
            const dNext = Math.hypot(nxt.x - cur.x, nxt.y - cur.y);
            const maxAllowNext = maxJumpForDt(nxt.idx - cur.idx);

            // Only keep a large jump if BOTH nxt and nxt2 stay smoothly anchored at the new position (real camera cut)
            let sustainedCut = false;
            if (!returnsNearPrev && dNext <= maxAllowNext && k + 2 < cleaned.length) {
              const nxt2 = cleaned[k + 2];
              const dNext2 = Math.hypot(nxt2.x - nxt.x, nxt2.y - nxt.y);
              if (dNext2 <= maxJumpForDt(nxt2.idx - nxt.idx)) {
                sustainedCut = true;
              }
            }

            if (returnsNearPrev || !sustainedCut) {
              drop = true;
            }
          }
        }
        if (!drop) {
          nextCleaned.push(cur);
        }
      }
      cleaned = nextCleaned;
    }

    if (cleaned.length === 0) return;

    // 2B. Bridge short tracking dropouts (1..6 frames) and homography glitch bursts (<= 36 frames)
    const bridged: RawObs[] = [];
    for (let k = 0; k < cleaned.length; k++) {
      const cur = cleaned[k];
      if (k > 0) {
        const prev = cleaned[k - 1];
        const gap = cur.idx - prev.idx;
        const dist = Math.hypot(cur.x - prev.x, cur.y - prev.y);

        let glitchMidCount = 0;
        if (gap > 10 && gap <= 36) {
          for (let fIdx = prev.idx + 1; fIdx < cur.idx; fIdx++) {
            if (isGlitchFrame[fIdx]) glitchMidCount++;
          }
        }

        const isShortDropout = gap > 1 && gap <= 10;
        const isGlitchBurst = gap > 10 && gap <= 36 && glitchMidCount >= Math.ceil((gap - 1) * 0.4);
        const maxBridgeDist = maxJumpForDt(gap) * 1.35;

        if ((isShortDropout || isGlitchBurst) && dist <= maxBridgeDist) {
          for (let g = 1; g < gap; g++) {
            const ratio = g / gap;
            const frameI = prev.idx + g;
            bridged.push({
              idx: frameI,
              f: rawFrames[frameI].f,
              x: prev.x + (cur.x - prev.x) * ratio,
              y: prev.y + (cur.y - prev.y) * ratio,
              spd: prev.spd + (cur.spd - prev.spd) * ratio,
              team: cur.team,
              role: cur.role,
              jn: cur.jn ?? prev.jn ?? null,
              jv: cur.jv ?? prev.jv ?? null,
            });
          }
        }
      }
      bridged.push(cur);
    }

    // 2C. Zero-Phase 7-Tap Gaussian + 3-Tap Polish Smoothing on continuous segments
    const pass1X = new Float64Array(bridged.length);
    const pass1Y = new Float64Array(bridged.length);
    const smoothedX = new Float64Array(bridged.length);
    const smoothedY = new Float64Array(bridged.length);
    const smoothedSpd = new Float64Array(bridged.length);
    const segBreak = new Uint8Array(bridged.length);

    for (let k = 0; k < bridged.length; k++) {
      if (k === 0) {
        segBreak[k] = 1;
      } else {
        const idxGap = bridged[k].idx - bridged[k - 1].idx;
        const stepDist = Math.hypot(
          bridged[k].x - bridged[k - 1].x,
          bridged[k].y - bridged[k - 1].y
        );
        if (idxGap > 6 || stepDist > Math.max(2.4, maxJumpForDt(idxGap) * 1.2)) {
          segBreak[k] = 1;
        }
      }
    }

    const isContinuous = (aIdx: number, bIdx: number): boolean => {
      const lo = Math.min(aIdx, bIdx);
      const hi = Math.max(aIdx, bIdx);
      for (let step = lo + 1; step <= hi; step++) {
        if (segBreak[step] || bridged[step].idx - bridged[step - 1].idx > 2) {
          return false;
        }
      }
      return true;
    };

    // Pass 1: 7-tap Gaussian on positions & 5-tap average on speed
    for (let k = 0; k < bridged.length; k++) {
      let sx = 0;
      let sy = 0;
      let wSum = 0;
      let spdSum = 0;
      let spdCnt = 0;

      for (let off = -3; off <= 3; off++) {
        const j = k + off;
        if (j < 0 || j >= bridged.length) continue;
        if (!isContinuous(k, j)) continue;
        const w = GAUSS_7[off + 3];
        sx += bridged[j].x * w;
        sy += bridged[j].y * w;
        wSum += w;
        if (Math.abs(off) <= 2) {
          spdSum += bridged[j].spd;
          spdCnt++;
        }
      }

      pass1X[k] = wSum > 0 ? sx / wSum : bridged[k].x;
      pass1Y[k] = wSum > 0 ? sy / wSum : bridged[k].y;
      smoothedSpd[k] = spdCnt > 0 ? spdSum / spdCnt : bridged[k].spd;
    }

    // Pass 2: Light 3-tap zero-phase polish eliminates residual micro-jitter
    for (let k = 0; k < bridged.length; k++) {
      let sx = 0;
      let sy = 0;
      let wSum = 0;
      for (let off = -1; off <= 1; off++) {
        const j = k + off;
        if (j < 0 || j >= bridged.length) continue;
        if (!isContinuous(k, j)) continue;
        const w = POLISH_3[off + 1];
        sx += pass1X[j] * w;
        sy += pass1Y[j] * w;
        wSum += w;
      }
      smoothedX[k] = clampSoftX(wSum > 0 ? sx / wSum : pass1X[k]);
      smoothedY[k] = clampSoftY(wSum > 0 ? sy / wSum : pass1Y[k]);
    }

    // Populate trajectory & per-frame players
    const trajPoints: CleanTrajectoryPoint[] = [];
    for (let k = 0; k < bridged.length; k++) {
      const item = bridged[k];
      const fx = smoothedX[k];
      const fy = smoothedY[k];
      const fspd = smoothedSpd[k];

      let vx = 0;
      let vy = 0;
      if (k > 0 && !segBreak[k]) {
        const dt = Math.max(1, item.idx - bridged[k - 1].idx) / fps;
        vx = (fx - smoothedX[k - 1]) / dt;
        vy = (fy - smoothedY[k - 1]) / dt;
      }

      perFramePlayers[item.idx].push({
        id: trackId,
        x: fx,
        y: fy,
        spd: fspd,
        team: item.team,
        role: item.role,
        jn: item.jn ?? null,
        jv: item.jv ?? null,
        vx,
        vy,
      });

      const prevPt = trajPoints.length > 0 ? trajPoints[trajPoints.length - 1] : null;
      const breakBefore =
        prevPt === null ||
        segBreak[k] === 1 ||
        item.f - prevPt.frame > 6 ||
        Math.hypot(fx - prevPt.x, fy - prevPt.y) > 2.8;

      trajPoints.push({
        x: fx,
        y: fy,
        frame: item.f,
        spd: fspd,
        breakBefore,
      });
    }

    trajectories.set(trackId, trajPoints);
  });

  // 2D. Anti-flicker short empty-frame hold/interpolation (1..10 frames) — skip explicit close-up frames (hv === false)
  let iEmpty = 0;
  while (iEmpty < nFrames) {
    if (perFramePlayers[iEmpty].length > 0 || rawFrames[iEmpty].hv === false) {
      iEmpty++;
      continue;
    }
    const startEmpty = iEmpty;
    while (iEmpty < nFrames && perFramePlayers[iEmpty].length === 0 && rawFrames[iEmpty].hv !== false) {
      iEmpty++;
    }
    const endEmpty = iEmpty - 1;
    const gapLen = endEmpty - startEmpty + 1;
    if (gapLen <= 10) {
      const prevIdx = startEmpty - 1;
      const nextIdx = endEmpty + 1;
      const hasPrev = prevIdx >= 0 && perFramePlayers[prevIdx].length > 0;
      const hasNext = nextIdx < nFrames && perFramePlayers[nextIdx].length > 0;
      if (hasPrev && hasNext) {
        const prevMap = new Map<number, StabilizedPlayer>();
        for (const p of perFramePlayers[prevIdx]) prevMap.set(p.id, p);
        const nextMap = new Map<number, StabilizedPlayer>();
        for (const p of perFramePlayers[nextIdx]) nextMap.set(p.id, p);

        for (let k = startEmpty; k <= endEmpty; k++) {
          const ratio = (k - prevIdx) / (nextIdx - prevIdx);
          const held: StabilizedPlayer[] = [];
          prevMap.forEach((pa, pid) => {
            const pb = nextMap.get(pid);
            if (pb && Math.hypot(pb.x - pa.x, pb.y - pa.y) <= maxJumpForDt(nextIdx - prevIdx) * 1.4) {
              held.push({
                ...pa,
                x: pa.x + (pb.x - pa.x) * ratio,
                y: pa.y + (pb.y - pa.y) * ratio,
                spd: pa.spd + (pb.spd - pa.spd) * ratio,
              });
            } else if (ratio < 0.5) {
              held.push({ ...pa });
            }
          });
          if (ratio >= 0.5) {
            nextMap.forEach((pb, pid) => {
              if (!prevMap.has(pid)) held.push({ ...pb });
            });
          }
          perFramePlayers[k] = held;
        }
      } else if (hasPrev) {
        for (let k = startEmpty; k <= endEmpty; k++) {
          perFramePlayers[k] = perFramePlayers[prevIdx].map((p) => ({ ...p }));
        }
      }
    }
  }

  // ── 3. Ball Outlier Rejection, Corner Snap Guard & 3-Tap Smoothing ──
  interface RawBallObs {
    idx: number;
    x: number;
    y: number;
  }
  const ballObs: RawBallObs[] = [];
  for (let i = 0; i < nFrames; i++) {
    if (isGlitchFrame[i]) continue;
    const b = rawFrames[i].ball;
    if (!b || b.length < 2 || !Number.isFinite(b[0]) || !Number.isFinite(b[1])) continue;
    if (isSentinelOob(b[0], b[1])) continue;
    const bx = Math.max(0, Math.min(PITCH_L, b[0]));
    const by = Math.max(0, Math.min(PITCH_W, b[1]));
    const isCornerSnap =
      (bx <= 0.3 || bx >= PITCH_L - 0.3) && (by <= 0.3 || by >= PITCH_W - 0.3);
    if (isCornerSnap && ballObs.length > 0) {
      const prevB = ballObs[ballObs.length - 1];
      if (Math.hypot(bx - prevB.x, by - prevB.y) > 8.0) continue;
    }
    ballObs.push({ idx: i, x: bx, y: by });
  }

  // Drop isolated ball spikes
  let cleanBallObs: RawBallObs[] = [];
  if (ballObs.length > 0) {
    cleanBallObs = [ballObs[0]];
    for (let k = 1; k < ballObs.length; k++) {
      const prev = cleanBallObs[cleanBallObs.length - 1];
      const cur = ballObs[k];
      const dtPrev = Math.max(1, cur.idx - prev.idx);
      const dPrev = Math.hypot(cur.x - prev.x, cur.y - prev.y);
      if (dPrev > Math.max(7.5, dtPrev * 2.4)) {
        if (k < ballObs.length - 1) {
          const nxt = ballObs[k + 1];
          const dtNext = Math.max(1, nxt.idx - cur.idx);
          const dNext = Math.hypot(nxt.x - cur.x, nxt.y - cur.y);
          if (dNext > Math.max(7.5, dtNext * 2.4)) {
            continue;
          }
        } else {
          continue;
        }
      }
      cleanBallObs.push(cur);
    }
  }

  const ballTimeline: Array<[number, number] | null> = new Array(nFrames).fill(null);
  const bridgedBall: RawBallObs[] = [];
  for (let k = 0; k < cleanBallObs.length; k++) {
    const cur = cleanBallObs[k];
    if (k > 0) {
      const prev = cleanBallObs[k - 1];
      const gap = cur.idx - prev.idx;
      const dist = Math.hypot(cur.x - prev.x, cur.y - prev.y);
      if (gap > 1 && gap <= 8 && dist <= Math.max(5.0, gap * 2.4)) {
        for (let g = 1; g < gap; g++) {
          const r = g / gap;
          bridgedBall.push({
            idx: prev.idx + g,
            x: prev.x + (cur.x - prev.x) * r,
            y: prev.y + (cur.y - prev.y) * r,
          });
        }
      }
    }
    bridgedBall.push(cur);
  }

  for (let k = 0; k < bridgedBall.length; k++) {
    const cur = bridgedBall[k];
    let bx = cur.x;
    let by = cur.y;
    const hasPrev = k > 0 && cur.idx - bridgedBall[k - 1].idx <= 2;
    const hasNext = k < bridgedBall.length - 1 && bridgedBall[k + 1].idx - cur.idx <= 2;
    if (hasPrev && hasNext) {
      const p = bridgedBall[k - 1];
      const n = bridgedBall[k + 1];
      if (Math.hypot(p.x - cur.x, p.y - cur.y) < 5.0 && Math.hypot(n.x - cur.x, n.y - cur.y) < 5.0) {
        bx = 0.2 * p.x + 0.6 * cur.x + 0.2 * n.x;
        by = 0.2 * p.y + 0.6 * cur.y + 0.2 * n.y;
      }
    }
    ballTimeline[cur.idx] = [clampSoftX(bx), clampSoftY(by)];
  }

  // ── 4. Temporally Damped Hull Areas, Defensive Lines & Voronoi Split ──
  const stabilizedFrames: StabilizedFrame[] = [];
  let emaDef0: number | null = null;
  let emaDef1: number | null = null;
  let emaVor0 = rawFrames[0]?.vor?.[0] ?? 50;
  let emaHull0 = 0;
  let emaHull1 = 0;
  const EMA_ALPHA = 0.18;
  const MAX_DEF_STEP_PER_FRAME = 1.15;

  // Estimate attacking direction hint for teams across entire clip
  let sumX0 = 0, c0 = 0, sumX1 = 0, c1 = 0;
  for (let i = 0; i < nFrames; i++) {
    for (const p of perFramePlayers[i]) {
      if (p.role !== "p") continue;
      if (p.team === 0) { sumX0 += p.x; c0++; }
      else { sumX1 += p.x; c1++; }
    }
  }
  const team0PlaysLeft = (c0 > 0 ? sumX0 / c0 : 35) <= (c1 > 0 ? sumX1 / c1 : 70);

  const computeCleanDef = (pts: { x: number }[], takeLowest: boolean): number | null => {
    if (pts.length === 0) return null;
    const xs = pts.map((pt) => pt.x).sort((a, b) => a - b);
    return takeLowest ? xs[0] : xs[xs.length - 1];
  };

  for (let i = 0; i < nFrames; i++) {
    const rawFr = rawFrames[i];
    const players = perFramePlayers[i];
    const glitch = isGlitchFrame[i] === 1;

    // Outfield convex hulls from clean stabilized players (strictly excludes GK & referees)
    const out0 = players.filter((p) => p.team === 0 && p.role === "p").map((p) => ({ x: p.x, y: p.y }));
    const out1 = players.filter((p) => p.team === 1 && p.role === "p").map((p) => ({ x: p.x, y: p.y }));

    const rawHull0 = out0.length >= 3 ? computePolygonArea(computeConvexHull(out0)) : emaHull0;
    const rawHull1 = out1.length >= 3 ? computePolygonArea(computeConvexHull(out1)) : emaHull1;

    if (i === 0 || emaHull0 === 0) {
      emaHull0 = rawHull0;
      emaHull1 = rawHull1;
    } else {
      const clampedTarget0 =
        Math.abs(rawHull0 - emaHull0) > 280
          ? emaHull0 + Math.sign(rawHull0 - emaHull0) * 280
          : rawHull0;
      const clampedTarget1 =
        Math.abs(rawHull1 - emaHull1) > 280
          ? emaHull1 + Math.sign(rawHull1 - emaHull1) * 280
          : rawHull1;
      emaHull0 = emaHull0 + EMA_ALPHA * (clampedTarget0 - emaHull0);
      emaHull1 = emaHull1 + EMA_ALPHA * (clampedTarget1 - emaHull1);
    }

    // Defensive line computed from stabilized players + rate-limited spike damping
    const stabDef0 = computeCleanDef(out0, team0PlaysLeft);
    let targetDef0: number | undefined = rawFr.def_x?.["0"];
    const rawDef0Valid =
      !glitch &&
      typeof targetDef0 === "number" &&
      Number.isFinite(targetDef0) &&
      targetDef0 >= 1.5 &&
      targetDef0 <= 103.5 &&
      (stabDef0 === null || Math.abs(targetDef0 - stabDef0) <= 5.5);
    if (!rawDef0Valid) {
      targetDef0 = stabDef0 ?? emaDef0 ?? undefined;
    }
    if (typeof targetDef0 === "number" && Number.isFinite(targetDef0)) {
      if (emaDef0 === null) {
        emaDef0 = targetDef0;
      } else {
        const delta0 = targetDef0 - emaDef0;
        const clampedStep0 = Math.max(
          -MAX_DEF_STEP_PER_FRAME,
          Math.min(MAX_DEF_STEP_PER_FRAME, delta0 * EMA_ALPHA)
        );
        emaDef0 = clampSoftX(emaDef0 + clampedStep0);
      }
    }

    const stabDef1 = computeCleanDef(out1, !team0PlaysLeft);
    let targetDef1: number | undefined = rawFr.def_x?.["1"];
    const rawDef1Valid =
      !glitch &&
      typeof targetDef1 === "number" &&
      Number.isFinite(targetDef1) &&
      targetDef1 >= 1.5 &&
      targetDef1 <= 103.5 &&
      (stabDef1 === null || Math.abs(targetDef1 - stabDef1) <= 5.5);
    if (!rawDef1Valid) {
      targetDef1 = stabDef1 ?? emaDef1 ?? undefined;
    }
    if (typeof targetDef1 === "number" && Number.isFinite(targetDef1)) {
      if (emaDef1 === null) {
        emaDef1 = targetDef1;
      } else {
        const delta1 = targetDef1 - emaDef1;
        const clampedStep1 = Math.max(
          -MAX_DEF_STEP_PER_FRAME,
          Math.min(MAX_DEF_STEP_PER_FRAME, delta1 * EMA_ALPHA)
        );
        emaDef1 = clampSoftX(emaDef1 + clampedStep1);
      }
    }

    const def_x: Record<string, number> = {};
    if (emaDef0 !== null) def_x["0"] = Number(emaDef0.toFixed(2));
    if (emaDef1 !== null) def_x["1"] = Number(emaDef1.toFixed(2));

    // Voronoi percentage smoothing
    const rawVor0 = !glitch && typeof rawFr.vor?.[0] === "number" ? rawFr.vor[0] : emaVor0;
    emaVor0 = i === 0 ? rawVor0 : emaVor0 + 0.2 * (rawVor0 - emaVor0);
    const cleanVor0 = Number(Math.max(5, Math.min(95, emaVor0)).toFixed(1));
    const cleanVor1 = Number((100 - cleanVor0).toFixed(1));

    const explicitInvalid = rawFr.hv === false;
    const homographyValid = !explicitInvalid && players.length > 0;
    stabilizedFrames.push({
      f: rawFr.f,
      t: rawFr.t,
      players: explicitInvalid ? [] : players,
      ball: explicitInvalid ? null : ballTimeline[i],
      def_x: explicitInvalid ? {} : def_x,
      vor: [cleanVor0, cleanVor1],
      hullArea: {
        team0_m2: Math.round(emaHull0),
        team1_m2: Math.round(emaHull1),
      },
      isGlitchBridged: glitch,
      homographyValid,
    });
  }

  // Zero-phase 5-tap smoothing pass over defensive lines to eliminate residual wobble
  const DEF_KERN = [0.12, 0.22, 0.32, 0.22, 0.12];
  for (const teamKey of ["0", "1"] as const) {
    const rawVals = stabilizedFrames.map((fr) => fr.def_x[teamKey]);
    for (let i = 0; i < stabilizedFrames.length; i++) {
      if (rawVals[i] === undefined) continue;
      let sum = 0;
      let wSum = 0;
      for (let off = -2; off <= 2; off++) {
        const j = i + off;
        if (j < 0 || j >= stabilizedFrames.length) continue;
        const v = rawVals[j];
        if (v === undefined) continue;
        const w = DEF_KERN[off + 2];
        sum += v * w;
        wSum += w;
      }
      if (wSum > 0) {
        stabilizedFrames[i].def_x[teamKey] = Number((sum / wSum).toFixed(2));
      }
    }
  }

  return {
    fps,
    duration_s: telemetry.summary?.duration_s || nFrames / fps,
    frames: stabilizedFrames,
    trajectories,
    ballTimeline,
    glitchFrameCount,
  };
}

/**
 * Computes a silky 60FPS linearly interpolated tactical snapshot at continuous floating frame `exactFrame`.
 */
export function interpolateStabilizedFrame(
  stabilized: StabilizedTelemetry | null,
  exactFrame: number
): StabilizedFrame | null {
  if (!stabilized || stabilized.frames.length === 0) return null;

  const frames = stabilized.frames;
  const maxIdx = frames.length - 1;
  const clamped = Math.max(0, Math.min(maxIdx, exactFrame));
  const f0 = Math.floor(clamped);
  const f1 = Math.min(maxIdx, f0 + 1);
  const alpha = clamped - f0;

  const a = frames[f0];
  const b = frames[f1];
  if (f0 === f1 || alpha <= 0.001) return a;
  if (alpha >= 0.999) return b;

  // Player interpolation by track ID (tight <= 2.4m continuity guard prevents lerping across cuts)
  const mapB = new Map<number, StabilizedPlayer>();
  for (const pb of b.players) {
    mapB.set(pb.id, pb);
  }

  const seenA = new Set<number>();
  const interpPlayers: StabilizedPlayer[] = [];
  for (const pa of a.players) {
    seenA.add(pa.id);
    const pb = mapB.get(pa.id);
    if (pb && Math.hypot(pb.x - pa.x, pb.y - pa.y) <= 2.4) {
      interpPlayers.push({
        id: pa.id,
        x: pa.x + (pb.x - pa.x) * alpha,
        y: pa.y + (pb.y - pa.y) * alpha,
        spd: pa.spd + (pb.spd - pa.spd) * alpha,
        team: pb.team,
        role: pb.role,
        jn: pb.jn ?? pa.jn ?? null,
        jv: pb.jv ?? pa.jv ?? null,
        vx: pa.vx,
        vy: pa.vy,
      });
    } else if (alpha < 0.5) {
      interpPlayers.push(pa);
    }
  }
  if (alpha >= 0.5) {
    for (const pb of b.players) {
      if (!seenA.has(pb.id)) {
        interpPlayers.push(pb);
      }
    }
  }

  // Ball interpolation
  let interpBall: [number, number] | null = a.ball;
  if (a.ball && b.ball) {
    if (Math.hypot(b.ball[0] - a.ball[0], b.ball[1] - a.ball[1]) <= 6.0) {
      interpBall = [
        a.ball[0] + (b.ball[0] - a.ball[0]) * alpha,
        a.ball[1] + (b.ball[1] - a.ball[1]) * alpha,
      ];
    }
  } else if (b.ball && alpha > 0.5) {
    interpBall = b.ball;
  }

  // Def lines & Voronoi interpolation
  const def_x: Record<string, number> = {};
  if (a.def_x["0"] !== undefined && b.def_x["0"] !== undefined) {
    def_x["0"] = a.def_x["0"] + (b.def_x["0"] - a.def_x["0"]) * alpha;
  } else if (a.def_x["0"] !== undefined) {
    def_x["0"] = a.def_x["0"];
  }

  if (a.def_x["1"] !== undefined && b.def_x["1"] !== undefined) {
    def_x["1"] = a.def_x["1"] + (b.def_x["1"] - a.def_x["1"]) * alpha;
  } else if (a.def_x["1"] !== undefined) {
    def_x["1"] = a.def_x["1"];
  }

  const vor0 = Number((a.vor[0] + (b.vor[0] - a.vor[0]) * alpha).toFixed(1));
  const vor1 = Number((100 - vor0).toFixed(1));

  return {
    f: Math.round(a.f + (b.f - a.f) * alpha),
    t: a.t + (b.t - a.t) * alpha,
    players: interpPlayers,
    ball: interpBall,
    def_x,
    vor: [vor0, vor1],
    hullArea: {
      team0_m2: Math.round(
        a.hullArea.team0_m2 + (b.hullArea.team0_m2 - a.hullArea.team0_m2) * alpha
      ),
      team1_m2: Math.round(
        a.hullArea.team1_m2 + (b.hullArea.team1_m2 - a.hullArea.team1_m2) * alpha
      ),
    },
    isGlitchBridged: a.isGlitchBridged || b.isGlitchBridged,
    homographyValid: alpha < 0.5 ? a.homographyValid : b.homographyValid,
  };
}

/**
 * Selects the cleanest, highest-tempo tactical window in `stabilized.frames`
 * for the Homepage Hero continuous replay showcase.
 */
export function findBestHeroWindow(
  stabilized: StabilizedTelemetry | null,
  windowFrames = 175
): HeroSequenceWindow {
  if (!stabilized || stabilized.frames.length === 0) {
    return { startFrame: 0, endFrame: 0, score: 0 };
  }

  const frames = stabilized.frames;
  const total = frames.length;
  const win = Math.min(windowFrames, total);
  if (total <= win) {
    return { startFrame: 0, endFrame: total - 1, score: 1 };
  }

  let best: HeroSequenceWindow = {
    startFrame: 0,
    endFrame: win - 1,
    score: -Infinity,
  };

  for (let start = 0; start <= total - win; start += 5) {
    let glitchCount = 0;
    let ballCount = 0;
    let sumPlayers = 0;
    let sumSpeed = 0;

    for (let i = start; i < start + win; i++) {
      const fr = frames[i];
      if (fr.isGlitchBridged) glitchCount++;
      if (fr.ball) ballCount++;
      sumPlayers += fr.players.length;
      for (const p of fr.players) {
        sumSpeed += Math.min(35, p.spd);
      }
    }

    const ballRatio = ballCount / win;
    const avgPlayers = sumPlayers / win;
    const avgCollectiveSpeed = sumSpeed / Math.max(1, sumPlayers);
    const score =
      ballRatio * 45 +
      avgPlayers * 2.2 +
      avgCollectiveSpeed * 1.4 -
      glitchCount * 12;

    if (score > best.score) {
      best = {
        startFrame: start,
        endFrame: start + win - 1,
        score: Number(score.toFixed(2)),
      };
    }
  }

  return best;
}
