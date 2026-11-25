import { FramePlayer } from "@/lib/types";
import { PitchViewport, pitchToCanvas } from "./pitchConfig";

export interface Point2D {
  x: number;
  y: number;
}

/**
 * 2D cross product of OA and OB vectors.
 */
function cross(o: Point2D, a: Point2D, b: Point2D): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

/**
 * Computes the 2D Convex Hull using Andrew's Monotone Chain algorithm.
 */
export function computeConvexHull(points: Point2D[]): Point2D[] {
  if (points.length <= 2) return [...points];

  const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));

  const lower: Point2D[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point2D[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();

  return lower.concat(upper);
}

/**
 * Computes polygon area in m² using the Shoelace formula.
 */
export function computePolygonArea(hull: Point2D[]): number {
  if (hull.length < 3) return 0;
  let area = 0;
  const n = hull.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += hull[i].x * hull[j].y;
    area -= hull[j].x * hull[i].y;
  }
  return Math.abs(area) * 0.5;
}

export interface HullAreasResult {
  team0Area_m2: number;
  team1Area_m2: number;
}

/**
 * Draws stabilized team outfield convex hull polygons with subtle vertex anchor pips
 * and returns compactness areas in m².
 */
export function drawHulls(
  ctx: CanvasRenderingContext2D,
  players: FramePlayer[],
  vp: PitchViewport
): HullAreasResult {
  const result: HullAreasResult = { team0Area_m2: 0, team1Area_m2: 0 };
  if (!players || players.length === 0) return result;

  const isCleanOutfield = (p: FramePlayer, team: 0 | 1) =>
    p.team === team &&
    p.role === "p" &&
    Number.isFinite(p.x) &&
    Number.isFinite(p.y) &&
    p.x > -1.2 &&
    p.x < 106.2 &&
    p.y > -1.2 &&
    p.y < 69.2;

  const clampPoint = (p: FramePlayer): Point2D => ({
    x: Math.max(0.2, Math.min(104.8, p.x)),
    y: Math.max(0.2, Math.min(67.8, p.y)),
  });

  const team0Outfield = players.filter((p) => isCleanOutfield(p, 0)).map(clampPoint);
  const team1Outfield = players.filter((p) => isCleanOutfield(p, 1)).map(clampPoint);

  const renderTeamHull = (
    pts: Point2D[],
    strokeColor: string,
    fillColor: string
  ): number => {
    if (pts.length < 3) return 0;
    const hull = computeConvexHull(pts);
    if (hull.length < 3) return 0;

    const area = computePolygonArea(hull);

    ctx.save();
    ctx.beginPath();
    const canvasPts = hull.map((h) => pitchToCanvas(h.x, h.y, vp));
    ctx.moveTo(canvasPts[0][0], canvasPts[0][1]);
    for (let i = 1; i < canvasPts.length; i++) {
      ctx.lineTo(canvasPts[i][0], canvasPts[i][1]);
    }
    ctx.closePath();

    ctx.fillStyle = fillColor;
    ctx.fill();

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.stroke();

    // Subtle vertex anchor pips
    ctx.setLineDash([]);
    ctx.fillStyle = strokeColor;
    for (const [vx, vy] of canvasPts) {
      ctx.beginPath();
      ctx.arc(vx, vy, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();

    return Math.round(area);
  };

  result.team0Area_m2 = renderTeamHull(
    team0Outfield,
    "rgba(56, 189, 248, 0.78)",
    "rgba(56, 189, 248, 0.08)"
  );

  result.team1Area_m2 = renderTeamHull(
    team1Outfield,
    "rgba(251, 113, 133, 0.78)",
    "rgba(251, 113, 133, 0.08)"
  );

  return result;
}
