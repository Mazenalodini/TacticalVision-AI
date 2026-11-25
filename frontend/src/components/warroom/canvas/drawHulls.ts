import { FramePlayer } from "@/lib/types";
import { PitchViewport, pitchToCanvas } from "./pitchConfig";

interface Point2D {
  x: number;
  y: number;
}

/**
 * 2D cross product of OA and OB vectors.
 * Returns positive for counter-clockwise turn, negative for clockwise, 0 for collinear.
 */
function cross(o: Point2D, a: Point2D, b: Point2D): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

/**
 * Computes the 2D Convex Hull of a set of points using Andrew's Monotone Chain algorithm.
 * Time complexity: O(n log n)
 */
export function computeConvexHull(points: Point2D[]): Point2D[] {
  if (points.length <= 2) return [...points];

  // 1. Sort points lexicographically (by x, then by y)
  const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));

  // 2. Build lower hull
  const lower: Point2D[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  // 3. Build upper hull
  const upper: Point2D[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  // Remove the last point of each half because it is repeated
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
 * Draws team convex hull polygons and returns live compactness areas in m².
 * Only considers outfield players (role === "p"), excluding goalkeepers.
 */
export function drawHulls(
  ctx: CanvasRenderingContext2D,
  players: FramePlayer[],
  vp: PitchViewport
): HullAreasResult {
  const result: HullAreasResult = { team0Area_m2: 0, team1Area_m2: 0 };
  if (!players || players.length === 0) return result;

  // Filter outfield players and clamp strictly into pitch bounds so computed m² matches rendered polygon
  const clampPoint = (p: FramePlayer): Point2D => ({
    x: Math.max(0, Math.min(105, p.x)),
    y: Math.max(0, Math.min(68, p.y)),
  });

  const team0Outfield = players
    .filter(
      (p) =>
        p.team === 0 &&
        p.role === "p" &&
        p.x >= -3 &&
        p.x <= 108 &&
        p.y >= -3 &&
        p.y <= 71
    )
    .map(clampPoint);

  const team1Outfield = players
    .filter(
      (p) =>
        p.team === 1 &&
        p.role === "p" &&
        p.x >= -3 &&
        p.x <= 108 &&
        p.y >= -3 &&
        p.y <= 71
    )
    .map(clampPoint);

  // Helper to render one team's polygon
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
    const [startX, startY] = pitchToCanvas(hull[0].x, hull[0].y, vp);
    ctx.moveTo(startX, startY);
    for (let i = 1; i < hull.length; i++) {
      const [hx, hy] = pitchToCanvas(hull[i].x, hull[i].y, vp);
      ctx.lineTo(hx, hy);
    }
    ctx.closePath();

    ctx.fillStyle = fillColor;
    ctx.fill();

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.restore();

    return Math.round(area);
  };

  // Team A Hull (Broadcast Sky)
  result.team0Area_m2 = renderTeamHull(
    team0Outfield,
    "#38BDF8",
    "rgba(56, 189, 248, 0.11)"
  );

  // Team B Hull (Luminous Rose-Coral)
  result.team1Area_m2 = renderTeamHull(
    team1Outfield,
    "#FB7185",
    "rgba(251, 113, 133, 0.11)"
  );

  return result;
}
