import type { Telemetry } from "@/lib/types";
import { resolvePlayerPositions } from "@/lib/telemetryHelpers";
import { PitchViewport, pitchToCanvas } from "./pitchConfig";

/**
 * Draws directed passing network graph on the 2D PitchCanvas without dropping short-clip players.
 * When live players layer is active (`showLivePlayers`), renders pass-anchor waypoints as subtle
 * translucent hollow rings (no duplicate `#id` text) so they never clash visually with live players.
 */
export function drawPassNetwork(
  ctx: CanvasRenderingContext2D,
  telemetry: Telemetry,
  showLivePlayers: boolean,
  vp: PitchViewport
) {
  const edges = telemetry?.passing_network;
  if (!edges || edges.length === 0) return;

  const playerMap = resolvePlayerPositions(telemetry);

  // 1. Draw Directed Curved Arrows
  ctx.save();
  for (const edge of edges) {
    const fromP = playerMap.get(edge.from_id);
    const toP = playerMap.get(edge.to_id);
    if (!fromP || !toP) continue;

    const [x1, y1] = pitchToCanvas(fromP.x, fromP.y, vp);
    const [x2, y2] = pitchToCanvas(toP.x, toP.y, vp);

    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.hypot(dx, dy);
    if (dist < 3) continue;

    // Slight curve so reciprocal passes (A->B and B->A) never collide
    const nx = -dy / dist;
    const ny = dx / dist;
    const bow = Math.min(20, Math.max(8, dist * 0.14));
    const ctrlX = (x1 + x2) / 2 + nx * bow;
    const ctrlY = (y1 + y2) / 2 + ny * bow;

    const isTeam0 = edge.team === 0;
    const color = isTeam0 ? "#38BDF8" : "#FB7185";
    const alpha = Math.min(0.88, 0.4 + edge.count * 0.14);

    ctx.strokeStyle = isTeam0
      ? `rgba(56, 189, 248, ${alpha})`
      : `rgba(251, 113, 133, ${alpha})`;
    ctx.lineWidth = Math.min(5, Math.max(1.6, edge.count * 1.4));

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(ctrlX, ctrlY, x2, y2);
    ctx.stroke();

    // Arrowhead near target anchor
    const angle = Math.atan2(y2 - ctrlY, x2 - ctrlX);
    const arrowDist = showLivePlayers ? 10 : 14;
    const ax = x2 - arrowDist * Math.cos(angle);
    const ay = y2 - arrowDist * Math.sin(angle);
    const headLen = Math.min(9.5, 6 + edge.count);

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(
      ax - headLen * Math.cos(angle - Math.PI / 6),
      ay - headLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      ax - headLen * Math.cos(angle + Math.PI / 6),
      ay - headLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();

  // 2. Draw Passing Anchor Nodes (subtle hollow rings when live dots are ON)
  const activeIds = new Set<number>();
  for (const e of edges) {
    activeIds.add(e.from_id);
    activeIds.add(e.to_id);
  }

  for (const tid of activeIds) {
    const p = playerMap.get(tid);
    if (!p) continue;

    const [nx, ny] = pitchToCanvas(p.x, p.y, vp);
    const isTeam0 = p.team === 0;
    const color = isTeam0 ? "#38BDF8" : "#FB7185";

    ctx.save();
    if (showLivePlayers) {
      // Subtle translucent hollow anchor ring (no duplicate #ID circle clash)
      ctx.beginPath();
      ctx.arc(nx, ny, 7.5, 0, Math.PI * 2);
      ctx.fillStyle = isTeam0
        ? "rgba(56, 189, 248, 0.14)"
        : "rgba(251, 113, 133, 0.14)";
      ctx.fill();
      ctx.strokeStyle = isTeam0
        ? "rgba(56, 189, 248, 0.78)"
        : "rgba(251, 113, 133, 0.78)";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([2, 2]);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(nx, ny, 10.5, 0, Math.PI * 2);
      ctx.fillStyle = "#080D18";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.stroke();

      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 9px var(--font-jetbrains), monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`${p.track_id}`, nx, ny);
    }
    ctx.restore();
  }
}
