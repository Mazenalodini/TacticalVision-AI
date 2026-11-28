import type { Telemetry } from "@/lib/types";
import { resolvePlayerPositions } from "@/lib/telemetryHelpers";
import { resolvePlayerIdentity, type MatchOverridesMap } from "@/lib/jerseyOverrides";
import { PitchViewport, pitchToCanvas } from "./pitchConfig";

/**
 * Draws calm, static directed passing corridors on the 2D PitchCanvas without dropping short-clip players.
 * Renders clean curved links scaled by pass count with crisp static directional arrowheads.
 */
export function drawPassNetwork(
  ctx: CanvasRenderingContext2D,
  telemetry: Telemetry,
  showLivePlayers: boolean,
  vp: PitchViewport,
  overrides?: MatchOverridesMap
) {
  const edges = telemetry?.passing_network;
  if (!edges || edges.length === 0) return;

  const playerMap = resolvePlayerPositions(telemetry);

  // 1. Draw Static Directed Curved Corridors
  ctx.save();
  ctx.lineCap = "round";
  edges.forEach((edge) => {
    const fromP = playerMap.get(edge.from_id);
    const toP = playerMap.get(edge.to_id);
    if (!fromP || !toP) return;

    const [x1, y1] = pitchToCanvas(fromP.x, fromP.y, vp);
    const [x2, y2] = pitchToCanvas(toP.x, toP.y, vp);

    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.hypot(dx, dy);
    if (dist < 3) return;

    // Slight curve so reciprocal passes (A->B and B->A) never collide
    const nx = -dy / dist;
    const ny = dx / dist;
    const bow = Math.min(20, Math.max(8, dist * 0.14));
    const ctrlX = (x1 + x2) / 2 + nx * bow;
    const ctrlY = (y1 + y2) / 2 + ny * bow;

    const isTeam0 = edge.team === 0;
    const color = isTeam0 ? "#38BDF8" : "#FB7185";
    const alpha = Math.min(0.88, 0.42 + edge.count * 0.14);

    ctx.strokeStyle = isTeam0
      ? `rgba(56, 189, 248, ${alpha})`
      : `rgba(251, 113, 133, ${alpha})`;
    ctx.lineWidth = Math.min(4.8, Math.max(1.6, edge.count * 1.35));

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(ctrlX, ctrlY, x2, y2);
    ctx.stroke();

    // Static directional arrowhead near target anchor
    const angle = Math.atan2(y2 - ctrlY, x2 - ctrlX);
    const arrowDist = showLivePlayers ? 11 : 14;
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
  });
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
    const identity = resolvePlayerIdentity(p.track_id, p.stat, overrides);

    ctx.save();
    if (showLivePlayers) {
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
      const nodeR = 11;
      ctx.beginPath();
      ctx.arc(nx, ny, nodeR, 0, Math.PI * 2);
      ctx.fillStyle = "#080D18";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.stroke();

      if (identity.isVerified) {
        ctx.beginPath();
        ctx.arc(nx, ny, nodeR + 1.6, 0, Math.PI * 2);
        ctx.strokeStyle = identity.isCoachOverride
          ? "#FBBF24"
          : "rgba(52, 211, 153, 0.85)";
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(nx + nodeR * 0.74, ny - nodeR * 0.74, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = identity.isCoachOverride ? "#FBBF24" : "#34D399";
        ctx.fill();
        ctx.strokeStyle = "#080D18";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      if (identity.canvasLabel) {
        const label = identity.canvasLabel;
        const fontSize = label.length >= 3 ? 7.6 : 9.0;
        ctx.fillStyle = identity.isVerified ? "#FFFFFF" : "#E2E8F0";
        ctx.font = `${identity.isVerified ? 800 : 700} ${fontSize}px var(--font-jetbrains), monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, nx, ny + 0.5);
      }
    }
    ctx.restore();
  }
}
