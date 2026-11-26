import { FramePlayer } from "@/lib/types";
import { FIFA_PITCH, PitchViewport, pitchToCanvas } from "./pitchConfig";

export interface BallHistoryPoint {
  x: number;
  y: number;
  frame: number;
}

export interface PlayerTrajectoryPoint {
  x: number;
  y: number;
  frame: number;
  breakBefore?: boolean;
}

/**
 * Draws stabilized players, goalkeepers, sprint halos, selected spotlight reticle,
 * discontinuity-safe fading comet player trajectory, and luminous ball comet trail.
 */
export function drawPlayers(
  ctx: CanvasRenderingContext2D,
  players: FramePlayer[] | undefined,
  ball: [number, number] | null | undefined,
  ballTrail: BallHistoryPoint[],
  selectedPlayerId: number | null,
  trajectory: PlayerTrajectoryPoint[] | undefined,
  currentFrameIndex: number,
  vp: PitchViewport,
  hoveredPlayerId: number | null = null
) {
  // ── 0. Selected Player Discontinuity-Safe Trajectory Path ──
  if (selectedPlayerId !== null && trajectory && trajectory.length > 1) {
    ctx.save();

    const isJump = (p1: PlayerTrajectoryPoint, p2: PlayerTrajectoryPoint) =>
      Boolean(p2.breakBefore) ||
      p2.frame - p1.frame > 6 ||
      Math.hypot(p2.x - p1.x, p2.y - p1.y) > 3.8;

    // 0A. Subtle full-clip ghost path (breaks cleanly across gaps/glitches)
    ctx.beginPath();
    ctx.lineWidth = 1.3;
    ctx.strokeStyle = "rgba(226, 232, 240, 0.16)";
    ctx.setLineDash([3, 4]);

    for (let i = 1; i < trajectory.length; i++) {
      const p1 = trajectory[i - 1];
      const p2 = trajectory[i];
      if (isJump(p1, p2)) continue;

      const [x1, y1] = pitchToCanvas(p1.x, p1.y, vp);
      const [x2, y2] = pitchToCanvas(p2.x, p2.y, vp);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();

    // 0B. Luminous Mint fading comet tail up to currentFrameIndex (last ~100 frames)
    const pastWindowStart = Math.max(0, currentFrameIndex - 100);
    const pastTrajectory = trajectory.filter(
      (p) => p.frame <= currentFrameIndex && p.frame >= pastWindowStart
    );

    if (pastTrajectory.length > 1) {
      ctx.setLineDash([]);
      ctx.lineCap = "round";
      const total = pastTrajectory.length;
      for (let i = 1; i < total; i++) {
        const p1 = pastTrajectory[i - 1];
        const p2 = pastTrajectory[i];
        if (isJump(p1, p2)) continue;

        const [x1, y1] = pitchToCanvas(p1.x, p1.y, vp);
        const [x2, y2] = pitchToCanvas(p2.x, p2.y, vp);
        const progress = i / total;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(52, 211, 153, ${(0.12 + progress * 0.82).toFixed(2)})`;
        ctx.lineWidth = 1.6 + progress * 1.9;
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  // ── 1. Ball Luminous Comet Trail ──
  if (ballTrail.length > 1) {
    ctx.save();
    const trailLen = ballTrail.length;
    ctx.lineCap = "round";
    for (let i = 1; i < trailLen; i++) {
      const p1 = ballTrail[i - 1];
      const p2 = ballTrail[i];

      const frameGap = p2.frame - p1.frame;
      const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);

      if (frameGap >= 0 && frameGap <= 3 && dist <= 7.5) {
        const [cx1, cy1] = pitchToCanvas(p1.x, p1.y, vp);
        const [cx2, cy2] = pitchToCanvas(p2.x, p2.y, vp);

        const progress = i / trailLen;
        const alpha = Math.max(0.08, progress * 0.78);
        const lineWidth = Math.max(1.4, progress * 3.6);

        ctx.beginPath();
        ctx.strokeStyle = `rgba(52, 211, 153, ${alpha.toFixed(2)})`;
        ctx.lineWidth = lineWidth;
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  // ── 2. Ball Marker with Bloom Halo ──
  if (ball && ball.length === 2 && Number.isFinite(ball[0]) && Number.isFinite(ball[1])) {
    const [bx, by] = pitchToCanvas(ball[0], ball[1], vp);

    ctx.save();
    const ballGlow = ctx.createRadialGradient(bx, by, 1, bx, by, 11);
    ballGlow.addColorStop(0, "rgba(255, 255, 255, 0.55)");
    ballGlow.addColorStop(0.5, "rgba(52, 211, 153, 0.35)");
    ballGlow.addColorStop(1, "rgba(52, 211, 153, 0)");
    ctx.fillStyle = ballGlow;
    ctx.beginPath();
    ctx.arc(bx, by, 11, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(bx, by, 4.4, 0, Math.PI * 2);
    ctx.fillStyle = "#FFFFFF";
    ctx.fill();
    ctx.strokeStyle = "#060B10";
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.restore();
  }

  // ── 3. Players & Goalkeepers ──
  if (!players || players.length === 0) return;

  const now = performance.now();
  const pulseScale = 1.0 + 0.14 * Math.sin(now * 0.006);

  for (const player of players) {
    if (
      !Number.isFinite(player.x) ||
      !Number.isFinite(player.y) ||
      player.x < -1.2 ||
      player.x > FIFA_PITCH.length + 1.2 ||
      player.y < -1.2 ||
      player.y > FIFA_PITCH.width + 1.2
    ) {
      continue;
    }

    const [cx, cy] = pitchToCanvas(player.x, player.y, vp);
    const isTeam0 = player.team === 0;
    const teamColor = isTeam0 ? "#38BDF8" : "#FB7185";
    const isSelected = selectedPlayerId === player.id;
    const isHovered = hoveredPlayerId === player.id;
    const isGK = player.role === "gk";
    const isSprint = player.spd >= FIFA_PITCH.sprintThresholdKmh;

    ctx.save();

    if (selectedPlayerId !== null && !isSelected && !isHovered) {
      ctx.globalAlpha = 0.34;
    }

    // Selected player tactical dual-ring reticle
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(cx, cy, 17.5 * pulseScale, 0, Math.PI * 2);
      ctx.strokeStyle = "#34D399";
      ctx.lineWidth = 2.2;
      ctx.setLineDash([4, 3]);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(cx, cy, 22 * pulseScale, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(52, 211, 153, 0.36)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([]);
      ctx.stroke();
    } else if (isHovered) {
      ctx.beginPath();
      ctx.arc(cx, cy, 14.5, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }

    // Active Sprint outer rim (> 24 km/h)
    if (isSprint && !isSelected) {
      ctx.beginPath();
      ctx.arc(cx, cy, 13.5, 0, Math.PI * 2);
      ctx.strokeStyle = "#34D399";
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }

    // Soft studio drop shadow
    const baseRadius = isGK ? 10.5 : 9.5;
    const radius = isSelected || isHovered ? baseRadius + 1.2 : baseRadius;

    ctx.beginPath();
    ctx.arc(cx, cy + 1.5, radius + 1.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(2, 6, 12, 0.55)";
    ctx.fill();

    if (isGK) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 2.5, 0, Math.PI * 2);
      ctx.strokeStyle = teamColor;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = teamColor;
    ctx.fill();
    ctx.strokeStyle = "#060B10";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // Subtle specular top crescent inside token
    ctx.beginPath();
    ctx.arc(cx, cy, radius - 1.2, -Math.PI * 0.85, -Math.PI * 0.15);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 1.1;
    ctx.stroke();

    // Verified Model 4 Jersey Ring + Mint Lock Indicator
    const hasVerifiedJersey = Boolean(player.jv && player.jn != null);
    if (hasVerifiedJersey) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 1.4, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(52, 211, 153, 0.75)";
      ctx.lineWidth = 1.25;
      ctx.stroke();

      // Tiny verified Mint badge pip at top-right shoulder
      const bx = cx + radius * 0.76;
      const by = cy - radius * 0.76;
      ctx.beginPath();
      ctx.arc(bx, by, 2.7, 0, Math.PI * 2);
      ctx.fillStyle = "#34D399";
      ctx.fill();
      ctx.strokeStyle = "#060B10";
      ctx.lineWidth = 0.9;
      ctx.stroke();
    }

    // Razor-sharp Player ID or Verified Jersey Label
    const displayLabel = hasVerifiedJersey ? `${player.jn}` : `${player.id}`;
    ctx.fillStyle = isTeam0 ? "#04101A" : "#FFFFFF";
    ctx.font = "bold 8.8px var(--font-jetbrains), monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(displayLabel, cx, cy + 0.5);

    ctx.restore();
  }
}
