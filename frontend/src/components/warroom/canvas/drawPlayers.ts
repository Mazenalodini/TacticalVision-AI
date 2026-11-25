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
}

/**
 * Draws active players, goalkeepers, active sprints, selected player spotlight,
 * player movement trajectory path, and the ball with its fading comet trail.
 */
export function drawPlayers(
  ctx: CanvasRenderingContext2D,
  players: FramePlayer[] | undefined,
  ball: [number, number] | null | undefined,
  ballTrail: BallHistoryPoint[],
  selectedPlayerId: number | null,
  trajectory: PlayerTrajectoryPoint[] | undefined,
  currentFrameIndex: number,
  vp: PitchViewport
) {
  // ── 0. Draw Selected Player Movement Trajectory Path ──
  if (selectedPlayerId !== null && trajectory && trajectory.length > 1) {
    ctx.save();

    // 0A. Full-clip subtle faded trajectory
    ctx.beginPath();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
    ctx.setLineDash([3, 4]);

    for (let i = 1; i < trajectory.length; i++) {
      const p1 = trajectory[i - 1];
      const p2 = trajectory[i];

      // Break cleanly if frame gap > 12 frames (player left & re-entered camera)
      if (p2.frame - p1.frame > 12) {
        continue;
      }

      const [x1, y1] = pitchToCanvas(p1.x, p1.y, vp);
      const [x2, y2] = pitchToCanvas(p2.x, p2.y, vp);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();

    // 0B. Solid vibrant trajectory up to current frame
    const pastTrajectory = trajectory.filter((p) => p.frame <= currentFrameIndex);
    if (pastTrajectory.length > 1) {
      ctx.beginPath();
      ctx.setLineDash([]);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#3CFF82";

      for (let i = 1; i < pastTrajectory.length; i++) {
        const p1 = pastTrajectory[i - 1];
        const p2 = pastTrajectory[i];

        if (p2.frame - p1.frame > 12) {
          continue;
        }

        const [x1, y1] = pitchToCanvas(p1.x, p1.y, vp);
        const [x2, y2] = pitchToCanvas(p2.x, p2.y, vp);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
    }

    ctx.restore();
  }

  // ── 1. Draw Ball Comet Trail ──
  if (ballTrail.length > 1) {
    ctx.save();
    const trailLen = ballTrail.length;
    for (let i = 1; i < trailLen; i++) {
      const p1 = ballTrail[i - 1];
      const p2 = ballTrail[i];

      const frameGap = p2.frame - p1.frame;
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (frameGap <= 2 && dist <= 15.0) {
        const [cx1, cy1] = pitchToCanvas(p1.x, p1.y, vp);
        const [cx2, cy2] = pitchToCanvas(p2.x, p2.y, vp);

        const progress = i / trailLen;
        const alpha = Math.max(0.08, progress * 0.75);
        const lineWidth = Math.max(1.5, progress * 4.0);

        ctx.beginPath();
        ctx.strokeStyle = `rgba(52, 211, 153, ${alpha})`;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = "round";
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  // ── 2. Draw Ball ──
  if (ball && ball.length === 2 && !isNaN(ball[0]) && !isNaN(ball[1])) {
    const [bx, by] = pitchToCanvas(ball[0], ball[1], vp);

    ctx.save();
    ctx.beginPath();
    ctx.arc(bx, by, 7, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(52, 211, 153, 0.35)";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(bx, by, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#FFFFFF";
    ctx.fill();
    ctx.strokeStyle = "#05090C";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }

  // ── 3. Draw Players ──
  if (!players || players.length === 0) return;

  const now = performance.now();
  const pulseScale = 1.0 + 0.15 * Math.sin(now * 0.007);

  for (const player of players) {
    // Allow boundary goalkeepers/wingers slightly outside line [-3..108, -3..71] and clamp visually
    if (
      player.x < -3 ||
      player.x > FIFA_PITCH.length + 3 ||
      player.y < -3 ||
      player.y > FIFA_PITCH.width + 3
    ) {
      continue;
    }

    const [cx, cy] = pitchToCanvas(player.x, player.y, vp);
    const isTeam0 = player.team === 0;
    const teamColor = isTeam0 ? "#38BDF8" : "#FB7185";
    const isSelected = selectedPlayerId === player.id;
    const isGK = player.role === "gk";
    const isSprint = player.spd >= FIFA_PITCH.sprintThresholdKmh;

    ctx.save();

    // Dim non-selected players if a player is spotlighted
    if (selectedPlayerId !== null && !isSelected) {
      ctx.globalAlpha = 0.32;
    }

    // ── Spotlight Halo Ring (if selected) ──
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(cx, cy, 18 * pulseScale, 0, Math.PI * 2);
      ctx.strokeStyle = "#34D399";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 3]);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(cx, cy, 23 * pulseScale, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(52, 211, 153, 0.42)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.stroke();
    }

    // ── Active Sprint Outer Ring (> 24 km/h) ──
    if (isSprint && !isSelected) {
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.strokeStyle = "#34D399";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // ── Player Dot / Marker ──
    const radius = 10;

    if (isGK) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 2, 0, Math.PI * 2);
      ctx.strokeStyle = teamColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = teamColor;
      ctx.fill();
      ctx.strokeStyle = "#070E0B";
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = teamColor;
      ctx.fill();
      ctx.strokeStyle = "#070E0B";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // ── Player Track ID Text ──
    ctx.fillStyle = isTeam0 ? "#05090C" : "#FFFFFF";
    ctx.font = "bold 9px var(--font-jetbrains), monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${player.id}`, cx, cy);

    ctx.restore();
  }
}
