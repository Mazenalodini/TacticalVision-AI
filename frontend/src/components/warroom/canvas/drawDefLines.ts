import { FIFA_PITCH, PitchViewport, pitchToCanvas } from "./pitchConfig";

/**
 * Helper to draw a rounded pill badge safely clamped inside the canvas bounds.
 */
function drawClampedPillBadge(
  ctx: CanvasRenderingContext2D,
  centerX: number,
  targetY: number,
  text: string,
  accentColor: string,
  vp: PitchViewport
) {
  ctx.save();
  ctx.font = "bold 10px var(--font-jetbrains), monospace";
  const measured = ctx.measureText(text).width;
  const badgeW = Math.max(52, Math.ceil(measured + 14));
  const badgeH = 18;

  const clampedCenterX = Math.max(
    badgeW / 2 + 6,
    Math.min(vp.width - badgeW / 2 - 6, centerX)
  );
  const clampedTopY = Math.max(3, Math.min(vp.height - badgeH - 3, targetY));
  const boxX = clampedCenterX - badgeW / 2;

  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(6, 11, 20, 0.92)";
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1.1;

  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(boxX, clampedTopY, badgeW, badgeH, 5);
  } else {
    ctx.rect(boxX, clampedTopY, badgeW, badgeH);
  }
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = accentColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, clampedCenterX, clampedTopY + badgeH / 2 + 0.5);
  ctx.restore();
}

/**
 * Draws virtual defensive lines (last outfield defender line X) for Team A and Team B.
 * Handles string keys ("0" / "1"), sparse frames, and guarantees zero badge clipping.
 */
export function drawDefLines(
  ctx: CanvasRenderingContext2D,
  def_x: Record<string, number> | undefined,
  vp: PitchViewport
) {
  if (!def_x) return;

  const W = FIFA_PITCH.width;

  // Team 0 Defensive Line (Cyan — badge anchored in safe top margin)
  const def0 = def_x["0"];
  if (def0 !== undefined && typeof def0 === "number" && !isNaN(def0)) {
    const [lineX] = pitchToCanvas(def0, 0, vp);
    const [, topY] = pitchToCanvas(0, 0, vp);
    const [, botY] = pitchToCanvas(0, W, vp);

    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
    ctx.lineWidth = 1.6;
    ctx.moveTo(lineX, topY);
    ctx.lineTo(lineX, botY);
    ctx.stroke();
    ctx.restore();

    drawClampedPillBadge(
      ctx,
      lineX,
      topY - 21,
      `A ${def0.toFixed(1)}m`,
      "#38BDF8",
      vp
    );
  }

  // Team 1 Defensive Line (Rose-Coral — badge anchored in safe bottom margin)
  const def1 = def_x["1"];
  if (def1 !== undefined && typeof def1 === "number" && !isNaN(def1)) {
    const [lineX] = pitchToCanvas(def1, 0, vp);
    const [, topY] = pitchToCanvas(0, 0, vp);
    const [, botY] = pitchToCanvas(0, W, vp);

    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "rgba(251, 113, 133, 0.85)";
    ctx.lineWidth = 1.6;
    ctx.moveTo(lineX, topY);
    ctx.lineTo(lineX, botY);
    ctx.stroke();
    ctx.restore();

    drawClampedPillBadge(
      ctx,
      lineX,
      botY + 3,
      `B ${def1.toFixed(1)}m`,
      "#FB7185",
      vp
    );
  }
}
