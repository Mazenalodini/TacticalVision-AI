import { FIFA_PITCH, PitchViewport, pitchToCanvas } from "./pitchConfig";

/**
 * Renders the stealth dark FIFA stadium turf (#0B1410),
 * alternating mowing stripes, white-muted boundary markings,
 * center circle, penalty areas, and tactical thirds dividers.
 */
export function drawPitch(ctx: CanvasRenderingContext2D, vp: PitchViewport) {
  const L = FIFA_PITCH.length;
  const W = FIFA_PITCH.width;

  const [originX, originY] = pitchToCanvas(0, 0, vp);
  const [pitchEndX, pitchEndY] = pitchToCanvas(L, W, vp);
  const pitchW = pitchEndX - originX;
  const pitchH = pitchEndY - originY;

  // 1. Dark Stadium Canvas Background (Surrounding turf)
  ctx.fillStyle = "#08120F";
  ctx.fillRect(0, 0, vp.width, vp.height);

  // 2. Main Pitch Turf Rectangle (Clean Slate-Emerald Broadcast Turf)
  ctx.fillStyle = "#0C1B17";
  ctx.fillRect(originX, originY, pitchW, pitchH);

  // 3. Subtle Alternating Mowing Stripes (10 stripes across length)
  const stripeWidth_m = L / 10;
  for (let i = 0; i < 10; i++) {
    if (i % 2 === 0) {
      const [sX1] = pitchToCanvas(i * stripeWidth_m, 0, vp);
      const [sX2] = pitchToCanvas((i + 1) * stripeWidth_m, 0, vp);
      ctx.fillStyle = "#10231E";
      ctx.fillRect(sX1, originY, sX2 - sX1, pitchH);
    }
  }

  // 4. Subtle Tactical Thirds Dividers (at 35m and 70m)
  ctx.save();
  ctx.setLineDash([4, 6]);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
  ctx.lineWidth = 1;
  const [third1X] = pitchToCanvas(35, 0, vp);
  const [third2X] = pitchToCanvas(70, 0, vp);
  ctx.beginPath();
  ctx.moveTo(third1X, originY);
  ctx.lineTo(third1X, originY + pitchH);
  ctx.moveTo(third2X, originY);
  ctx.lineTo(third2X, originY + pitchH);
  ctx.stroke();
  ctx.restore();

  // 5. Crisp White-Muted FIFA Pitch Markings
  ctx.strokeStyle = "rgba(255, 255, 255, 0.24)";
  ctx.lineWidth = Math.max(1, Math.min(2.2, vp.scaleX * 0.18));

  // Outer boundary
  ctx.strokeRect(originX, originY, pitchW, pitchH);

  // Halfway line
  const [halfX] = pitchToCanvas(L / 2, 0, vp);
  ctx.beginPath();
  ctx.moveTo(halfX, originY);
  ctx.lineTo(halfX, originY + pitchH);
  ctx.stroke();

  // Center circle & spot
  const [centerX, centerY] = pitchToCanvas(L / 2, W / 2, vp);
  const circleRadiusPx = FIFA_PITCH.centerCircleRadius * vp.scaleX;
  ctx.beginPath();
  ctx.arc(centerX, centerY, circleRadiusPx, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(226, 232, 240, 0.85)";
  ctx.beginPath();
  ctx.arc(centerX, centerY, 3, 0, Math.PI * 2);
  ctx.fill();

  // Left & Right Penalty Box (16.5m x 40.32m)
  const pBoxY = (W - FIFA_PITCH.penaltyBoxWidth) / 2;
  const [pBoxL_x1, pBoxL_y1] = pitchToCanvas(0, pBoxY, vp);
  const [pBoxL_x2, pBoxL_y2] = pitchToCanvas(FIFA_PITCH.penaltyBoxLength, pBoxY + FIFA_PITCH.penaltyBoxWidth, vp);
  ctx.strokeRect(pBoxL_x1, pBoxL_y1, pBoxL_x2 - pBoxL_x1, pBoxL_y2 - pBoxL_y1);

  const [pBoxR_x1, pBoxR_y1] = pitchToCanvas(L - FIFA_PITCH.penaltyBoxLength, pBoxY, vp);
  const [pBoxR_x2, pBoxR_y2] = pitchToCanvas(L, pBoxY + FIFA_PITCH.penaltyBoxWidth, vp);
  ctx.strokeRect(pBoxR_x1, pBoxR_y1, pBoxR_x2 - pBoxR_x1, pBoxR_y2 - pBoxR_y1);

  // Left & Right 6-Yard Goal Box (5.5m x 18.32m)
  const gBoxY = (W - FIFA_PITCH.goalBoxWidth) / 2;
  const [gBoxL_x1, gBoxL_y1] = pitchToCanvas(0, gBoxY, vp);
  const [gBoxL_x2, gBoxL_y2] = pitchToCanvas(FIFA_PITCH.goalBoxLength, gBoxY + FIFA_PITCH.goalBoxWidth, vp);
  ctx.strokeRect(gBoxL_x1, gBoxL_y1, gBoxL_x2 - gBoxL_x1, gBoxL_y2 - gBoxL_y1);

  const [gBoxR_x1, gBoxR_y1] = pitchToCanvas(L - FIFA_PITCH.goalBoxLength, gBoxY, vp);
  const [gBoxR_x2, gBoxR_y2] = pitchToCanvas(L, gBoxY + FIFA_PITCH.goalBoxWidth, vp);
  ctx.strokeRect(gBoxR_x1, gBoxR_y1, gBoxR_x2 - gBoxR_x1, gBoxR_y2 - gBoxR_y1);

  // Penalty Spots (11m from goal line, center of pitch Y)
  const [spotLX, spotLY] = pitchToCanvas(FIFA_PITCH.penaltySpotDist, W / 2, vp);
  const [spotRX, spotRY] = pitchToCanvas(L - FIFA_PITCH.penaltySpotDist, W / 2, vp);

  ctx.beginPath();
  ctx.arc(spotLX, spotLY, 2.5, 0, Math.PI * 2);
  ctx.arc(spotRX, spotRY, 2.5, 0, Math.PI * 2);
  ctx.fill();

  // Corner arcs (1m radius)
  const cornerR = 1.0 * vp.scaleX;
  ctx.beginPath();
  ctx.arc(originX, originY, cornerR, 0, Math.PI / 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(originX, originY + pitchH, cornerR, -Math.PI / 2, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(originX + pitchW, originY, cornerR, Math.PI / 2, Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(originX + pitchW, originY + pitchH, cornerR, Math.PI, Math.PI * 1.5);
  ctx.stroke();
}
