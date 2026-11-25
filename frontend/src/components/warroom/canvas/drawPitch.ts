import { FIFA_PITCH, PitchViewport, pitchToCanvas } from "./pitchConfig";

/**
 * Renders the Broadcast Night-Emerald Slate FIFA pitch (#0A141A / #0D1920),
 * subtle stadium center spotlight depth vignette, alternating mowing stripes,
 * crisp white-ice FIFA markings, 9.15m penalty D-arcs, goal frames, and thirds dividers.
 */
export function drawPitch(ctx: CanvasRenderingContext2D, vp: PitchViewport) {
  const L = FIFA_PITCH.length;
  const W = FIFA_PITCH.width;

  const [originX, originY] = pitchToCanvas(0, 0, vp);
  const [pitchEndX, pitchEndY] = pitchToCanvas(L, W, vp);
  const pitchW = pitchEndX - originX;
  const pitchH = pitchEndY - originY;

  // 1. Deep Stadium Apron Surround
  ctx.fillStyle = "#060B10";
  ctx.fillRect(0, 0, vp.width, vp.height);

  // 2. Main Pitch Turf Base (Broadcast Night-Emerald Slate — eliminates muddy red/green clashes)
  ctx.fillStyle = "#0A141A";
  ctx.fillRect(originX, originY, pitchW, pitchH);

  // 3. Alternating Mowing Stripes (10 crisp turf bands)
  const stripeWidth_m = L / 10;
  for (let i = 0; i < 10; i++) {
    if (i % 2 === 1) {
      const [sX1] = pitchToCanvas(i * stripeWidth_m, 0, vp);
      const [sX2] = pitchToCanvas((i + 1) * stripeWidth_m, 0, vp);
      ctx.fillStyle = "#0D1920";
      ctx.fillRect(sX1, originY, sX2 - sX1, pitchH);
    }
  }

  // 4. Subtle Overhead Stadium Spotlight Radial Depth Vignette
  const [centerX, centerY] = pitchToCanvas(L / 2, W / 2, vp);
  const turfGlow = ctx.createRadialGradient(
    centerX,
    centerY,
    pitchW * 0.08,
    centerX,
    centerY,
    pitchW * 0.65
  );
  turfGlow.addColorStop(0, "rgba(56, 189, 248, 0.035)");
  turfGlow.addColorStop(0.55, "rgba(16, 185, 129, 0.015)");
  turfGlow.addColorStop(1, "rgba(2, 6, 12, 0.24)");
  ctx.fillStyle = turfGlow;
  ctx.fillRect(originX, originY, pitchW, pitchH);

  // 5. Subtle Tactical Thirds Dividers (35m and 70m)
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

  // 6. Crisp Luminous White-Ice FIFA Markings
  ctx.save();
  ctx.strokeStyle = "rgba(226, 240, 255, 0.27)";
  ctx.lineWidth = Math.max(1.1, Math.min(2.1, vp.scaleX * 0.18));

  // Outer pitch boundary
  ctx.strokeRect(originX, originY, pitchW, pitchH);

  // Halfway line
  const [halfX] = pitchToCanvas(L / 2, 0, vp);
  ctx.beginPath();
  ctx.moveTo(halfX, originY);
  ctx.lineTo(halfX, originY + pitchH);
  ctx.stroke();

  // Center circle & center spot
  const circleRadiusPx = FIFA_PITCH.centerCircleRadius * vp.scaleX;
  ctx.beginPath();
  ctx.arc(centerX, centerY, circleRadiusPx, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(226, 240, 255, 0.85)";
  ctx.beginPath();
  ctx.arc(centerX, centerY, 2.8, 0, Math.PI * 2);
  ctx.fill();

  // Left & Right 18-Yard Penalty Boxes (16.5m x 40.32m)
  const pBoxY = (W - FIFA_PITCH.penaltyBoxWidth) / 2;
  const [pBoxL_x1, pBoxL_y1] = pitchToCanvas(0, pBoxY, vp);
  const [pBoxL_x2, pBoxL_y2] = pitchToCanvas(
    FIFA_PITCH.penaltyBoxLength,
    pBoxY + FIFA_PITCH.penaltyBoxWidth,
    vp
  );
  ctx.strokeRect(pBoxL_x1, pBoxL_y1, pBoxL_x2 - pBoxL_x1, pBoxL_y2 - pBoxL_y1);

  const [pBoxR_x1, pBoxR_y1] = pitchToCanvas(L - FIFA_PITCH.penaltyBoxLength, pBoxY, vp);
  const [pBoxR_x2, pBoxR_y2] = pitchToCanvas(L, pBoxY + FIFA_PITCH.penaltyBoxWidth, vp);
  ctx.strokeRect(pBoxR_x1, pBoxR_y1, pBoxR_x2 - pBoxR_x1, pBoxR_y2 - pBoxR_y1);

  // Left & Right 6-Yard Goal Boxes (5.5m x 18.32m)
  const gBoxY = (W - FIFA_PITCH.goalBoxWidth) / 2;
  const [gBoxL_x1, gBoxL_y1] = pitchToCanvas(0, gBoxY, vp);
  const [gBoxL_x2, gBoxL_y2] = pitchToCanvas(
    FIFA_PITCH.goalBoxLength,
    gBoxY + FIFA_PITCH.goalBoxWidth,
    vp
  );
  ctx.strokeRect(gBoxL_x1, gBoxL_y1, gBoxL_x2 - gBoxL_x1, gBoxL_y2 - gBoxL_y1);

  const [gBoxR_x1, gBoxR_y1] = pitchToCanvas(L - FIFA_PITCH.goalBoxLength, gBoxY, vp);
  const [gBoxR_x2, gBoxR_y2] = pitchToCanvas(L, gBoxY + FIFA_PITCH.goalBoxWidth, vp);
  ctx.strokeRect(gBoxR_x1, gBoxR_y1, gBoxR_x2 - gBoxR_x1, gBoxR_y2 - gBoxR_y1);

  // Penalty Spots & 9.15m Penalty D-Arcs
  const [spotLX, spotLY] = pitchToCanvas(FIFA_PITCH.penaltySpotDist, W / 2, vp);
  const [spotRX, spotRY] = pitchToCanvas(L - FIFA_PITCH.penaltySpotDist, W / 2, vp);

  ctx.beginPath();
  ctx.arc(spotLX, spotLY, 2.3, 0, Math.PI * 2);
  ctx.arc(spotRX, spotRY, 2.3, 0, Math.PI * 2);
  ctx.fill();

  const dAngle = Math.acos((16.5 - 11.0) / 9.15); // ~0.926 rad
  ctx.beginPath();
  ctx.arc(spotLX, spotLY, circleRadiusPx, -dAngle, dAngle);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(spotRX, spotRY, circleRadiusPx, Math.PI - dAngle, Math.PI + dAngle);
  ctx.stroke();

  // Goal Net Frames (7.32m wide, 2.4m deep extending slightly into touch margin)
  const goalHalfW_m = 3.66;
  const [, goalTopY] = pitchToCanvas(0, W / 2 - goalHalfW_m, vp);
  const [, goalBotY] = pitchToCanvas(0, W / 2 + goalHalfW_m, vp);
  const goalDepthPx = Math.max(4, Math.min(8, 2.0 * vp.scaleX));

  ctx.fillStyle = "rgba(226, 240, 255, 0.05)";
  ctx.fillRect(originX - goalDepthPx, goalTopY, goalDepthPx, goalBotY - goalTopY);
  ctx.strokeRect(originX - goalDepthPx, goalTopY, goalDepthPx, goalBotY - goalTopY);

  ctx.fillRect(pitchEndX, goalTopY, goalDepthPx, goalBotY - goalTopY);
  ctx.strokeRect(pitchEndX, goalTopY, goalDepthPx, goalBotY - goalTopY);

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

  ctx.restore();
}
