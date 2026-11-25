/**
 * Pitch coordinate transformation & geometry constants.
 * Standard FIFA football pitch: 105.0m length x 68.0m width.
 */

export const FIFA_PITCH = {
  length: 105.0, // X: 0 (Team A goal) -> 105 (Team B goal)
  width: 68.0, // Y: 0 (Top touchline) -> 68 (Bottom touchline)
  penaltyBoxLength: 16.5,
  penaltyBoxWidth: 40.32,
  goalBoxLength: 5.5,
  goalBoxWidth: 18.32,
  centerCircleRadius: 9.15,
  penaltySpotDist: 11.0,
  sprintThresholdKmh: 24.0,
} as const;

export interface PitchViewport {
  width: number;
  height: number;
  padX: number;
  padY: number;
  scaleX: number;
  scaleY: number;
}

/**
 * Computes centered uniform-scale viewport geometry preserving the exact 105:68 FIFA ratio
 * with safe top/bottom margins so defensive-line pills never clip.
 */
export function computePitchViewport(
  canvasWidth: number,
  canvasHeight: number,
  paddingPercent = 0.045
): PitchViewport {
  const minPadX = Math.max(16, canvasWidth * paddingPercent);
  const minPadY = Math.max(24, canvasHeight * 0.068);
  const usableW = Math.max(10, canvasWidth - 2 * minPadX);
  const usableH = Math.max(10, canvasHeight - 2 * minPadY);

  // Single uniform scale preserves exact 105:68 metric proportions with zero stretching
  const scale = Math.min(usableW / FIFA_PITCH.length, usableH / FIFA_PITCH.width);
  const pitchPixelW = FIFA_PITCH.length * scale;
  const pitchPixelH = FIFA_PITCH.width * scale;

  const padX = (canvasWidth - pitchPixelW) / 2;
  const padY = (canvasHeight - pitchPixelH) / 2;

  return {
    width: canvasWidth,
    height: canvasHeight,
    padX,
    padY,
    scaleX: scale,
    scaleY: scale,
  };
}

/**
 * Transforms real-world pitch meters [0..105, 0..68] to canvas pixel coordinates.
 */
export function pitchToCanvas(
  x_m: number,
  y_m: number,
  vp: PitchViewport
): [number, number] {
  const cx = vp.padX + Math.max(0, Math.min(FIFA_PITCH.length, x_m)) * vp.scaleX;
  const cy = vp.padY + Math.max(0, Math.min(FIFA_PITCH.width, y_m)) * vp.scaleY;
  return [cx, cy];
}

/**
 * Transforms canvas pixel coordinates to real-world pitch meters [0..105, 0..68].
 */
export function canvasToPitch(
  canvasX: number,
  canvasY: number,
  vp: PitchViewport
): [number, number] {
  const px = (canvasX - vp.padX) / vp.scaleX;
  const py = (canvasY - vp.padY) / vp.scaleY;
  return [
    Math.max(0, Math.min(FIFA_PITCH.length, px)),
    Math.max(0, Math.min(FIFA_PITCH.width, py)),
  ];
}
