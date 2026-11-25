import { Heatmaps } from "@/lib/types";
import { FIFA_PITCH, PitchViewport, pitchToCanvas } from "./pitchConfig";

export type HeatmapMode = "team_0" | "team_1" | "both";

/**
 * Renders smooth bilinearly-upscaled positional heatmaps from 12x8 zonal grid.
 * Offscreen 12x8 canvas smoothed onto the full pitch area.
 */
export class HeatmapRenderer {
  private offscreen: HTMLCanvasElement;
  private offCtx: CanvasRenderingContext2D;
  private imgData: ImageData;

  constructor() {
    this.offscreen = document.createElement("canvas");
    this.offscreen.width = 12;
    this.offscreen.height = 8;
    const ctx = this.offscreen.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Failed to get 2D context for Heatmap buffer");
    this.offCtx = ctx;
    this.imgData = this.offCtx.createImageData(12, 8);
  }

  public render(
    ctx: CanvasRenderingContext2D,
    heatmaps: Heatmaps | undefined,
    mode: HeatmapMode,
    vp: PitchViewport
  ) {
    if (!heatmaps) return;

    const grid0 = heatmaps.team_0; // 8 rows x 12 cols
    const grid1 = heatmaps.team_1;
    if (!grid0 || !grid1 || grid0.length !== 8) return;

    // Find max value for normalization
    let maxVal = 1;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 12; c++) {
        if (mode === "team_0" || mode === "both") maxVal = Math.max(maxVal, grid0[r][c] || 0);
        if (mode === "team_1" || mode === "both") maxVal = Math.max(maxVal, grid1[r][c] || 0);
      }
    }

    const data = this.imgData.data;
    let ptr = 0;

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 12; c++) {
        const val0 = (grid0[r]?.[c] || 0) / maxVal;
        const val1 = (grid1[r]?.[c] || 0) / maxVal;

        if (mode === "team_0") {
          // Team A Sky
          data[ptr] = 56;
          data[ptr + 1] = 189;
          data[ptr + 2] = 248;
          data[ptr + 3] = Math.min(180, Math.floor(val0 * 220));
        } else if (mode === "team_1") {
          // Team B Rose-Coral
          data[ptr] = 251;
          data[ptr + 1] = 113;
          data[ptr + 2] = 133;
          data[ptr + 3] = Math.min(180, Math.floor(val1 * 220));
        } else {
          // Both teams combined
          const total = val0 + val1;
          const blend0 = total > 0 ? val0 / total : 0.5;
          data[ptr] = Math.floor(56 * blend0 + 251 * (1 - blend0));
          data[ptr + 1] = Math.floor(189 * blend0 + 113 * (1 - blend0));
          data[ptr + 2] = Math.floor(248 * blend0 + 133 * (1 - blend0));
          data[ptr + 3] = Math.min(180, Math.floor(Math.max(val0, val1) * 220));
        }

        ptr += 4;
      }
    }

    this.offCtx.putImageData(this.imgData, 0, 0);

    const [originX, originY] = pitchToCanvas(0, 0, vp);
    const [pitchEndX, pitchEndY] = pitchToCanvas(FIFA_PITCH.length, FIFA_PITCH.width, vp);
    const destW = pitchEndX - originX;
    const destH = pitchEndY - originY;

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.globalCompositeOperation = "screen";
    ctx.drawImage(this.offscreen, originX, originY, destW, destH);
    ctx.restore();
  }
}
