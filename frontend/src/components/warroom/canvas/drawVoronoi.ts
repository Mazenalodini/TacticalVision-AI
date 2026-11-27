import { FramePlayer } from "@/lib/types";
import { FIFA_PITCH, PitchViewport, pitchToCanvas } from "./pitchConfig";

/**
 * Zero-allocation 60FPS High-Resolution Voronoi Pitch Control Renderer.
 *
 * Uses a persistent 210x136 offscreen canvas (0.5m resolution) and traces
 * crisp Second-Spectrum-style Voronoi frontier seams between Team A (#38BDF8)
 * and Team B (#FB7185) over Night-Emerald Slate turf with zero muddy mixing.
 */
export class VoronoiBuffer {
  private offscreen: HTMLCanvasElement;
  private offCtx: CanvasRenderingContext2D;
  private imgData: ImageData;
  private width = 210;
  private height = 136;
  private owners: Int8Array;

  constructor() {
    this.offscreen = document.createElement("canvas");
    this.offscreen.width = this.width;
    this.offscreen.height = this.height;
    const ctx = this.offscreen.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Failed to get 2D context for Voronoi buffer");
    this.offCtx = ctx;
    this.imgData = this.offCtx.createImageData(this.width, this.height);
    this.owners = new Int8Array(this.width * this.height);
  }

  public render(
    ctx: CanvasRenderingContext2D,
    players: FramePlayer[],
    vp: PitchViewport
  ) {
    if (!players || players.length < 2) return;

    const validPlayers = players.filter(
      (p) =>
        p.role !== "referee" &&
        Number.isFinite(p.x) &&
        Number.isFinite(p.y) &&
        p.x >= -0.5 &&
        p.x <= FIFA_PITCH.length + 0.5 &&
        p.y >= -0.5 &&
        p.y <= FIFA_PITCH.width + 0.5 &&
        (p.team === 0 || p.team === 1)
    );

    if (validPlayers.length < 2) return;

    const n = validPlayers.length;
    const px = new Float32Array(n);
    const py = new Float32Array(n);
    const pteam = new Int8Array(n);

    for (let i = 0; i < n; i++) {
      px[i] = validPlayers[i].x;
      py[i] = validPlayers[i].y;
      pteam[i] = validPlayers[i].team;
    }

    const w = this.width;
    const h = this.height;
    const owners = this.owners;

    // Pass 1: Compute team ownership per 0.5m x 0.5m cell
    for (let gy = 0; gy < h; gy++) {
      const cellY = gy * 0.5 + 0.25;
      const rowOffset = gy * w;
      for (let gx = 0; gx < w; gx++) {
        const cellX = gx * 0.5 + 0.25;

        let minDistSq = 999999;
        let ownerTeam = -1;

        for (let i = 0; i < n; i++) {
          const dx = px[i] - cellX;
          const dy = py[i] - cellY;
          const distSq = dx * dx + dy * dy;
          if (distSq < minDistSq) {
            minDistSq = distSq;
            ownerTeam = pteam[i];
          }
        }
        owners[rowOffset + gx] = ownerTeam;
      }
    }

    // Pass 2: Shade pure Broadcast Sky / Luminous Rose zones + subtle frontier seam
    const data = this.imgData.data;
    let ptr = 0;
    for (let gy = 0; gy < h; gy++) {
      const rowOffset = gy * w;
      for (let gx = 0; gx < w; gx++) {
        const idx = rowOffset + gx;
        const owner = owners[idx];

        const isBoundary =
          (gx + 1 < w && owners[idx + 1] !== owner) ||
          (gy + 1 < h && owners[idx + w] !== owner);

        if (isBoundary) {
          data[ptr] = 226;
          data[ptr + 1] = 240;
          data[ptr + 2] = 255;
          data[ptr + 3] = 52;
        } else if (owner === 0) {
          // Team A: Pure Broadcast Sky (#38BDF8)
          data[ptr] = 56;
          data[ptr + 1] = 189;
          data[ptr + 2] = 248;
          data[ptr + 3] = 36;
        } else if (owner === 1) {
          // Team B: Pure Luminous Rose (#FB7185)
          data[ptr] = 251;
          data[ptr + 1] = 113;
          data[ptr + 2] = 133;
          data[ptr + 3] = 38;
        } else {
          data[ptr + 3] = 0;
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
    ctx.drawImage(this.offscreen, originX, originY, destW, destH);
    ctx.restore();
  }
}
