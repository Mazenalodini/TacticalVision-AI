import { FramePlayer } from "@/lib/types";
import { FIFA_PITCH, PitchViewport, pitchToCanvas } from "./pitchConfig";

/**
 * Zero-allocation 60FPS Voronoi pitch control renderer.
 *
 * Uses ONE persistent offscreen canvas (105x68) and ONE persistent ImageData(105, 68).
 * Computes nearest active player per 1m^2 cell using squared Euclidean distance (dx*dx + dy*dy, no Math.sqrt).
 * Scales smoothly onto the main pitch canvas using imageSmoothingEnabled = true.
 */
export class VoronoiBuffer {
  private offscreen: HTMLCanvasElement;
  private offCtx: CanvasRenderingContext2D;
  private imgData: ImageData;
  private width = 105;
  private height = 68;

  constructor() {
    this.offscreen = document.createElement("canvas");
    this.offscreen.width = this.width;
    this.offscreen.height = this.height;
    const ctx = this.offscreen.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Failed to get 2D context for Voronoi buffer");
    this.offCtx = ctx;
    this.imgData = this.offCtx.createImageData(this.width, this.height);
  }

  public render(
    ctx: CanvasRenderingContext2D,
    players: FramePlayer[],
    vp: PitchViewport
  ) {
    if (!players || players.length < 2) return;

    // Filter players with valid pitch coordinates
    const validPlayers = players.filter(
      (p) =>
        p.x >= 0 &&
        p.x <= FIFA_PITCH.length &&
        p.y >= 0 &&
        p.y <= FIFA_PITCH.width &&
        (p.team === 0 || p.team === 1)
    );

    if (validPlayers.length < 2) return;

    // Pre-extract arrays for ultra-fast loop
    const n = validPlayers.length;
    const px = new Float32Array(n);
    const py = new Float32Array(n);
    const pteam = new Int8Array(n);

    for (let i = 0; i < n; i++) {
      px[i] = validPlayers[i].x;
      py[i] = validPlayers[i].y;
      pteam[i] = validPlayers[i].team;
    }

    const data = this.imgData.data;
    let ptr = 0;

    // 105 columns x 68 rows (1 cell = 1m x 1m)
    for (let y = 0; y < 68; y++) {
      const cellY = y + 0.5;
      for (let x = 0; x < 105; x++) {
        const cellX = x + 0.5;

        let minDistSq = 999999;
        let ownerTeam = -1;

        for (let i = 0; i < n; i++) {
          const dx = px[i] - cellX;
          const dy = py[i] - cellY;
          const distSq = dx * dx + dy * dy; // squared Euclidean — zero Math.sqrt!
          if (distSq < minDistSq) {
            minDistSq = distSq;
            ownerTeam = pteam[i];
          }
        }

        if (ownerTeam === 0) {
          // Team A: Broadcast Sky (#38BDF8)
          data[ptr] = 56;
          data[ptr + 1] = 189;
          data[ptr + 2] = 248;
          data[ptr + 3] = 42;
        } else if (ownerTeam === 1) {
          // Team B: Luminous Rose-Coral (#FB7185 — crisp rose, zero muddy brown on turf)
          data[ptr] = 251;
          data[ptr + 1] = 113;
          data[ptr + 2] = 133;
          data[ptr + 3] = 44;
        } else {
          data[ptr + 3] = 0;
        }

        ptr += 4;
      }
    }

    // Put image data into offscreen canvas
    this.offCtx.putImageData(this.imgData, 0, 0);

    // Blit smoothly onto target pitch canvas area
    const [originX, originY] = pitchToCanvas(0, 0, vp);
    const [pitchEndX, pitchEndY] = pitchToCanvas(FIFA_PITCH.length, FIFA_PITCH.width, vp);
    const destW = pitchEndX - originX;
    const destH = pitchEndY - originY;

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "medium";
    ctx.drawImage(this.offscreen, originX, originY, destW, destH);
    ctx.restore();
  }
}
