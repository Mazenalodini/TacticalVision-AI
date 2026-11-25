import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from .config import EngineConfig, FIFAPitchConfig


class BroadcastRenderer:
    """
    Produces two broadcast-grade videos:
      1. broadcast_analyzed.mp4: 3D-style ground ellipses, ID & speed badges,
         comet ball trail, inverse-projected defensive lines on turf, and top HUD.
      2. tactical_radar.mp4: Live 2D FIFA pitch with Voronoi Pitch Control overlay,
         Team Convex Hull polygons, player dots & ball trajectory.
    """

    def __init__(self, cfg: EngineConfig, pitch: FIFAPitchConfig):
        self.cfg = cfg
        self.pitch = pitch
        self.ball_trail = deque(maxlen=16)
        self.radar_ball_trail = deque(maxlen=20)
        # Radar canvas dimensions (px): 10px per meter + padding
        self.scale = 8
        self.pad = 40
        self.radar_w = int(pitch.length * self.scale + 2 * self.pad)
        self.radar_h = int(pitch.width * self.scale + 2 * self.pad)
        self.base_pitch_img = self._build_base_pitch()

    def _m_to_radar(self, x_m: float, y_m: float) -> Tuple[int, int]:
        rx = int(self.pad + np.clip(x_m, 0.0, self.pitch.length) * self.scale)
        ry = int(self.pad + np.clip(y_m, 0.0, self.pitch.width) * self.scale)
        return rx, ry

    def _build_base_pitch(self) -> np.ndarray:
        img = np.full((self.radar_h, self.radar_w, 3), (22, 34, 28), dtype=np.uint8)
        # Subtle alternating pitch grass stripes
        stripe_w = int((self.pitch.length / 10.0) * self.scale)
        for i in range(10):
            if i % 2 == 0:
                x1 = self.pad + i * stripe_w
                x2 = min(self.pad + int(self.pitch.length * self.scale), x1 + stripe_w)
                cv2.rectangle(img, (x1, self.pad), (x2, self.radar_h - self.pad), (26, 42, 34), -1)

        verts = self.pitch.vertices
        line_col = (190, 210, 200)
        for a, b in self.pitch.edges:
            p1 = self._m_to_radar(verts[a, 0], verts[a, 1])
            p2 = self._m_to_radar(verts[b, 0], verts[b, 1])
            cv2.line(img, p1, p2, line_col, 2, cv2.LINE_AA)

        # Center circle & spots
        center = self._m_to_radar(self.pitch.length / 2.0, self.pitch.width / 2.0)
        cv2.circle(img, center, int(self.pitch.centre_circle_radius * self.scale), line_col, 2, cv2.LINE_AA)
        cv2.circle(img, center, 4, line_col, -1, cv2.LINE_AA)
        cv2.circle(img, self._m_to_radar(self.pitch.penalty_spot_distance, self.pitch.width / 2.0), 4, line_col, -1)
        cv2.circle(img, self._m_to_radar(self.pitch.length - self.pitch.penalty_spot_distance, self.pitch.width / 2.0), 4, line_col, -1)
        return img

    def draw_broadcast_frame(
        self,
        frame_bgr: np.ndarray,
        rec: dict,
        fps: float,
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # 1. Inverse-projected Virtual Defensive Lines directly on turf (using H^-1)
        H_inv = rec.get("H_inv")
        def_lines = rec.get("def_lines", {})
        if H_inv is not None and def_lines:
            line_overlay = canvas.copy()
            for tm, x_m in def_lines.items():
                col = self.cfg.team_colors_bgr.get(tm, (255, 255, 255))
                pts_m = np.array([[x_m, 2.0], [x_m, self.pitch.width - 2.0]], dtype=np.float32).reshape(-1, 1, 2)
                pts_px = cv2.perspectiveTransform(pts_m, H_inv).reshape(-1, 2)
                p1 = (int(np.clip(pts_px[0, 0], -w, 2 * w)), int(np.clip(pts_px[0, 1], -h, 2 * h)))
                p2 = (int(np.clip(pts_px[1, 0], -w, 2 * w)), int(np.clip(pts_px[1, 1], -h, 2 * h)))
                cv2.line(line_overlay, p1, p2, col, 3, cv2.LINE_AA)
            cv2.addWeighted(line_overlay, 0.45, canvas, 0.55, 0, canvas)

        # 2. Broadcast-style Ground Ellipses + Track ID + Live Speed
        for ent in rec["entities"]:
            x1, y1, x2, y2 = ent["box"].astype(int)
            cx = (x1 + x2) // 2
            bw = max(16, x2 - x1)
            cls_id = int(ent["cls"])
            tid = int(ent["track_id"])
            tm = int(ent.get("team", -1))

            if cls_id == self.cfg.referee_id:
                col = self.cfg.referee_color_bgr
            else:
                col = self.cfg.team_colors_bgr.get(tm, (200, 200, 200))

            # 3D-perspective ellipse under player feet
            cv2.ellipse(
                canvas,
                center=(cx, y2),
                axes=(int(bw * 0.52), max(5, int(bw * 0.18))),
                angle=0.0,
                startAngle=-40,
                endAngle=220,
                color=col,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Compact dark badge with Track ID
            badge_text = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bx1, by1 = cx - tw // 2 - 4, y2 + 5
            bx2, by2 = cx + tw // 2 + 4, y2 + th + 10
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (18, 22, 28), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), col, 1)
            cv2.putText(canvas, badge_text, (bx1 + 4, by2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)

            spd = ent.get("speed_kmh", 0.0)
            if spd >= 14.0 and cls_id != self.cfg.referee_id:
                spd_col = (60, 255, 130) if spd < self.cfg.sprint_threshold_kmh else (60, 140, 255)
                cv2.putText(canvas, f"{spd:.0f} km/h", (cx - 24, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, spd_col, 1, cv2.LINE_AA)

        # 3. Ball Comet Trail & Inverted Triangle Marker
        ball_xy = rec.get("ball_xy")
        if ball_xy is not None:
            self.ball_trail.append((int(ball_xy[0]), int(ball_xy[1])))
            pts_trail = list(self.ball_trail)
            for i in range(1, len(pts_trail)):
                if np.linalg.norm(np.array(pts_trail[i]) - np.array(pts_trail[i - 1])) < 140:
                    thick = max(1, int(4 * (i / len(pts_trail))))
                    cv2.line(canvas, pts_trail[i - 1], pts_trail[i], self.cfg.ball_color_bgr, thick, cv2.LINE_AA)
            bx, by = int(ball_xy[0]), int(ball_xy[1])
            tri = np.array([[bx, by - 10], [bx - 8, by - 24], [bx + 8, by - 24]], np.int32)
            cv2.drawContours(canvas, [tri], 0, self.cfg.ball_color_bgr, -1)
            cv2.drawContours(canvas, [tri], 0, (15, 20, 25), 1)
        else:
            self.ball_trail.clear()

        # 4. Top Translucent Broadcast HUD Bar
        hud = rec.get("hud_stats", {})
        hud_w, hud_h = 680, 56
        hx1 = max(10, (w - hud_w) // 2)
        hy1 = 16
        overlay = canvas.copy()
        cv2.rectangle(overlay, (hx1, hy1), (hx1 + hud_w, hy1 + hud_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (hx1, hy1), (hx1 + hud_w, hy1 + hud_h), (55, 70, 90), 1)

        t_sec = rec["frame_idx"] / max(1.0, fps)
        pos0, pos1 = hud.get("pos_0", 50.0), hud.get("pos_1", 50.0)
        ctrl0, ctrl1 = hud.get("ctrl_0", 50.0), hud.get("ctrl_1", 50.0)
        p0, p1 = hud.get("passes_0", 0), hud.get("passes_1", 0)

        cv2.putText(canvas, "TACTICALVISION AI", (hx1 + 14, hy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{t_sec:05.1f}s", (hx1 + 14, hy1 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 175, 200), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"POSSESSION:  A {pos0:4.1f}%  |  {pos1:4.1f}% B", (hx1 + 185, hy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 245, 250), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"PITCH CTRL:  A {ctrl0:4.1f}%  |  {ctrl1:4.1f}% B   PASSES: {p0}-{p1}", (hx1 + 185, hy1 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (185, 210, 235), 1, cv2.LINE_AA)

        # Mini team color indicators
        cv2.circle(canvas, (hx1 + 170, hy1 + 18), 5, self.cfg.team_colors_bgr[0], -1)
        cv2.circle(canvas, (hx1 + 170, hy1 + 40), 5, self.cfg.team_colors_bgr[1], -1)

        return canvas

    def draw_radar_frame(self, rec: dict) -> np.ndarray:
        radar = self.base_pitch_img.copy()

        # 1. Voronoi Pitch Control semi-transparent layer
        vor = rec.get("voronoi_grid")
        if vor is not None and np.any(vor >= 0):
            vor_rgb = np.zeros((68, 105, 3), dtype=np.uint8)
            vor_rgb[vor == 0] = self.cfg.team_colors_bgr[0]
            vor_rgb[vor == 1] = self.cfg.team_colors_bgr[1]
            vor_resized = cv2.resize(
                vor_rgb,
                (int(self.pitch.length * self.scale), int(self.pitch.width * self.scale)),
                interpolation=cv2.INTER_NEAREST,
            )
            roi = radar[self.pad:self.radar_h - self.pad, self.pad:self.radar_w - self.pad]
            blended = cv2.addWeighted(vor_resized, 0.24, roi, 0.76, 0)
            radar[self.pad:self.radar_h - self.pad, self.pad:self.radar_w - self.pad] = blended

        # 2. Team Convex Hull Polygons (Compactness)
        for tm, hull_pts in rec.get("hulls", {}).items():
            col = self.cfg.team_colors_bgr.get(tm, (255, 255, 255))
            poly = np.array([self._m_to_radar(p[0], p[1]) for p in hull_pts], dtype=np.int32)
            cv2.polylines(radar, [poly], True, col, 2, cv2.LINE_AA)

        # 3. Defensive Lines on 2D Radar
        for tm, x_m in rec.get("def_lines", {}).items():
            col = self.cfg.team_colors_bgr.get(tm, (255, 255, 255))
            p1 = self._m_to_radar(x_m, 0.0)
            p2 = self._m_to_radar(x_m, self.pitch.width)
            cv2.line(radar, p1, p2, col, 1, cv2.LINE_AA)

        # 4. Players & Referees
        for ent in rec["entities"]:
            px = ent.get("pitch_xy")
            if px is None:
                continue
            rx, ry = self._m_to_radar(px[0], px[1])
            cls_id = int(ent["cls"])
            tm = int(ent.get("team", -1))
            if cls_id == self.cfg.referee_id:
                cv2.circle(radar, (rx, ry), 6, self.cfg.referee_color_bgr, -1, cv2.LINE_AA)
            else:
                col = self.cfg.team_colors_bgr.get(tm, (200, 200, 200))
                radius = 10 if cls_id == self.cfg.player_id else 11
                cv2.circle(radar, (rx, ry), radius, col, -1, cv2.LINE_AA)
                cv2.circle(radar, (rx, ry), radius, (15, 20, 25), 2, cv2.LINE_AA)
                cv2.putText(radar, str(ent["track_id"]), (rx - 7, ry + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (15, 20, 25), 1, cv2.LINE_AA)

        # 5. Ball Position & Trail
        bp = rec.get("ball_pitch_xy")
        if bp is not None:
            r_ball = self._m_to_radar(bp[0], bp[1])
            self.radar_ball_trail.append(r_ball)
            pts = list(self.radar_ball_trail)
            for i in range(1, len(pts)):
                cv2.line(radar, pts[i - 1], pts[i], self.cfg.ball_color_bgr, 2, cv2.LINE_AA)
            cv2.circle(radar, r_ball, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(radar, r_ball, 6, (10, 10, 10), 2, cv2.LINE_AA)

        # Footer label
        vm = rec.get("voronoi_metrics", {})
        cv2.putText(
            radar,
            f"2D TACTICAL RADAR + VORONOI CONTROL  |  TEAM A: {vm.get('team_0_pct', 50.0)}%   TEAM B: {vm.get('team_1_pct', 50.0)}%",
            (self.pad, self.radar_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 235, 245),
            1,
            cv2.LINE_AA,
        )
        return radar


def transcode_to_h264(raw_mp4: Path, final_mp4: Path):
    """Re-encodes OpenCV mp4v video into web-ready H.264 (yuv420p + faststart)."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        shutil.move(str(raw_mp4), str(final_mp4))
        return
    cmd = [
        ffmpeg_bin, "-y", "-i", str(raw_mp4),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(final_mp4),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0 and final_mp4.exists():
        raw_mp4.unlink(missing_ok=True)
    else:
        shutil.move(str(raw_mp4), str(final_mp4))
