from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import ConvexHull

from .config import EngineConfig, FIFAPitchConfig, XT_GRID


def lookup_xt(x_m: float, y_m: float, team: int, pitch: FIFAPitchConfig) -> float:
    """
    Computes Expected Threat (xT) from Karun Singh's 12x8 pitch value matrix.
    Team 0 attacks Left -> Right (X: 0 -> 105); Team 1 attacks Right -> Left (X: 105 -> 0).
    Strictly guards against NaN/Inf and out-of-bounds coordinates (goals, corners, throw-ins).
    """
    try:
        x_val = float(x_m)
        y_val = float(y_m)
    except (TypeError, ValueError):
        return 0.0
    if not (np.isfinite(x_val) and np.isfinite(y_val)):
        return 0.0

    pitch_len = max(1e-6, float(pitch.length))
    pitch_wid = max(1e-6, float(pitch.width))
    x_norm = float(np.clip(x_val / pitch_len, 0.0, 1.0))
    y_norm = float(np.clip(y_val / pitch_wid, 0.0, 1.0))
    if int(team) == 1:
        x_norm = 1.0 - x_norm

    n_rows, n_cols = XT_GRID.shape
    col = max(0, min(n_cols - 1, int(x_norm * n_cols)))
    row = max(0, min(n_rows - 1, int(y_norm * n_rows)))
    return float(XT_GRID[row, col])


def bridge_ball_pitch_trajectory(
    raw_ball_pitch: List[Optional[np.ndarray]],
    fps: float,
    max_gap_frames: int = 8,
    max_speed_mps: float = 38.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> List[Optional[np.ndarray]]:
    """
    Linearly bridges short ball-detection dropouts (1..max_gap_frames missing frames) in metric
    pitch space when the implied flight speed is physically plausible, so fast passes in flight
    do not reset the possession/pass state machine.
    """
    out: List[Optional[np.ndarray]] = [
        None if b is None else np.asarray(b, dtype=np.float32).copy() for b in raw_ball_pitch
    ]
    fps = max(1.0, float(fps))
    last_idx: Optional[int] = None
    for i, b in enumerate(out):
        if b is None:
            continue
        if last_idx is not None:
            missing = i - last_idx - 1
            if 1 <= missing <= max_gap_frames:
                a = out[last_idx]
                dist = float(np.linalg.norm(b - a))
                dt = (i - last_idx) / fps
                if dt > 0 and dist / dt <= max_speed_mps:
                    for g in range(1, missing + 1):
                        r = g / float(i - last_idx)
                        p = a + (b - a) * r
                        p[0] = float(np.clip(p[0], 0.0, pitch_length))
                        p[1] = float(np.clip(p[1], 0.0, pitch_width))
                        out[last_idx + g] = p.astype(np.float32)
        last_idx = i
    return out


def filter_and_interpolate_ball_pitch(
    raw_ball_pitch: List[Optional[np.ndarray]],
    fps: float,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    invalid_mask: Optional[List[bool]] = None,
    max_speed_mps: float = 38.0,
    max_interp_speed_mps: float = 35.0,
    max_interp_sec: float = 0.40,
    max_interp_frames: int = 8,
) -> List[Optional[np.ndarray]]:
    """
    Offline Physics-Gated Ball Trajectory Filter (General-Purpose):
      1. Strict Out-of-Bounds Rejection (ZERO Corner Clamping):
         Discards any point with x < -1.5, x > L+1.5, y < -1.5, y > W+1.5, or non-finite;
         also discards legacy border/corner-clamped outliers jumping > 8m to pitch edges.
      2. 1-2 Frame Teleport Rejection:
         Rejects false positives jumping > max(2.2m, 38 m/s * dt) or > 10m in 1 frame
         unless followed by a physically coherent multi-frame trajectory.
      3. Short-Gap Physics Interpolation (1..8 frames, <= 0.40s, <= 35 m/s):
         Smoothly bridges brief occlusions/blur without bridging across scene cuts/paused spans.
      4. Light Zero-Phase Smoothing (5-tap kernel on continuous segments >= 5 frames).
    """
    n = len(raw_ball_pitch)
    if n == 0:
        return []
    fps = max(1.0, float(fps))
    L = float(pitch_length)
    W = float(pitch_width)
    inv_m = invalid_mask if (invalid_mask is not None and len(invalid_mask) == n) else [False] * n

    # Step 1: Strict OOB rejection (never clamp wild OOB points onto (0,0) / (105,68)!)
    obs: List[Tuple[int, np.ndarray]] = []
    for i, b in enumerate(raw_ball_pitch):
        if inv_m[i] or b is None:
            continue
        pt = np.asarray(b, dtype=np.float32).reshape(-1)
        if len(pt) < 2 or not np.all(np.isfinite(pt[:2])):
            continue
        bx, by = float(pt[0]), float(pt[1])
        if bx < -1.5 or bx > L + 1.5 or by < -1.5 or by > W + 1.5:
            continue
        # Soft-clip genuine near-touchline balls slightly inside pitch lines
        bx = float(np.clip(bx, 0.1, L - 0.1))
        by = float(np.clip(by, 0.1, W - 0.1))
        obs.append((i, np.array([bx, by], dtype=np.float32)))

    if not obs:
        return [None] * n

    # Drop exact border/corner-clamped artifacts from legacy inputs jumping > 8m to a pitch edge
    def _is_border_snap(pt: np.ndarray) -> bool:
        x, y = float(pt[0]), float(pt[1])
        return x <= 0.25 or x >= L - 0.25 or y <= 0.25 or y >= W - 0.25

    def _is_corner_snap(pt: np.ndarray) -> bool:
        x, y = float(pt[0]), float(pt[1])
        return (x <= 0.25 or x >= L - 0.25) and (y <= 0.25 or y >= W - 0.25)

    def _step_ok(i_a: int, p_a: np.ndarray, i_b: int, p_b: np.ndarray, spd_lim: float = max_speed_mps) -> bool:
        df = abs(i_b - i_a)
        if df == 0:
            return True
        dt = df / fps
        d = float(np.linalg.norm(p_b - p_a))
        if df == 1 and d > min(10.0, max(2.2, spd_lim * dt)):
            return False
        return d <= max(2.2, spd_lim * dt)

    # Step 2: 1-2 frame teleport & isolated false-positive rejection
    # First check if initial 1-2 points are isolated outliers before the real track starts
    start_k = 0
    while start_k + 2 < len(obs):
        i0, p0 = obs[start_k]
        i1, p1 = obs[start_k + 1]
        i2, p2 = obs[start_k + 2]
        if _is_corner_snap(p0) and float(np.linalg.norm(p1 - p0)) > 8.0:
            start_k += 1
            continue
        if (not _step_ok(i0, p0, i1, p1)) and _step_ok(i1, p1, i2, p2):
            start_k += 1
            continue
        break

    clean_obs: List[Tuple[int, np.ndarray]] = []
    if start_k < len(obs):
        if not (_is_corner_snap(obs[start_k][1]) and len(obs) > start_k + 1 and float(np.linalg.norm(obs[start_k + 1][1] - obs[start_k][1])) > 8.0):
            clean_obs.append(obs[start_k])

    for k in range(start_k + 1, len(obs)):
        i_cur, p_cur = obs[k]
        if not clean_obs:
            clean_obs.append((i_cur, p_cur))
            continue

        i_prev, p_prev = clean_obs[-1]
        d_prev = float(np.linalg.norm(p_cur - p_prev))

        # Reject corner/border snap false positives that jump > 8m onto a boundary
        if _is_border_snap(p_cur) and d_prev > 8.0:
            continue

        # Check if there was a scene cut / invalid gap between i_prev and i_cur
        has_cut = any(inv_m[t] for t in range(i_prev + 1, i_cur))
        if has_cut:
            clean_obs.append((i_cur, p_cur))
            continue

        if _step_ok(i_prev, p_prev, i_cur, p_cur):
            clean_obs.append((i_cur, p_cur))
            continue

        # Teleport candidate! Accept ONLY if corroborated by >= 1 coherent next frame
        # that does NOT snap right back to p_prev (i.e. genuine track re-acquisition after long gap)
        coherent_next = False
        if k + 1 < len(obs):
            i_nxt, p_nxt = obs[k + 1]
            # Must be close to cur AND further from prev than from cur (not a 1-frame spike!)
            if _step_ok(i_cur, p_cur, i_nxt, p_nxt) and (i_nxt - i_cur) <= max(3, int(round(0.25 * fps))):
                # If gap from prev was tiny (<= 3 frames), require 2 consecutive coherent frames so 2-frame spikes are killed
                if (i_cur - i_prev) <= 3:
                    if k + 2 < len(obs):
                        i_n2, p_n2 = obs[k + 2]
                        if _step_ok(i_nxt, p_nxt, i_n2, p_n2) and float(np.linalg.norm(p_n2 - p_prev)) > 4.0:
                            coherent_next = True
                else:
                    coherent_next = True

        if coherent_next:
            clean_obs.append((i_cur, p_cur))

    # Step 3: Short-gap physics interpolation (1..max_gap frames, <= 0.40s, <= 35 m/s)
    max_gap = min(int(max_interp_frames), max(3, int(round(max_interp_sec * fps))))
    out: List[Optional[np.ndarray]] = [None] * n
    for idx_c, (fi, pt) in enumerate(clean_obs):
        out[fi] = pt.copy()
        if idx_c > 0:
            prev_fi, prev_pt = clean_obs[idx_c - 1]
            missing = fi - prev_fi - 1
            if 1 <= missing <= max_gap:
                if not any(inv_m[t] for t in range(prev_fi + 1, fi)):
                    dt = (fi - prev_fi) / fps
                    dist = float(np.linalg.norm(pt - prev_pt))
                    if dt > 0.0 and (dist / dt) <= max_interp_speed_mps:
                        for g in range(1, missing + 1):
                            r = g / float(fi - prev_fi)
                            p_mid = prev_pt + (pt - prev_pt) * r
                            p_mid[0] = float(np.clip(p_mid[0], 0.1, L - 0.1))
                            p_mid[1] = float(np.clip(p_mid[1], 0.1, W - 0.1))
                            out[prev_fi + g] = p_mid.astype(np.float32)

    # Step 4: Light zero-phase 5-tap smoothing on continuous ball segments (>= 5 frames)
    kern = np.array([0.12, 0.22, 0.32, 0.22, 0.12], dtype=np.float32)
    seg_start: Optional[int] = None
    segments: List[Tuple[int, int]] = []
    for i in range(n):
        if out[i] is not None:
            if seg_start is None:
                seg_start = i
            elif i > seg_start and float(np.linalg.norm(out[i] - out[i - 1])) > 5.5:
                segments.append((seg_start, i - 1))
                seg_start = i
        else:
            if seg_start is not None:
                segments.append((seg_start, i - 1))
                seg_start = None
    if seg_start is not None:
        segments.append((seg_start, n - 1))

    for s0, s1 in segments:
        seg_len = s1 - s0 + 1
        if seg_len < 5:
            continue
        arr = np.stack([out[t] for t in range(s0, s1 + 1)], axis=0)
        pad_x = np.pad(arr[:, 0], (2, 2), mode="edge")
        pad_y = np.pad(arr[:, 1], (2, 2), mode="edge")
        sx = np.convolve(pad_x, kern, mode="valid")
        sy = np.convolve(pad_y, kern, mode="valid")
        for off in range(seg_len):
            out[s0 + off] = np.array(
                [float(np.clip(sx[off], 0.1, L - 0.1)), float(np.clip(sy[off], 0.1, W - 0.1))],
                dtype=np.float32,
            )

    return out


def is_progressive_pass(
    start_xy: Any,
    end_xy: Any,
    team: int,
    pitch_length: float = 105.0,
) -> bool:
    """
    Progressive pass: advances the ball toward the opponent goal by >= 10m ending in the
    opponent half, or by >= 15m anywhere. Team 0 attacks L->R, Team 1 attacks R->L.
    """
    half = pitch_length / 2.0
    sx, ex = float(start_xy[0]), float(end_xy[0])
    if int(team) == 0:
        advance = ex - sx
        in_opp_half = ex >= half
    else:
        advance = sx - ex
        in_opp_half = ex <= half
    return bool((advance >= 10.0 and in_opp_half) or advance >= 15.0)


def extract_possession_and_passes(
    frames_players: List[List[Tuple[int, int, float, float]]],
    ball_xy: List[Optional[np.ndarray]],
    fps: float,
    cfg: EngineConfig,
    pitch: FIFAPitchConfig,
    frames_image_fallback: Optional[List[Tuple[Optional[np.ndarray], List[Tuple[int, int, np.ndarray]]]]] = None,
) -> Dict[str, Any]:
    """
    Possession + pass + turnover state machine with asymmetric hysteresis.

    frames_players: per-frame list of (track_id, team, x_m, y_m) for players/goalkeepers.
    ball_xy: per-frame ball pitch position (already bridged) or None.
    frames_image_fallback: optional per-frame `(ball_px, [(tid, team, box_xyxy), ...])`
      used strictly for possession continuity on frames where the ball is visibly detected
      in image space while pitch calibration is briefly unavailable/recovering.

    - Same-team reception confirms after `min_owner_frames`.
    - Opponent takeover requires `min_turnover_frames` (or 3 frames within 1.6m for a clean
      tackle), so 2-4 frame defender brushes during a pass never break a same-team pass chain.
    - The passer's release point is updated on every frame they keep the ball.
    """
    fps = max(1.0, float(fps))
    radius = float(cfg.possession_radius_m)
    min_same = max(1, int(cfg.min_owner_frames))
    min_turn = max(min_same, int(getattr(cfg, "min_turnover_frames", 5)))
    min_pass_d = float(cfg.min_pass_distance_m)

    possession_frames = {0: 0, 1: 0}
    owner_team_per_frame: List[int] = []
    passes: List[dict] = []
    interceptions: List[dict] = []
    passing_edges: Dict[Tuple[int, int, int], int] = defaultdict(int)
    team_xt = {0: 0.0, 1: 0.0}
    player_pass_stats: Dict[int, Dict[str, float]] = defaultdict(lambda: {"passes": 0, "xt": 0.0})

    confirmed: Optional[Tuple[int, int, np.ndarray, int]] = None  # (tid, team, release_xy, frame)
    cand_tid: Optional[int] = None
    cand_streak = 0
    cand_min_d = 1e9
    cand_start_f = 0

    for f_idx, players in enumerate(frames_players):
        bp = ball_xy[f_idx] if f_idx < len(ball_xy) else None
        if bp is None or not players:
            # Image-space possession fallback: if ball & players are clearly visible in pixels
            # during a brief calibration gap, credit possession to the nearest player's feet
            # (within 0.90 * player box height) without fabricating metric passes.
            fb_owner = -1
            if frames_image_fallback is not None and f_idx < len(frames_image_fallback):
                b_px, px_players = frames_image_fallback[f_idx]
                if b_px is not None and px_players:
                    bx, by = float(b_px[0]), float(b_px[1])
                    best_norm_d = 1e9
                    for _tid, tm, box in px_players:
                        if tm not in (0, 1):
                            continue
                        fx = 0.5 * (float(box[0]) + float(box[2]))
                        fy = float(box[3])
                        bh = max(20.0, float(box[3]) - float(box[1]))
                        nd = float(np.hypot(fx - bx, fy - by)) / bh
                        if nd < best_norm_d:
                            best_norm_d = nd
                            fb_owner = int(tm)
                    if best_norm_d <= 0.95 and fb_owner in (0, 1):
                        possession_frames[fb_owner] += 1
                    else:
                        fb_owner = -1
            owner_team_per_frame.append(fb_owner)
            continue
        bp = np.asarray(bp, dtype=np.float32)

        best_d, best_tid, best_tm = 1e9, None, -1
        for tid, tm, x, y in players:
            d = float(np.hypot(x - bp[0], y - bp[1]))
            if d < best_d:
                best_d, best_tid, best_tm = d, int(tid), int(tm)

        if best_tid is None or best_d > radius:
            owner_team_per_frame.append(-1)
            continue

        possession_frames[best_tm] += 1
        owner_team_per_frame.append(best_tm)

        if best_tid == cand_tid:
            cand_streak += 1
            cand_min_d = min(cand_min_d, best_d)
        else:
            cand_tid = best_tid
            cand_streak = 1
            cand_min_d = best_d
            cand_start_f = f_idx

        # Current owner retains possession: only advance the pass release anchor while the ball is
        # still under close control (<= 1.8m). Do not drag start_xy out along mid-flight toward a receiver.
        if confirmed is not None and confirmed[0] == best_tid:
            prev_tid, prev_tm, anchor_xy, _ = confirmed
            new_anchor = bp.copy() if best_d <= 1.8 else anchor_xy
            confirmed = (prev_tid, prev_tm, new_anchor, f_idx)
            continue

        if confirmed is None or confirmed[1] == best_tm:
            required = min_same
        else:
            required = 3 if cand_min_d <= 1.6 else min_turn

        if cand_streak < required:
            continue

        if confirmed is not None:
            prev_tid, prev_tm, prev_xy, _prev_f = confirmed
            if prev_tm == best_tm:
                pass_dist = float(np.linalg.norm(bp - prev_xy))
                if pass_dist >= min_pass_d:
                    xt_delta = lookup_xt(bp[0], bp[1], best_tm, pitch) - lookup_xt(prev_xy[0], prev_xy[1], best_tm, pitch)
                    start_xy = [round(float(prev_xy[0]), 2), round(float(prev_xy[1]), 2)]
                    end_xy = [round(float(bp[0]), 2), round(float(bp[1]), 2)]
                    recv_f = int(cand_start_f)
                    passes.append({
                        "type": "pass",
                        "frame": recv_f,
                        "time_s": round(recv_f / fps, 2),
                        "team": best_tm,
                        "from_id": prev_tid,
                        "to_id": best_tid,
                        "start_xy": start_xy,
                        "end_xy": end_xy,
                        "distance_m": round(pass_dist, 1),
                        "xt_delta": round(float(xt_delta), 4),
                        "is_progressive": is_progressive_pass(start_xy, end_xy, best_tm, pitch.length),
                    })
                    passing_edges[(best_tm, prev_tid, best_tid)] += 1
                    team_xt[best_tm] += max(0.0, float(xt_delta))
                    player_pass_stats[prev_tid]["passes"] += 1
                    player_pass_stats[prev_tid]["xt"] += max(0.0, float(xt_delta))
            else:
                tf = int(cand_start_f)
                interceptions.append({
                    "type": "turnover",
                    "frame": tf,
                    "time_s": round(tf / fps, 2),
                    "from_team": prev_tm,
                    "to_team": best_tm,
                    "by_id": best_tid,
                    "xy": [round(float(bp[0]), 2), round(float(bp[1]), 2)],
                })
        confirmed = (best_tid, best_tm, bp.copy(), f_idx)

    return {
        "possession_frames": possession_frames,
        "owner_team_per_frame": owner_team_per_frame,
        "passes": passes,
        "interceptions": interceptions,
        "passing_edges": dict(passing_edges),
        "team_xt": team_xt,
        "player_pass_stats": {k: dict(v) for k, v in player_pass_stats.items()},
    }


class TacticalAnalyticsEngine:
    """
    Extracts complete physical & tactical intelligence from stabilized metric coordinates:
      - Smoothed Speed (km/h), Max Speed, Distance (m), and High-Intensity Sprints (>24 km/h)
      - Ball Possession (%)
      - Automated Pass Detection, Passing Network Graph & Interceptions
      - Team Compactness & Convex Hull Area (m^2) + Virtual Last-Defender Line X
      - Zonal Voronoi Pitch Control (Overall, Defensive/Middle/Attacking Thirds)
      - Expected Threat (xT) generated by passes
    """

    def __init__(self, cfg: EngineConfig, pitch: FIFAPitchConfig, fps: float):
        self.cfg = cfg
        self.pitch = pitch
        self.fps = max(1.0, float(fps))
        # Precompute 1m-resolution pitch grid (105 x 68) for ultra-fast vectorized Voronoi control
        gx = np.arange(0.5, pitch.length, 1.0, dtype=np.float32)
        gy = np.arange(0.5, pitch.width, 1.0, dtype=np.float32)
        self.grid_xx, self.grid_yy = np.meshgrid(gx, gy)  # (68, 105)
        self.grid_pts = np.stack([self.grid_xx.ravel(), self.grid_yy.ravel()], axis=-1)  # (7140, 2)

    def _smooth_series(self, arr: np.ndarray, window: int = 5) -> np.ndarray:
        if len(arr) < window:
            return arr
        kernel = np.ones(window, dtype=np.float32) / window
        pad = window // 2
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(arr)
        out[:, 0] = np.convolve(padded[:, 0], kernel, mode="valid")
        out[:, 1] = np.convolve(padded[:, 1], kernel, mode="valid")
        return out

    def compute_voronoi_control(
        self,
        players_xy: np.ndarray,
        players_team: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Returns:
          owner_grid: (68, 105) int8 array of 0/1 indicating nearest team for each 1m^2 cell
          metrics: dict with team_0_pct, team_1_pct, and zonal thirds control for Team 0
        """
        if len(players_xy) > 0:
            finite_mask = np.all(np.isfinite(players_xy), axis=-1)
            players_xy = players_xy[finite_mask]
            players_team = players_team[finite_mask]

        if len(players_xy) < 2 or len(np.unique(players_team)) < 2:
            empty = np.full(self.grid_xx.shape, -1, dtype=np.int8)
            return empty, {
                "team_0_pct": 50.0,
                "team_1_pct": 50.0,
                "team_0_def_third_pct": 50.0,
                "team_0_mid_third_pct": 50.0,
                "team_0_att_third_pct": 50.0,
            }

        # Vectorized squared Euclidean distance from all 7140 pitch cells to active players
        diff = self.grid_pts[:, None, :] - players_xy[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        nearest_idx = np.argmin(dist2, axis=1)
        owner = players_team[nearest_idx].reshape(self.grid_xx.shape).astype(np.int8)

        t0_all = float(np.mean(owner == 0) * 100.0)
        t0_def = float(np.mean(owner[:, :35] == 0) * 100.0)
        t0_mid = float(np.mean(owner[:, 35:70] == 0) * 100.0)
        t0_att = float(np.mean(owner[:, 70:] == 0) * 100.0)

        return owner, {
            "team_0_pct": round(t0_all, 1),
            "team_1_pct": round(100.0 - t0_all, 1),
            "team_0_def_third_pct": round(t0_def, 1),
            "team_0_mid_third_pct": round(t0_mid, 1),
            "team_0_att_third_pct": round(t0_att, 1),
        }

    def analyze(
        self,
        frame_records: List[dict],
        jersey_assignments: Optional[Dict[int, dict]] = None,
    ) -> Dict[str, Any]:
        # 1. Collect per-player pitch trajectories for smooth kinematics
        track_frames: Dict[int, List[int]] = defaultdict(list)
        track_coords: Dict[int, List[np.ndarray]] = defaultdict(list)
        track_team: Dict[int, int] = {}
        track_role: Dict[int, str] = {}
        jersey_map = jersey_assignments or {}

        role_names = {
            self.cfg.player_id: "player",
            self.cfg.goalkeeper_id: "goalkeeper",
            self.cfg.referee_id: "referee",
        }
        cls_votes: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for f_idx, rec in enumerate(frame_records):
            for ent in rec["entities"]:
                tid = int(ent["track_id"])
                cls_votes[tid][int(ent["cls"])] += 1
                px = ent.get("pitch_xy")
                if px is None:
                    continue
                pt = np.array(px, dtype=np.float32)
                # Filter out clearly out-of-bounds sideline staff (y < -2.5 or y > 70.5)
                if not np.all(np.isfinite(pt)) or float(pt[1]) < -2.5 or float(pt[1]) > 70.5:
                    continue
                track_frames[tid].append(f_idx)
                track_coords[tid].append(pt)
                if ent.get("team", -1) in (0, 1):
                    track_team[tid] = int(ent["team"])

        for tid, cnt_map in cls_votes.items():
            total_c = max(1, sum(cnt_map.values()))
            if cnt_map.get(self.cfg.referee_id, 0) / total_c >= 0.35:
                track_role[tid] = "referee"
                track_team.pop(tid, None)
            else:
                best_cls = max(cnt_map.items(), key=lambda kv: kv[1])[0]
                track_role[tid] = role_names.get(best_cls, "player")

        # Ensure referee tracks always carry team=-1 and cls=referee_id across every frame
        for rec in frame_records:
            for ent in rec["entities"]:
                tid = int(ent["track_id"])
                if track_role.get(tid) == "referee":
                    ent["cls"] = self.cfg.referee_id
                    ent["team"] = -1

        # 2. Detect ONLY genuine scene cuts, invalid frames, or severe whole-frame teleports
        # (NEVER wipe sprint streaks on routine PnLCalib/homography keyframe updates!)
        n_total_frames = len(frame_records)
        exclude_radius = max(1, int(round(0.25 * self.fps)))
        homography_excluded = np.zeros(n_total_frames, dtype=bool)
        prev_frame_pts: Dict[int, np.ndarray] = {}
        for f_idx, rec in enumerate(frame_records):
            cur_frame_pts: Dict[int, np.ndarray] = {}
            big_jumps = 0
            for ent in rec["entities"]:
                px = ent.get("pitch_xy")
                if px is None:
                    continue
                pt = np.asarray(px, dtype=np.float32)
                if not np.all(np.isfinite(pt)):
                    continue
                tid = int(ent["track_id"])
                cur_frame_pts[tid] = pt
                prev_pt = prev_frame_pts.get(tid)
                if prev_pt is not None and float(np.linalg.norm(pt - prev_pt)) > 4.5:
                    big_jumps += 1
            is_hard_invalid = (
                bool(rec.get("is_scene_cut", False))
                or rec.get("H") is None
                or (rec.get("homography_valid") is False)
                or big_jumps >= 4
            )
            if is_hard_invalid:
                lo = max(0, f_idx - exclude_radius)
                hi = min(n_total_frames, f_idx + exclude_radius + 1)
                homography_excluded[lo:hi] = True
            prev_frame_pts = cur_frame_pts

        # 3. Compute Smoothed Positions, Windowed + Median-Filtered Speeds, Distances & Calibrated Sprints
        pos_win = max(1, int(self.cfg.position_moving_avg_window))
        vel_win = max(1, int(self.cfg.speed_window_frames))
        med_k = max(1, int(self.cfg.speed_median_filter))
        max_spd_cap = float(self.cfg.max_valid_speed_kmh)
        max_human_mps = 11.5  # > 11.5 m/s (41.4 km/h) single-frame jump is a glitch, NOT a sprint
        # Broadcast High-Speed Sprint standard: >= 6.2 m/s (22.3 km/h) sustained for >= 0.55s
        sprint_thresh_kmh = 22.32
        min_sprint_frames = max(5, int(round(0.55 * self.fps)))
        max_blink_gap_s = max(0.22, 3.2 / self.fps)  # Tolerate 1-2 missing detection frames
        sprint_cooldown_s = 4.0

        player_stats: Dict[int, dict] = {}
        speed_lookup: Dict[Tuple[int, int], float] = {}
        sprint_events = []

        for tid, f_list in track_frames.items():
            pts = np.array(track_coords[tid], dtype=np.float32)
            smooth_pts = self._smooth_series(pts, window=pos_win)
            n_obs = len(f_list)
            raw_speeds = np.zeros(n_obs, dtype=np.float32)
            step_dists = np.zeros(n_obs, dtype=np.float32)
            obs_excluded = np.zeros(n_obs, dtype=bool)

            # Exclude brief warmup at track start and genuine scene cuts / invalid frames
            start_warmup = max(1, int(round(0.20 * self.fps)))
            for k in range(n_obs):
                if (f_list[k] - f_list[0]) < start_warmup:
                    obs_excluded[k] = True
                if 0 <= f_list[k] < n_total_frames and homography_excluded[f_list[k]]:
                    obs_excluded[k] = True

            for k in range(1, n_obs):
                dt_step = (f_list[k] - f_list[k - 1]) / self.fps
                if dt_step <= 0 or dt_step > 0.5:
                    # Genuine track re-entry gap (> 0.5s): exclude +-0.25s around re-entry
                    for j in range(max(0, k - exclude_radius), min(n_obs, k + exclude_radius + 1)):
                        if abs(f_list[j] - f_list[k]) <= exclude_radius:
                            obs_excluded[j] = True
                    continue

                raw_d_step = float(np.linalg.norm(pts[k] - pts[k - 1]))
                # Reject genuine single-frame teleport glitches (> 1.35m in 1 frame OR > 16 m/s raw jump)
                # without letting normal 4-pixel (0.20m) YOLO foot jitter at 25fps wipe real sprints!
                if raw_d_step > max(1.35, 16.0 * dt_step):
                    for j in range(max(0, k - exclude_radius), min(n_obs, k + exclude_radius + 1)):
                        if abs(f_list[j] - f_list[k]) <= exclude_radius:
                            obs_excluded[j] = True
                    continue

                d_step = float(np.linalg.norm(smooth_pts[k] - smooth_pts[k - 1]))
                inst_mps = d_step / dt_step
                if inst_mps > 13.5:
                    obs_excluded[k] = True
                    continue
                if inst_mps * 3.6 > max_spd_cap:
                    d_step = (max_spd_cap / 3.6) * dt_step
                step_dists[k] = d_step

                # Multi-frame rolling windowed speed suppresses single-frame projection noise
                k_lo = max(0, k - vel_win)
                while k_lo < k - 1 and (f_list[k] - f_list[k_lo]) > max(vel_win * 2, int(self.fps * 0.8)):
                    k_lo += 1
                dt_win = (f_list[k] - f_list[k_lo]) / self.fps
                if dt_win > 0:
                    d_win = float(np.linalg.norm(smooth_pts[k] - smooth_pts[k_lo]))
                    win_mps = d_win / dt_win
                    if win_mps <= max_human_mps:
                        raw_speeds[k] = min(max_spd_cap, win_mps * 3.6)
                    else:
                        obs_excluded[k] = True

            if n_obs >= med_k and med_k > 1:
                speeds = np.clip(median_filter(raw_speeds, size=med_k, mode="nearest"), 0.0, max_spd_cap)
            else:
                speeds = np.clip(raw_speeds, 0.0, max_spd_cap)

            total_dist = float(np.sum(step_dists))
            max_speed = float(np.max(speeds)) if n_obs > 0 else 0.0
            sprint_streak = 0
            sprint_count = 0
            in_sprint = False
            last_sprint_t = -1e9

            for k in range(1, n_obs):
                frame_gap = f_list[k] - f_list[k - 1]
                dt_step = frame_gap / self.fps
                if dt_step <= 0 or dt_step > max_blink_gap_s or obs_excluded[k]:
                    sprint_streak = 0
                    in_sprint = False
                    continue
                spd_kmh = float(speeds[k])
                if spd_kmh >= sprint_thresh_kmh:
                    # Tolerate 1-2 frame tracker blinks by crediting bridged frames when continuous
                    sprint_streak += min(3, max(1, int(frame_gap)))
                    cur_t = f_list[k] / self.fps
                    if (
                        not in_sprint
                        and sprint_streak >= min_sprint_frames
                        and (cur_t - last_sprint_t) >= sprint_cooldown_s
                    ):
                        in_sprint = True
                        last_sprint_t = cur_t
                        if track_role.get(tid) != "referee" and tid in track_team:
                            sprint_count += 1
                            sprint_events.append({
                                "type": "sprint",
                                "frame": int(f_list[k]),
                                "time_s": round(cur_t, 2),
                                "track_id": tid,
                                "team": track_team.get(tid, -1),
                                "speed_kmh": round(spd_kmh, 1),
                            })
                else:
                    sprint_streak = 0
                    in_sprint = False

            for f_i, sp in zip(f_list, speeds):
                speed_lookup[(f_i, tid)] = round(float(sp), 1)

            if track_role.get(tid) == "referee" or track_role.get(tid) not in ("player", "goalkeeper") or tid not in track_team:
                continue
            jinfo = jersey_map.get(tid, {})
            player_stats[tid] = {
                "track_id": tid,
                "team": track_team[tid],
                "role": track_role[tid],
                "distance_m": round(total_dist, 1),
                "max_speed_kmh": round(max_speed, 1),
                "avg_speed_kmh": round(float(np.mean(speeds)), 1),
                "sprints": sprint_count,
                "passes_completed": 0,
                "xt_generated": 0.0,
                "avg_pitch_xy": [round(float(np.mean(smooth_pts[:, 0])), 2), round(float(np.mean(smooth_pts[:, 1])), 2)],
                "frames_visible": len(f_list),
                "jersey_number": jinfo.get("jersey_number"),
                "jersey_confidence": jinfo.get("jersey_confidence"),
                "jersey_verified": bool(jinfo.get("jersey_verified", False)),
                "legibility_score": jinfo.get("legibility_score"),
            }

        # 3. Possession, Passes & Turnovers (bridged ball + asymmetric hysteresis state machine)
        voronoi_history = {0: [], 1: [], "def": [], "mid": [], "att": []}
        hull_areas = {0: [], 1: []}

        frames_players_pos: List[List[Tuple[int, int, float, float]]] = []
        frames_image_fallback: List[Tuple[Optional[np.ndarray], List[Tuple[int, int, np.ndarray]]]] = []
        raw_ball_pitch: List[Optional[np.ndarray]] = []
        for rec in frame_records:
            fp_list = []
            px_list = []
            for ent in rec["entities"]:
                px = ent.get("pitch_xy")
                tm = int(ent.get("team", -1))
                if tm not in (0, 1):
                    continue
                if ent["cls"] not in (self.cfg.player_id, self.cfg.goalkeeper_id):
                    continue
                box = ent.get("box")
                if box is not None:
                    px_list.append((int(ent["track_id"]), tm, np.asarray(box, dtype=np.float32)))
                if px is not None:
                    fp_list.append((int(ent["track_id"]), tm, float(px[0]), float(px[1])))
            frames_players_pos.append(fp_list)
            b_img = rec.get("ball_xy")
            frames_image_fallback.append(
                (None if b_img is None else np.asarray(b_img, dtype=np.float32), px_list)
            )
            bpx = rec.get("ball_pitch_xy")
            raw_ball_pitch.append(None if bpx is None else np.asarray(bpx, dtype=np.float32))

        invalid_mask = [
            bool(rec.get("is_scene_cut", False))
            or (rec.get("homography_valid") is False)
            or (rec.get("H") is None and rec.get("ball_pitch_xy") is None)
            for rec in frame_records
        ]
        bridged_ball = filter_and_interpolate_ball_pitch(
            raw_ball_pitch,
            self.fps,
            pitch_length=self.pitch.length,
            pitch_width=self.pitch.width,
            invalid_mask=invalid_mask,
            max_speed_mps=float(getattr(self.cfg, "ball_bridge_max_speed_mps", 38.0)),
            max_interp_speed_mps=35.0,
            max_interp_sec=0.40,
            max_interp_frames=int(getattr(self.cfg, "ball_bridge_max_gap_frames", 8)),
        )
        # Keep frame_records synchronized so tactical_radar.mp4 matches telemetry.json 1:1
        for rec, bp_clean in zip(frame_records, bridged_ball):
            rec["ball_pitch_xy"] = bp_clean.copy() if bp_clean is not None else None
        poss_res = extract_possession_and_passes(
            frames_players_pos,
            bridged_ball,
            self.fps,
            self.cfg,
            self.pitch,
            frames_image_fallback=frames_image_fallback,
        )
        possession_frames = poss_res["possession_frames"]
        owner_team_per_frame = poss_res["owner_team_per_frame"]
        passes = poss_res["passes"]
        interceptions = poss_res["interceptions"]
        passing_edges = poss_res["passing_edges"]
        team_xt = poss_res["team_xt"]
        for tid, ps in poss_res["player_pass_stats"].items():
            if tid in player_stats:
                player_stats[tid]["passes_completed"] = int(ps["passes"])
                player_stats[tid]["xt_generated"] = round(float(ps["xt"]), 4)

        # Cumulative per-frame counters for broadcast HUD
        n_rec = len(frame_records)
        cum_pass = {0: np.zeros(n_rec + 1, dtype=np.int32), 1: np.zeros(n_rec + 1, dtype=np.int32)}
        for p in passes:
            fi = min(n_rec, max(0, int(p["frame"])))
            cum_pass[int(p["team"])][fi] += 1
        for tm in (0, 1):
            cum_pass[tm] = np.cumsum(cum_pass[tm])
        cum_poss = {0: 0, 1: 0}

        # Heatmaps (12x8 grid per team)
        heatmaps = {0: np.zeros((8, 12), dtype=np.int32), 1: np.zeros((8, 12), dtype=np.int32)}
        timeline_frames = []

        for f_idx, rec in enumerate(frame_records):
            t0_pts, t1_pts = [], []
            all_xy, all_tm = [], []
            frame_players_json = []

            for ent in rec["entities"]:
                tid = int(ent["track_id"])
                ent["speed_kmh"] = speed_lookup.get((f_idx, tid), 0.0)
                px = ent.get("pitch_xy")
                tm = int(ent.get("team", -1))

                if px is not None and tm in (0, 1) and ent["cls"] in (self.cfg.player_id, self.cfg.goalkeeper_id):
                    pt = np.array(px, dtype=np.float32)
                    if not np.all(np.isfinite(pt)) or float(pt[1]) < -2.5 or float(pt[1]) > 70.5:
                        continue
                    all_xy.append(pt)
                    all_tm.append(tm)
                    if ent["cls"] == self.cfg.player_id:
                        if tm == 0:
                            t0_pts.append(pt)
                        else:
                            t1_pts.append(pt)
                    # Accumulate heatmap with strict finite + grid-index clamping
                    if np.isfinite(pt[0]) and np.isfinite(pt[1]):
                        h_rows, h_cols = heatmaps[tm].shape
                        hx_norm = float(np.clip(float(pt[0]) / max(1e-6, float(self.pitch.length)), 0.0, 1.0))
                        hy_norm = float(np.clip(float(pt[1]) / max(1e-6, float(self.pitch.width)), 0.0, 1.0))
                        hx = max(0, min(h_cols - 1, int(hx_norm * h_cols)))
                        hy = max(0, min(h_rows - 1, int(hy_norm * h_rows)))
                        heatmaps[tm][hy, hx] += 1

                    jinfo = jersey_map.get(tid, {})
                    fp_entry = {
                        "id": tid,
                        "team": tm,
                        "role": "gk" if ent["cls"] == self.cfg.goalkeeper_id else "p",
                        "x": round(float(pt[0]), 2),
                        "y": round(float(pt[1]), 2),
                        "spd": ent["speed_kmh"],
                    }
                    if jinfo.get("jersey_number") is not None:
                        fp_entry["jn"] = jinfo["jersey_number"]
                        fp_entry["jv"] = bool(jinfo.get("jersey_verified", False))
                    frame_players_json.append(fp_entry)

            # Convex Hull & Defensive Line (Last outfield defender X per team)
            frame_hulls = {}
            def_lines = {}
            for tm, pts_list in [(0, t0_pts), (1, t1_pts)]:
                if len(pts_list) >= 3:
                    pts_arr = np.array(pts_list, dtype=np.float32)
                    try:
                        hull = ConvexHull(pts_arr)
                        hull_pts = pts_arr[hull.vertices]
                        hull_areas[tm].append(float(hull.volume))  # In 2D scipy, volume = area
                        frame_hulls[tm] = hull_pts
                    except Exception:
                        pass
                if len(pts_list) >= 2:
                    xs = sorted([float(p[0]) for p in pts_list])
                    # Team 0 defends Left (x=0) -> lowest X outfield player; Team 1 defends Right (x=105) -> highest X
                    def_lines[tm] = xs[0] if tm == 0 else xs[-1]

            rec["hulls"] = frame_hulls
            rec["def_lines"] = def_lines

            # Voronoi Pitch Control (computed every frame for smooth radar rendering)
            all_xy_arr = np.array(all_xy, dtype=np.float32) if all_xy else np.zeros((0, 2), dtype=np.float32)
            all_tm_arr = np.array(all_tm, dtype=np.int32) if all_tm else np.zeros((0,), dtype=np.int32)
            owner_grid, vor_metrics = self.compute_voronoi_control(all_xy_arr, all_tm_arr)
            rec["voronoi_grid"] = owner_grid
            rec["voronoi_metrics"] = vor_metrics

            voronoi_history[0].append(vor_metrics["team_0_pct"])
            voronoi_history[1].append(vor_metrics["team_1_pct"])
            voronoi_history["def"].append(vor_metrics["team_0_def_third_pct"])
            voronoi_history["mid"].append(vor_metrics["team_0_mid_third_pct"])
            voronoi_history["att"].append(vor_metrics["team_0_att_third_pct"])

            # Ball Ownership & Cumulative HUD stats (from unified bridged state machine)
            ball_p = bridged_ball[f_idx] if f_idx < len(bridged_ball) else rec.get("ball_pitch_xy")
            ball_owner_team = owner_team_per_frame[f_idx] if f_idx < len(owner_team_per_frame) else -1
            if ball_owner_team in (0, 1):
                cum_poss[ball_owner_team] += 1
            rec["ball_owner_team"] = ball_owner_team

            tot_pos_so_far = cum_poss[0] + cum_poss[1]
            # Require >= 5 possession frames before leaving neutral 50/50 on broadcast HUD
            # so a single initial touch does not flash 0.0% | 100.0% on the opening seconds.
            if tot_pos_so_far >= 5:
                hud_pos_0 = round(100.0 * cum_poss[0] / float(tot_pos_so_far), 1)
                hud_pos_1 = round(100.0 - hud_pos_0, 1)
            else:
                hud_pos_0, hud_pos_1 = 50.0, 50.0
            rec["hud_stats"] = {
                "pos_0": hud_pos_0,
                "pos_1": hud_pos_1,
                "ctrl_0": vor_metrics["team_0_pct"],
                "ctrl_1": vor_metrics["team_1_pct"],
                "passes_0": int(cum_pass[0][f_idx]),
                "passes_1": int(cum_pass[1][f_idx]),
            }

            hv = bool(rec.get("homography_valid", rec.get("H") is not None or len(frame_players_json) > 0))
            timeline_frames.append({
                "f": f_idx,
                "t": round(f_idx / self.fps, 2),
                "hv": hv,
                "players": frame_players_json if hv else [],
                "ball": [round(float(ball_p[0]), 2), round(float(ball_p[1]), 2)] if (hv and ball_p is not None) else None,
                "vor": [vor_metrics["team_0_pct"], vor_metrics["team_1_pct"]],
                "def_x": {str(k): round(float(v), 2) for k, v in def_lines.items()} if hv else {},
            })

        # 4. Assemble Clean Frontend-Ready Telemetry Contract
        tot_pos = possession_frames[0] + possession_frames[1]
        pos_0 = round(100.0 * possession_frames[0] / max(1, tot_pos), 1) if tot_pos else 50.0

        leaderboard = sorted(
            [p for p in player_stats.values() if p["frames_visible"] >= int(self.fps * 0.8)],
            key=lambda x: (-x["distance_m"], -x["max_speed_kmh"]),
        )

        passing_network = [
            {"team": tm, "from_id": u, "to_id": v, "count": cnt}
            for (tm, u, v), cnt in passing_edges.items()
        ]

        events = sorted(passes + interceptions + sprint_events, key=lambda e: e["frame"])

        return {
            "summary": {
                "total_frames": len(frame_records),
                "fps": round(self.fps, 2),
                "duration_s": round(len(frame_records) / self.fps, 2),
                "possession_pct": {"team_0": pos_0, "team_1": round(100.0 - pos_0, 1)},
                "pitch_control_pct": {
                    "team_0": round(float(np.mean(voronoi_history[0])), 1),
                    "team_1": round(float(np.mean(voronoi_history[1])), 1),
                    "team_0_zonal": {
                        "defensive_third": round(float(np.mean(voronoi_history["def"])), 1),
                        "middle_third": round(float(np.mean(voronoi_history["mid"])), 1),
                        "attacking_third": round(float(np.mean(voronoi_history["att"])), 1),
                    },
                },
                "passes": {
                    "team_0": sum(1 for p in passes if p["team"] == 0),
                    "team_1": sum(1 for p in passes if p["team"] == 1),
                },
                "turnovers": len(interceptions),
                "expected_threat_xt": {
                    "team_0": round(team_xt[0], 4),
                    "team_1": round(team_xt[1], 4),
                },
                "avg_convex_hull_area_m2": {
                    "team_0": round(float(np.mean(hull_areas[0])), 1) if hull_areas[0] else 0.0,
                    "team_1": round(float(np.mean(hull_areas[1])), 1) if hull_areas[1] else 0.0,
                },
                "top_speed_kmh": max([p["max_speed_kmh"] for p in leaderboard], default=0.0),
            },
            "players": leaderboard,
            "passing_network": passing_network,
            "events": events,
            "heatmaps": {
                "team_0": heatmaps[0].tolist(),
                "team_1": heatmaps[1].tolist(),
            },
            "frames": timeline_frames,
        }
