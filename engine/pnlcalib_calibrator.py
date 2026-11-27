"""SoccerNet GSR SOTA 3D Broadcast Pitch Calibration — PnLCalib Integration.

Integrates the official `mguti97/PnLCalib` dual HRNetV2-w48 pipeline:
  - 57 semantic pitch keypoints (`SV_kp`) + 23 pitch line extremities (`SV_lines`)
  - Physically constrained 3D camera parameter estimation `(K, R, t)` + nonlinear
    Points-and-Lines (PnL) refinement via `FramebyFrameCalib.heuristic_voting(refine_lines=True)`
  - Exact ground-plane (`Z = 0`) projection converted from PnLCalib's centered pitch frame
    `[-52.5..52.5, -34..34]` to FIFA pitch coordinates `[0..105, 0..68]`:
        H_img_to_fifa = T_shift @ inv(P[:, [0, 1, 3]])
  - High-speed Keyframe (every 2–3 frames + scene cuts) + Bidirectional Masked Background
    Lucas-Kanade Optical Flow propagation in-between for broadcast-grade speed & smoothness
  - Universal 3D camera pose + Jacobian monotonicity + Player In-Bounds sanity gates
  - Seamless, zero-crash graceful fallback to `yolo_pitch_fallback` (StabilizedHomography)
    on local CPU or when PnLCalib weights are absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import FIFAPitchConfig
from .homography import StabilizedHomography

PNLCALIB_RELEASE_BASE = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"
PNLCALIB_WEIGHT_URLS = {
    "SV_kp": f"{PNLCALIB_RELEASE_BASE}/SV_kp",
    "SV_lines": f"{PNLCALIB_RELEASE_BASE}/SV_lines",
}


def resolve_pnlcalib_paths(
    models_dir: Optional[Path] = None,
) -> Dict[str, Optional[Path]]:
    """Locate official PnLCalib repo and SV_kp / SV_lines weight files cleanly."""
    project_root = Path(__file__).resolve().parent.parent

    # 1. Locate PnLCalib repo directory (containing model/cls_hrnet.py & utils/utils_calib.py)
    repo_candidates = []
    env_repo = os.environ.get("PNLCALIB_REPO_DIR", "").strip()
    if env_repo:
        repo_candidates.append(Path(env_repo))
    repo_candidates.extend([
        Path("/kaggle/working/PnLCalib"),
        project_root / "third_party" / "PnLCalib",
        project_root.parent / "PnLCalib",
    ])
    repo_dir: Optional[Path] = None
    for cand in repo_candidates:
        if (cand / "model" / "cls_hrnet.py").exists() and (cand / "utils" / "utils_calib.py").exists():
            repo_dir = cand
            break

    # 2. Locate SV_kp and SV_lines weights
    weight_search_dirs = []
    env_wdir = os.environ.get("PNLCALIB_WEIGHTS_DIR", "").strip()
    if env_wdir:
        weight_search_dirs.append(Path(env_wdir))
    if models_dir is not None:
        weight_search_dirs.append(Path(models_dir))
    env_mdir = os.environ.get("MODELS_DIR", "").strip()
    if env_mdir:
        weight_search_dirs.append(Path(env_mdir))
    weight_search_dirs.extend([
        Path("/kaggle/working/pnlcalib_weights"),
        project_root / "models" / "pnlcalib",
        project_root / "models",
    ])
    if repo_dir is not None:
        weight_search_dirs.append(repo_dir / "weights")

    kp_path: Optional[Path] = None
    lines_path: Optional[Path] = None
    for d in weight_search_dirs:
        if not d.exists():
            continue
        if kp_path is None:
            for name in ("SV_kp", "SV_kp.pth", "SV_kp.pt"):
                p = d / name
                if p.is_file() and p.stat().st_size > 1_000_000:
                    kp_path = p
                    break
        if lines_path is None:
            for name in ("SV_lines", "SV_lines.pth", "SV_lines.pt"):
                p = d / name
                if p.is_file() and p.stat().st_size > 1_000_000:
                    lines_path = p
                    break
        if kp_path is not None and lines_path is not None:
            break

    return {
        "repo_dir": repo_dir,
        "weights_kp": kp_path,
        "weights_lines": lines_path,
    }


def check_pnlcalib_readiness(models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Check whether PnLCalib 3D is ready to run on CUDA GPU (used by GET /health & Pipeline)."""
    paths = resolve_pnlcalib_paths(models_dir=models_dir)
    cuda_ok = bool(torch.cuda.is_available())
    files_ok = (
        paths["repo_dir"] is not None
        and paths["weights_kp"] is not None
        and paths["weights_lines"] is not None
    )
    ready = bool(cuda_ok and files_ok)
    return {
        "pnlcalib_ready": ready,
        "calibrator_backend": "pnlcalib_3d" if ready else "yolo_pitch_fallback",
        "cuda_available": cuda_ok,
        "repo_dir": str(paths["repo_dir"]) if paths["repo_dir"] else None,
        "weights_kp": str(paths["weights_kp"]) if paths["weights_kp"] else None,
        "weights_lines": str(paths["weights_lines"]) if paths["weights_lines"] else None,
    }


def cam_params_to_fifa_homography(
    final_params_dict: Dict[str, Any],
    pitch_cfg: Optional[FIFAPitchConfig] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    """
    Convert PnLCalib 3D `final_params_dict` (`cam_params`: K, R, position_meters)
    into a validated 3x3 image-to-FIFA-pitch homography `H_img_to_fifa` (0..105, 0..68).

    Also validates physical 3D broadcast camera plausibility:
      - Focal length in `[150, 18000]` px
      - Camera height above pitch `|z_cam| >= 1.5m` (broadcast gantry, not underground/on-turf)
      - Camera position outside pitch interior or elevated cleanly
      - Invertible ground-plane projection matrix `P[:, [0, 1, 3]]`

    Returns `(H_img_to_fifa, pose_meta)` or `None` if physically implausible.
    """
    if not isinstance(final_params_dict, dict):
        return None
    cam_params = final_params_dict.get("cam_params")
    if not isinstance(cam_params, dict):
        return None

    try:
        fx = float(cam_params["x_focal_length"])
        fy = float(cam_params["y_focal_length"])
        pp = np.asarray(cam_params["principal_point"], dtype=np.float64).reshape(2)
        pos = np.asarray(cam_params["position_meters"], dtype=np.float64).reshape(3)
        rot = np.asarray(cam_params["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError):
        return None

    if not (
        np.isfinite(fx)
        and np.isfinite(fy)
        and np.all(np.isfinite(pp))
        and np.all(np.isfinite(pos))
        and np.all(np.isfinite(rot))
    ):
        return None

    # Physical 3D broadcast camera sanity gates (match-agnostic)
    if not (150.0 <= fx <= 18000.0 and 150.0 <= fy <= 18000.0):
        return None
    aspect_f = fx / max(1e-6, fy)
    if not (0.75 <= aspect_f <= 1.35):
        return None
    # Broadcast cameras are elevated above ground (Z_pnl is negative-up in PnLCalib; check |Z| >= 1.5m)
    if abs(float(pos[2])) < 1.5 or abs(float(pos[2])) > 120.0:
        return None
    if float(np.linalg.norm(pos[:2])) > 260.0:
        return None

    # Build 3x4 camera projection matrix P = Q @ R @ [I | -pos]
    It = np.eye(4, dtype=np.float64)[:-1]
    It[:, -1] = -pos
    Q = np.array(
        [[fx, 0.0, pp[0]], [0.0, fy, pp[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    P = Q @ (rot @ It)

    # Ground plane Z=0 homography: maps centered pitch [X_pnl, Y_pnl, 1] -> image [u, v, w]
    H_pnl_to_img = P[:, [0, 1, 3]]
    if abs(float(np.linalg.det(H_pnl_to_img))) < 1e-10:
        return None
    try:
        H_img_to_pnl = np.linalg.inv(H_pnl_to_img)
    except np.linalg.LinAlgError:
        return None

    # Shift from PnLCalib centered coordinates [-L/2..L/2, -W/2..W/2] -> FIFA [0..L, 0..W]
    L = float(pitch_cfg.length) if pitch_cfg is not None else 105.0
    W = float(pitch_cfg.width) if pitch_cfg is not None else 68.0
    T_shift = np.array(
        [[1.0, 0.0, L * 0.5], [0.0, 1.0, W * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_img_to_fifa = T_shift @ H_img_to_pnl
    if not np.all(np.isfinite(H_img_to_fifa)) or abs(float(H_img_to_fifa[2, 2])) < 1e-10:
        return None
    H_img_to_fifa = H_img_to_fifa / float(H_img_to_fifa[2, 2])

    pose_meta = {
        "fx": round(fx, 1),
        "fy": round(fy, 1),
        "cam_x_m": round(float(pos[0]), 2),
        "cam_y_m": round(float(pos[1]), 2),
        "cam_z_m": round(float(pos[2]), 2),
    }
    return H_img_to_fifa, pose_meta


class PnLCalib3DCalibrator:
    """
    Production Keyframe-3D + Masked-Optical-Flow Broadcast Pitch Calibrator.

    - On CUDA GPU with `SV_kp` + `SV_lines` + official `mguti97/PnLCalib` available:
      runs dual HRNetV2-w48 (57 keypoints + 23 lines + 3D PnL refinement) on keyframes
      and propagates smoothly in-between via bidirectional masked background optical flow.
    - On CPU or when PnLCalib assets are unavailable:
      operates in `yolo_pitch_fallback` mode delegating directly to hardened `StabilizedHomography`.
    """

    def __init__(
        self,
        pitch_cfg: FIFAPitchConfig,
        stabilizer: StabilizedHomography,
        models_dir: Optional[Path] = None,
        keyframe_stride: int = 3,
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        pnl_refine: bool = True,
        force_fallback: bool = False,
    ):
        self.pitch_cfg = pitch_cfg
        self.stabilizer = stabilizer
        self.keyframe_stride = max(1, int(keyframe_stride))
        self.kp_threshold = float(kp_threshold)
        self.line_threshold = float(line_threshold)
        self.pnl_refine = bool(pnl_refine)

        self.backend: str = "yolo_pitch_fallback"
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_kp: Any = None
        self._model_lines: Any = None
        self._cam: Any = None
        self._cam_dims: Tuple[int, int] = (0, 0)
        self._pnl_fns: Dict[str, Any] = {}

        # Diagnostics
        self.pnl_keyframes_attempted: int = 0
        self.pnl_keyframes_succeeded: int = 0
        self.pnl_last_pose: Optional[Dict[str, float]] = None

        if not force_fallback:
            self._try_load_pnlcalib(models_dir=models_dir)

    @property
    def is_pnlcalib_active(self) -> bool:
        return self.backend == "pnlcalib_3d" and self._model_kp is not None and self._model_lines is not None

    def _try_load_pnlcalib(self, models_dir: Optional[Path] = None) -> None:
        info = check_pnlcalib_readiness(models_dir=models_dir)
        if not info["pnlcalib_ready"]:
            return
        repo_dir = Path(info["repo_dir"])
        kp_path = Path(info["weights_kp"])
        lines_path = Path(info["weights_lines"])

        try:
            import yaml

            repo_str = str(repo_dir)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

            from model.cls_hrnet import get_cls_net
            from model.cls_hrnet_l import get_cls_net as get_cls_net_l
            from utils.utils_calib import FramebyFrameCalib
            from utils.utils_heatmap import (
                complete_keypoints,
                coords_to_dict,
                get_keypoints_from_heatmap_batch_maxpool,
                get_keypoints_from_heatmap_batch_maxpool_l,
            )

            cfg_kp = yaml.safe_load((repo_dir / "config" / "hrnetv2_w48.yaml").read_text(encoding="utf-8"))
            cfg_l = yaml.safe_load((repo_dir / "config" / "hrnetv2_w48_l.yaml").read_text(encoding="utf-8"))

            state_kp = torch.load(str(kp_path), map_location=self.device)
            model_kp = get_cls_net(cfg_kp)
            model_kp.load_state_dict(state_kp)
            model_kp.to(self.device).eval()

            state_l = torch.load(str(lines_path), map_location=self.device)
            model_l = get_cls_net_l(cfg_l)
            model_l.load_state_dict(state_l)
            model_l.to(self.device).eval()

            self._model_kp = model_kp
            self._model_lines = model_l
            self._pnl_fns = {
                "FramebyFrameCalib": FramebyFrameCalib,
                "get_kp": get_keypoints_from_heatmap_batch_maxpool,
                "get_l": get_keypoints_from_heatmap_batch_maxpool_l,
                "coords_to_dict": coords_to_dict,
                "complete_keypoints": complete_keypoints,
            }
            self.backend = "pnlcalib_3d"
            print(
                f"[PnLCalib] Loaded official 3D HRNetV2-w48 (SV_kp + SV_lines) on {self.device.upper()} "
                f"(keyframe_stride={self.keyframe_stride}, pnl_refine={self.pnl_refine})"
            )
        except Exception as e:
            print(f"[PnLCalib] Unavailable ({e}); falling back cleanly to yolo_pitch_fallback.")
            self.close()
            self.backend = "yolo_pitch_fallback"

    def close(self) -> None:
        self._model_kp = None
        self._model_lines = None
        self._cam = None
        self._pnl_fns.clear()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _infer_pnlcalib_homography(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        """Run dual HRNetV2-w48 + FramebyFrameCalib PnL refinement on a single keyframe."""
        if not self.is_pnlcalib_active:
            return None
        h_orig, w_orig = frame_bgr.shape[:2]
        if self._cam is None or self._cam_dims != (w_orig, h_orig):
            self._cam = self._pnl_fns["FramebyFrameCalib"](
                iwidth=w_orig, iheight=h_orig, denormalize=True
            )
            self._cam_dims = (w_orig, h_orig)

        self.pnl_keyframes_attempted += 1
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if w_orig != 960 or h_orig != 540:
                rgb_in = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_LINEAR)
            else:
                rgb_in = rgb
            tensor = (
                torch.from_numpy(rgb_in)
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
                .unsqueeze(0)
                .to(self.device)
            )
            _, _, h_in, w_in = tensor.shape

            with torch.no_grad():
                hm_kp = self._model_kp(tensor)
                hm_l = self._model_lines(tensor)

            kp_coords = self._pnl_fns["get_kp"](hm_kp[:, :-1, :, :])
            line_coords = self._pnl_fns["get_l"](hm_l[:, :-1, :, :])
            kp_dict = self._pnl_fns["coords_to_dict"](kp_coords, threshold=self.kp_threshold)
            lines_dict = self._pnl_fns["coords_to_dict"](line_coords, threshold=self.line_threshold)
            kp_dict, lines_dict = self._pnl_fns["complete_keypoints"](
                kp_dict[0], lines_dict[0], w=w_in, h=h_in, normalize=True
            )

            self._cam.update(kp_dict, lines_dict)
            final_params_dict = self._cam.heuristic_voting(refine_lines=self.pnl_refine)
            if final_params_dict is None:
                return None

            res = cam_params_to_fifa_homography(final_params_dict, self.pitch_cfg)
            if res is not None:
                self.pnl_keyframes_succeeded += 1
                self.pnl_last_pose = res[1]
            return res
        except Exception:
            return None

    def update_frame(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        prev_gray: Optional[np.ndarray],
        cur_gray: np.ndarray,
        person_boxes: np.ndarray,
        player_feet_px: Optional[np.ndarray],
        is_scene_cut: bool = False,
        fallback_kpts_xy: Optional[np.ndarray] = None,
        fallback_kpts_conf: Optional[np.ndarray] = None,
        run_fallback_pitch: bool = True,
    ) -> bool:
        """
        Unified per-frame calibration entry point.
        Updates `self.stabilizer` (`H`, `H_inv`, `homography_valid`, `last_was_reset`) cleanly.
        """
        h_img, w_img = frame_bgr.shape[:2]
        closeup_now = StabilizedHomography.is_genuine_closeup(person_boxes, h_img)

        # ── PATH A: Official PnLCalib 3D (Keyframe HRNetV2-w48 + Optical Flow In-Between) ──
        if self.is_pnlcalib_active:
            self.stabilizer.last_was_reset = False
            if is_scene_cut:
                self.stabilizer.reset_scene_cut()

            # Propagate existing H smoothly via background optical flow first
            flow_ok = False
            if not is_scene_cut and self.stabilizer.H is not None and prev_gray is not None:
                flow_ok = self.stabilizer.propagate_optical_flow(
                    prev_gray,
                    cur_gray,
                    exclude_boxes=person_boxes,
                    player_feet_px=player_feet_px,
                )

            run_keyframe = (
                is_scene_cut
                or (self.stabilizer.H is None)
                or (not flow_ok)
                or (frame_idx % self.keyframe_stride == 0)
            )
            pnl_ok = False
            if run_keyframe:
                res = self._infer_pnlcalib_homography(frame_bgr)
                if res is not None:
                    H_cand, _pose = res
                    pnl_ok = self.stabilizer.apply_external_homography(
                        H_cand,
                        player_feet_px=player_feet_px,
                        is_scene_cut=False,  # already reset above
                        img_width=w_img,
                        img_height=h_img,
                        reproj_err_m=0.0,
                        ema_alpha_override=0.45 if self.stabilizer._consecutive_flow_frames > 0 else 0.35,
                    )

            if pnl_ok or flow_ok:
                self.stabilizer.homography_valid = True
                return True

            # Neither PnLCalib nor optical flow succeeded:
            # Clear immediately on genuine close-ups; brief grace hold on wide tactical shots
            self.stabilizer._consecutive_failures += 1
            self.stabilizer.rejected_updates += 1
            hold_limit = 2 if closeup_now else self.stabilizer.temporal_hold_frames
            if self.stabilizer._consecutive_failures > hold_limit and self.stabilizer.H is not None:
                self.stabilizer.H = None
                self.stabilizer.H_inv = None
                self.stabilizer.homography_valid = False
                self.stabilizer.last_was_reset = True
            self.stabilizer.homography_valid = self.stabilizer.H is not None
            return self.stabilizer.homography_valid

        # ── PATH B: Hardened YOLOv11-pose Fallback (Local CPU / No PnLCalib Weights) ──
        if run_fallback_pitch and fallback_kpts_xy is not None and fallback_kpts_conf is not None:
            ok = self.stabilizer.update(
                fallback_kpts_xy,
                fallback_kpts_conf,
                player_feet_px=player_feet_px,
                is_scene_cut=is_scene_cut,
                prev_gray=prev_gray,
                cur_gray=cur_gray,
                exclude_boxes=person_boxes,
                img_width=w_img,
            )
            # If this frame is an obvious extreme close-up/celebration and keypoints failed, pause promptly
            if not ok or (closeup_now and self.stabilizer._consecutive_failures >= 3):
                if closeup_now and self.stabilizer._consecutive_failures >= 3:
                    self.stabilizer.H = None
                    self.stabilizer.H_inv = None
                    self.stabilizer.homography_valid = False
                    self.stabilizer.last_was_reset = True
            return self.stabilizer.homography_valid

        if self.stabilizer.H is not None and prev_gray is not None:
            flow_ok = self.stabilizer.propagate_optical_flow(
                prev_gray,
                cur_gray,
                exclude_boxes=person_boxes,
                player_feet_px=player_feet_px,
            )
            if flow_ok:
                return True

        return self.stabilizer.homography_valid

    def summary(self) -> Dict[str, Any]:
        base = self.stabilizer.summary()
        base.update({
            "calibrator_backend": self.backend,
            "pnlcalib_keyframes_attempted": self.pnl_keyframes_attempted,
            "pnlcalib_keyframes_succeeded": self.pnl_keyframes_succeeded,
            "pnlcalib_last_pose": self.pnl_last_pose,
        })
        return base
