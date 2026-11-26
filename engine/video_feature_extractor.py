from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

# Exact TensorFlow-2 Keras Applications ResNet Caffe BGR mean (matches resnet152_tf2_backbone.onnx)
CAFFE_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)


class SoccerNetVideoFeatureExtractor:
    """
    Extracts SoccerNet-compatible 2FPS `(T, 512)` ResNet-152 + PCA-512 feature matrices
    directly from raw `.mp4` broadcast videos using ONNX Runtime CPU:
      1. Resize BGR frame to `224x224` and subtract TF2 Caffe BGR mean `[103.939, 116.779, 123.68]`
      2. Forward NHWC `bgr_input` `[B, 224, 224, 3]` -> `features_2048` `[B, 2048]`
      3. Apply verified SoccerNet PCA-512 projection: `(f2048 - average - pca_mean) @ proj_matrix`
    """

    def __init__(
        self,
        resnet_onnx_path: Optional[str | Path] = None,
        pca_weights_path: Optional[str | Path] = None,
        target_fps: float = 2.0,
        batch_size: int = 16,
    ):
        root = Path(__file__).resolve().parent.parent
        models_dir = root / "models"
        self.resnet_onnx_path = (
            Path(resnet_onnx_path)
            if resnet_onnx_path
            else (models_dir / "resnet152_tf2_backbone.onnx")
        )
        self.pca_weights_path = (
            Path(pca_weights_path)
            if pca_weights_path
            else (models_dir / "soccernet_pca512_tf2_weights.npz")
        )
        self.target_fps = float(target_fps)
        self.batch_size = max(1, int(batch_size))

        self._ort_session: Any = None
        self._ort_input_name: str = "bgr_input"
        self._average: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._proj_matrix: Optional[np.ndarray] = None

    def is_available(self) -> bool:
        return self.resnet_onnx_path.exists() and self.pca_weights_path.exists()

    def _ensure_loaded(self) -> None:
        if self._ort_session is not None and self._proj_matrix is not None:
            return
        if not self.is_available():
            raise FileNotFoundError(
                f"Missing ResNet152 ONNX ({self.resnet_onnx_path}) or PCA512 weights ({self.pca_weights_path})."
            )

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(self.resnet_onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._ort_session = sess
        self._ort_input_name = sess.get_inputs()[0].name

        with np.load(str(self.pca_weights_path), allow_pickle=False) as z:
            self._average = np.asarray(z["average"], dtype=np.float32).reshape(1, -1)
            self._pca_mean = np.asarray(z["pca_mean"], dtype=np.float32).reshape(1, -1)
            self._proj_matrix = np.asarray(z["proj_matrix"], dtype=np.float32)

    @staticmethod
    def preprocess_frames_bgr(frames_bgr: List[np.ndarray]) -> np.ndarray:
        """Resize BGR frames to 224x224 and apply TF2 Caffe BGR mean subtraction (NHWC float32)."""
        if not frames_bgr:
            return np.zeros((0, 224, 224, 3), dtype=np.float32)

        batch = []
        for fr in frames_bgr:
            if fr.shape[0] != 224 or fr.shape[1] != 224:
                resized = cv2.resize(fr, (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                resized = fr
            batch.append(resized.astype(np.float32) - CAFFE_BGR_MEAN)
        return np.ascontiguousarray(np.stack(batch, axis=0), dtype=np.float32)

    def extract_from_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """Extract contiguous `(T, 512)` float32 SoccerNet PCA features from BGR frames."""
        if not frames_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        self._ensure_loaded()
        assert self._average is not None
        assert self._pca_mean is not None
        assert self._proj_matrix is not None

        chunks: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self.batch_size):
            xb = self.preprocess_frames_bgr(frames_bgr[i:i + self.batch_size])
            feats_2048 = self._ort_session.run(None, {self._ort_input_name: xb})[0]
            centered = np.asarray(feats_2048, dtype=np.float32) - self._average - self._pca_mean
            proj_512 = centered @ self._proj_matrix
            chunks.append(np.asarray(proj_512, dtype=np.float32))

        return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)

    def extract_from_video(
        self,
        video_path: str | Path,
        max_seconds: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Sample video at ~2.0 FPS and return `(features (T, 512), effective_fps)`."""
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Video file not found: {vpath}")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {vpath}")

        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_seconds is not None:
            total_frames = min(total_frames, max(1, int(round(video_fps * max_seconds))))

        stride = max(1, int(round(video_fps / max(0.1, self.target_fps))))
        eff_fps = video_fps / float(stride)

        sampled: List[np.ndarray] = []
        f_idx = 0
        while True:
            if total_frames > 0 and f_idx >= total_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if f_idx % stride == 0:
                sampled.append(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR))
            f_idx += 1
        cap.release()

        if not sampled:
            return np.zeros((0, 512), dtype=np.float32), eff_fps

        return self.extract_from_frames(sampled), eff_fps
