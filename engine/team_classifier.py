from collections import defaultdict, Counter
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class SigLIPTeamClassifier:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_siglip = False
        self.processor = None
        self.model = None
        self.reducer = None
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.fitted = False

        try:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.use_siglip = True
            print(f"[TeamClassifier] Loaded SigLIP ({model_name}) on {self.device}.")
        except Exception as e:
            print(f"[TeamClassifier] SigLIP unavailable ({e}); using LAB non-green jersey fallback.")

    @staticmethod
    def crop_jersey(frame_bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 14:
            return None
        jy1 = y1 + int(bh * 0.14)
        jy2 = y1 + int(bh * 0.52)
        jx1 = x1 + int(bw * 0.18)
        jx2 = x1 + int(bw * 0.82)
        crop = frame_bgr[jy1:jy2, jx1:jx2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        return crop

    def _extract_lab_features(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        feats = []
        for crop in crops_bgr:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            grass_mask = (hsv[:, :, 0] >= 32) & (hsv[:, :, 0] <= 88) & (hsv[:, :, 1] >= 40)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            fg = lab[~grass_mask]
            if len(fg) < 8:
                fg = lab.reshape(-1, 3)
            feats.append(np.concatenate([fg.mean(axis=0), fg.std(axis=0)]))
        return np.array(feats, dtype=np.float32)

    def extract_embeddings(self, crops_bgr: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 6), dtype=np.float32)
        if not self.use_siglip:
            return self._extract_lab_features(crops_bgr)

        all_embs = []
        pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        with torch.no_grad():
            for i in range(0, len(pil_imgs), batch_size):
                batch = pil_imgs[i:i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                if hasattr(self.model, "get_image_features"):
                    out = self.model.get_image_features(**inputs)
                else:
                    out = self.model.vision_model(**inputs).pooler_output
                if not isinstance(out, torch.Tensor):
                    out = out.pooler_output if hasattr(out, "pooler_output") else out[0]
                out = torch.nn.functional.normalize(out.float(), p=2, dim=-1)
                all_embs.append(out.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    def fit(self, sample_crops_bgr: List[np.ndarray]):
        if len(sample_crops_bgr) < 6:
            return
        if len(sample_crops_bgr) > 300:
            idx = np.linspace(0, len(sample_crops_bgr) - 1, 300, dtype=int)
            sample_crops_bgr = [sample_crops_bgr[i] for i in idx]

        embs = self.extract_embeddings(sample_crops_bgr)
        n_comp = min(3, embs.shape[1], max(2, len(embs) - 1))
        try:
            import umap
            self.reducer = umap.UMAP(n_components=n_comp, random_state=42, n_neighbors=min(15, len(embs) - 1))
            proj = self.reducer.fit_transform(embs)
            print("[TeamClassifier] Fitted SigLIP + UMAP + KMeans(k=2).")
        except Exception as e:
            print(f"[TeamClassifier] UMAP fallback to PCA ({e}).")
            self.reducer = PCA(n_components=n_comp, random_state=42)
            proj = self.reducer.fit_transform(embs)

        self.kmeans.fit(proj)
        self.fitted = True

    def predict(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0,), dtype=np.int32)
        if not self.fitted:
            return np.zeros((len(crops_bgr),), dtype=np.int32)
        embs = self.extract_embeddings(crops_bgr)
        proj = self.reducer.transform(embs)
        return self.kmeans.predict(proj).astype(np.int32)


def stabilize_team_assignments(
    frame_records: List[dict],
    classifier: SigLIPTeamClassifier,
    player_id: int = 2,
    goalkeeper_id: int = 1,
    max_crops_per_track: int = 12,
) -> Dict[int, int]:
    crops_by_tid: Dict[int, List[np.ndarray]] = defaultdict(list)
    for rec in frame_records:
        for ent in rec["entities"]:
            if ent["cls"] == player_id and ent.get("crop") is not None:
                crops_by_tid[int(ent["track_id"])].append(ent["crop"])

    batched_crops, batched_tids = [], []
    for tid, clist in crops_by_tid.items():
        if len(clist) > max_crops_per_track:
            idxs = np.linspace(0, len(clist) - 1, max_crops_per_track, dtype=int)
            chosen = [clist[i] for i in idxs]
        else:
            chosen = clist
        for c in chosen:
            batched_crops.append(c)
            batched_tids.append(tid)

    votes: Dict[int, Counter] = defaultdict(Counter)
    if batched_crops:
        preds = classifier.predict(batched_crops)
        for tid, tm in zip(batched_tids, preds):
            votes[tid][int(tm)] += 1

    track_to_team: Dict[int, int] = {tid: cnt.most_common(1)[0][0] for tid, cnt in votes.items()}

    gk_positions: Dict[int, List[float]] = defaultdict(list)
    team_x_positions: Dict[int, List[float]] = defaultdict(list)
    for rec in frame_records:
        for ent in rec["entities"]:
            px = ent.get("pitch_xy")
            if px is None:
                continue
            tid = int(ent["track_id"])
            if ent["cls"] == player_id and tid in track_to_team:
                team_x_positions[track_to_team[tid]].append(float(px[0]))
            elif ent["cls"] == goalkeeper_id:
                gk_positions[tid].append(float(px[0]))

    c0 = float(np.median(team_x_positions[0])) if team_x_positions[0] else 25.0
    c1 = float(np.median(team_x_positions[1])) if team_x_positions[1] else 80.0

    for gk_tid, xs in gk_positions.items():
        gx = float(np.median(xs))
        track_to_team[gk_tid] = 0 if abs(gx - c0) <= abs(gx - c1) else 1

    if c0 > c1:
        track_to_team = {tid: (1 - tm) for tid, tm in track_to_team.items()}

    for rec in frame_records:
        for ent in rec["entities"]:
            tid = int(ent["track_id"])
            ent["team"] = track_to_team.get(tid, -1)
            ent.pop("crop", None)

    return track_to_team
