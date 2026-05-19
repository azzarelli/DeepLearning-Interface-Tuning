try:
    import dearpygui.dearpygui as dpg
except Exception:
    print("No dpg running")
    dpg = None

import sys
import numpy as np
import os
import glob
import copy
import psutil
import torch
from tqdm import tqdm
import time
import json
import cv2
from torchvision import transforms
import threading

to_tensor = transforms.ToTensor()  # auto converts HWC uint8 -> CHW float32 in [0,1]

# Supported image extensions for dataset browsing
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# =========================================================================== #
# Model registry
# =========================================================================== #
# Each registered model is a dict with these keys:
#   - "name":   display name shown in the dropdown
#   - "load":   callable(device) -> nn.Module (or any object usable in `predict`)
#   - "predict":callable(model, chw_tensor_float01, device) -> HxW or CxHxW tensor
#               returning a depth / segmentation / image. Will be colorized for
#               display if single-channel.
#
# To add a new model, just append to MODEL_REGISTRY below.
# =========================================================================== #

def _load_unidepth_v2(device, backbone="vitl14"):
    """Loader for UniDepthV2.

    Tries (in order):
        1. The local clone at ./UniDepth (added to sys.path)
        2. The installed `unidepth` package
        3. torch.hub
    """
    # 1+2) try local clone / installed package
    local_clone = os.path.abspath("UniDepth")
    if os.path.isdir(local_clone) and local_clone not in sys.path:
        sys.path.insert(0, local_clone)

    try:
        from unidepth.models import UniDepthV2
        # Map backbone string to HF repo name used by the authors
        repo_map = {
            "vits14": "lpiccinelli/unidepth-v2-vits14",
            "vitb14": "lpiccinelli/unidepth-v2-vitb14",
            "vitl14": "lpiccinelli/unidepth-v2-vitl14",
        }
        repo = repo_map.get(backbone, repo_map["vitl14"])
        model = UniDepthV2.from_pretrained(repo)
    except Exception as e_local:
        # 3) fallback: torch.hub
        try:
            model = torch.hub.load(
                "lpiccinelli-eth/UniDepth",
                "UniDepth",
                version="v2", backbone=backbone,
                pretrained=True, trust_repo=True,
            )
        except Exception as e_hub:
            raise RuntimeError(
                f"UniDepthV2 load failed.\n  local: {e_local}\n  hub: {e_hub}"
            )

    model.to(device).eval()
    return model


@torch.no_grad()
def _predict_unidepth_v2(model, chw_float01, device):
    """UniDepthV2 inference. Returns a HxW float tensor (metric depth)."""
    # UniDepthV2 expects a uint8 CHW tensor on the same device as the model.
    # (It does the ImageNet-style normalization internally.)
    rgb_uint8 = (chw_float01.clamp(0, 1) * 255).to(torch.uint8).to(device)
    out = model.infer(rgb_uint8)  # dict with 'depth', 'points', 'intrinsics', ...
    depth = out["depth"]                                # 1x1xHxW or 1xHxW
    depth = depth.squeeze().detach().float().cpu()      # HxW
    return depth


MODEL_REGISTRY = [
    {
        "name":    "UniDepthV2 (ViT-L)",
        "load":    lambda dev: _load_unidepth_v2(dev, backbone="vitl14"),
        "predict": _predict_unidepth_v2,
    },
    {
        "name":    "UniDepthV2 (ViT-B)",
        "load":    lambda dev: _load_unidepth_v2(dev, backbone="vitb14"),
        "predict": _predict_unidepth_v2,
    },
    {
        "name":    "UniDepthV2 (ViT-S)",
        "load":    lambda dev: _load_unidepth_v2(dev, backbone="vits14"),
        "predict": _predict_unidepth_v2,
    },
]


# --------------------------------------------------------------------------- #
# DINOv3 backbones (loaded from local ./dinov3 clone, weights from
# ./checkpoints/dinov3/dinov3_vit{s,b,l}16_pretrain_lvd1689m*.pth)
# --------------------------------------------------------------------------- #
# ImageNet normalization stats for LVD-1689M pretraining
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Patch size for all ViT/16 variants
_DINO_PATCH = 16


def _find_dinov3_weights(variant):
    """Find a weight file for the given DINOv3 variant.

    Searches ./checkpoints/dinov3/ for files matching
    'dinov3_{variant}_pretrain_*.pth'. Returns the first match.
    """
    pattern = os.path.join("checkpoints", "dinov3",
                           f"dinov3_{variant}_pretrain_*.pth")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # Be lenient: also accept any pth in that folder containing the variant
        matches = sorted(glob.glob(
            os.path.join("checkpoints", "dinov3", f"*{variant}*.pth")))
    return matches[0] if matches else None


def _load_dinov3(device, variant="vits16"):
    """Load a DINOv3 backbone via torch.hub from the local ./dinov3 clone."""
    repo_dir = os.path.abspath("dinov3")
    if not os.path.isdir(repo_dir):
        raise RuntimeError(
            f"DINOv3 repo not found at {repo_dir}. "
            "Clone it into ./dinov3 (git clone https://github.com/facebookresearch/dinov3 dinov3).")

    weights = _find_dinov3_weights(variant)
    if weights is None:
        raise RuntimeError(
            f"No weights for dinov3_{variant} found in ./checkpoints/dinov3/. "
            f"Expected something like 'dinov3_{variant}_pretrain_lvd1689m-XXXXXXXX.pth'.")

    model = torch.hub.load(
        repo_dir, f"dinov3_{variant}",
        source="local", weights=weights,
    )
    model.to(device).eval()
    return model


def _pad_to_multiple(chw, m):
    """Pad CHW tensor on bottom/right so H,W are multiples of m. Returns (padded, (pad_h, pad_w))."""
    c, h, w = chw.shape
    pad_h = (m - h % m) % m
    pad_w = (m - w % m) % m
    if pad_h == 0 and pad_w == 0:
        return chw, (0, 0)
    return torch.nn.functional.pad(chw, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)


@torch.no_grad()
def _extract_dinov3_patches(model, chw_float01, device):
    """Run DINOv3 on a single image and return (patch_tokens, h_tok, w_tok, H, W, pad).

    patch_tokens: (N, D) float32 CPU tensor
    """
    c, H, W = chw_float01.shape
    x = chw_float01.float()
    x = (x - _DINO_MEAN) / _DINO_STD
    x, (pad_h, pad_w) = _pad_to_multiple(x, _DINO_PATCH)
    Hp, Wp = x.shape[1], x.shape[2]
    h_tok, w_tok = Hp // _DINO_PATCH, Wp // _DINO_PATCH

    x = x.unsqueeze(0).to(device)
    with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                        dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        feats_dict = model.forward_features(x)

    if isinstance(feats_dict, dict) and "x_norm_patchtokens" in feats_dict:
        patches = feats_dict["x_norm_patchtokens"]
    elif isinstance(feats_dict, dict) and "x_patchtokens" in feats_dict:
        patches = feats_dict["x_patchtokens"]
    else:
        patches = feats_dict if torch.is_tensor(feats_dict) else feats_dict[0]
    patches = patches.squeeze(0).float().cpu()             # (N, D)
    return patches, h_tok, w_tok, H, W, (pad_h, pad_w), (Hp, Wp)


def _project_to_panels(proj_n9, h_tok, w_tok, H, W, pad, Hp_Wp, basis):
    """Reshape (N, 9) projection -> upsampled 9xHxW image, then per-channel normalize.

    `basis` provides the lo/hi percentiles for normalization. If basis.lo_hi is None,
    falls back to per-image percentiles.
    """
    Hp, Wp = Hp_Wp
    pad_h, pad_w = pad
    feat_img = proj_n9.permute(1, 0).reshape(9, h_tok, w_tok)
    feat_img = torch.nn.functional.interpolate(
        feat_img.unsqueeze(0), size=(Hp, Wp),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    if pad_h or pad_w:
        feat_img = feat_img[:, :H, :W]

    out = torch.zeros_like(feat_img)
    if basis is not None and basis.lo_hi is not None:
        # Use the locked normalization ranges for cross-frame consistency
        for i in range(9):
            lo, hi = basis.lo_hi[i]
            if abs(hi - lo) < 1e-6:
                hi = lo + 1e-6
            out[i] = ((feat_img[i] - lo) / (hi - lo)).clamp(0, 1)
    else:
        for i in range(9):
            ch = feat_img[i]
            lo = torch.quantile(ch, 0.02)
            hi = torch.quantile(ch, 0.98)
            if (hi - lo).abs() < 1e-6:
                hi = lo + 1e-6
            out[i] = ((ch - lo) / (hi - lo)).clamp(0, 1)
    return out


class DINOPCABasis:
    """Stores a locked PCA basis fitted on a sample of images.

    Attributes:
        mean: (1, D) - feature mean used for centering
        V:    (D, 9) - top-9 principal components (orthonormal columns)
        lo_hi: list of 9 (lo, hi) float tuples for per-component normalization
        n_imgs: how many images contributed
        variant: 'vits16' / 'vitb16' / 'vitl16'
    """
    __slots__ = ("mean", "V", "lo_hi", "n_imgs", "variant")

    def __init__(self, mean, V, lo_hi, n_imgs, variant):
        self.mean = mean
        self.V = V
        self.lo_hi = lo_hi
        self.n_imgs = n_imgs
        self.variant = variant

    def save(self, path):
        torch.save({
            "mean":    self.mean,
            "V":       self.V,
            "lo_hi":   self.lo_hi,
            "n_imgs":  self.n_imgs,
            "variant": self.variant,
        }, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["mean"], d["V"], d["lo_hi"], d["n_imgs"], d["variant"])


@torch.no_grad()
def _fit_dinov3_basis(model, image_paths, device, variant,
                      n_sample=32, status_cb=None):
    """Sample images, run DINOv3, fit PCA basis (mean, V, normalization ranges).

    Args:
        model:        loaded DINOv3 backbone matching `variant`.
        image_paths:  list of full paths to images in the dataset.
        device:       torch device.
        variant:      'vits16' / 'vitb16' / 'vitl16'.
        n_sample:     number of images to sample (random, without replacement).
        status_cb:    optional callable(str) to report progress.

    Returns:
        DINOPCABasis instance.
    """
    if not image_paths:
        raise RuntimeError("No images to fit basis on.")
    n = min(n_sample, len(image_paths))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(image_paths), size=n, replace=False)

    feature_pool = []         # list of (Ni, D) tensors
    projected_for_ranges = [] # list of (Ni, 9) - filled after V is known

    for k, idx in enumerate(sample_idx):
        path = image_paths[int(idx)]
        if status_cb is not None:
            status_cb(f"Fitting basis: extracting {k+1}/{n}")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        chw = to_tensor(img)
        patches, *_ = _extract_dinov3_patches(model, chw, device)
        feature_pool.append(patches)

    if not feature_pool:
        raise RuntimeError("Failed to extract features from any sampled image.")

    if status_cb is not None:
        status_cb(f"Fitting basis: computing PCA on {len(feature_pool)} images...")

    all_feats = torch.cat(feature_pool, dim=0)        # (sum_Ni, D)
    mean = all_feats.mean(dim=0, keepdim=True)        # (1, D)
    centered = all_feats - mean
    # Same q=9 PCA, fitted once on the pooled feature set
    U, S, V = torch.pca_lowrank(centered, q=9, center=False)   # V: (D, 9)

    # Now project every sampled image's features through V to compute the
    # per-component percentile ranges across the whole sample.
    proj_all = centered @ V                            # (sum_Ni, 9)
    lo_hi = []
    for i in range(9):
        col = proj_all[:, i]
        lo = float(torch.quantile(col, 0.02))
        hi = float(torch.quantile(col, 0.98))
        lo_hi.append((lo, hi))

    if status_cb is not None:
        status_cb(f"Basis fitted on {len(feature_pool)} images.")

    return DINOPCABasis(mean=mean, V=V, lo_hi=lo_hi,
                        n_imgs=len(feature_pool), variant=variant)


def _make_dinov3_predict(get_basis):
    """Factory returning a predict() that uses get_basis() if available.

    get_basis() should return either a DINOPCABasis or None (per-image PCA).
    """
    @torch.no_grad()
    def _predict(model, chw_float01, device):
        basis = get_basis()
        patches, h_tok, w_tok, H, W, pad, hpwp = _extract_dinov3_patches(
            model, chw_float01, device)

        if basis is not None:
            # Locked basis: center with stored mean, project through stored V.
            centered = patches - basis.mean
            proj = centered @ basis.V                  # (N, 9)
        else:
            # Per-image PCA (original behavior)
            mean = patches.mean(dim=0, keepdim=True)
            centered = patches - mean
            U, S, V = torch.pca_lowrank(centered, q=9, center=False)
            proj = centered @ V

        return _project_to_panels(proj, h_tok, w_tok, H, W, pad, hpwp, basis)
    return _predict


# Module-level holder so the GUI can plug in a fitted basis without changing the
# registry contract. Keyed nothing — only one DINOv3 basis active at a time.
_DINOV3_BASIS = {"basis": None}


def _get_dinov3_basis():
    return _DINOV3_BASIS["basis"]


def _set_dinov3_basis(b):
    _DINOV3_BASIS["basis"] = b


# A single predict callable used by all three ViT variants. Which variant
# generated the basis (if locked) is enforced at fit time.
_dinov3_predict = _make_dinov3_predict(_get_dinov3_basis)


MODEL_REGISTRY.extend([
    {
        "name":    "DINOv3 PCA (ViT-S/16)",
        "load":    lambda dev: _load_dinov3(dev, variant="vits16"),
        "predict": _dinov3_predict,
        "variant": "vits16",
    },
    {
        "name":    "DINOv3 PCA (ViT-B/16)",
        "load":    lambda dev: _load_dinov3(dev, variant="vitb16"),
        "predict": _dinov3_predict,
        "variant": "vitb16",
    },
    {
        "name":    "DINOv3 PCA (ViT-L/16)",
        "load":    lambda dev: _load_dinov3(dev, variant="vitl16"),
        "predict": _dinov3_predict,
        "variant": "vitl16",
    },
])


# --------------------------------------------------------------------------- #
# Depth Anything 3 (single-image inference via the official Python API)
# --------------------------------------------------------------------------- #
# These entries are for the live "Predict" button. For consistent depth across
# 1500+ frames, use the "Consistent batch (DA3-Streaming)" panel which spawns
# the official streaming CLI as a subprocess.

def _load_da3(device, repo_id="depth-anything/DA3-Large"):
    """Load a Depth Anything 3 model via its `from_pretrained` API."""
    local_clone = os.path.abspath("Depth-Anything-3")
    if os.path.isdir(local_clone):
        src = os.path.join(local_clone, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
        if local_clone not in sys.path:
            sys.path.insert(0, local_clone)
    try:
        from depth_anything_3.api import DepthAnything3
    except Exception as e:
        raise RuntimeError(
            f"Could not import depth_anything_3. Clone the repo into "
            f"./Depth-Anything-3 or `pip install depth-anything-3`. ({e})")

    model = DepthAnything3.from_pretrained(repo_id)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def _predict_da3(model, chw_float01, device):
    """Single-image DA3 inference. Returns HxW float depth on CPU."""
    import tempfile
    H, W = chw_float01.shape[1], chw_float01.shape[2]
    rgb_u8 = (chw_float01.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, bgr)
        try:
            pred = model.inference(images=[tmp.name])
        finally:
            try: os.unlink(tmp.name)
            except Exception: pass

    depth = np.asarray(pred.depth)
    if depth.ndim == 3:
        depth = depth[0]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(depth).float()


MODEL_REGISTRY.extend([
    {
        "name":    "DA3 (Large) - single image",
        "load":    lambda dev: _load_da3(dev, "depth-anything/DA3-Large"),
        "predict": _predict_da3,
    },
    {
        "name":    "DA3 (Base) - single image",
        "load":    lambda dev: _load_da3(dev, "depth-anything/DA3-Base"),
        "predict": _predict_da3,
    },
    {
        "name":    "DA3 (Small) - single image",
        "load":    lambda dev: _load_da3(dev, "depth-anything/DA3-Small"),
        "predict": _predict_da3,
    },
    {
        "name":    "DA3 Metric (Large) - single image",
        "load":    lambda dev: _load_da3(dev, "depth-anything/DA3Metric-Large"),
        "predict": _predict_da3,
    },
])


def _colorize_depth(depth_hw, invert=True, cmap=cv2.COLORMAP_TURBO,
                    pct_lo=2.0, pct_hi=98.0):
    """Map a HxW float depth tensor to a CxHxW RGB tensor in [0,1].

    Args:
        depth_hw: HxW depth (torch tensor or numpy array). Near = small, far = big.
        invert:   If True, NEAR pixels are bright, FAR pixels are dark.
                  This matches conventional depth viz and avoids the foreground
                  going dark on cool-palette cmaps like TURBO/VIRIDIS.
        cmap:     Any cv2 COLORMAP_* constant. TURBO has good contrast end-to-end
                  and is the right default for depth.
        pct_lo, pct_hi: Percentile clipping bounds for robust normalization.

    Returns:
        CxHxW float32 tensor in [0,1].
    """
    d = depth_hw.numpy() if torch.is_tensor(depth_hw) else np.asarray(depth_hw)
    d = d.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)  # ignore invalid / zero (often "no data")
    if not valid.any():
        return torch.zeros(3, *d.shape, dtype=torch.float32)

    lo, hi = np.percentile(d[valid], [pct_lo, pct_hi])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    d_norm = np.clip((d - lo) / (hi - lo), 0, 1)
    if invert:
        d_norm = 1.0 - d_norm  # near = high value = bright end of colormap

    d_u8 = (d_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_u8, cmap)              # BGR HxWx3 uint8
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    chw = torch.from_numpy(colored).permute(2, 0, 1).float() / 255.0
    return chw


# --------------------------------------------------------------------------- #
# COLMAP binary readers + per-frame least-squares scale/shift refinement
# --------------------------------------------------------------------------- #
# Format reference: https://colmap.github.io/format.html
# We only read the fields we actually need (camera intrinsics, image poses,
# 2D-3D correspondences, 3D point coordinates). Code adapted from the standard
# colmap/scripts/python/read_write_model.py logic.

import struct as _struct


def _colmap_read_next_bytes(fid, num_bytes, format_char_sequence, endian="<"):
    data = fid.read(num_bytes)
    return _struct.unpack(endian + format_char_sequence, data)


def _colmap_read_cameras_bin(path):
    """Returns dict: camera_id -> (model, width, height, params [fx, fy, cx, cy, ...])."""
    cameras = {}
    with open(path, "rb") as f:
        num = _colmap_read_next_bytes(f, 8, "Q")[0]
        for _ in range(num):
            cam_id, model_id, w, h = _colmap_read_next_bytes(f, 24, "iiQQ")
            # Number of params depends on model; here we read until next camera
            # Model param counts (subset): SIMPLE_PINHOLE=3, PINHOLE=4, RADIAL=5,
            # OPENCV=8, OPENCV_FISHEYE=8, FULL_OPENCV=12, ...
            param_counts = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5,
                            8: 4, 9: 5, 10: 12}
            n_params = param_counts.get(model_id, 4)
            params = _colmap_read_next_bytes(f, 8 * n_params, "d" * n_params)
            cameras[cam_id] = {
                "model_id": model_id, "width": w, "height": h,
                "params": np.array(params, dtype=np.float64),
            }
    return cameras


def _colmap_qvec_to_R(qvec):
    """COLMAP stores quaternions as (w, x, y, z). Returns 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _colmap_read_images_bin(path):
    """Returns dict: image_id -> dict(name, R, t, point3D_ids [array of int])."""
    images = {}
    with open(path, "rb") as f:
        num = _colmap_read_next_bytes(f, 8, "Q")[0]
        for _ in range(num):
            img_id, *qt, cam_id = _colmap_read_next_bytes(f, 64, "idddddddi")
            qvec = np.array(qt[:4], dtype=np.float64)
            tvec = np.array(qt[4:7], dtype=np.float64)
            # Image name (null-terminated string)
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            n_pts2d = _colmap_read_next_bytes(f, 8, "Q")[0]
            # Each 2D point: (x, y, point3D_id). We only care about point3D_id.
            arr = _colmap_read_next_bytes(f, 24 * n_pts2d, "ddq" * n_pts2d) if n_pts2d else ()
            xys = np.array(arr, dtype=np.float64).reshape(-1, 3) if n_pts2d else np.zeros((0, 3))
            images[img_id] = {
                "name": name, "camera_id": cam_id,
                "R": _colmap_qvec_to_R(qvec), "t": tvec,
                "xys": xys[:, :2] if n_pts2d else np.zeros((0, 2)),
                "point3D_ids": xys[:, 2].astype(np.int64) if n_pts2d else np.zeros((0,), dtype=np.int64),
            }
    return images


def _colmap_read_points3d_bin(path):
    """Returns dict: point3D_id -> xyz (np.float64 length 3)."""
    pts = {}
    with open(path, "rb") as f:
        num = _colmap_read_next_bytes(f, 8, "Q")[0]
        for _ in range(num):
            pid, x, y, z, r, g, b, err = _colmap_read_next_bytes(f, 43, "QdddBBBd")
            n_track = _colmap_read_next_bytes(f, 8, "Q")[0]
            f.read(8 * n_track)   # skip track (image_id, point2D_idx) pairs
            pts[pid] = np.array([x, y, z], dtype=np.float64)
    return pts


def _project_points_to_frame(pts3d_world, R_wc, t_wc, K, W, H):
    """Project 3D world points into a camera, return Nx3 (u, v, z_cam) for valid ones."""
    cam = (R_wc @ pts3d_world.T).T + t_wc   # (N, 3) in camera frame
    z = cam[:, 2]
    valid = z > 1e-3
    cam = cam[valid]
    uv = cam[:, :2] / cam[:, 2:3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = uv[:, 0] * fx + cx
    v = uv[:, 1] * fy + cy
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack([u[inside], v[inside], cam[inside, 2]], axis=1)


def _intrinsic_matrix_from_colmap(cam):
    """Build a 3x3 K from a COLMAP camera record (PINHOLE / SIMPLE_PINHOLE)."""
    p = cam["params"]
    model = cam["model_id"]
    if model == 0:   # SIMPLE_PINHOLE: f, cx, cy
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif model == 1: # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        # Other models: first 4 params are usually fx, fy, cx, cy or close enough
        fx, fy, cx, cy = p[0], p[0] if len(p) < 4 else p[1], p[1 if model == 0 else 2], p[2 if model == 0 else 3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _fit_scale_shift_lsq(d_pred_vec, z_gt_vec):
    """Solve d_gt = s * d_pred + t in least squares. Returns (s, t)."""
    if len(d_pred_vec) < 2:
        return 1.0, 0.0
    A = np.stack([d_pred_vec, np.ones_like(d_pred_vec)], axis=1)
    sol, *_ = np.linalg.lstsq(A, z_gt_vec, rcond=None)
    return float(sol[0]), float(sol[1])


def refine_depths_with_colmap(predictions_dir, image_paths, colmap_sparse_dir,
                              status_cb=None):
    """For every .pt file in predictions_dir, fit a scale+shift using COLMAP
    sparse points visible in that frame, and overwrite the file.

    Args:
        predictions_dir: directory of *.pt files (one per source image stem).
        image_paths:     list of source image paths (used to match stems).
        colmap_sparse_dir: dir containing cameras.bin, images.bin, points3D.bin.
        status_cb:       optional callable(str) for progress messages.
    """
    cam_bin = os.path.join(colmap_sparse_dir, "cameras.bin")
    img_bin = os.path.join(colmap_sparse_dir, "images.bin")
    pts_bin = os.path.join(colmap_sparse_dir, "points3D.bin")
    for p in (cam_bin, img_bin, pts_bin):
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing COLMAP file: {p}")

    if status_cb: status_cb("Reading COLMAP cameras/images/points...")
    cameras = _colmap_read_cameras_bin(cam_bin)
    images = _colmap_read_images_bin(img_bin)
    points3d = _colmap_read_points3d_bin(pts_bin)

    # Build basename -> COLMAP image record
    name_to_record = {os.path.basename(rec["name"]): rec for rec in images.values()}

    # Build basename -> source image path
    name_to_path = {os.path.basename(p): p for p in image_paths}

    refined_ok = 0
    skipped = 0
    matched_files = sorted(glob.glob(os.path.join(predictions_dir, "*.pt")))
    for k, pt_path in enumerate(matched_files):
        stem = os.path.splitext(os.path.basename(pt_path))[0]
        # Find matching COLMAP record (try a few extensions)
        rec = None
        for ext in (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"):
            if (stem + ext) in name_to_record:
                rec = name_to_record[stem + ext]
                break
        if rec is None:
            skipped += 1
            continue

        if status_cb and k % 25 == 0:
            status_cb(f"Refining {k+1}/{len(matched_files)} ({refined_ok} ok)")

        K = _intrinsic_matrix_from_colmap(cameras[rec["camera_id"]])
        H_cam = cameras[rec["camera_id"]]["height"]
        W_cam = cameras[rec["camera_id"]]["width"]

        # Gather 3D points seen in this frame
        ids = rec["point3D_ids"]
        valid_ids = ids[ids >= 0]
        if len(valid_ids) < 8:
            skipped += 1
            continue
        pts3d = np.array([points3d[i] for i in valid_ids if i in points3d])
        if pts3d.shape[0] < 8:
            skipped += 1
            continue

        # Project, sample predicted depth at those pixels
        proj = _project_points_to_frame(pts3d, rec["R"], rec["t"], K, W_cam, H_cam)
        if proj.shape[0] < 8:
            skipped += 1
            continue

        depth = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(depth, dict):  # in case anyone stored a dict
            depth = depth.get("depth", depth)
        depth = depth.float()
        # Depth may have been saved at a different resolution than the COLMAP cam.
        # Resize proj coords to depth resolution if needed.
        dH, dW = depth.shape[-2:]
        if (dH, dW) != (H_cam, W_cam):
            proj[:, 0] *= dW / W_cam
            proj[:, 1] *= dH / H_cam

        u = np.clip(proj[:, 0].astype(int), 0, dW - 1)
        v = np.clip(proj[:, 1].astype(int), 0, dH - 1)
        d_pred = depth.numpy()[v, u].astype(np.float64)
        z_gt   = proj[:, 2]
        finite = np.isfinite(d_pred) & np.isfinite(z_gt) & (d_pred > 0)
        if finite.sum() < 8:
            skipped += 1
            continue
        s, t = _fit_scale_shift_lsq(d_pred[finite], z_gt[finite])
        refined = (depth.float() * s + t)
        torch.save(refined.contiguous(), pt_path)
        refined_ok += 1

    if status_cb:
        status_cb(f"Refinement done: {refined_ok} ok, {skipped} skipped.")
    return refined_ok, skipped


# --------------------------------------------------------------------------- #
# DA3-Streaming subprocess wrapper
# --------------------------------------------------------------------------- #
def _find_da3_streaming_script():
    """Locate da3_streaming.py from the local Depth-Anything-3 clone."""
    for cand in (
        os.path.join("Depth-Anything-3", "da3_streaming", "da3_streaming.py"),
        os.path.join("Depth-Anything-3", "da3_streaming.py"),
    ):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def run_da3_streaming(image_dir, out_dir, chunk_size=60, save_results=True,
                      python_exe=None, log_cb=None):
    """Launch DA3-Streaming as a subprocess. Returns the Popen object.

    DA3-Streaming's CLI only accepts --image_dir, --config, --output_dir.
    Knobs like chunk_size / overlap live in the YAML config (nested under
    `Model:`), so we load the default config, patch the relevant keys, write
    a temp YAML, and pass that via --config.
    """
    import subprocess, tempfile
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required. `pip install pyyaml`.")

    script = _find_da3_streaming_script()
    if script is None:
        raise RuntimeError(
            "Could not find da3_streaming.py. Clone the DA3 repo with "
            "--recursive into ./Depth-Anything-3 and ensure the da3_streaming/ "
            "subfolder exists.")

    streaming_dir = os.path.dirname(script)
    default_cfg = os.path.join(streaming_dir, "configs", "base_config.yaml")
    if not os.path.isfile(default_cfg):
        raise RuntimeError(f"DA3-Streaming default config not found at {default_cfg}")

    # Load default config as a dict
    with open(default_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides. The schema (as of Dec 2025) nests both chunk_size and
    # overlap under `Model:`. overlap MUST be strictly less than chunk_size.
    if "Model" not in cfg or not isinstance(cfg["Model"], dict):
        cfg["Model"] = {}
    if chunk_size is not None:
        cs = int(chunk_size)
        cfg["Model"]["chunk_size"] = cs
        cfg["Model"]["overlap"] = max(1, cs // 2)   # half, clamped > 0
    if save_results:
        cfg["Model"]["save_depth_conf_result"] = True

    # Sanity check: overlap < chunk_size or DA3-Streaming will refuse to run
    if cfg["Model"].get("overlap", 0) >= cfg["Model"].get("chunk_size", 1):
        cfg["Model"]["overlap"] = max(1, cfg["Model"]["chunk_size"] - 1)

    # Write the patched config to a temp file inside configs/ so any relative
    # paths inside the YAML still resolve from the streaming dir cwd.
    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        dir=os.path.join(streaming_dir, "configs"))
    yaml.safe_dump(cfg, tmp_cfg, sort_keys=False)
    tmp_cfg.close()
    tmp_cfg_path = tmp_cfg.name

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        python_exe or sys.executable, "-u", script,
        "--image_dir",  os.path.abspath(image_dir),
        "--output_dir", os.path.abspath(out_dir),
        "--config",     tmp_cfg_path,
    ]

    if log_cb:
        log_cb(f"Config: chunk_size={cfg['Model'].get('chunk_size')}, "
               f"overlap={cfg['Model'].get('overlap')}, "
               f"save_depth_conf_result={cfg['Model'].get('save_depth_conf_result')}")
        log_cb("Launching: " + " ".join(cmd))

    # Pass an env with expandable_segments enabled - reduces fragmentation-driven
    # OOMs in long-running CUDA processes. Preserve any pre-existing user setting.
    env = os.environ.copy()
    prev = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in prev:
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            (prev + ",") if prev else "") + "expandable_segments:True"

    proc = subprocess.Popen(
        cmd, cwd=streaming_dir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    proc._tmp_cfg_path = tmp_cfg_path
    return proc


def collect_da3_streaming_outputs(streaming_out_dir, predictions_dir,
                                  image_paths, save_fp16=False, status_cb=None):
    """Convert DA3-Streaming's `results_output/*.npz` files to per-image `.pt`s
    in `predictions_dir`, matching stems with `image_paths`.

    Each npz contains keys like 'depth', 'rgb', 'conf', 'intrinsic'.
    We save only depth (raw model output) so the rest of the pipeline (refine,
    display) treats it identically to single-image predictions.
    """
    results_dir = os.path.join(streaming_out_dir, "results_output")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"DA3-Streaming results_output not found at {results_dir}. "
                           "Did the streaming run complete? Check `save_depth_conf_result: True` in config.")
    npz_files = sorted(glob.glob(os.path.join(results_dir, "*.npz")))
    if not npz_files:
        raise RuntimeError(f"No .npz files in {results_dir}.")

    os.makedirs(predictions_dir, exist_ok=True)
    # Map stems
    stem_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    ok, fail = 0, 0
    for k, npz_path in enumerate(npz_files):
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        if stem not in stem_to_path:
            # Frame names may differ if DA3 renumbered them; fall back to index match
            pass
        try:
            data = np.load(npz_path)
            depth = data["depth"] if "depth" in data.files else data[data.files[0]]
            depth = np.asarray(depth, dtype=np.float32)
            t = torch.from_numpy(depth).contiguous()
            if save_fp16:
                t = t.to(torch.float16)
            out = os.path.join(predictions_dir, stem + ".pt")
            torch.save(t, out)
            ok += 1
        except Exception as e:
            fail += 1
            if status_cb: status_cb(f"Failed on {os.path.basename(npz_path)}: {e}")
        if status_cb and k % 25 == 0:
            status_cb(f"Collecting outputs: {k+1}/{len(npz_files)}")
    if status_cb:
        status_cb(f"Collected {ok} .pt files ({fail} failed) -> {predictions_dir}")
    return ok, fail


def _downscale_dataset(src_dir, target_width, status_cb=None):
    """Downscale all images in src_dir to target_width (preserving aspect).

    Writes to a sibling folder named <basename>_downscaled_<W>/ next to src_dir.
    Returns the path to the new folder. Idempotent: if the folder already
    exists with the same image count, it's reused.
    """
    src_dir = os.path.abspath(src_dir.rstrip(os.sep))
    parent = os.path.dirname(src_dir)
    base = os.path.basename(src_dir)
    out_dir = os.path.join(parent, f"{base}_downscaled_{target_width}")

    src_files = []
    for ext in IMAGE_EXTS:
        src_files.extend(glob.glob(os.path.join(src_dir, f"*{ext}")))
    src_files.sort()
    if not src_files:
        raise RuntimeError(f"No images found in {src_dir}")

    # Reuse if already done
    if os.path.isdir(out_dir):
        existing = sum(
            len(glob.glob(os.path.join(out_dir, f"*{ext}"))) for ext in IMAGE_EXTS)
        if existing == len(src_files):
            if status_cb: status_cb(f"Reusing existing downscaled dir ({existing} images)")
            return out_dir

    os.makedirs(out_dir, exist_ok=True)
    for k, p in enumerate(src_files):
        if status_cb and k % 50 == 0:
            status_cb(f"Downscaling {k+1}/{len(src_files)} -> width={target_width}")
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w == target_width:
            cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), img)
            continue
        new_h = int(round(h * target_width / w))
        # Ensure even dimensions (some codecs / models care)
        new_h -= new_h % 2
        resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), resized)
    if status_cb:
        status_cb(f"Downscaled {len(src_files)} images -> {out_dir}")
    return out_dir


class GUIBase:
    """DPG visualization with a 720p main viewer, a control panel, and 3 side panels.

    Layout:
        - Left:  1280 x 720 main image viewer (current dataset image)
        - Mid:   400  x 720 control panel
        - Right: three stacked 512 x 240 panels (top = model prediction, others reserved)

    Features:
        - Text-input dataset path; browse through images with prev/next
        - Dropdown to pick a model from MODEL_REGISTRY
        - Synchronized mouse-wheel zoom across all 4 image panels
    """

    # ------------------------------------------------------------------ #
    # Layout constants
    # ------------------------------------------------------------------ #
    MAIN_W, MAIN_H = 1280, 720          # 720p left viewer
    CTRL_W = 400                        # middle control panel width
    SIDE_W, SIDE_H = 512, 240           # each right-side panel (3 * 240 = 720)

    def __init__(self, runname="viewer"):
        self.gui = True
        self.runname = runname

        # ------------------------------------------------------------------ #
        # Image buffers (float32 RGB in [0,1], HWC)
        # ------------------------------------------------------------------ #
        self.buffer_main  = np.ones((self.MAIN_H, self.MAIN_W, 3), dtype=np.float32)
        self.buffer_pred  = np.zeros((self.SIDE_H, self.SIDE_W, 3), dtype=np.float32)
        self.buffer_aux1  = np.zeros((self.SIDE_H, self.SIDE_W, 3), dtype=np.float32)
        self.buffer_aux2  = np.zeros((self.SIDE_H, self.SIDE_W, 3), dtype=np.float32)

        # ------------------------------------------------------------------ #
        # Dataset state
        # ------------------------------------------------------------------ #
        self.dataset_path = ""
        self.image_paths = []
        self.current_idx = 0
        self.current_image = None       # CHW float tensor [0,1]
        self.prediction_image = None    # CHW float tensor [0,1] (top panel)
        self.prediction_image_2 = None  # CHW float tensor [0,1] (mid panel, PC 4-6)
        self.prediction_image_3 = None  # CHW float tensor [0,1] (bot panel, PC 7-9)

        # ------------------------------------------------------------------ #
        # Model state (registry-driven)
        # ------------------------------------------------------------------ #
        self.model_names = [m["name"] for m in MODEL_REGISTRY] or ["(none)"]
        self.selected_model_name = self.model_names[0]
        self.model = None
        self.model_entry = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------ #
        # Zoom state (synchronized across all panels)
        # ------------------------------------------------------------------ #
        self.zoom = 1.0
        self.zoom_min = 1.0
        self.zoom_max = 20.0
        self.zoom_center = [0.5, 0.5]   # (u, v) in [0,1]

        # ------------------------------------------------------------------ #
        # Drag state (left-mouse drag to pan)
        # ------------------------------------------------------------------ #
        self._drag_active = False
        self._drag_panel_tag = None     # which image item the drag started on
        self._drag_panel_size = (1, 1)  # rect size of that panel at drag start
        self._drag_start_mouse = (0, 0)
        self._drag_start_center = [0.5, 0.5]

        # ------------------------------------------------------------------ #
        # Save options
        # ------------------------------------------------------------------ #
        self.save_fp16 = False

        # ------------------------------------------------------------------ #
        # Consistent batch (DA3-Streaming) state
        # ------------------------------------------------------------------ #
        self._da3stream_proc = None      # active subprocess.Popen, if any
        self._da3stream_thread = None    # log-reader thread
        self._da3stream_out_dir = None   # tmp output dir for the run
        self._da3stream_log_lines = []   # rolling log buffer (last N)

        # ------------------------------------------------------------------ #
        # Misc
        # ------------------------------------------------------------------ #
        self.save_frame = False
        self.mous_loc = [0, 0]
        self._dirty = True

        print("DPG initializing ...")
        dpg.create_context()
        self.register_dpg()

    def __del__(self):
        if self.gui and dpg is not None:
            try:
                dpg.destroy_context()
            except Exception:
                pass

    # ====================================================================== #
    # Model handling
    # ====================================================================== #
    def _find_entry(self, name):
        for m in MODEL_REGISTRY:
            if m["name"] == name:
                return m
        return None

    def load_model(self, model_name):
        entry = self._find_entry(model_name)
        if entry is None:
            dpg.set_value("_log_model_status", f"Unknown model: {model_name}")
            return
        dpg.set_value("_log_model_status", f"Loading '{model_name}' ...")
        try:
            self.model = entry["load"](self.device)
            self.model_entry = entry
            dpg.set_value("_log_model_status", f"Loaded '{model_name}'")
        except Exception as e:
            self.model = None
            self.model_entry = None
            dpg.set_value("_log_model_status", f"Load failed: {e}")
            print(f"[load_model] {e}")
            return

        # If a locked PCA basis exists from a different variant, clear it so
        # we don't crash on a feature-dim mismatch on the next predict.
        b = _get_dinov3_basis()
        if b is not None and b.variant != entry.get("variant"):
            _set_dinov3_basis(None)
            dpg.set_value("_log_basis_status",
                          f"basis cleared (was for {b.variant}, model is {entry.get('variant')})")
        # Always sync the status indicator with reality
        if dpg.does_item_exist("_log_basis_status"):
            self._update_basis_status()

    @torch.no_grad()
    def run_prediction(self):
        if self.model is None or self.model_entry is None:
            dpg.set_value("_log_model_status", "No model loaded.")
            return
        if self.current_image is None:
            dpg.set_value("_log_model_status", "No image loaded.")
            return
        try:
            y = self.model_entry["predict"](self.model, self.current_image, self.device)
            # Reset all three panels first (so a depth model clears stale PCA viz)
            self.prediction_image_2 = None
            self.prediction_image_3 = None
            # Normalize / colorize for display
            if torch.is_tensor(y) and y.ndim == 2:
                # Single-channel output (depth, etc.): colorize into top panel only
                self.prediction_image = _colorize_depth(y)
            elif torch.is_tensor(y) and y.ndim == 3:
                y = y.float().clamp(0, 1)
                if y.shape[0] == 1:
                    self.prediction_image = y.repeat(3, 1, 1)
                elif y.shape[0] == 9:
                    # Split into PC 1-3, 4-6, 7-9 across the three side panels
                    self.prediction_image   = y[0:3]
                    self.prediction_image_2 = y[3:6]
                    self.prediction_image_3 = y[6:9]
                else:
                    # Other multi-channel: clip to first 3
                    self.prediction_image = y[:3]
            else:
                raise ValueError(f"Unexpected prediction shape: {tuple(y.shape)}")
            self._dirty = True
            dpg.set_value("_log_model_status", "Prediction OK")
        except Exception as e:
            dpg.set_value("_log_model_status", f"Inference failed: {e}")
            print(f"[run_prediction] {e}")

    @torch.no_grad()
    def process_all(self):
        """Run the loaded model on every image in the dataset and save raw outputs.

        Outputs go to <parent_of_dataset_dir>/predictions/ as .pt files
        (same stem as the source image). Raw CPU tensor is saved — NOT the
        colorized display version. Use `torch.load(path, map_location='cuda')`
        to bring it straight onto the GPU.

        If `self.save_fp16` is True, tensors are cast to float16 before saving
        (~2x smaller, ~2x faster host->device transfer).
        """
        if self.model is None or self.model_entry is None:
            dpg.set_value("_log_model_status", "No model loaded.")
            return
        if not self.image_paths:
            dpg.set_value("_log_model_status", "No dataset loaded.")
            return

        # Output dir: parent of the dataset dir / predictions
        dataset_dir = os.path.abspath(self.dataset_path).rstrip(os.sep)
        parent_dir = os.path.dirname(dataset_dir)
        out_dir = os.path.join(parent_dir, "predictions")
        os.makedirs(out_dir, exist_ok=True)
        dtype_str = "fp16" if self.save_fp16 else "fp32"
        print(f"[process_all] Saving to {out_dir} as .pt ({dtype_str})")

        n = len(self.image_paths)
        ok = 0
        fail = 0
        t_start = time.time()

        for i, path in enumerate(self.image_paths):
            dpg.set_value("_log_model_status",
                          f"Processing {i+1}/{n}: {os.path.basename(path)}")
            try:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise IOError("cv2.imread returned None")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                chw = to_tensor(img)
                y = self.model_entry["predict"](self.model, chw, self.device)
                if not torch.is_tensor(y):
                    y = torch.as_tensor(y)
                tensor = y.detach().cpu().contiguous()
                if self.save_fp16:
                    tensor = tensor.to(torch.float16)
                stem = os.path.splitext(os.path.basename(path))[0]
                torch.save(tensor, os.path.join(out_dir, stem + ".pt"))
                ok += 1
            except Exception as e:
                print(f"[process_all] {path}: {e}")
                fail += 1

        dt = time.time() - t_start
        msg = f"Done: {ok} saved, {fail} failed, {dt:.1f}s ({dtype_str}) -> {out_dir}"
        print(f"[process_all] {msg}")
        dpg.set_value("_log_model_status", msg)

    # ====================================================================== #
    # Dataset handling
    # ====================================================================== #
    def load_dataset(self, path):
        self.dataset_path = path
        if not os.path.isdir(path):
            self.image_paths = []
            dpg.set_value("_log_dataset_status", f"Not a directory: {path}")
            return

        files = []
        for ext in IMAGE_EXTS:
            files.extend(glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True))
        files = sorted(files)
        self.image_paths = files
        self.current_idx = 0

        if files:
            dpg.set_value("_log_dataset_status", f"Found {len(files)} images")
            self.load_current_image()
        else:
            dpg.set_value("_log_dataset_status", "No images found")

    def load_current_image(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            dpg.set_value("_log_image_info", f"Failed to read: {path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_image = to_tensor(img)
        self.prediction_image = None
        self.prediction_image_2 = None
        self.prediction_image_3 = None
        self._dirty = True
        dpg.set_value("_log_image_info",
                      f"[{self.current_idx + 1}/{len(self.image_paths)}] "
                      f"{os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    def next_image(self):
        if not self.image_paths:
            return
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self.load_current_image()

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_idx = (self.current_idx - 1) % len(self.image_paths)
        self.load_current_image()

    # ====================================================================== #
    # DPG callbacks
    # ====================================================================== #
    def _cb_load_dataset(self, sender, app_data):
        self.load_dataset(dpg.get_value("_input_dataset_path"))

    def _cb_prev(self, sender, app_data): self.prev_image()
    def _cb_next(self, sender, app_data): self.next_image()

    def _cb_model_changed(self, sender, app_data):
        self.selected_model_name = app_data

    def _cb_load_model(self, sender, app_data):
        self.load_model(self.selected_model_name)

    def _cb_unload_model(self, sender, app_data):
        """Drop the loaded model and free its GPU memory.

        Useful before running DA3-Streaming so the subprocess gets the full
        VRAM budget instead of competing with model weights still resident
        from the live Predict button.
        """
        if self.model is None:
            dpg.set_value("_log_model_status", "No model loaded.")
            return
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        self.model_entry = None
        # Best-effort VRAM cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        dpg.set_value("_log_model_status", "Unloaded model; VRAM freed.")
        print("[gui] Unloaded model, called torch.cuda.empty_cache()", flush=True)

    def _cb_predict(self, sender, app_data):
        self.run_prediction()

    def _cb_reset_zoom(self, sender, app_data):
        self.zoom = 1.0
        self.zoom_center = [0.5, 0.5]
        self._dirty = True

    def _cb_mouse_wheel(self, sender, app_data):
        for tag in ("_img_main", "_img_pred", "_img_aux1", "_img_aux2"):
            if not dpg.does_item_exist(tag):
                continue
            if dpg.is_item_hovered(tag):
                rect_min = dpg.get_item_rect_min(tag)
                rect_size = dpg.get_item_rect_size(tag)
                mx, my = dpg.get_mouse_pos(local=False)
                u = (mx - rect_min[0]) / max(rect_size[0], 1)
                v = (my - rect_min[1]) / max(rect_size[1], 1)
                u = float(np.clip(u, 0.0, 1.0))
                v = float(np.clip(v, 0.0, 1.0))

                half = 0.5 / self.zoom
                cx, cy = self.zoom_center
                img_u = (cx - half) + u * (2 * half)
                img_v = (cy - half) + v * (2 * half)

                factor = 1.2 if app_data > 0 else 1 / 1.2
                self.zoom = float(np.clip(self.zoom * factor, self.zoom_min, self.zoom_max))

                half = 0.5 / self.zoom
                self.zoom_center = [
                    float(np.clip(img_u, half, 1 - half)),
                    float(np.clip(img_v, half, 1 - half)),
                ]
                self._dirty = True
                break

    def _cb_mouse_down(self, sender, app_data):
        """Begin a drag on left-button press over any image panel."""
        # app_data for mouse_down handler is [button, duration]; button 0 = left
        button = app_data[0] if isinstance(app_data, (list, tuple)) else app_data
        if button != 0:
            return
        if self._drag_active:
            return
        for tag in ("_img_main", "_img_pred", "_img_aux1", "_img_aux2"):
            if not dpg.does_item_exist(tag):
                continue
            if dpg.is_item_hovered(tag):
                self._drag_active = True
                self._drag_panel_tag = tag
                self._drag_panel_size = tuple(dpg.get_item_rect_size(tag))
                self._drag_start_mouse = tuple(dpg.get_mouse_pos(local=False))
                self._drag_start_center = list(self.zoom_center)
                break

    def _cb_mouse_drag(self, sender, app_data):
        """Pan the zoom window while the left button is held."""
        if not self._drag_active:
            return
        # No need to recompute hover; we track the panel from mouse-down.
        mx, my = dpg.get_mouse_pos(local=False)
        sx, sy = self._drag_start_mouse
        pw, ph = self._drag_panel_size
        # Pixels moved -> fraction of the panel -> fraction of the zoom window
        # (zoom window width in image-space coords is 1/zoom).
        du_screen = (mx - sx) / max(pw, 1)
        dv_screen = (my - sy) / max(ph, 1)
        # Dragging right should move the view content right, which means the
        # zoom center moves LEFT in image coords -> negate.
        du_img = -du_screen / self.zoom
        dv_img = -dv_screen / self.zoom

        half = 0.5 / self.zoom
        new_cx = self._drag_start_center[0] + du_img
        new_cy = self._drag_start_center[1] + dv_img
        self.zoom_center = [
            float(np.clip(new_cx, half, 1 - half)),
            float(np.clip(new_cy, half, 1 - half)),
        ]
        self._dirty = True

    def _cb_mouse_release(self, sender, app_data):
        button = app_data if not isinstance(app_data, (list, tuple)) else app_data[0]
        if button == 0:
            self._drag_active = False
            self._drag_panel_tag = None

    def _cb_process_all(self, sender, app_data):
        self.process_all()

    def _cb_toggle_fp16(self, sender, app_data):
        self.save_fp16 = not self.save_fp16
        label = "FP16: ON" if self.save_fp16 else "FP16: OFF"
        dpg.set_item_label("_btn_fp16", label)
        theme = "_theme_btn_on" if self.save_fp16 else "_theme_btn_off"
        dpg.bind_item_theme("_btn_fp16", theme)

    # -------------- DINOv3 PCA basis callbacks --------------
    def _is_dinov3_loaded(self):
        return (self.model is not None and self.model_entry is not None
                and self.model_entry.get("variant") in ("vits16", "vitb16", "vitl16"))

    def _update_basis_status(self):
        b = _get_dinov3_basis()
        if b is None:
            dpg.set_value("_log_basis_status", "basis: per-image (not locked)")
            dpg.bind_item_theme("_btn_fit_basis", "_theme_btn_off")
        else:
            dpg.set_value("_log_basis_status",
                          f"basis: LOCKED on {b.n_imgs} imgs ({b.variant})")
            dpg.bind_item_theme("_btn_fit_basis", "_theme_btn_on")

    def _cb_fit_basis(self, sender, app_data):
        if not self._is_dinov3_loaded():
            dpg.set_value("_log_basis_status", "Load a DINOv3 model first.")
            return
        if not self.image_paths:
            dpg.set_value("_log_basis_status", "Load a dataset first.")
            return
        n = int(dpg.get_value("_input_basis_sample"))
        variant = self.model_entry["variant"]
        try:
            basis = _fit_dinov3_basis(
                self.model, self.image_paths, self.device,
                variant=variant, n_sample=n,
                status_cb=lambda m: dpg.set_value("_log_basis_status", m),
            )
            _set_dinov3_basis(basis)
            self._dirty = True
        except Exception as e:
            dpg.set_value("_log_basis_status", f"Fit failed: {e}")
            print(f"[fit_basis] {e}")
            return
        self._update_basis_status()

    def _cb_clear_basis(self, sender, app_data):
        _set_dinov3_basis(None)
        self._dirty = True
        self._update_basis_status()

    def _cb_save_basis(self, sender, app_data):
        b = _get_dinov3_basis()
        if b is None:
            dpg.set_value("_log_basis_status", "No basis to save.")
            return
        # Save next to the dataset for portability
        if self.dataset_path and os.path.isdir(self.dataset_path):
            parent = os.path.dirname(os.path.abspath(self.dataset_path).rstrip(os.sep))
            out = os.path.join(parent, f"dinov3_pca_basis_{b.variant}.pt")
        else:
            out = f"dinov3_pca_basis_{b.variant}.pt"
        try:
            b.save(out)
            dpg.set_value("_log_basis_status", f"Saved -> {out}")
        except Exception as e:
            dpg.set_value("_log_basis_status", f"Save failed: {e}")

    def _cb_load_basis(self, sender, app_data):
        # Look for a matching basis file next to the dataset; fall back to cwd.
        variant = self.model_entry["variant"] if self._is_dinov3_loaded() else None
        candidates = []
        if self.dataset_path and os.path.isdir(self.dataset_path):
            parent = os.path.dirname(os.path.abspath(self.dataset_path).rstrip(os.sep))
            if variant:
                candidates.append(os.path.join(parent, f"dinov3_pca_basis_{variant}.pt"))
            candidates.extend(sorted(glob.glob(os.path.join(parent, "dinov3_pca_basis_*.pt"))))
        if variant:
            candidates.append(f"dinov3_pca_basis_{variant}.pt")
        candidates.extend(sorted(glob.glob("dinov3_pca_basis_*.pt")))

        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            dpg.set_value("_log_basis_status",
                          "No basis file found (expected dinov3_pca_basis_*.pt).")
            return
        try:
            b = DINOPCABasis.load(path)
            if variant and b.variant != variant:
                dpg.set_value("_log_basis_status",
                              f"Basis variant '{b.variant}' != loaded model '{variant}'.")
                return
            _set_dinov3_basis(b)
            self._dirty = True
            dpg.set_value("_log_basis_status", f"Loaded <- {os.path.basename(path)}")
        except Exception as e:
            dpg.set_value("_log_basis_status", f"Load failed: {e}")

    # -------------- Consistent batch (DA3-Streaming) callbacks --------------
    def _da3stream_set_status(self, msg):
        """Print full log to terminal, show last line in GUI status."""
        msg = (msg or "").rstrip()
        if not msg:
            return
        # Full message to terminal (flushed so it appears in real time)
        print(f"[da3stream] {msg}", flush=True)
        # GUI shows only a truncated single-line status
        gui_msg = msg if len(msg) <= 90 else msg[:87] + "..."
        if dpg.does_item_exist("_log_da3stream_status"):
            dpg.set_value("_log_da3stream_status", gui_msg)

    def _da3stream_log_reader(self, proc):
        """Background thread: read subprocess stdout line by line and post to GUI."""
        try:
            for line in proc.stdout:
                self._da3stream_set_status(line.rstrip())
        except Exception as e:
            self._da3stream_set_status(f"Log reader error: {e}")
        proc.wait()
        rc = proc.returncode
        # Clean up the temp YAML config we wrote
        tmp_cfg = getattr(proc, "_tmp_cfg_path", None)
        if tmp_cfg and os.path.isfile(tmp_cfg):
            try: os.unlink(tmp_cfg)
            except Exception: pass
        self._da3stream_set_status(f"Subprocess exited with code {rc}")
        # If it succeeded, collect outputs into predictions/ and optionally refine.
        if rc == 0 and self._da3stream_out_dir is not None:
            try:
                self._da3stream_collect_and_refine()
            except Exception as e:
                self._da3stream_set_status(f"Post-processing failed: {e}")

    def _da3stream_collect_and_refine(self):
        if not self.image_paths or not self.dataset_path:
            self._da3stream_set_status("No dataset to map outputs to; skipping.")
            return
        # Output dir convention: predictions/ next to dataset
        dataset_dir = os.path.abspath(self.dataset_path).rstrip(os.sep)
        parent_dir = os.path.dirname(dataset_dir)
        predictions_dir = os.path.join(parent_dir, "predictions")

        collect_da3_streaming_outputs(
            self._da3stream_out_dir, predictions_dir, self.image_paths,
            save_fp16=self.save_fp16,
            status_cb=self._da3stream_set_status,
        )

        # Optionally refine using COLMAP
        if dpg.get_value("_chk_use_colmap"):
            sparse_dir = dpg.get_value("_input_colmap_path").strip()
            if not sparse_dir:
                self._da3stream_set_status("COLMAP refinement on but no path given.")
                return
            try:
                refine_depths_with_colmap(
                    predictions_dir, self.image_paths, sparse_dir,
                    status_cb=self._da3stream_set_status,
                )
            except Exception as e:
                self._da3stream_set_status(f"COLMAP refine failed: {e}")

    def _cb_run_consistent_batch(self, sender, app_data):
        if self._da3stream_proc is not None and self._da3stream_proc.poll() is None:
            self._da3stream_set_status("A run is already in progress.")
            return
        if not self.dataset_path or not os.path.isdir(self.dataset_path):
            self._da3stream_set_status("Load a dataset first.")
            return

        chunk = int(dpg.get_value("_slider_chunk_size"))
        ds_w = int(dpg.get_value("_input_downscale_width"))

        # Optionally pre-downscale the dataset, then point DA3 at the new folder
        image_dir_for_da3 = self.dataset_path
        if ds_w > 0:
            try:
                image_dir_for_da3 = _downscale_dataset(
                    self.dataset_path, ds_w,
                    status_cb=self._da3stream_set_status,
                )
            except Exception as e:
                self._da3stream_set_status(f"Downscale failed: {e}")
                return

        # Output to a temp dir under the (possibly downscaled) dataset's parent
        parent = os.path.dirname(os.path.abspath(image_dir_for_da3).rstrip(os.sep))
        out_dir = os.path.join(parent, "da3_streaming_out")
        self._da3stream_out_dir = out_dir
        # Stash the actual dir we fed to DA3 for the collect step
        self._da3stream_input_dir = image_dir_for_da3
        self._da3stream_log_lines = []
        self._da3stream_set_status(
            f"Starting DA3-Streaming (chunk={chunk}, downscale_w={ds_w if ds_w > 0 else 'off'})...")

        try:
            proc = run_da3_streaming(
                image_dir=image_dir_for_da3,
                out_dir=out_dir,
                chunk_size=chunk,
                log_cb=self._da3stream_set_status,
            )
        except Exception as e:
            self._da3stream_set_status(f"Launch failed: {e}")
            return

        self._da3stream_proc = proc
        self._da3stream_thread = threading.Thread(
            target=self._da3stream_log_reader, args=(proc,), daemon=True)
        self._da3stream_thread.start()

    def _cb_stop_consistent_batch(self, sender, app_data):
        proc = self._da3stream_proc
        if proc is None or proc.poll() is not None:
            self._da3stream_set_status("No active run.")
            return
        try:
            proc.terminate()
            self._da3stream_set_status("Sent terminate signal.")
        except Exception as e:
            self._da3stream_set_status(f"Terminate failed: {e}")

    def _cb_refine_only(self, sender, app_data):
        """Run COLMAP refinement on existing predictions/ without re-running DA3."""
        if not self.dataset_path or not os.path.isdir(self.dataset_path):
            self._da3stream_set_status("Load a dataset first.")
            return
        sparse_dir = dpg.get_value("_input_colmap_path").strip()
        if not sparse_dir:
            self._da3stream_set_status("Set COLMAP sparse/0 path first.")
            return
        parent = os.path.dirname(os.path.abspath(self.dataset_path).rstrip(os.sep))
        predictions_dir = os.path.join(parent, "predictions")
        if not os.path.isdir(predictions_dir):
            self._da3stream_set_status(f"No predictions/ found at {predictions_dir}.")
            return
        # Run in a thread so the GUI stays responsive
        def _go():
            try:
                refine_depths_with_colmap(
                    predictions_dir, self.image_paths, sparse_dir,
                    status_cb=self._da3stream_set_status,
                )
            except Exception as e:
                self._da3stream_set_status(f"Refine failed: {e}")
        threading.Thread(target=_go, daemon=True).start()

    # ====================================================================== #
    # Rendering
    # ====================================================================== #
    def _apply_zoom(self, chw_tensor, out_h, out_w):
        if chw_tensor is None:
            return np.zeros((out_h, out_w, 3), dtype=np.float32)

        c, h, w = chw_tensor.shape
        half = 0.5 / self.zoom
        cu, cv = self.zoom_center
        u0 = int(np.clip((cu - half) * w, 0, w - 1))
        u1 = int(np.clip((cu + half) * w, u0 + 1, w))
        v0 = int(np.clip((cv - half) * h, 0, h - 1))
        v1 = int(np.clip((cv + half) * h, v0 + 1, h))

        crop = chw_tensor[:, v0:v1, u0:u1].unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            crop, size=(out_h, out_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        return (resized.permute(1, 2, 0).clamp(0, 1).contiguous()
                .detach().cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def viewer_step(self):
        if not self._dirty:
            return
        t0 = time.time()

        self.buffer_main[:] = self._apply_zoom(
            self.current_image, self.MAIN_H, self.MAIN_W)
        dpg.set_value("_tex_main", self.buffer_main)

        self.buffer_pred[:] = self._apply_zoom(
            self.prediction_image, self.SIDE_H, self.SIDE_W)
        dpg.set_value("_tex_pred", self.buffer_pred)

        self.buffer_aux1[:] = self._apply_zoom(
            self.prediction_image_2, self.SIDE_H, self.SIDE_W)
        dpg.set_value("_tex_aux1", self.buffer_aux1)

        self.buffer_aux2[:] = self._apply_zoom(
            self.prediction_image_3, self.SIDE_H, self.SIDE_W)
        dpg.set_value("_tex_aux2", self.buffer_aux2)

        t1 = time.time()
        dt = t1 - t0
        if dt > 0:
            dpg.set_value("_log_infer_time", f"{1.0/dt:.1f} fps")
        dpg.set_value("_log_zoom",
                      f"{self.zoom:.2f}x  ({self.zoom_center[0]:.2f}, {self.zoom_center[1]:.2f})")
        self._dirty = False

    def run(self):
        while dpg.is_dearpygui_running():
            with torch.no_grad():
                self.viewer_step()
                dpg.render_dearpygui_frame()
        dpg.destroy_context()

    # ====================================================================== #
    # DPG registration
    # ====================================================================== #
    def register_dpg(self):
        # ---------- toggle button themes (used by FP16 toggle) ----------
        with dpg.theme(tag="_theme_btn_on"):
            with dpg.theme_component(dpg.mvButton):
                # green when active
                dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 140, 60))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55, 170, 75))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (30, 110, 50))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
        with dpg.theme(tag="_theme_btn_off"):
            with dpg.theme_component(dpg.mvButton):
                # subdued gray when inactive
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 60))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (85, 85, 85))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (45, 45, 45))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (200, 200, 200))

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.MAIN_W, self.MAIN_H, self.buffer_main,
                                format=dpg.mvFormat_Float_rgb, tag="_tex_main")
            dpg.add_raw_texture(self.SIDE_W, self.SIDE_H, self.buffer_pred,
                                format=dpg.mvFormat_Float_rgb, tag="_tex_pred")
            dpg.add_raw_texture(self.SIDE_W, self.SIDE_H, self.buffer_aux1,
                                format=dpg.mvFormat_Float_rgb, tag="_tex_aux1")
            dpg.add_raw_texture(self.SIDE_W, self.SIDE_H, self.buffer_aux2,
                                format=dpg.mvFormat_Float_rgb, tag="_tex_aux2")

        with dpg.window(tag="_primary_window",
                        width=self.MAIN_W, height=self.MAIN_H,
                        pos=[0, 0],
                        no_move=True, no_title_bar=True, no_scrollbar=True):
            dpg.add_image("_tex_main", tag="_img_main")

        with dpg.window(label="Control", tag="_control_window",
                        width=self.CTRL_W, height=self.MAIN_H,
                        pos=[self.MAIN_W, 0],
                        no_move=True, no_title_bar=True):

            dpg.add_text("Dataset")
            dpg.add_input_text(tag="_input_dataset_path",
                               default_value="", width=-1,
                               hint="path to image folder")
            dpg.add_button(label="Load dataset", width=-1,
                           callback=self._cb_load_dataset)
            dpg.add_text("(no dataset loaded)", tag="_log_dataset_status")
            dpg.add_separator()

            dpg.add_text("Browse")
            with dpg.group(horizontal=True):
                dpg.add_button(label="< Prev", width=120, callback=self._cb_prev)
                dpg.add_button(label="Next >", width=120, callback=self._cb_next)
            dpg.add_text("(no image)", tag="_log_image_info")
            dpg.add_separator()

            dpg.add_text("Model")
            dpg.add_combo(items=self.model_names,
                          default_value=self.selected_model_name,
                          tag="_combo_model", width=-1,
                          callback=self._cb_model_changed)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load model", width=120,
                               callback=self._cb_load_model)
                dpg.add_button(label="Predict", width=120,
                               callback=self._cb_predict)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Unload model (free VRAM)", width=-1,
                               callback=self._cb_unload_model)
            dpg.add_button(label="Process all -> predictions/",
                           width=-1, callback=self._cb_process_all)
            with dpg.group(horizontal=True):
                dpg.add_text("Save dtype: ")
                dpg.add_button(label="FP16: OFF", tag="_btn_fp16",
                               width=120, callback=self._cb_toggle_fp16)
            dpg.bind_item_theme("_btn_fp16", "_theme_btn_off")
            dpg.add_text("(no model loaded)", tag="_log_model_status")
            dpg.add_separator()

            # ---------------- DINOv3 PCA basis controls ----------------
            dpg.add_text("DINOv3 PCA basis")
            with dpg.group(horizontal=True):
                dpg.add_text("Sample size:")
                dpg.add_input_int(tag="_input_basis_sample", default_value=32,
                                  min_value=2, max_value=512, width=80,
                                  step=8, on_enter=False)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Fit basis", tag="_btn_fit_basis",
                               width=120, callback=self._cb_fit_basis)
                dpg.add_button(label="Clear basis", width=120,
                               callback=self._cb_clear_basis)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", width=80,
                               callback=self._cb_save_basis)
                dpg.add_button(label="Load", width=80,
                               callback=self._cb_load_basis)
            dpg.add_text("basis: per-image (not locked)", tag="_log_basis_status")
            dpg.bind_item_theme("_btn_fit_basis", "_theme_btn_off")
            dpg.add_separator()

            # ---------------- Consistent batch (DA3-Streaming) ----------------
            dpg.add_text("Consistent batch (DA3-Streaming)")
            with dpg.group(horizontal=True):
                dpg.add_text("Chunk size:")
                dpg.add_slider_int(tag="_slider_chunk_size",
                                   default_value=60, min_value=30, max_value=120,
                                   width=180)
            with dpg.group(horizontal=True):
                dpg.add_text("Downscale W:")
                dpg.add_input_int(tag="_input_downscale_width",
                                  default_value=0, min_value=0, max_value=4096,
                                  step=64, width=100)
                dpg.add_text("(0 = off)")
            dpg.add_input_text(tag="_input_colmap_path",
                               default_value="", width=-1,
                               hint="path to sparse/0/ (COLMAP .bin files)")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Refine with COLMAP",
                                 tag="_chk_use_colmap", default_value=False)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run consistent batch", width=180,
                               callback=self._cb_run_consistent_batch)
                dpg.add_button(label="Stop", width=60,
                               callback=self._cb_stop_consistent_batch)
            dpg.add_button(label="Refine existing only (no DA3 rerun)",
                           width=-1, callback=self._cb_refine_only)
            dpg.add_text("(idle - see terminal for full log)", tag="_log_da3stream_status")
            dpg.add_separator()

            dpg.add_text("Zoom")
            dpg.add_text("1.00x", tag="_log_zoom")
            dpg.add_button(label="Reset zoom", width=-1,
                           callback=self._cb_reset_zoom)
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_text("FPS: ")
                dpg.add_text("N/A", tag="_log_infer_time")

        side_x = self.MAIN_W + self.CTRL_W
        with dpg.window(tag="_side_top",
                        width=self.SIDE_W, height=self.SIDE_H,
                        pos=[side_x, 0],
                        no_move=True, no_title_bar=True, no_scrollbar=True):
            dpg.add_text("Prediction / PCA 1-3")
            dpg.add_image("_tex_pred", tag="_img_pred")

        with dpg.window(tag="_side_mid",
                        width=self.SIDE_W, height=self.SIDE_H,
                        pos=[side_x, self.SIDE_H],
                        no_move=True, no_title_bar=True, no_scrollbar=True):
            dpg.add_text("PCA 4-6")
            dpg.add_image("_tex_aux1", tag="_img_aux1")

        with dpg.window(tag="_side_bot",
                        width=self.SIDE_W, height=self.SIDE_H,
                        pos=[side_x, 2 * self.SIDE_H],
                        no_move=True, no_title_bar=True, no_scrollbar=True):
            dpg.add_text("PCA 7-9")
            dpg.add_image("_tex_aux2", tag="_img_aux2")

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self._cb_mouse_wheel)
            dpg.add_mouse_down_handler(callback=self._cb_mouse_down)
            dpg.add_mouse_drag_handler(callback=self._cb_mouse_drag, threshold=0.0)
            dpg.add_mouse_release_handler(callback=self._cb_mouse_release)

        total_w = self.MAIN_W + self.CTRL_W + self.SIDE_W
        total_h = self.MAIN_H + (45 if os.name == "nt" else 0)
        dpg.create_viewport(title=f"{self.runname}",
                            width=total_w, height=total_h, resizable=False)

        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0,
                                    category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()


if __name__ == "__main__":
    app = GUIBase(runname="Image Browser")
    app.run()