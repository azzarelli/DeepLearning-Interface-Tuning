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