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
        self.prediction_image = None    # CHW float tensor [0,1] (display-ready)

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
            # Normalize / colorize for display
            if torch.is_tensor(y) and y.ndim == 2:
                # Single-channel output (depth, segmentation logits, ...): colorize
                disp = _colorize_depth(y)
            elif torch.is_tensor(y) and y.ndim == 3:
                disp = y.float().clamp(0, 1)
                if disp.shape[0] == 1:
                    disp = disp.repeat(3, 1, 1)
                elif disp.shape[0] > 3:
                    disp = disp[:3]
            else:
                raise ValueError(f"Unexpected prediction shape: {tuple(y.shape)}")
            self.prediction_image = disp
            self._dirty = True
            dpg.set_value("_log_model_status", "Prediction OK")
        except Exception as e:
            dpg.set_value("_log_model_status", f"Inference failed: {e}")
            print(f"[run_prediction] {e}")

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
            dpg.add_text("(no model loaded)", tag="_log_model_status")
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
            dpg.add_text("Prediction")
            dpg.add_image("_tex_pred", tag="_img_pred")

        with dpg.window(tag="_side_mid",
                        width=self.SIDE_W, height=self.SIDE_H,
                        pos=[side_x, self.SIDE_H],
                        no_move=True, no_title_bar=True, no_scrollbar=True):
            dpg.add_text("Reserved")
            dpg.add_image("_tex_aux1", tag="_img_aux1")

        with dpg.window(tag="_side_bot",
                        width=self.SIDE_W, height=self.SIDE_H,
                        pos=[side_x, 2 * self.SIDE_H],
                        no_move=True, no_title_bar=True, no_scrollbar=True):
            dpg.add_text("Reserved")
            dpg.add_image("_tex_aux2", tag="_img_aux2")

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self._cb_mouse_wheel)

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