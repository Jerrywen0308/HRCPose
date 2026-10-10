
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch pose inference with MMPose using GT bboxes from a COCO annotation file,
and visualization consistent with draw_coco_monkey_pose.py.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from mmengine.config import Config

# Optional OpenCV; fallback to PIL
try:
    import cv2
    has_cv2 = True
except Exception:
    has_cv2 = False
    from PIL import Image, ImageDraw

# Try new API first
_INFERENCER_OK = True
try:
    from mmpose.apis import MMPoseInferencer
except Exception:
    _INFERENCER_OK = False
    # legacy API will be lazy-imported when used

# ---------- Color utils ----------
def default_palette(n: int) -> List[Tuple[int, int, int]]:
    import math
    colors = []
    for i in range(max(1, n)):
        hue = i / max(1, n)
        r = int(127.5 * (1 + math.sin(2 * math.pi * hue)))
        g = int(127.5 * (1 + math.sin(2 * math.pi * hue + 2.094)))
        b = int(127.5 * (1 + math.sin(2 * math.pi * hue + 4.188)))
        colors.append((b, g, r) if has_cv2 else (r, g, b))
    return colors

def parse_color(s: str) -> Tuple[int,int,int]:
    named = {
        "black": (0,0,0), "white": (255,255,255),
        "red": (255,0,0), "green": (0,255,0), "blue": (0,0,255),
        "yellow": (255,255,0), "cyan": (0,255,255), "magenta": (255,0,255),
        "grey": (128,128,128), "gray": (128,128,128)
    }
    s = s.strip().lower()
    if s in named:
        rgb = named[s]
    elif s.startswith("#") and len(s)==7:
        rgb = (int(s[1:3],16), int(s[3:5],16), int(s[5:7],16))
    else:
        raise ValueError(f"Unrecognized color: {s}")
    return (rgb[2], rgb[1], rgb[0]) if has_cv2 else rgb

# ---------- Drawing helpers ----------
def draw_circle(img, center: Tuple[int,int], radius: int, color: Tuple[int,int,int], thickness: int):
    if has_cv2:
        cv2.circle(img, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    else:
        draw = ImageDraw.Draw(img, "RGBA")
        x, y = center
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color+(255,))

def draw_line(img, p1: Tuple[int,int], p2: Tuple[int,int], color: Tuple[int,int,int], thickness: int, alpha: float = 0.9):
    if has_cv2:
        overlay = img.copy()
        cv2.line(overlay, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    else:
        draw = ImageDraw.Draw(img, "RGBA")
        draw.line([p1, p2], fill=color+(int(255*alpha),), width=thickness)

def draw_text(img, text: str, org: Tuple[int,int], color: Tuple[int,int,int]):
    if has_cv2:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    else:
        draw = ImageDraw.Draw(img, "RGBA")
        draw.text(org, text, fill=color+(255,))

# ---------- COCO helpers ----------
def load_coco(coco_json: str) -> Dict:
    with open(coco_json, "r", encoding="utf-8") as f:
        return json.load(f)

def build_image_index(coco: Dict) -> Dict[str, Dict]:
    idx = {}
    for im in coco.get("images", []):
        fn = im.get("file_name")
        if fn is not None:
            idx[fn] = im
    return idx

def collect_bboxes_for_image(coco: Dict, image_id: int) -> List[Tuple[float,float,float,float]]:
    bboxes = []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") == image_id and "bbox" in ann:
            b = ann["bbox"]
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                bboxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))  # xywh
    return bboxes

def _normalize_edges_to_1based(edge_tokens, num_kpt):
    parsed = []
    for tok in edge_tokens or []:
        tok = str(tok).replace(",", "-").replace(":", "-")
        parts = tok.split("-")
        if len(parts) >= 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            parsed.append((int(parts[0].strip()), int(parts[1].strip())))
    if not parsed:
        return []
    max_idx = max(max(a,b) for a,b in parsed)
    if 0 <= max_idx <= num_kpt-1:
        parsed = [(a+1, b+1) for a,b in parsed]
    return parsed

# ---------- Visualization ----------
def draw_pose(img, kpts_xy: np.ndarray, kpt_scores: np.ndarray, skeleton: List[Tuple[int,int]], names: List[str],
              kpt_thr: float, radius: int, thickness: int, alpha: float,
              label_mode: str, label_color: Tuple[int,int,int]):
    n = kpts_xy.shape[0]
    kp_colors = default_palette(n)
    edge_colors = default_palette(max(1, len(skeleton)))

    # edges
    for ei, (a, b) in enumerate(skeleton):
        ia, ib = a - 1, b - 1
        if ia < 0 or ib < 0 or ia >= n or ib >= n:
            continue
        if kpt_scores[ia] < kpt_thr or kpt_scores[ib] < kpt_thr:
            continue
        pa = (int(round(float(kpts_xy[ia,0]))), int(round(float(kpts_xy[ia,1]))))
        pb = (int(round(float(kpts_xy[ib,0]))), int(round(float(kpts_xy[ib,1]))))
        draw_line(img, pa, pb, edge_colors[ei % len(edge_colors)], thickness, alpha)

    # keypoints
    for i in range(n):
        if kpt_scores[i] < kpt_thr:
            continue
        p = (int(round(float(kpts_xy[i,0]))), int(round(float(kpts_xy[i,1]))))
        draw_circle(img, p, radius, kp_colors[i % len(kp_colors)], thickness)
        if label_mode != "none":
            if label_mode == "index":
                label = f"{i}"
            elif label_mode == "name":
                label = f"{names[i]}" if i < len(names) else f"{i}"
            else:
                label = f"{i}:{names[i]}" if i < len(names) else f"{i}"
            text_org = (p[0] + radius + 2, p[1] - radius - 2)
            draw_text(img, label, text_org, label_color)

# ---------- load dataset_meta from cfg (for legacy path) ----------
def load_meta_from_cfg(cfg_file: str):
    cfg = Config.fromfile(cfg_file)
    if hasattr(cfg, 'dataset_info'): return cfg.dataset_info
    if hasattr(cfg, 'dataset_meta'): return cfg.dataset_meta

    def _pick(d):
        try:
            meta = d['dataset']['metainfo']
            if isinstance(meta, dict) and 'from_file' in meta:
                c2 = Config.fromfile(meta['from_file'])
                if hasattr(c2, 'dataset_info'): return c2.dataset_info
                if hasattr(c2, 'dataset_meta'): return c2.dataset_meta
        except Exception:
            pass
        return None
    for key in ['train_dataloader','val_dataloader','test_dataloader']:
        if key in cfg:
            m = _pick(cfg[key])
            if m is not None:
                return m
    return None

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Batch MMPose inference using GT bboxes from COCO, with visualization.")
    ap.add_argument("--config", required=True, help="MMPose config file path")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint file path")
    ap.add_argument("--coco-json", default='img/val.json', help="Path to COCO annotations json (provides GT bboxes)")
    ap.add_argument("--image-dir", default='img/demo_2', help="Directory containing images referenced by the COCO json")
    ap.add_argument("--out-dir", default='img/ours', help="Output directory for visualized images")
    ap.add_argument("--device", default="cpu", help="Device string, e.g., 'cpu', 'cuda:0'")
    ap.add_argument("--kpt-thr", type=float, default=0.6, help="Confidence threshold for drawing keypoints/edges")
    ap.add_argument("--radius", type=int, default=3, help="Keypoint circle radius")
    ap.add_argument("--thickness", type=int, default=2, help="Line thickness")
    ap.add_argument("--alpha", type=float, default=0.9, help="Alpha for edge blending")
    ap.add_argument("--label", choices=["index", "name", "both", "none"], default="none", help="Label style for keypoints")
    ap.add_argument("--label-color", default="white", help="Label text color name or #RRGGBB (default: white)")
    ap.add_argument("--drop-edge", nargs="*", default=[], help="Edges to remove from skeleton, e.g. 5-8 0-3")
    ap.add_argument("--force-legacy", action="store_true", help="Force use of legacy API (init_model + inference_topdown) to avoid Inferencer/detector overhead.")
    ap.add_argument("--out-size", nargs=2, type=int, default=None,
                metavar=("W","H"), help="Resize output to fixed size (W H).")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith('cuda') and torch.cuda.is_available()) else "cpu")

    coco = load_coco(args.coco_json)
    categories = coco.get("categories", [])
    if not categories:
        raise KeyError("No categories found in COCO json.")
    cat = categories[0]
    names = cat.get("keypoints", [])
    num_kpt = len(names)
    raw_skel = cat.get("skeleton", [])
    skeleton = [tuple(map(int, e)) for e in raw_skel]

    drop_edges = _normalize_edges_to_1based(args.drop_edge, num_kpt)
    if drop_edges:
        drop_set = {tuple(sorted(e)) for e in drop_edges}
        skeleton = [e for e in skeleton if tuple(sorted(e)) not in drop_set]

    img_index = build_image_index(coco)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label_color = parse_color(args.label_color)

    # Build model/inferencer
    use_inferencer = False
    if _INFERENCER_OK and (not args.force_legacy):
        inferencer = MMPoseInferencer(
            pose2d=args.config,
            pose2d_weights=args.checkpoint,
            det_model=None,
            device=str(device)
        )
        use_inferencer = True
    else:
        from mmpose.apis import init_model, inference_topdown  # lazy import
        model = init_model(args.config, args.checkpoint, device=device)
        # inject dataset meta (very important for custom datasets)
        meta = load_meta_from_cfg(args.config)
        if meta is not None:
            model.dataset_meta = meta
        model.eval()

    # Iterate over images in COCO
    image_root = Path(args.image_dir)
    count = 0
    for fn, imrec in img_index.items():
        img_path = image_root / fn
        if not img_path.exists():
            continue

        bboxes = collect_bboxes_for_image(coco, imrec["id"])
        if not bboxes:
            continue

        # Load image for drawing
        if has_cv2:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
        else:
            from PIL import Image
            img = Image.open(str(img_path)).convert("RGB").copy()

        # Run inference per bbox
        all_instances = []
        if use_inferencer:
            for bbox in bboxes:
                x, y, w, h = bbox
                xyxy = [x, y, x + w, y + h]
                with torch.inference_mode():
                    result_gen = inferencer(inputs=str(img_path),
                                            bboxes=[xyxy],
                                            bbox_format='xyxy',
                                            return_vis=False, show=False)
                    out = next(result_gen)
                preds = out.get("predictions", [])
                if not preds:
                    continue
                pred0 = preds[0]
                kpts = np.asarray(pred0.get("keypoints", []), dtype=float)        # [N,K,2]
                ksc  = np.asarray(pred0.get("keypoint_scores", []), dtype=float)  # [N,K]
                if kpts.ndim == 3 and kpts.shape[0] >= 1:
                    all_instances.append((kpts[0], ksc[0]))
        else:
            for bbox in bboxes:
                x, y, w, h = bbox
                xyxy = [x, y, x + w, y + h]
                with torch.inference_mode():
                    res = inference_topdown(model, str(img_path),
                                            bboxes=[xyxy], bbox_format='xyxy')
                if not res:
                    continue
                pred = res[0].pred_instances
                kpts = np.asarray(pred.keypoints, dtype=float)        # [N,K,2]
                ksc  = np.asarray(pred.keypoint_scores, dtype=float)  # [N,K]
                if kpts.ndim == 3 and kpts.shape[0] >= 1:
                    all_instances.append((kpts[0], ksc[0]))

        # Draw
        for kpts, ksc in all_instances:
            draw_pose(img, kpts, ksc, skeleton, names,
                      kpt_thr=args.kpt_thr, radius=args.radius, thickness=args.thickness, alpha=args.alpha,
                      label_mode=args.label, label_color=label_color)

        # Save
        out_path = out_dir / (Path(fn).stem + "_pred.jpg")
        if has_cv2:
            cv2.imwrite(str(out_path), img)
        else:
            img.save(str(out_path))
        count += 1

    print(f"[OK] Processed {count} images. Saved to: {out_dir}")

if __name__ == "__main__":
    main()

#ap10k
#python infer_demo.py --config configs/animal_2d_keypoint/HRCPose/ap10k/HRCPose_ap10k-256x256.py --checkpoint weights/ap10k-pretrain.pth --image-dir data/ap10k/data --coco-json data/ap10k/annotations/ap10k-test-split1.json --out-dir img/ap10k_output --force-legacy

#cm9k(monkey)
#python infer_demo.py --config configs/monkey/HRCPose/HRCPose/HRCPose_monkey-384x288.py --checkpoint weights/monkey.pth --image-dir data/monkey/data --coco-json data/monkey/annotations/val_demo.json --out-dir img/monkey_output --force-legacy --drop-edge 5-8 3-2
