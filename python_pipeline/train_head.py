"""
train_head.py — Fast Feature-Cached Training & Evaluation for EnhancedCoSpyHead
===============================================================================
Fixes the 41,357 ms/img bottleneck and random untrained weights in evaluation_metrics.ipynb:
  1. Freezes heavy CLIP/VAE backbones, pre-extracts spatial features + quality vectors + native crops
     with JPEG (q=30..95) & Resize augmentation.
  2. Trains FrequencyBranch + Quality-Conditioned EnhancedCoSpyHead in seconds with AdamW.
  3. Evaluates Accuracy, Precision, Recall, Specificity, F1, and AUC-ROC on test_images/manifest.csv.

Usage:
  # Benchmark the calibrated Multi-Domain CAGU directly on test_images/manifest.csv (no GPU needed):
  python train_head.py --eval-only --manifest test_images/manifest.csv

  # Train EnhancedCoSpyHead + FrequencyBranch on a dataset folder (real/ and fake/ subfolders):
  python train_head.py --data-dir path/to/train_data --epochs 25 --out pretrained/enhanced_head.pth
"""

import argparse
import csv
import io
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image

from native_forensics import extract_multi_domain_features, predict_multi_domain_cagu


def _compute_metrics(labels: List[int], probs: List[float], threshold: float = 0.5) -> dict:
    tp = sum(1 for y, p in zip(labels, probs) if y == 1 and p >= threshold)
    fp = sum(1 for y, p in zip(labels, probs) if y == 0 and p >= threshold)
    tn = sum(1 for y, p in zip(labels, probs) if y == 0 and p < threshold)
    fn = sum(1 for y, p in zip(labels, probs) if y == 1 and p < threshold)

    total = max(len(labels), 1)
    acc = (tp + tn) / total * 100.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec / 10000.0) / ((prec + rec) / 100.0) if (prec + rec) > 0 else 0.0

    # Exact trapezoidal ROC-AUC
    th_list = sorted(set([1.01, -0.01] + probs), reverse=True)
    pos = max(sum(labels), 1)
    neg = max(len(labels) - sum(labels), 1)
    pts = []
    for th in th_list:
        t_tp = sum(1 for y, p in zip(labels, probs) if y == 1 and p >= th)
        t_fp = sum(1 for y, p in zip(labels, probs) if y == 0 and p >= th)
        pts.append((t_fp / neg, t_tp / pos))
    auc = sum(
        (pts[i][0] - pts[i - 1][0]) * 0.5 * (pts[i][1] + pts[i - 1][1])
        for i in range(1, len(pts))
    )
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "auc_roc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_manifest(manifest_path: str):
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    labels: List[int] = []
    probs: List[float] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(base_dir, row["file"])
            if not os.path.exists(img_path):
                continue
            y = 1 if row["label"].strip().lower() in ("fake", "ai", "1") else 0
            pil_img = Image.open(img_path).convert("RGB")
            res = predict_multi_domain_cagu(np.array(pil_img))
            p = float(res["probability_ai"])
            labels.append(y)
            probs.append(p)
            mark = "OK" if ((p >= 0.5) == (y == 1)) else "MISS"
            print(f"[{mark:4s}] {row['file']:32s} | y={y} | prob_ai={p*100:5.1f}% | lead={res['primary_lead']}")

    m = _compute_metrics(labels, probs)
    print("\n=== SUMMARY ===")
    print(
        f"Accuracy={m['accuracy']:.2f}% | Precision={m['precision']:.2f}% | "
        f"Recall={m['recall']:.2f}% | Specificity={m['specificity']:.2f}% | "
        f"F1={m['f1']:.4f} | AUC-ROC={m['auc_roc']:.4f}"
    )
    print(f"Confusion Matrix: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")


def _apply_degradation_aug(pil_img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        scale = random.uniform(0.5, 0.9)
        nw = max(128, int(pil_img.width * scale))
        nh = max(128, int(pil_img.height * scale))
        pil_img = pil_img.resize((nw, nh), Image.BILINEAR)
    if random.random() < 0.65:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=random.randint(30, 95))
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")
    return pil_img


def train_head(data_dir: str, out_path: str, epochs: int = 25, lr: float = 3e-4):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from adaptive_fusion import EnhancedCoSpyHead
    from engine import CoSpyEngine
    from frequency_branch import FrequencyBranch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eng = CoSpyEngine()
    sd_det = eng.get_sd_artifact_detector()

    samples: List[Tuple[str, int]] = []
    for cls_name, label in [("real", 0), ("fake", 1)]:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                samples.append((os.path.join(cls_dir, fname), label))

    if not samples:
        raise RuntimeError(f"No images found under {data_dir}/real or {data_dir}/fake")

    print(f"Extracting frozen spatial features + native crops for {len(samples)} images (with JPEG augmentation)...")
    sp_feats, native_crops, q_feats, mses, ys = [], [], [], [], []

    with torch.no_grad():
        for path, y in samples:
            for aug_idx in range(2):  # 1 clean + 1 degraded JPEG/resize copy
                img = Image.open(path).convert("RGB")
                if aug_idx == 1:
                    img = _apply_degradation_aug(img)
                np_rgb = np.array(img)

                sd_t = sd_det.test_transform(img).unsqueeze(0).to(device)
                sp_f, _ = sd_det(sd_t, return_feat=True)
                recon = sd_det.artifact_encoder.reconstruct(sd_t)
                mse_val = float(torch.mean((sd_t - recon) ** 2).item())

                md_f = extract_multi_domain_features(np_rgb)
                q_vec = [
                    md_f["jpeg_blockiness"],
                    md_f["saturation_mean"],
                    md_f["chroma_to_luma"],
                    md_f["min_channel_corr"],
                ]

                h, w, _ = np_rgb.shape
                cs = 128
                if h < cs or w < cs:
                    img_pad = img.resize((max(w, cs), max(h, cs)), Image.BICUBIC)
                    np_rgb = np.array(img_pad)
                    h, w, _ = np_rgb.shape
                y0, x0 = (h - cs) // 2, (w - cs) // 2
                crop = torch.from_numpy(np_rgb[y0 : y0 + cs, x0 : x0 + cs]).permute(2, 0, 1).float() / 255.0

                sp_feats.append(sp_f.squeeze(0).cpu())
                native_crops.append(crop)
                q_feats.append(torch.tensor(q_vec, dtype=torch.float32))
                mses.append(torch.tensor([mse_val], dtype=torch.float32))
                ys.append(torch.tensor([float(y)], dtype=torch.float32))

    dataset = TensorDataset(
        torch.stack(sp_feats),
        torch.stack(native_crops),
        torch.stack(q_feats),
        torch.stack(mses),
        torch.stack(ys),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    freq_branch = FrequencyBranch(r0=16, out_features=128, native_crop_size=128).to(device)
    head = EnhancedCoSpyHead(spatial_dim=512, freq_dim=128, quality_dim=4).to(device)

    optimizer = torch.optim.AdamW(
        list(freq_branch.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce = nn.BCELoss()

    for ep in range(1, epochs + 1):
        freq_branch.train()
        head.train()
        total_loss = 0.0
        for sp_b, crop_b, q_b, mse_b, y_b in loader:
            sp_b = sp_b.to(device)
            crop_b = crop_b.to(device)
            q_b = q_b.to(device)
            mse_b = mse_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            fr_b = freq_branch(crop_b)
            prob = head(sp_b, fr_b, mse_b, quality_feat=q_b)
            loss = bce(prob, y_b)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * sp_b.size(0)
        scheduler.step()
        if ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs} | Loss = {total_loss / len(dataset):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(
        {
            "r0": 16,
            "freq_dim": 128,
            "spatial_dim": 512,
            "quality_dim": 4,
            "native_crop_size": 128,
            "freq_branch": freq_branch.state_dict(),
            "head": head.state_dict(),
        },
        out_path,
    )
    print(f"Saved trained EnhancedCoSpyHead checkpoint to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--manifest", default="test_images/manifest.csv")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--out", default="pretrained/enhanced_head.pth")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    if args.eval_only or not args.data_dir:
        evaluate_manifest(args.manifest)
    else:
        train_head(args.data_dir, args.out, epochs=args.epochs)
