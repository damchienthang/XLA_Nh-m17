"""
cli.py — Standalone Command-Line Forensic Inspector & Benchmark Tool
===================================================================
Run forensic inspection on a single image or benchmark the full 29-image dataset
in pure Python without requiring Streamlit or PyTorch checkpoints.

Examples:
  python cli.py --image ../public/dataset/degraded/degraded_04.jpg
  python cli.py --benchmark --manifest ../public/dataset/manifest.csv
"""

import argparse
import csv
import json
import os
from typing import List

from PIL import Image

from engine import CoSpyEngine, _build_analysis_log
from train_head import _compute_metrics


def inspect_single(image_path: str, threshold: float = 0.5, as_json: bool = False):
    eng = CoSpyEngine()
    pil_img = Image.open(image_path).convert("RGB")
    res = eng.analyze_image(pil_img)
    base = res["baseline_info"]

    if as_json:
        payload = {
            "file": os.path.basename(image_path),
            "size": [pil_img.width, pil_img.height],
            "baseline_prob_ai": base["probability_ai"],
            "enhanced_prob_ai": res["probability_ai"],
            "decision": "AI-Generated" if res["probability_ai"] >= threshold else "Real Image",
            "cagu_stats": res["cagu_stats"],
            "sub_scores": res["sub_scores"],
            "native_features": res["native_features"],
            "reasoning": res["verdict_reasoning"],
            "log": _build_analysis_log(res, threshold=threshold),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    verdict = "AI-GENERATED" if res["probability_ai"] >= threshold else "REAL IMAGE"
    print(f"=== CO-SPY Forensic Studio CLI ===")
    print(f"File       : {image_path} ({pil_img.width}x{pil_img.height})")
    print(f"Baseline v1: {base['probability_ai']*100:.1f}% AI  ({base['reasoning']})")
    print(f"Enhanced v2: {res['probability_ai']*100:.1f}% AI  -> {verdict}")
    print(f"CAGU Gate  : Spatial={res['cagu_stats']['spatial_pct']:.1f}% | Freq={res['cagu_stats']['freq_pct']:.1f}% | Chroma={res['cagu_stats']['chroma_pct']:.1f}% [{res['cagu_stats']['compression_mode']}]")
    print(f"Telemetry  : {res['metrics']}")
    print(f"Reasoning  : {res['verdict_reasoning']}")


def run_benchmark(manifest_path: str, threshold: float = 0.5):
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    eng = CoSpyEngine()

    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full = os.path.join(base_dir, row["file"])
            if os.path.exists(full):
                rows.append((row["file"], 1 if row["label"].strip().lower() in ("fake", "ai", "1") else 0, full))

    # Also include samples/ if present alongside manifest
    samples_dir = os.path.join(base_dir, "samples")
    if not os.path.isdir(samples_dir):
        samples_dir = os.path.join(os.path.dirname(base_dir), "samples")
    if os.path.isdir(samples_dir):
        for name in sorted(os.listdir(samples_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                rel = f"samples/{name}"
                if not any(r[0] == rel for r in rows):
                    y = 1 if ("ai" in name.lower() or "fake" in name.lower()) else 0
                    rows.append((rel, y, os.path.join(samples_dir, name)))

    labels: List[int] = []
    base_probs: List[float] = []
    enh_probs: List[float] = []

    for rel, y, full_path in rows:
        pil_img = Image.open(full_path).convert("RGB")
        res = eng.analyze_image(pil_img)
        bp = float(res["baseline_info"]["probability_ai"])
        ep = float(res["probability_ai"])
        labels.append(y)
        base_probs.append(bp)
        enh_probs.append(ep)
        ok = (ep >= threshold) == (y == 1)
        print(f"[{'OK' if ok else 'MISS':4s}] {rel:34s} | y={y} | Base={bp*100:5.1f}% | Enh={ep*100:5.1f}%")

    bm = _compute_metrics(labels, base_probs, threshold)
    em = _compute_metrics(labels, enh_probs, threshold)
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Baseline v1 : Acc={bm['accuracy']:.2f}% | Prec={bm['precision']:.2f}% | Rec={bm['recall']:.2f}% | Spec={bm['specificity']:.2f}% | F1={bm['f1']:.4f} | AUC={bm['auc_roc']:.4f}")
    print(f"Enhanced v2 : Acc={em['accuracy']:.2f}% | Prec={em['precision']:.2f}% | Rec={em['recall']:.2f}% | Spec={em['specificity']:.2f}% | F1={em['f1']:.4f} | AUC={em['auc_roc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CO-SPY Forensic Studio CLI")
    parser.add_argument("--image", help="Path to a single image to inspect")
    parser.add_argument("--benchmark", action="store_true", help="Run full dataset benchmark")
    parser.add_argument("--manifest", default="test_images/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--threshold", type=float, default=0.5, help="AI probability threshold")
    parser.add_argument("--json", action="store_true", help="Output JSON for single image")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.manifest, threshold=args.threshold)
    elif args.image:
        inspect_single(args.image, threshold=args.threshold, as_json=args.json)
    else:
        parser.print_help()
