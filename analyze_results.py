#!/usr/bin/env python3
"""Analyze and compare quantization experiment results."""

import json
import os
from pathlib import Path
import sys


def find_latest_outputs():
    """Find all output directories sorted by time."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return []

    # Get all dated directories
    all_outputs = []
    for date_dir in outputs_dir.iterdir():
        if date_dir.is_dir():
            for time_dir in date_dir.iterdir():
                if time_dir.is_dir():
                    all_outputs.append(time_dir)

    # Sort by modification time (newest first)
    all_outputs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return all_outputs


def extract_metrics(output_dir):
    """Extract metrics from an output directory."""
    metrics_file = output_dir / "compression_metrics.json"

    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        data = json.load(f)

    return {
        "bit_width": data.get("bit_width", 32),
        "original_size_mb": data.get("original_size_mb", 0),
        "compressed_size_mb": data.get("compressed_size_mb", 0),
        "compression_ratio": data.get("compression_ratio", 1.0),
        "size_reduction_percent": data.get("size_reduction_percent", 0),
        "output_dir": str(output_dir),
    }


def main():
    """Analyze quantization results."""
    print("\n" + "="*70)
    print("📊 Quantization Experiment Results Analysis")
    print("="*70 + "\n")

    outputs = find_latest_outputs()

    if not outputs:
        print("❌ No output directories found")
        sys.exit(1)

    # Collect all results
    results = []
    for output_dir in outputs:
        metrics = extract_metrics(output_dir)
        if metrics:
            results.append(metrics)

    if not results:
        print("❌ No compression metrics found in outputs")
        sys.exit(1)

    # Sort by bit-width (descending)
    results.sort(key=lambda x: x["bit_width"], reverse=True)

    # Print comparison table
    print("Bit-Width | Size (MB) | Compression | Reduction | Output Directory")
    print("-" * 70)

    for r in results:
        print(f"{r['bit_width']:9d} | {r['compressed_size_mb']:9.2f} | "
              f"{r['compression_ratio']:11.2f}x | {r['size_reduction_percent']:9.1f}% | "
              f"{Path(r['output_dir']).name}")

    print("\n" + "="*70)
    print(f"📁 Total experiments found: {len(results)}")
    print("="*70 + "\n")

    # Find best compression
    best_compression = max(results, key=lambda x: x["compression_ratio"])
    print(f"🏆 Best compression: {best_compression['bit_width']}-bit "
          f"({best_compression['compression_ratio']:.2f}x)")

    # Calculate average
    avg_compression = sum(r["compression_ratio"] for r in results) / len(results)
    print(f"📊 Average compression: {avg_compression:.2f}x\n")


if __name__ == "__main__":
    main()
