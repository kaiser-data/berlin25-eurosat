#!/usr/bin/env python3
"""Analyze and compare quantization experiment results."""

import json
import csv
from pathlib import Path
import sys
from datetime import datetime


def find_all_results():
    """Find all training results across all bit-widths."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return []

    results = []

    # Scan new folder structure: outputs/{bit_width}bit/run_*/
    for bit_dir in outputs_dir.glob("*bit"):
        for run_dir in bit_dir.glob("run_*"):
            result_file = run_dir / "training_results.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                        data["output_dir"] = str(run_dir)
                        results.append(data)
                except Exception as e:
                    print(f"⚠️  Warning: Could not read {result_file}: {e}")

    # Also check old folder structure for backward compatibility
    for date_dir in outputs_dir.glob("20*"):
        if date_dir.is_dir():
            for time_dir in date_dir.iterdir():
                if time_dir.is_dir():
                    result_file = time_dir / "training_results.json"
                    if result_file.exists():
                        try:
                            with open(result_file) as f:
                                data = json.load(f)
                                data["output_dir"] = str(time_dir)
                                results.append(data)
                        except:
                            pass

    # Sort by bit-width (descending)
    results.sort(key=lambda x: x.get("bit_width", 32), reverse=True)

    return results


def calculate_accuracy_drop(results):
    """Calculate accuracy drop compared to 32-bit baseline."""
    # Find 32-bit baseline
    baseline = next((r for r in results if r.get("bit_width") == 32), None)
    if not baseline:
        return results

    baseline_acc = baseline.get("final_test_accuracy", 0)

    for result in results:
        acc = result.get("final_test_accuracy", 0)
        result["accuracy_drop_percent"] = (acc - baseline_acc) * 100

    return results


def print_comparison_table(results):
    """Print comprehensive comparison table."""
    print("\n" + "="*140)
    print("📊 Quantization Experiment Results - Comprehensive Analysis")
    print("="*140)
    print()

    # Header
    header = (
        f"{'Bit-Width':>10} | {'Size (MB)':>10} | {'Compression':>12} | "
        f"{'Reduction':>10} | {'Test Acc':>10} | {'Train Acc':>10} | {'Overfit':>8} | "
        f"{'Δ Acc':>8} | {'Time (s)':>9} | {'Output Dir':>20}"
    )
    print(header)
    print("-" * 140)

    # Rows
    for r in results:
        bit_width = r.get("bit_width", 32)
        comp_metrics = r.get("compression_metrics", {})
        size_mb = comp_metrics.get("compressed_size_mb", 0)
        compression = comp_metrics.get("compression_ratio", 1.0)
        reduction = comp_metrics.get("size_reduction_percent", 0)

        test_accuracy = r.get("final_test_accuracy", 0) * 100  # Convert to percentage

        # Get final training accuracy (from last round)
        train_accuracy = 0.0
        results_per_round = r.get("results_per_round", [])
        if results_per_round and "train_accuracy" in results_per_round[-1]:
            train_accuracy = results_per_round[-1]["train_accuracy"] * 100

        # Calculate overfitting gap (positive means overfitting)
        overfit_gap = train_accuracy - test_accuracy

        acc_drop = r.get("accuracy_drop_percent", 0)
        time_s = r.get("training_time_seconds", 0)
        output_dir = Path(r.get("output_dir", "")).name

        row = (
            f"{bit_width:>10} | {size_mb:>10.2f} | {compression:>11.2f}x | "
            f"{reduction:>9.1f}% | {test_accuracy:>9.2f}% | {train_accuracy:>9.2f}% | "
            f"{overfit_gap:>+7.2f}% | {acc_drop:>+7.2f}% | {time_s:>9.1f} | {output_dir:>20}"
        )
        print(row)

    print("="*140)
    print()


def save_results_csv(results, filename="outputs/results_summary.csv"):
    """Save results to CSV file."""
    Path(filename).parent.mkdir(exist_ok=True)

    with open(filename, "w", newline="") as f:
        fieldnames = [
            "bit_width",
            "model_size_mb",
            "compression_ratio",
            "size_reduction_percent",
            "final_accuracy_percent",
            "final_loss",
            "accuracy_drop_percent",
            "training_time_seconds",
            "num_rounds",
            "learning_rate",
            "timestamp",
            "output_directory"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            comp_metrics = r.get("compression_metrics", {})
            writer.writerow({
                "bit_width": r.get("bit_width", 32),
                "model_size_mb": comp_metrics.get("compressed_size_mb", 0),
                "compression_ratio": comp_metrics.get("compression_ratio", 1.0),
                "size_reduction_percent": comp_metrics.get("size_reduction_percent", 0),
                "final_accuracy_percent": r.get("final_test_accuracy", 0) * 100,
                "final_loss": r.get("final_test_loss", 0),
                "accuracy_drop_percent": r.get("accuracy_drop_percent", 0),
                "training_time_seconds": r.get("training_time_seconds", 0),
                "num_rounds": r.get("num_rounds", 0),
                "learning_rate": r.get("learning_rate", 0),
                "timestamp": r.get("timestamp", ""),
                "output_directory": r.get("output_dir", "")
            })

    print(f"✅ Results saved to CSV: {filename}")


def save_results_json(results, filename="outputs/results_summary.json"):
    """Save results to JSON file."""
    Path(filename).parent.mkdir(exist_ok=True)

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to JSON: {filename}")


def print_summary_stats(results):
    """Print summary statistics."""
    if not results:
        return

    print(f"📁 Total experiments found: {len(results)}")
    print()

    # Best compression
    best_comp = max(results, key=lambda x: x.get("compression_metrics", {}).get("compression_ratio", 0))
    best_comp_ratio = best_comp.get("compression_metrics", {}).get("compression_ratio", 0)
    print(f"🏆 Best compression: {best_comp.get('bit_width')}-bit ({best_comp_ratio:.2f}x)")

    # Best accuracy
    best_acc = max(results, key=lambda x: x.get("final_test_accuracy", 0))
    best_acc_val = best_acc.get("final_test_accuracy", 0) * 100
    print(f"🎯 Best test accuracy: {best_acc.get('bit_width')}-bit ({best_acc_val:.2f}%)")

    # Average compression
    avg_comp = sum(r.get("compression_metrics", {}).get("compression_ratio", 0) for r in results) / len(results)
    print(f"📊 Average compression: {avg_comp:.2f}x")

    # Overfitting analysis
    overfit_results = []
    for r in results:
        results_per_round = r.get("results_per_round", [])
        if results_per_round and "train_accuracy" in results_per_round[-1]:
            train_acc = results_per_round[-1]["train_accuracy"] * 100
            test_acc = r.get("final_test_accuracy", 0) * 100
            overfit_gap = train_acc - test_acc
            overfit_results.append((r.get("bit_width"), overfit_gap))

    if overfit_results:
        avg_overfit = sum(gap for _, gap in overfit_results) / len(overfit_results)
        max_overfit_bit, max_overfit_gap = max(overfit_results, key=lambda x: x[1])
        print(f"🔍 Average overfitting: {avg_overfit:.2f}% (train - test)")
        print(f"⚠️  Most overfitting: {max_overfit_bit}-bit ({max_overfit_gap:+.2f}%)")

    # Find sweet spot (best accuracy/compression trade-off)
    for r in results:
        comp_ratio = r.get("compression_metrics", {}).get("compression_ratio", 1.0)
        acc = r.get("final_test_accuracy", 0) * 100
        acc_drop = abs(r.get("accuracy_drop_percent", 0))

        # Sweet spot: >2x compression with <5% accuracy drop
        if comp_ratio >= 2.0 and acc_drop < 5.0 and r.get("bit_width") != 32:
            print(f"💡 Sweet spot: {r.get('bit_width')}-bit - {comp_ratio:.1f}x compression, {acc_drop:.2f}% accuracy drop")
            break

    print()


def main():
    """Analyze quantization results."""
    results = find_all_results()

    if not results:
        print("❌ No results found")
        print("\nMake sure experiments have completed and saved training_results.json files")
        sys.exit(1)

    # Calculate accuracy drops
    results = calculate_accuracy_drop(results)

    # Print comparison table
    print_comparison_table(results)

    # Print summary statistics
    print_summary_stats(results)

    # Save to files
    save_results_csv(results)
    save_results_json(results)

    print("\n✨ Analysis complete!")
    print()


if __name__ == "__main__":
    main()
