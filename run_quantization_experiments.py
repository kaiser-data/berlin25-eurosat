#!/usr/bin/env python3
"""Automated quantization experiments - trains models with different bit-widths."""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import argparse


def run_experiment(bit_width: int, federation: str, output_dir: Path) -> dict:
    """
    Run a single quantization experiment.

    Args:
        bit_width: Quantization bit-width
        federation: Federation config (local-simulation, cluster-cpu, cluster-gpu)
        output_dir: Directory to save results

    Returns:
        Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {bit_width}-bit quantization")
    print(f"Federation: {federation}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Set environment variable for bit-width
    env = {
        "QUANTIZATION_BITS": str(bit_width),
        "PATH": subprocess.os.environ.get("PATH", ""),
        "HOME": subprocess.os.environ.get("HOME", ""),
    }

    # Run Flower
    cmd = ["flwr", "run", ".", federation]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,  # 10 minute timeout
        )

        elapsed_time = time.time() - start_time

        # Save logs
        log_file = output_dir / f"experiment_{bit_width}bit.log"
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Bit-width: {bit_width}\n")
            f.write(f"Federation: {federation}\n")
            f.write(f"Duration: {elapsed_time:.2f}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        success = result.returncode == 0

        return {
            "bit_width": bit_width,
            "success": success,
            "duration_seconds": elapsed_time,
            "return_code": result.returncode,
            "log_file": str(log_file),
        }

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"⚠️  Experiment timed out after {elapsed_time:.2f}s")
        return {
            "bit_width": bit_width,
            "success": False,
            "duration_seconds": elapsed_time,
            "error": "timeout",
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Experiment failed: {e}")
        return {
            "bit_width": bit_width,
            "success": False,
            "duration_seconds": elapsed_time,
            "error": str(e),
        }


def main():
    """Run quantization experiments for all bit-widths."""
    parser = argparse.ArgumentParser(description="Run quantization experiments")
    parser.add_argument(
        "--federation",
        default="local-simulation",
        choices=["local-simulation", "cluster-cpu", "cluster-gpu"],
        help="Federation configuration to use",
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=[32, 16, 8, 4, 2, 1],
        help="Bit-widths to test (default: 32 16 8 4 2 1)",
    )
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"quantization_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    print(f"\n🚀 Starting Quantization Experiments")
    print(f"📊 Bit-widths: {args.bits}")
    print(f"🖥️  Federation: {args.federation}")
    print(f"📁 Output directory: {output_dir}\n")

    results = []
    total_start = time.time()

    for bit_width in args.bits:
        result = run_experiment(bit_width, args.federation, output_dir)
        results.append(result)

        # Print result
        status = "✅" if result["success"] else "❌"
        print(f"{status} {bit_width}-bit: {result['duration_seconds']:.2f}s")

        # Short pause between experiments
        if bit_width != args.bits[-1]:
            print("Waiting 5s before next experiment...")
            time.sleep(5)

    total_time = time.time() - total_start

    # Save summary
    summary = {
        "total_duration_seconds": total_time,
        "federation": args.federation,
        "timestamp": timestamp,
        "experiments": results,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"\nResults:")

    for r in results:
        status = "✅ Success" if r["success"] else "❌ Failed"
        print(f"  {r['bit_width']:2d}-bit: {status:12s} ({r['duration_seconds']:6.2f}s)")

    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📄 Summary: {summary_file}")

    # Success rate
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\n✅ Success rate: {successful}/{total} ({successful/total*100:.0f}%)")


if __name__ == "__main__":
    main()
