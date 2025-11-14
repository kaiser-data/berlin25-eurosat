#!/usr/bin/env python3
"""Real quantization comparison: 32-bit vs INT8 with actual speedup.

This script:
1. Trains ONE model in FP32 (32-bit)
2. Applies PyTorch native INT8 quantization
3. Compares: accuracy, model size, inference speed

This is Post-Training Quantization (PTQ) - real INT8 operations.
"""

import torch
import torch.nn as nn
from pathlib import Path
import time
import json
from datasets import load_dataset
from torch.utils.data import DataLoader

from eurosat.task import Net, apply_transforms, test


def get_model_size_mb(model):
    """Calculate actual model size in MB."""
    torch.save(model.state_dict(), "temp_model.pt")
    size_mb = Path("temp_model.pt").stat().st_size / (1024 * 1024)
    Path("temp_model.pt").unlink()
    return size_mb


def train_fp32_model(num_epochs=10, lr=0.001, batch_size=64):
    """Train a single FP32 model."""
    print("="*70)
    print("🚀 Training FP32 (32-bit) Baseline Model")
    print("="*70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load full training dataset
    train_dataset = load_dataset("tanganke/eurosat", split="train")
    test_dataset = load_dataset("tanganke/eurosat", split="test")

    trainloader = DataLoader(
        train_dataset.with_transform(apply_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    testloader = DataLoader(
        test_dataset.with_transform(apply_transforms),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    # Create and train model
    model = Net(bit_width=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total

        # Test accuracy
        test_loss, test_acc = test(model, testloader, device)

        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%")

    training_time = time.time() - start_time

    print(f"\n✅ Training complete in {training_time:.1f}s")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    return model, testloader, {
        "training_time": training_time,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
    }


def quantize_to_int8(model_fp32, testloader):
    """Apply PyTorch native INT8 quantization."""
    print("\n" + "="*70)
    print("🔧 Applying INT8 Dynamic Quantization")
    print("="*70)

    device = torch.device("cpu")  # Quantization requires CPU
    model_fp32_cpu = model_fp32.to(device)

    # Apply dynamic quantization to Linear layers
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32_cpu,
        {nn.Linear},  # Quantize only Linear layers
        dtype=torch.qint8
    )

    print("✅ INT8 quantization applied")

    return model_int8


def measure_inference_speed(model, testloader, device, num_batches=50):
    """Measure inference speed."""
    model.eval()

    times = []
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            if i >= num_batches:
                break

            images = batch["image"].to(device)

            start = time.time()
            _ = model(images)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.time()

            times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time * 1000  # Convert to ms


def compare_models(model_fp32, model_int8, testloader):
    """Compare FP32 vs INT8 models."""
    print("\n" + "="*70)
    print("📊 Comparison: FP32 vs INT8")
    print("="*70)

    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    # Model sizes
    fp32_size = get_model_size_mb(model_fp32)
    int8_size = get_model_size_mb(model_int8)

    print("\n📦 Model Size:")
    print(f"   FP32: {fp32_size:.2f} MB")
    print(f"   INT8: {int8_size:.2f} MB")
    print(f"   Compression: {fp32_size/int8_size:.2f}x")
    print(f"   Size reduction: {(1 - int8_size/fp32_size)*100:.1f}%")

    # Accuracy comparison
    print("\n🎯 Accuracy:")
    model_fp32_cpu = model_fp32.to(device_cpu)

    loss_fp32, acc_fp32 = test(model_fp32_cpu, testloader, device_cpu)
    loss_int8, acc_int8 = test(model_int8, testloader, device_cpu)

    print(f"   FP32: {acc_fp32*100:.2f}%")
    print(f"   INT8: {acc_int8*100:.2f}%")
    print(f"   Accuracy drop: {(acc_fp32 - acc_int8)*100:.2f}%")

    # Inference speed (CPU only - INT8 optimized for CPU)
    print("\n⚡ Inference Speed (CPU):")
    fp32_time = measure_inference_speed(model_fp32_cpu, testloader, device_cpu)
    int8_time = measure_inference_speed(model_int8, testloader, device_cpu)

    print(f"   FP32: {fp32_time:.2f} ms/batch")
    print(f"   INT8: {int8_time:.2f} ms/batch")
    print(f"   Speedup: {fp32_time/int8_time:.2f}x")

    return {
        "fp32": {
            "size_mb": fp32_size,
            "accuracy": acc_fp32,
            "loss": loss_fp32,
            "inference_time_ms": fp32_time,
        },
        "int8": {
            "size_mb": int8_size,
            "accuracy": acc_int8,
            "loss": loss_int8,
            "inference_time_ms": int8_time,
        },
        "comparison": {
            "compression_ratio": fp32_size / int8_size,
            "size_reduction_percent": (1 - int8_size/fp32_size) * 100,
            "accuracy_drop_percent": (acc_fp32 - acc_int8) * 100,
            "speedup": fp32_time / int8_time,
        }
    }


def main():
    """Run real quantization comparison."""
    print("\n" + "="*70)
    print("🔬 Real Quantization Comparison: FP32 vs INT8")
    print("="*70)
    print("\nThis uses PyTorch native INT8 quantization for REAL speedup!\n")

    # Train FP32 model
    model_fp32, testloader, train_results = train_fp32_model(
        num_epochs=10,
        lr=0.001,
        batch_size=64
    )

    # Quantize to INT8
    model_int8 = quantize_to_int8(model_fp32, testloader)

    # Compare models
    comparison_results = compare_models(model_fp32, model_int8, testloader)

    # Save results
    output_dir = Path("outputs/real_quantization_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "training": train_results,
        "comparison": comparison_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save models
    torch.save(model_fp32.state_dict(), output_dir / "model_fp32.pt")
    torch.save(model_int8.state_dict(), output_dir / "model_int8.pt")

    print(f"\n✅ Results saved to {output_dir}/")
    print("\n" + "="*70)
    print("Summary:")
    print(f"  Training time: {train_results['training_time']:.1f}s")
    print(f"  Compression: {comparison_results['comparison']['compression_ratio']:.2f}x")
    print(f"  Speedup: {comparison_results['comparison']['speedup']:.2f}x")
    print(f"  Accuracy drop: {comparison_results['comparison']['accuracy_drop_percent']:.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
