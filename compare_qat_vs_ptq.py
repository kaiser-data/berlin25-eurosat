#!/usr/bin/env python3
"""Compare Quantization-Aware Training (QAT) vs Post-Training Quantization (PTQ).

This script compares TWO approaches to get 8-bit models:

Approach 1 (QAT): Train with 8-bit quantization from the start
Approach 2 (PTQ): Train in 32-bit, then quantize to INT8

Which gives better accuracy?
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


def train_model(bit_width, num_epochs=10, lr=0.001, batch_size=64):
    """Train a model with specified bit-width."""
    approach = "QAT (8-bit training)" if bit_width == 8 else "FP32 baseline"

    print("\n" + "="*70)
    print(f"🚀 Training: {approach}")
    print("="*70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Bit-width: {bit_width}")

    # Load datasets
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

    # Create model with specified bit-width
    model = Net(bit_width=bit_width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining for {num_epochs} epochs...")
    start_time = time.time()

    best_test_acc = 0.0

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

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_acc*100:.2f}%, Test: {test_acc*100:.2f}%")

    training_time = time.time() - start_time

    print(f"\n✅ Training complete in {training_time:.1f}s")
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")

    return model, testloader, {
        "approach": approach,
        "bit_width": bit_width,
        "training_time": training_time,
        "final_test_accuracy": test_acc,
        "best_test_accuracy": best_test_acc,
        "final_test_loss": test_loss,
        "final_train_accuracy": train_acc,
    }


def apply_ptq_int8(model_fp32, testloader):
    """Apply Post-Training Quantization to INT8."""
    print("\n" + "="*70)
    print("🔧 Applying Post-Training Quantization (PTQ) to INT8")
    print("="*70)

    device = torch.device("cpu")
    model_fp32_cpu = model_fp32.to(device)

    # Apply PyTorch dynamic quantization
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32_cpu,
        {nn.Linear},
        dtype=torch.qint8
    )

    # Measure accuracy after quantization
    loss, acc = test(model_int8, testloader, device)

    print(f"✅ PTQ INT8 applied")
    print(f"   Accuracy after quantization: {acc*100:.2f}%")

    return model_int8, {
        "approach": "PTQ (FP32→INT8)",
        "bit_width": 8,
        "test_accuracy": acc,
        "test_loss": loss,
    }


def compare_approaches(qat_results, ptq_results, qat_model, ptq_model):
    """Compare QAT vs PTQ approaches."""
    print("\n" + "="*70)
    print("📊 COMPARISON: QAT vs PTQ for 8-bit Models")
    print("="*70)

    # Model sizes
    qat_size = get_model_size_mb(qat_model)
    ptq_size = get_model_size_mb(ptq_model)

    print("\n📦 Model Size:")
    print(f"   QAT (8-bit training):  {qat_size:.2f} MB")
    print(f"   PTQ (32→8 bit):        {ptq_size:.2f} MB")

    print("\n⏱️  Training Time:")
    print(f"   QAT (8-bit training):  {qat_results['training_time']:.1f}s")
    print(f"   PTQ (32-bit training): {ptq_results['training_time']:.1f}s")
    print(f"   Difference: {abs(qat_results['training_time'] - ptq_results['training_time']):.1f}s")

    print("\n🎯 Final Test Accuracy:")
    qat_acc = qat_results['final_test_accuracy'] * 100
    ptq_acc = ptq_results['test_accuracy'] * 100

    print(f"   QAT (8-bit training):  {qat_acc:.2f}%")
    print(f"   PTQ (32→8 bit):        {ptq_acc:.2f}%")
    print(f"   Difference: {ptq_acc - qat_acc:+.2f}%")

    # Determine winner
    print("\n🏆 Winner:")
    if ptq_acc > qat_acc + 0.5:
        print(f"   PTQ wins by {ptq_acc - qat_acc:.2f}%")
        print("   → Training in FP32 then quantizing is better!")
    elif qat_acc > ptq_acc + 0.5:
        print(f"   QAT wins by {qat_acc - ptq_acc:.2f}%")
        print("   → Training with 8-bit from start is better!")
    else:
        print("   Tie - both approaches give similar accuracy")
        print("   → Choose based on training time preference")

    return {
        "qat": {
            "size_mb": qat_size,
            "training_time": qat_results['training_time'],
            "final_accuracy": qat_acc,
            "best_accuracy": qat_results['best_test_accuracy'] * 100,
        },
        "ptq": {
            "size_mb": ptq_size,
            "training_time": ptq_results['training_time'],
            "final_accuracy": ptq_acc,
            "quantization_accuracy_drop": (ptq_results['baseline_accuracy'] - ptq_acc),
        },
        "comparison": {
            "accuracy_difference": ptq_acc - qat_acc,
            "time_difference": ptq_results['training_time'] - qat_results['training_time'],
            "winner": "PTQ" if ptq_acc > qat_acc else "QAT" if qat_acc > ptq_acc else "Tie",
        }
    }


def main():
    """Run QAT vs PTQ comparison."""
    print("\n" + "="*70)
    print("🔬 Experiment: QAT vs PTQ for 8-bit Quantization")
    print("="*70)
    print("\nQuestion: Which approach gives better 8-bit accuracy?")
    print("  1. QAT: Train with 8-bit from the start")
    print("  2. PTQ: Train in 32-bit, then quantize to INT8")
    print()

    # Approach 1: QAT - Train with 8-bit
    qat_model, testloader, qat_results = train_model(
        bit_width=8,
        num_epochs=10,
        lr=0.001,
        batch_size=64
    )

    # Approach 2: PTQ - Train in 32-bit, then quantize
    fp32_model, _, fp32_results = train_model(
        bit_width=32,
        num_epochs=10,
        lr=0.001,
        batch_size=64
    )

    # Quantize the FP32 model to INT8
    ptq_model, ptq_quant_results = apply_ptq_int8(fp32_model, testloader)

    # Combine PTQ results
    ptq_results = {
        **fp32_results,
        **ptq_quant_results,
        'baseline_accuracy': fp32_results['final_test_accuracy'] * 100,
    }

    # Compare the two approaches
    comparison = compare_approaches(qat_results, ptq_results, qat_model, ptq_model)

    # Save results
    output_dir = Path("outputs/qat_vs_ptq_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "experiment": "QAT vs PTQ for 8-bit quantization",
        "qat_results": qat_results,
        "ptq_results": ptq_results,
        "comparison": comparison,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save models
    torch.save(qat_model.state_dict(), output_dir / "model_qat_8bit.pt")
    torch.save(ptq_model.state_dict(), output_dir / "model_ptq_int8.pt")
    torch.save(fp32_model.state_dict(), output_dir / "model_baseline_fp32.pt")

    print(f"\n✅ Results saved to {output_dir}/")

    # Final summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    print(f"QAT (8-bit training):  {comparison['qat']['final_accuracy']:.2f}%")
    print(f"PTQ (32-bit→INT8):     {comparison['ptq']['final_accuracy']:.2f}%")
    print(f"Winner: {comparison['comparison']['winner']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
