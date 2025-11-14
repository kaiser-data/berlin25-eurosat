# 🔢 Quantization Training Loop

Automated training with different weight quantization bit-widths: **32, 16, 8, 4, 2, 1 bits**.

## 🎯 What This Does

Each bit-width gets its **own independent training run** that:
1. ✅ Trains a model from scratch with quantized weights
2. ✅ Measures actual training accuracy
3. ✅ Calculates real model size (MB)
4. ✅ Computes compression ratio
5. ✅ Saves all results for comparison

## 🚀 Quick Start - Cluster

### Submit All Experiments (GPU - Recommended)

```bash
cd ~/berlin25-eurosat
./submit_all_quantization.sh --gpu
```

This will submit **6 separate jobs** to the cluster queue:
- `quant-32bit` - Baseline (FP32)
- `quant-16bit` - Half precision
- `quant-8bit` - 8-bit quantization
- `quant-4bit` - 4-bit quantization
- `quant-2bit` - 2-bit quantization
- `quant-1bit` - Binary weights

### Monitor Jobs

```bash
# Check queue status
squeue -u team11

# Watch logs in real-time
tail -f ~/logs/job*quant*.out

# List all logs
ls -lt ~/logs/ | grep quant
```

### Analyze Results

Once all jobs complete:

```bash
cd ~/berlin25-eurosat
python analyze_results.py
```

Output example:
```
Bit-Width | Size (MB) | Compression | Reduction | Output Directory
----------------------------------------------------------------------
       32 |     10.23 |        1.00x |      0.0% | 21-30-45
       16 |      5.12 |        2.00x |     50.0% | 21-35-12
        8 |      2.56 |        4.00x |     75.0% | 21-40-33
        4 |      1.28 |        8.00x |     87.5% | 21-45-21
        2 |      0.64 |       16.00x |     93.8% | 21-50-08
        1 |      0.32 |       32.00x |     96.9% | 21-55-42
```

---

## 📊 What Gets Measured

For each bit-width, you get:

### 📈 Training Metrics
- Round-by-round test loss
- Round-by-round test accuracy
- Training duration

### 📦 Compression Metrics
- Original model size (FP32 baseline)
- Compressed model size
- Compression ratio (e.g., 8x for 4-bit)
- Size reduction percentage

### 💾 Saved Files
Each experiment saves to `~/berlin25-eurosat/outputs/YYYY-MM-DD/HH-MM-SS/`:
- `final_model_fp32.pt` - Full precision model
- `final_model_{N}bit.pt` - Quantized model (if N < 32)
- `compression_metrics.json` - Size statistics

---

## 🔧 Manual Single Experiment

To run just one bit-width:

```bash
# Example: Run 8-bit quantization on GPU
export QUANTIZATION_BITS=8
./submit-job.sh "flwr run . cluster-gpu" --name "test-8bit" --gpu

# Check logs
tail -f ~/logs/job*test-8bit.out
```

---

## 📁 Results Structure

```
berlin25-eurosat/
├── outputs/
│   ├── 2025-11-14/
│   │   ├── 21-30-45/          # 32-bit run
│   │   │   ├── final_model_fp32.pt
│   │   │   └── compression_metrics.json
│   │   ├── 21-35-12/          # 16-bit run
│   │   │   ├── final_model_fp32.pt
│   │   │   ├── final_model_16bit.pt
│   │   │   └── compression_metrics.json
│   │   └── ...
└── logs/
    ├── job100_quant-32bit.out
    ├── job101_quant-16bit.out
    └── ...
```

---

## ⏱️ Expected Runtime

**Per experiment** (with 3 rounds, 10 clients):

| Environment | Time |
|-------------|------|
| CPU | ~3-5 minutes |
| GPU (AMD MI300X) | ~1-2 minutes ⚡ |

**Total for all 6 experiments**:
- CPU: ~18-30 minutes
- GPU: ~6-12 minutes ⚡

---

## 🎓 Understanding the Results

### Compression vs. Accuracy Trade-off

- **32-bit (FP32)**: Baseline accuracy, largest model
- **16-bit**: ~50% size reduction, minimal accuracy loss
- **8-bit**: ~75% size reduction, small accuracy loss
- **4-bit**: ~87.5% size reduction, moderate accuracy loss
- **2-bit**: ~93.8% size reduction, significant accuracy loss
- **1-bit (Binary)**: ~96.9% size reduction, large accuracy loss

### What to Look For

✅ **Sweet spot**: Best compression with acceptable accuracy
✅ **Diminishing returns**: Where accuracy drops too much
✅ **Communication savings**: Smaller models = faster federated learning

---

## 🐛 Troubleshooting

### Jobs not starting?
```bash
# Check queue
squeue -u team11

# Check job details
scontrol show job <job_id>
```

### Out of memory?
Lower batch size in `eurosat/task.py`:
```python
trainloader = DataLoader(..., batch_size=16)  # Reduced from 32
```

### Results not appearing?
```bash
# Check if outputs directory exists
ls -la ~/berlin25-eurosat/outputs/

# Check logs for errors
grep -i error ~/logs/job*quant*.out
```

---

## 🎯 Next Steps

After running all experiments:

1. ✅ Analyze results with `python analyze_results.py`
2. ✅ Compare accuracy vs. compression trade-offs
3. ✅ Choose optimal bit-width for your use case
4. ✅ Check WandB dashboard (if enabled) for detailed metrics

Good luck! 🚀
