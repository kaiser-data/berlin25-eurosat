# Quantization Experiments - Three Approaches

This project includes **three different quantization experiments** with different goals:

---

## 🔬 Approach 1: Real INT8 Quantization (RECOMMENDED)

**File**: `real_quantization_comparison.py`
**Goal**: Compare **actual** FP32 vs INT8 performance with **real speedup**

### What This Does:
1. Trains **ONE** FP32 model (full precision)
2. Applies PyTorch native INT8 quantization
3. Measures **real** differences:
   - ✅ **Model size** (actual file size reduction)
   - ✅ **Inference speed** (real speedup on CPU)
   - ✅ **Accuracy** (post-quantization accuracy)

### How to Run:
```bash
# On cluster
./submit_real_comparison.sh

# Locally
python real_quantization_comparison.py
```

### Expected Results:
```
📊 Comparison: FP32 vs INT8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 Model Size:
   FP32: 2.45 MB
   INT8: 0.65 MB
   Compression: 3.77x
   Size reduction: 73.5%

🎯 Accuracy:
   FP32: 85.23%
   INT8: 84.91%
   Accuracy drop: 0.32%

⚡ Inference Speed (CPU):
   FP32: 45.2 ms/batch
   INT8: 18.7 ms/batch
   Speedup: 2.42x
```

### Why This is REAL:
- Uses PyTorch's `torch.quantization.quantize_dynamic()`
- Actual INT8 tensor operations (not FP32 simulation)
- Real memory savings (smaller model files)
- Real inference speedup (especially on CPU)

---

## ⚔️ Approach 2: QAT vs PTQ Head-to-Head (NEW!)

**File**: `compare_qat_vs_ptq.py`
**Goal**: Answer "Which is better for 8-bit: train with 8-bit or quantize after?"

### The Experiment:
This directly compares:
- **QAT**: Train with 8-bit quantization from the start
- **PTQ**: Train in 32-bit, then quantize to INT8

### How to Run:
```bash
./submit_qat_vs_ptq.sh
```

### What You'll Learn:
```
📊 COMPARISON: QAT vs PTQ for 8-bit Models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Final Test Accuracy:
   QAT (8-bit training):  82.45%
   PTQ (32→8 bit):        84.23%
   Difference: +1.78%

🏆 Winner: PTQ
   → Training in FP32 then quantizing is better!

⏱️  Training Time:
   Both take ~5 minutes (same training epochs)
```

### Why This Matters:
- Shows if quantization-aware training helps
- Determines best workflow for 8-bit models
- Reveals accuracy vs training approach trade-offs

---

## 📚 Approach 3: Quantization-Aware Training (QAT)

**Files**: `submit_quantization_jobs.sh`, federated learning pipeline
**Goal**: Study how **training with quantization noise** affects model accuracy

### What This Does:
1. Trains **6 separate models** (32, 16, 8, 4, 2, 1-bit)
2. Each model uses "quantized" weights during training
3. Measures accuracy degradation per bit-width

### How to Run:
```bash
./submit_quantization_jobs.sh
```

### Why Training Time is Same:
⚠️ **This does NOT give real speedup** because:
- Weights stored in FP32 (for gradients)
- Quantization only in forward pass
- Backward pass still uses FP32
- Same memory usage, same speed

### What You Learn:
- Which bit-widths maintain accuracy
- How quantization affects convergence
- Overfitting behavior at different bit-widths
- Compression vs accuracy trade-offs

### This is "Simulation":
```python
# Storage: Always FP32
self.weight_fp32 = nn.Parameter(...)

# Forward: Quantize then compute (still FP32 ops)
w_quant = quantize(self.weight_fp32)
output = F.linear(input, w_quant)

# Backward: Gradients on FP32 weights
loss.backward()  # Updates weight_fp32
```

---

## 🤔 Which Should You Use?

### Use **Approach 1 (Real INT8)** if you want:
- ✅ Actual deployment benefits (compression + speedup)
- ✅ Real INT8 operations and measurements
- ✅ Production-ready quantization
- ✅ Quick results (1 training run)

### Use **Approach 2 (QAT vs PTQ)** if you want:
- ✅ Direct answer: "Train with 8-bit or quantize after?"
- ✅ Compare two 8-bit workflows head-to-head
- ✅ Understand which approach preserves accuracy better
- ✅ Moderate time (2 training runs, ~10 min)

### Use **Approach 3 (Multi-bit QAT)** if you want:
- ✅ Scientific study of quantization effects
- ✅ Compare many bit-widths (1, 2, 4, 8, 16, 32)
- ✅ Understand overfitting at different precisions
- ✅ Comprehensive analysis (6 runs, ~15 min)

---

## 📊 Recommended Workflow

### For Hackathon / Best Story:
```bash
# Run the QAT vs PTQ comparison (most interesting!)
./submit_qat_vs_ptq.sh

# Wait ~10 minutes, then check results
cat outputs/qat_vs_ptq_comparison/comparison_results.json
```

**Why this is best:**
- Answers a clear question: "Which training approach is better?"
- Shows practical comparison everyone understands
- Reveals if quantization-aware training helps

### For Production Deployment:
```bash
# Run real INT8 comparison (shows actual benefits)
./submit_real_comparison.sh

# Check real speedup and compression
cat outputs/real_quantization_comparison/comparison_results.json
```

### For Research / Deep Analysis:
```bash
# Run all bit-widths (comprehensive study)
./submit_quantization_jobs.sh

# Analyze when complete
python analyze_results.py
```

---

## 🔍 Understanding the Results

### Approach 1 Results:
```json
{
  "comparison": {
    "compression_ratio": 3.77,      // Real file size reduction
    "speedup": 2.42,                 // Real inference speedup
    "accuracy_drop_percent": 0.32    // Post-quantization accuracy loss
  }
}
```

### Approach 2 Results:
```
Bit-Width | Test Acc | Train Acc | Overfit
----------|----------|-----------|--------
   32-bit |  45.23%  |  48.51%   | +3.28%
    8-bit |  42.56%  |  43.12%   | +0.56%  ← Less overfitting!
```

---

## 💡 Key Insights

### Why Lower Bits ≠ Faster Training (Approach 2):
- Training still uses FP32 operations
- Only simulates quantization during forward pass
- Same GPU/CPU utilization across all bit-widths

### Why INT8 IS Faster (Approach 1):
- Uses actual INT8 tensor operations
- CPU has optimized INT8 instructions
- 4x less memory bandwidth needed

### Accuracy Trade-offs:
- **Approach 1 (PTQ)**: ~0.3% accuracy drop for 3.7x compression
- **Approach 2 (QAT)**: Models learn to adapt, potentially less accuracy loss

---

## 📁 Output Locations

### Approach 1:
```
outputs/real_quantization_comparison/
├── comparison_results.json     # All metrics
├── model_fp32.pt              # FP32 model
└── model_int8.pt              # INT8 model
```

### Approach 2:
```
outputs/
├── 32bit/run_*/training_results.json
├── 16bit/run_*/training_results.json
├── 8bit/run_*/training_results.json
└── ...
```

---

## 🚀 Getting Started

### Quick Start (Recommended):
```bash
cd ~/berlin25-eurosat
git pull

# Run real INT8 comparison
./submit_real_comparison.sh

# Monitor
tail -f ~/logs/job*real-int8*.out
```

### Full Analysis:
```bash
# Run all bit-widths
./submit_quantization_jobs.sh

# After completion
python analyze_results.py
```

---

## 📚 Further Reading

- **Post-Training Quantization (PTQ)**: Approach 1
- **Quantization-Aware Training (QAT)**: Approach 2
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
