#!/bin/bash
# Submit real INT8 quantization comparison job
# This trains ONE FP32 model, then quantizes to INT8 for true speedup comparison

GPU_FLAG="--gpu"
JOB_NAME="real-int8-comparison"

echo "========================================="
echo "🔬 Submitting Real Quantization Comparison"
echo "========================================="
echo "Job: $JOB_NAME"
echo ""

# Create command
CMD="cd ~/berlin25-eurosat && \
export PYTHONPATH=/home/team11/berlin25-eurosat && \
python real_quantization_comparison.py"

# Submit using the cluster's submit-job.sh
~/submit-job.sh "$CMD" --name "$JOB_NAME" $GPU_FLAG

echo "✅ Job submitted: $JOB_NAME"
echo ""
echo "This will:"
echo "  1. Train ONE FP32 model (~3-5 minutes)"
echo "  2. Apply PyTorch native INT8 quantization"
echo "  3. Measure REAL compression, speedup, accuracy"
echo ""
echo "Monitor job:"
echo "  squeue -u team11"
echo "  tail -f ~/logs/job*${JOB_NAME}*.out"
echo ""
echo "Results will be in:"
echo "  ~/berlin25-eurosat/outputs/real_quantization_comparison/"
