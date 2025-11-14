#!/bin/bash
# Submit QAT vs PTQ comparison job
# Compares training with 8-bit vs training in 32-bit then quantizing

GPU_FLAG="--gpu"
JOB_NAME="qat-vs-ptq-8bit"

echo "========================================="
echo "🔬 Submitting QAT vs PTQ Comparison"
echo "========================================="
echo "Job: $JOB_NAME"
echo ""

# Create command
CMD="cd ~/berlin25-eurosat && \
export PYTHONPATH=/home/team11/berlin25-eurosat && \
python compare_qat_vs_ptq.py"

# Submit using the cluster's submit-job.sh
~/submit-job.sh "$CMD" --name "$JOB_NAME" $GPU_FLAG

echo "✅ Job submitted: $JOB_NAME"
echo ""
echo "This experiment will:"
echo "  1. Train with 8-bit QAT (quantization-aware training)"
echo "  2. Train with 32-bit FP32"
echo "  3. Quantize FP32 model to INT8 (post-training quantization)"
echo "  4. Compare which approach gives better 8-bit accuracy"
echo ""
echo "Expected time: ~10 minutes (2 training runs)"
echo ""
echo "Monitor job:"
echo "  squeue -u team11"
echo "  tail -f ~/logs/job*${JOB_NAME}*.out"
echo ""
echo "Results will be in:"
echo "  ~/berlin25-eurosat/outputs/qat_vs_ptq_comparison/"
echo ""
echo "Question being answered:"
echo "  Does training with 8-bit from start (QAT) beat"
echo "  training in 32-bit then quantizing (PTQ)?"
