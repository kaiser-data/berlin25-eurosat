#!/bin/bash
# Submit quantization experiments to cluster queue
# Usage: ./submit_all_quantization.sh [--gpu]

GPU_FLAG=""
if [[ "$1" == "--gpu" ]]; then
    GPU_FLAG="--gpu"
    FEDERATION="cluster-gpu"
    echo "🎮 Using GPU acceleration"
else
    FEDERATION="cluster-cpu"
    echo "💻 Using CPU only"
fi

echo "========================================="
echo "🚀 Submitting Quantization Experiments"
echo "========================================="
echo "Federation: $FEDERATION"
echo ""

# Array of bit-widths to test
BIT_WIDTHS=(32 16 8 4 2 1)

# Submit each experiment as a separate job
for BITS in "${BIT_WIDTHS[@]}"; do
    JOB_NAME="quant-${BITS}bit"

    echo "📤 Submitting ${BITS}-bit experiment..."

    # Create a wrapper command that sets the environment variable
    CMD="export QUANTIZATION_BITS=${BITS} && flwr run . ${FEDERATION}"

    # Submit using the cluster's submit-job.sh
    ./submit-job.sh "$CMD" --name "$JOB_NAME" $GPU_FLAG

    echo "✅ Job submitted: $JOB_NAME"

    # Small delay to avoid overwhelming the queue
    sleep 2
done

echo ""
echo "========================================="
echo "📊 All jobs submitted!"
echo "========================================="
echo ""
echo "Monitor your jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  ls -lt ~/logs/"
echo "  tail -f ~/logs/job*quant*.out"
echo ""
echo "Results will be saved to:"
echo "  ~/berlin25-eurosat/outputs/"
