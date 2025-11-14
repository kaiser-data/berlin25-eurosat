#!/bin/bash
# Automated script to run training for all bit-widths

set -e  # Exit on error

FEDERATION=${1:-"local-simulation"}  # Default to local-simulation
RESULTS_DIR="quantization_results_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "🚀 Quantization Training Loop"
echo "========================================="
echo "Federation: $FEDERATION"
echo "Results directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Array of bit-widths to test
BIT_WIDTHS=(32 16 8 4 2 1)

# Run training for each bit-width
for BITS in "${BIT_WIDTHS[@]}"; do
    echo ""
    echo "========================================="
    echo "🔢 Training with $BITS-bit quantization"
    echo "========================================="

    # Set environment variable for bit-width
    export QUANTIZATION_BITS=$BITS

    # Run Flower training
    START_TIME=$(date +%s)

    if flwr run . $FEDERATION; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "✅ $BITS-bit training completed in ${DURATION}s"

        # Log result
        echo "$BITS-bit: SUCCESS (${DURATION}s)" >> "$RESULTS_DIR/summary.txt"
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "❌ $BITS-bit training failed after ${DURATION}s"

        # Log failure
        echo "$BITS-bit: FAILED (${DURATION}s)" >> "$RESULTS_DIR/summary.txt"
    fi

    # Copy outputs to results directory
    LATEST_OUTPUT=$(ls -td outputs/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_OUTPUT" ]; then
        cp -r "$LATEST_OUTPUT" "$RESULTS_DIR/${BITS}bit_output/"
        echo "📁 Results saved to $RESULTS_DIR/${BITS}bit_output/"
    fi

    # Wait between experiments
    if [ "$BITS" != "1" ]; then
        echo "⏳ Waiting 5 seconds before next experiment..."
        sleep 5
    fi
done

echo ""
echo "========================================="
echo "📊 All experiments complete!"
echo "========================================="
echo "Results saved to: $RESULTS_DIR"
cat "$RESULTS_DIR/summary.txt"
