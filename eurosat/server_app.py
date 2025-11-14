"""eurosat: A Flower / PyTorch app."""

import os
import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader
import logging
import json

from eurosat.task import Net, test, apply_transforms, create_run_dir
from eurosat.quantization import WeightQuantizer, get_compression_metrics

# Optional WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create ServerApp
app = ServerApp()

PROJECT_NAME = "Hackathon-Berlin25-Eurosat"

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Get quantization bit-width from environment variable
    bit_width = int(os.getenv("QUANTIZATION_BITS", "32"))

    # Create run directory
    run_dir, save_path = create_run_dir()

    # Initialize Weights & Biases logging (optional)
    run_name = f"{str(run_dir)}-{bit_width}bit"
    if WANDB_AVAILABLE:
        wandb.init(
            project=PROJECT_NAME,
            name=run_name,
            config={"bit_width": bit_width}
        )

    print(f"\n{'='*70}")
    print(f"🔢 Quantization Training: {bit_width}-bit weights")
    print(f"📁 Output directory: {save_path}")
    print(f"{'='*70}\n")

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()

    # Calculate and log compression metrics
    compression_metrics = get_compression_metrics(global_model, bit_width)
    print(f"📊 Model Compression Metrics:")
    print(f"   Bit-width: {bit_width} bits")
    print(f"   Original size (FP32): {compression_metrics['original_size_mb']:.2f} MB")
    print(f"   Compressed size: {compression_metrics['compressed_size_mb']:.2f} MB")
    print(f"   Compression ratio: {compression_metrics['compression_ratio']:.2f}x")
    print(f"   Size reduction: {compression_metrics['size_reduction_percent']:.1f}%\n")

    # Save compression metrics
    with open(f"{save_path}/compression_metrics.json", "w") as f:
        json.dump(compression_metrics, f, indent=2)

    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(),
    )

    # Save final model to disk
    print(f"\nSaving final model to disk at {save_path}...")
    state_dict = result.arrays.to_torch_state_dict()

    # Save full precision model
    torch.save(state_dict, f"{save_path}/final_model_fp32.pt")

    # Apply quantization and save quantized model
    if bit_width < 32:
        quantizer = WeightQuantizer(bit_width)
        model_for_quant = Net()
        model_for_quant.load_state_dict(state_dict)

        quantized_state, quant_params = quantizer.quantize_model(model_for_quant)

        # Save quantized model and parameters
        torch.save({
            'quantized_state': quantized_state,
            'quant_params': quant_params,
            'bit_width': bit_width
        }, f"{save_path}/final_model_{bit_width}bit.pt")

        print(f"✅ Saved {bit_width}-bit quantized model")

        # Calculate actual file sizes
        import os as os_module
        fp32_size = os_module.path.getsize(f"{save_path}/final_model_fp32.pt") / (1024 * 1024)
        quant_size = os_module.path.getsize(f"{save_path}/final_model_{bit_width}bit.pt") / (1024 * 1024)

        print(f"\n📦 Actual File Sizes:")
        print(f"   FP32 model: {fp32_size:.2f} MB")
        print(f"   {bit_width}-bit model: {quant_size:.2f} MB")
        print(f"   Actual compression: {fp32_size/quant_size:.2f}x")


def get_global_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # This is the exact same dataset as the one downloaded by the clients via
        # FlowerDatasets. However, we don't use FlowerDatasets for the server since
        # partitioning is not needed.
        # We make use of the "test" split only
        global_test_set = load_dataset("tanganke/eurosat")["test"]

        testloader = DataLoader(
            global_test_set.with_transform(apply_transforms),
            batch_size=32,
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Apply global model parameters
        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        net.to(device)
        # Evaluate global model on test set
        loss, accuracy = test(net, testloader, device=device)

        # Log to WandB if available
        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "Global Test Loss": loss,
                    "Global Test Accuracy": accuracy,
                }
            )

        # Always print metrics
        print(f"Round {server_round} - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate