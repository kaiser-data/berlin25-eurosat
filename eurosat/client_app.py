"""eurosat: A Flower / PyTorch app."""

import os
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from eurosat.task import Net, load_data
from eurosat.task import test as test_fn
from eurosat.task import train as train_fn
from eurosat.quantization import WeightQuantizer

# Enable PyTorch multi-threading for better CPU utilization
torch.set_num_threads(6)  # Match cluster vCPU allocation

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Get quantization bit-width from config (with environment variable fallback)
    bit_width = context.run_config.get("quantization-bits", int(os.getenv("QUANTIZATION_BITS", "32")))

    # Load the model with quantized layers and initialize with received weights
    model = Net(bit_width=bit_width)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data with configurable batch size
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config.get("batch-size", 32)
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Model already trained with quantized weights in forward pass
    # No need for post-training quantization

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Get quantization bit-width from config (with environment variable fallback)
    bit_width = context.run_config.get("quantization-bits", int(os.getenv("QUANTIZATION_BITS", "32")))

    # Load the model with quantized layers and initialize with received weights
    model = Net(bit_width=bit_width)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data with configurable batch size
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config.get("batch-size", 32)
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
