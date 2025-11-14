# EuroSAT — Flower / PyTorch App (Track 2)

This app demonstrates federated learning on the **EuroSAT** remote-sensing dataset using Flower and PyTorch.  
It’s designed to help participants explore and understand how the dataset and federated setup work.

---

## 🚀 Quickstart

### 1) Install

First, install dependencies in editable mode so that changes you make to the app code are immediately reflected in your environment.

```bash
pip install -e .
```

### 2) Run (Simulation Engine)

You can run the app using Flower’s **Simulation Runtime**, which lets you simulate multiple clients locally.  
This is the easiest way to experiment and debug your app before scaling it to real devices.

Run with default configuration:

```bash
flwr run .
```

> **Tip:** Your `pyproject.toml` file can define more than just dependencies — it can also include **hyperparameters** (like `lr` or `num-server-rounds`) and control which **Flower Runtime** is used.  
> By default, this app uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.  
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

---

## 🌸 Explore Flower Datasets

The code below shows how to **load**, **transform**, and **inspect** the EuroSAT dataset using `flwr_datasets`.  
Each client receives a separate partition of the dataset which is to simulate data distributed across multiple devices.

```python
from flwr_datasets import FederatedDataset
from torchvision import transforms

# Define a simple transform for a single image
img_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Create a federated dataset with 10 IID partitions
fds = FederatedDataset(dataset="tanganke/eurosat", partitioners={"train": 10})

# Load one partition for local exploration
partition = fds.load_partition(0, split="train")

# HF Datasets' with_transform gets a *batch* dict: each value is a list
def hf_transform(batch):
    # Apply the torchvision transform to each image in the batch
    batch["image"] = [img_transform(img) for img in batch["image"]]
    return batch

partition = partition.with_transform(hf_transform)

# Inspect a few samples
for i in range(3):
    sample = partition[i]
    print(sample["image"].shape, sample["label"])

```

> 💡 **What this does:**  
> - Loads EuroSAT (RGB version) from Hugging Face using Flower’s dataset abstraction.  
> - Partitions it into 10 federated clients (simulated).  
> - Applies a transformation (resize + tensor) to images.  
> - Prints out a few samples so you can confirm everything works before training.

---

### 📊 Visualize Label Distributions

Understanding **how data is distributed** across clients is key to federated learning — it helps explain differences in model performance and convergence.

In the following, we can observe how to load the dataset, apply a transformation to each image, and then visualize how the class labels are distributed across all client partitions. It also allows plotting the label distribution for a single client to inspect the local data balance.

```python
from flwr_datasets import FederatedDataset
from flwr_datasets.visualization import plot_label_distributions
import matplotlib.pyplot as plt

fds = FederatedDataset(
    dataset="tanganke/eurosat",
    partitioners={"train": 10}, 
)

# Get the partitioner used for "train"
partitioner = fds.partitioners["train"]

# Plot label distribution across *all* 10 partitions
fig, ax, df = plot_label_distributions(
    partitioner=partitioner,
    label_name="label",
    plot_type="bar",          
    size_unit="percent",      
    partition_id_axis="x",    
    legend=True,
    verbose_labels=False,     
    title="EuroSAT Label Distribution per Client (train)",
)

plt.show()
print(df.head())
```
> 💡 **What this does:**  
> - Uses Flower’s built-in `plot_label_distributions` utility to visualize the dataset split across clients.  
> - Quickly shows whether partitions are IID or skewed.  
> - Returns a pandas DataFrame (`df`) so you can further analyze counts or percentages per class.

---

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

---

## 📚 Learn More

- [Flower PyTorch Quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
- [Simulation Docs](https://flower.ai/docs/framework/how-to-run-simulations.html)
- [Dataset Visualization Guide](https://flower.ai/docs/datasets/tutorial-visualize-label-distribution.html)
- [Join the Flower Slack](https://flower.ai/join-slack)
- [Flower GitHub](https://github.com/adap/flower)
- [Flower Website](https://flower.ai/)

