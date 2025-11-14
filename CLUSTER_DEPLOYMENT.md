# 🚀 Cluster Deployment Guide

## Quick Start

### 1️⃣ Upload Code to Cluster

```bash
# From your local machine, upload the project
scp -r /Users/marty/AIHack-Berlin/flwr/berlin25-eurosat <username>@<cluster-address>:~/

# OR use rsync for faster sync
rsync -avz --progress /Users/marty/AIHack-Berlin/flwr/berlin25-eurosat/ <username>@<cluster-address>:~/berlin25-eurosat/
```

### 2️⃣ SSH into Cluster

```bash
ssh <username>@<cluster-address>
```

### 3️⃣ Install Dependencies

```bash
cd ~/berlin25-eurosat

# Install in editable mode
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 4️⃣ Set Up WandB Authentication

```bash
# Set your WandB API key
export WANDB_API_KEY="your-api-key-here"

# Or configure interactively
wandb login
```

### 5️⃣ Test Locally on Cluster (CPU)

```bash
# Quick test with local simulation
flwr run . local-simulation

# Check outputs
ls -lh ~/berlin25-eurosat/outputs/
```

### 6️⃣ Submit GPU Job

```bash
# Submit job to GPU queue
./submit-job.sh "flwr run . cluster-gpu" --gpu --name "eurosat-baseline"

# Monitor job status
squeue -u $USER

# Check logs (replace job_id with actual ID)
tail -f ~/logs/job<id>_eurosat-baseline.out
```

---

## 📋 Pre-Deployment Checklist

- [ ] **Code uploaded** to `~/berlin25-eurosat/`
- [ ] **Dependencies installed** (`pip install -e .`)
- [ ] **WandB configured** (`export WANDB_API_KEY=...`)
- [ ] **Cluster federation configs added** (already done in pyproject.toml)
- [ ] **Test run completed** (`flwr run . local-simulation`)
- [ ] **Dataset cached** (automatic on first run to `~/.cache/huggingface/`)

---

## 🎯 Running Different Configurations

### CPU Training (Debugging)
```bash
./submit-job.sh "flwr run . cluster-cpu" --name "test-cpu"
```

### GPU Training (Production)
```bash
./submit-job.sh "flwr run . cluster-gpu" --gpu --name "baseline-gpu"
```

### Custom Hyperparameters
Edit `pyproject.toml` before submitting:
```toml
[tool.flwr.app.config]
num-server-rounds = 10    # Increase rounds
fraction-train = 1.0      # Use all clients
local-epochs = 3          # More local training
lr = 0.01                 # Higher learning rate
```

Then submit:
```bash
./submit-job.sh "flwr run . cluster-gpu" --gpu --name "optimized-config"
```

---

## 📊 Monitoring & Debugging

### Check Job Status

**Essential Slurm Commands**:

| Command | Description |
|---------|-------------|
| `squeue -u $USER` | Show your running and queued jobs |
| `squeue -l` | Detailed view with runtime, node, and resource usage |
| `sacct -u $USER` | View history of completed jobs |
| `sinfo` | Show available nodes and partition status |
| `scancel <job_id>` | Cancel a specific job |
| `scontrol show job <job_id>` | Display detailed job info |
| `tail -f slurm-<job_id>.out` | Stream your job's live output |

**Common Usage**:
```bash
# View your jobs
squeue -u $USER

# Detailed view
squeue -u $USER -l

# View job history
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed

# Cancel a job
scancel <job_id>

# Check cluster status
sinfo
```

### Monitor Logs
```bash
# Real-time log monitoring
tail -f ~/logs/job<id>_<name>.out

# Search for errors
grep -i "error\|exception\|failed" ~/logs/job<id>_<name>.out

# Check WandB uploads
grep -i "wandb" ~/logs/job<id>_<name>.out
```

### Check Outputs
```bash
# List saved models
ls -lh ~/berlin25-eurosat/outputs/

# Check latest run
ls -lh ~/berlin25-eurosat/outputs/$(ls -t ~/berlin25-eurosat/outputs/ | head -1)/
```

### WandB Dashboard
Access your runs at: https://wandb.ai/<your-username>/Hackathon-Berlin25-Eurosat

---

## 🔧 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `task.py`
```python
# task.py:73
trainloader = DataLoader(partition_train_test["train"], batch_size=16, shuffle=True)  # Reduced from 32
testloader = DataLoader(partition_train_test["test"], batch_size=16)  # Reduced from 32
```

### Issue: "Dataset download timeout"
**Solution**: Pre-download dataset
```bash
# SSH into cluster
python -c "from flwr_datasets import FederatedDataset; FederatedDataset(dataset='tanganke/eurosat', partitioners={'train': 10})"
```

### Issue: "WandB not logging"
**Solution**: Check API key and network
```bash
# Verify WandB is configured
wandb status

# Test connection
python -c "import wandb; wandb.init(project='test', mode='online')"
```

### Issue: "Job stuck in queue"
**Solution**: Check cluster load
```bash
# View all jobs
squeue

# Check node availability
sinfo
```

### Issue: "Module not found"
**Solution**: Reinstall dependencies
```bash
cd ~/berlin25-eurosat
pip install -e . --force-reinstall --no-cache-dir
```

---

## 🚀 Performance Optimization Tips

### 1. **Dataset Caching**
First run downloads dataset to `~/.cache/huggingface/`. Subsequent runs are faster.

### 2. **Use /scratch for Data**
For large datasets, consider copying to `/scratch/`:
```bash
# In your job script (future optimization)
cp -r ~/.cache/huggingface/datasets /scratch/datasets
export HF_DATASETS_CACHE=/scratch/datasets
```

### 3. **Increase DataLoader Workers**
Modify `task.py` for parallel data loading:
```python
trainloader = DataLoader(..., num_workers=4, pin_memory=True)
```

### 4. **Enable Mixed Precision Training**
For faster GPU training (future enhancement):
```python
from torch.cuda.amp import autocast, GradScaler
# Use in training loop
```

---

## 📈 Expected Performance

### Local Simulation (CPU)
- **Time per round**: ~2-5 minutes
- **Total time (3 rounds)**: ~6-15 minutes
- **GPU utilization**: 0%

### Cluster CPU
- **Time per round**: ~1-3 minutes (better CPU)
- **Total time (3 rounds)**: ~3-9 minutes
- **GPU utilization**: 0%

### Cluster GPU (AMD MI300X)
- **Time per round**: ~30-60 seconds ⚡
- **Total time (3 rounds)**: ~2-3 minutes ⚡
- **GPU utilization**: 30-60%
- **Speedup vs CPU**: 3-5x faster

---

## 🎯 Next Steps After Cluster Deployment

Once your baseline run completes successfully on the cluster:

1. ✅ **Verify outputs** in `~/berlin25-eurosat/outputs/`
2. ✅ **Check WandB dashboard** for metrics
3. ✅ **Review logs** for any warnings
4. 🚀 **Implement quantization loop** (next phase)

---

## 📞 Getting Help

- **Check logs first**: `tail -f ~/logs/job<id>_<name>.out`
- **Ask organizers** for cluster-specific issues
- **Review Flower docs**: https://flower.ai/docs/
- **Check WandB status**: https://status.wandb.ai/

Good luck with your cluster deployment! 🚀
