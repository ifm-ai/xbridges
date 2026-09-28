## Environment
```
pip install transformers==5.13.0 accelerate fire
```
Run all commands from the repository root with `export PYTHONPATH=$PWD:$PYTHONPATH`.
Validation scripts also need [xLLM](https://github.com/ifm-ai/xllm#installation) installed.

## xLLM to HuggingFace
```
python -m xbridges.huggingface.xllm_to_hf_main \
  --xllm_dir /path/to/xllm/checkpoints/checkpoint_00005500 \
  --tokenizer_dir /path/to/tokenizer \
  --save_dir hf_ckpts/my-model
```
* Output defaults to BF16 safetensors. For FP32 `.bin` shards (e.g., numerical
  validation), add `--dtype float32 --safe_serialization false`.
* `--layers_per_load` is optional and limits how many layers are loaded at once.
* Output uses the bundled IFM `K2HorizonForCausalLM` implementation. Missing or
  unexpected weights and incompatible shapes cause an error. Use a new output
  directory; existing nonempty directories are never overwritten.
* RoPE is derived from the source checkpoint. Conversion does not implicitly
  apply the additional YaRN context extension used by the published 0.9B model.

### Parallel Conversion
For large models, `xllm_to_hf_parallel` converts layers in parallel and writes
FP32 `.bin` shards:
```
python -m xbridges.huggingface.xllm_to_hf_parallel \
  --xllm-dir /path/to/xllm/checkpoints/checkpoint_00005500 \
  --tokenizer-dir /path/to/tokenizer \
  --save-dir hf_ckpts/my-model \
  --workers 8
```
* To spread layers across a Slurm allocation, run `--mode prepare` once, one
  `--mode layer --layer N` per layer, then `--mode finalize`.

## HuggingFace to xLLM
```
python -m xbridges.huggingface.hf_to_xllm_main \
  --hf_dir hf_ckpts/my-model \
  --save_dir xllm_ckpts/my-model
```
* TP size is inferred from checkpoints written by xBridges; otherwise pass `--tp_size`.
* Run with `--help` for dtype, rank batching, Slurm-array, and resume options.
* Output is one `full_model.tpNN` directory per TP rank plus `config.json`. Load
  it in xLLM with `--base_model_dir`; it is not a resumable training checkpoint.

## Validating Conversion

Print out and compare logprobs for the same document with the xLLM and HuggingFace models separately.

### Single-GPU Models
No conversion needed beforehand. Uses `cuda:0` for xLLM and `cuda:1` for
HuggingFace by default (`--xllm_device`, `--hf_device`).
```
python -m xbridges.huggingface.validate \
  --xllm_dir /path/to/xllm/checkpoints/checkpoint_00010000 \
  --tokenizer_dir /path/to/tokenizer
```
* Make sure the model fits on a single GPU, e.g., `Qwen/Qwen3-30B-A3B` on a H200 is doable.

### Multi-GPU/Node Models

[Conversion](#xllm-to-huggingface) should be completed in advance, preferably
with `--dtype float32` to match the FP32 xLLM logprobs.

#### Print xLLM Logprobs
```
#!/bin/bash
#SBATCH --job-name=xllm_logprobs
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-xllm.out
#SBATCH --error=slurm-xllm.err

# Checkpoint paths
XLLM_DIR="/path/to/xllm/checkpoints/checkpoint_00125000"
TOKENIZER="/path/to/tokenizer"

srun python -u xbridges/huggingface/get_xllm_logprobs.py \
  --xllm_dir=$XLLM_DIR \
  --tokenizer_dir=$TOKENIZER \
  --model_parallel_size=8
```
* Make sure to update `XLLM_DIR` & `TOKENIZER`, and set `--model_parallel_size`
  to the checkpoint's TP size.

#### Print HuggingFace Logprobs
```
#!/bin/bash
#SBATCH --job-name=hf_logprobs
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --output=slurm-hf.out
#SBATCH --error=slurm-hf.err

# Get the IP address of the first node for rendezvous
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500 # Choose an available port

HF_DIR="/path/to/hf_ckpts/my-model"

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  xbridges/huggingface/get_hf_logprobs.py --ckpt_dir $HF_DIR
```
* Make sure to update the `HF_DIR` path.

> [!IMPORTANT]
> Released K2Horizon Hugging Face checkpoints (e.g. [Huggingface](https://huggingface.co/IFM/K2-Horizon-375B-A23B/tree/main)) can be used for inference, evaluation, and downstream fine-tuning (including SFT).
> Training behavior in the bundled Hugging Face implementation may differ from native xLLM, including the auxiliary load-balancing loss.
> To continue the original pretraining with xLLM's training behavior, use the native xLLM checkpoint and xLLM runtime.
