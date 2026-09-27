## Environment
```
pip install transformers==5.13.0 fire
```

## Conversion
```
PYTHONPATH=./ python xllm_bridges/huggingface/xllm_to_hf_main.py \
  --xllm_dir /lustrefs/users/bbqbyte/workspace/checkpoints/xllm/k2mova-36B_mid3_v3_110B_jais250k_bsz20M_seq512k_lr4e-5_constant_wd0.06_rope128_dot_te/checkpoints/checkpoint_0005500 \
  --tokenizer_dir /lustrefs/users/bbqbyte/workspace/checkpoints/huggingface/k2mova-36B_mid3_v3_110B_jais250k_bsz20M_seq512k_lr4e-5_constant_wd0.06_rope128_dot_te/checkpoints/checkpoint_0005500 \
  --save_dir hf_ckpts/mova-36b
```
* `--layers_per_load` is optional and limits loading to the requested layers.
* For IFM-style BF16 safetensors, add `--dtype bfloat16 --safe_serialization true`.
  The default remains FP32 `.bin` for numerical validation and existing callers.
* Output uses the bundled IFM `K2HorizonForCausalLM` implementation. Missing or
  unexpected weights and incompatible shapes cause an error. Use a new output
  directory; existing nonempty directories are never overwritten.
* RoPE is derived from the source checkpoint. Conversion does not implicitly
  apply the additional YaRN context extension used by the published 0.9B model.
* An example converted checkpoint can be found at `/lustrefs/users/bowen.tan/xllm_bridges_converted/huggingface/mova-36b-ckpt-5500`

## Validating Conversion

Print out and compare logprobs for the same document with the xLLM and HuggingFace models separately.

### Single-GPU Models
No need conversion, but need `xllm` installed.
```
PYTHONPATH=./ python xllm_bridges/huggingface/validate.py \
	--xllm_dir /lustrefs/users/bbqshort/workspace/checkpoints/xllm/k2v3-4B_mid4_v2_200B_jais250k_bsz20M_seq512k_lr4e-5_constant_wd0.06_rope128/checkpoints/checkpoint_0010000 \
	--tokenizer_dir /lustrefs/users/bbqshort/workspace/checkpoints/huggingface/k2v3-4B_mid4_v2_200B_jais250k_bsz20M_seq512k_lr4e-5_constant_wd0.06_rope128/checkpoints/checkpoint_0010000
```
* Make sure the model runs on a single GPU, e.g., `Qwen/Qwen3-30B-A3B` on a H200 is doable. 

### Multi-GPU/Node Models

[Conversion](#conversion) should be completed in advance.

#### Print xllm logprobs
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
XLLM_DIR="/mnt/weka/shrd/k2m/runner/checkpoints/k2moe375B_txt360v2_256nodes_seed42_bsz32M_seq8k_jais250k_ep8_dot_te_bestfit/checkpoints/checkpoint_0125000"
TOKENIZER="/mnt/weka/shrd/k2m/xuezhe.ma/data/tokenizers/jais250k_enx10_codex6.5_arax5_3digits"

srun python -u xllm_bridges/huggingface/get_xllm_logprobs.py  \
  --xllm_dir=$XLLM_DIR \
  --tokenizer_dir=$TOKENIZER \
  --model_parallel_size=8
```
* Make sure to update paths of `XLLM_CKPT` & `TOKENIZER`.

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

HF_DIR="/mnt/weka/shrd/k2m/runner/checkpoints/k2moe375B_txt360v2_256nodes_seed42_bsz32M_seq8k_jais250k_ep8_dot_te_bestfit/huggingface/checkpoint_0125000"

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  xllm_bridges/huggingface/get_hf_logprobs.py --ckpt_dir $HF_DIR
```
* Make sure to update the `HF_DIR` path.

> [!IMPORTANT]
> Released K2Horizon Hugging Face checkpoints (e.g. [Huggingface](https://huggingface.co/IFM/K2-Horizon-375B-A23B/tree/main)) can be used for inference, evaluation, and downstream fine-tuning (including SFT).
> Training behavior in the bundled Hugging Face implementation may differ from native xLLM, including the auxiliary load-balancing loss.
> To continue the original pretraining with xLLM's training behavior, use the native xLLM checkpoint and xLLM runtime.
