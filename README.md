<div align="center">

<h1>xBridges</h1>

<p><strong>Checkpoint Conversion & Serving for xLLM.</strong></p>

<p>
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.11+"></a>
  <img src="https://img.shields.io/badge/CUDA-12.8%2B-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.8+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>
</p>

<p>
  <a href="#highlights">Highlights</a> &nbsp;|&nbsp;
  <a href="#installation">Installation</a> &nbsp;|&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;|&nbsp;
  <a href="#documentation">Documentation</a> &nbsp;|&nbsp;
  <a href="https://github.com/ifm-ai/xllm/issues">Issues</a>
</p>

</div>

xBridges takes [xLLM](https://github.com/ifm-ai/xllm) checkpoints to the wider
ecosystem: convert them to and from Hugging Face format, check that the
converted model matches the original, and serve it with vLLM.

## Highlights

| Area | Capabilities |
| --- | --- |
| **xLLM → Hugging Face** | Converts tensor-parallel xLLM checkpoints to `K2HorizonForCausalLM` (BF16 safetensors by default), on one host or with layers spread across a Slurm allocation. |
| **Hugging Face → xLLM** | Converts Hugging Face checkpoints back to xLLM TP checkpoints, with rank batching, Slurm arrays, and resume. |
| **Validation** | Prints per-token logprobs from xLLM and Hugging Face for the same document, on one GPU or across nodes. |
| **Serving & evaluation** | Registers K2 Horizon with vLLM 0.24, launches a multi-node Ray + vLLM server on Slurm, and runs `lm-eval` against it. |

## Installation

Start with **PyTorch >= 2.11** and **CUDA >= 12.8**. Follow the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) for your
environment.

### 1. Install xLLM

xBridges uses xLLM to read checkpoints and compute reference logprobs:

```bash
git clone https://github.com/ifm-ai/xllm.git
cd xllm

python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation --config-settings editable_mode=compat
python -m pip install flash-attn --no-build-isolation
cd ..
```

See the [xLLM installation guide](https://github.com/ifm-ai/xllm#installation)
for other attention backends (FlashAttention 3/4, xattn).

### 2. Install xBridges

Clone xBridges and add it to `PYTHONPATH`. The commands below run from the repository root.

```bash
git clone https://github.com/ifm-ai/xbridges.git
cd xbridges

python -m pip install transformers==5.13.0 accelerate fire
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 3. vLLM Environment (optional)

Serving needs vLLM 0.24.0 in a separate environment. Follow the
[vLLM guide](xbridges/vllm/README.md#environment).

## Quick Start

This walkthrough takes the checkpoint from the
[xLLM Quick Start](https://github.com/ifm-ai/xllm#quick-start) (K2 Horizon
0.9B, 100 steps, one GPU) through conversion, validation, and inference. To use
your own model, point `XLLM_CKPT` at any `checkpoints/checkpoint_XXXXXXXX`
directory and `TOKENIZER` at its tokenizer.

```bash
XLLM_CKPT=/path/to/xllm/saved_models/quickstart/checkpoints/checkpoint_00000100
TOKENIZER=/path/to/tokenizer
HF_CKPT=hf_ckpts/quickstart
```

### 1. Convert xLLM to Hugging Face

```bash
python -m xbridges.huggingface.xllm_to_hf_main \
  --xllm_dir "$XLLM_CKPT" \
  --tokenizer_dir "$TOKENIZER" \
  --save_dir "$HF_CKPT"
```

This writes BF16 safetensors to a new directory. For large models, convert
layers in parallel:

```bash
python -m xbridges.huggingface.xllm_to_hf_parallel \
  --xllm-dir "$XLLM_CKPT" --tokenizer-dir "$TOKENIZER" \
  --save-dir "$HF_CKPT" --workers 8
```

### 2. Validate the Conversion

Run the xLLM and Hugging Face models on the same document and compare their
logprobs (xLLM on `cuda:0`, Hugging Face on `cuda:1`):

```bash
python -m xbridges.huggingface.validate \
  --xllm_dir "$XLLM_CKPT" --tokenizer_dir "$TOKENIZER"
```

For models that need multiple GPUs or nodes, see
[Multi-GPU/Node Models](xbridges/huggingface/README.md#multi-gpunode-models).

### 3. Run with Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "hf_ckpts/quickstart"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 4. Convert Back to xLLM (optional)

```bash
python -m xbridges.huggingface.hf_to_xllm_main \
  --hf_dir "$HF_CKPT" \
  --save_dir xllm_ckpts/quickstart
```

Start xLLM training from the result with
`--base_model_dir xllm_ckpts/quickstart`.

### 5. Serve with vLLM (optional)

In the vLLM environment:

```bash
bash xbridges/vllm/add_xllm_to_vllm.sh
MODEL=$(realpath "$HF_CKPT") sbatch xbridges/vllm/launch_vllm_server.sh
```

Then query the server (`vllm_head` is printed in `slurm.out`):

```bash
curl http://<vllm_head>/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/abs/path/to/hf_ckpts/quickstart", "prompt": "The capital of France is", "max_tokens": 32}'
```

## Documentation

| I want to... | Start here |
| --- | --- |
| **Convert and validate checkpoints** | [Hugging Face bridge](xbridges/huggingface/README.md): conversion options and multi-node logprob validation. |
| **Serve and evaluate with vLLM** | [vLLM bridge](xbridges/vllm/README.md): environment, server launch, querying, and `lm-eval`. |
| **Train a model** | [xLLM](https://github.com/ifm-ai/xllm): training, data, and evaluation. |

## Repository Layout

```text
xbridges/
  huggingface/     xLLM <-> Hugging Face conversion and logprob validation
    k2_horizon/    Hugging Face K2HorizonForCausalLM implementation
  vllm/            vLLM model registration and Slurm server launch
tests/             Conversion tests
```

## License

xBridges is released under the [Apache 2.0 License](LICENSE).
