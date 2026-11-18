## Environment
```shell
conda create --name vllm024 python==3.12
conda activate vllm024
pip install vllm==0.24.0+cu129 --extra-index-url https://wheels.vllm.ai/0.24.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129
pip install --upgrade transformers==5.13.0 fire ray lm-eval[api]==0.4.10
```

## Add Xllm to Vllm
```shell
bash xbridges/vllm/add_xllm_to_vllm.sh
```

## Launch Vllm Server
```shell
MODEL=/abs/path/to/hf_ckpts/my-model sbatch xbridges/vllm/launch_vllm_server.sh
```
* `MODEL` must be an absolute path to the HuggingFace model folder; requests use it as the model name.

## Speculative Decoding with the MTP Layers
A checkpoint exported with multi-token prediction modules (`num_nextn_predict_layers = D > 0`, see
`xbridges/huggingface/mtp.py`) can use them as the draft model of vLLM's MTP speculative decoding, as DeepSeek-V3
does. The draft, `K2HorizonMTPModel` (`k2_horizon/modeling_k2_horizon_mtp_vllm.py`), runs the K2-Horizon decoder
block of each MTP layer with the main model's embedding and lm_head.

vLLM's own files are not changed: `k2_horizon/mtp_plugin.py` registers the draft when imported (the draft model
type `k2_horizon_mtp`, its architecture, and a K2-Horizon model that leaves the MTP layers to the draft). Use the
`vllm` command through `xbridges.vllm.vllm_mtp`, with xBridges on `PYTHONPATH` (the engine and worker processes
import the plugin from it):
```shell
export PYTHONPATH=/abs/path/to/xbridges:$PYTHONPATH
python -m xbridges.vllm.vllm_mtp serve "${MODEL}" \
  --trust-remote-code \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```
From Python, import the plugin before creating the engine:
```python
import xbridges.vllm.k2_horizon.mtp_plugin  # noqa: F401
from vllm import LLM

llm = LLM(model=MODEL, trust_remote_code=True,
          speculative_config={"method": "mtp", "num_speculative_tokens": 2})
```
* With vLLM 0.31, `num_speculative_tokens` up to D runs MTP modules 1, ..., `num_speculative_tokens`; more reuses
  them in turn. vLLM 0.24 runs module 1 for every speculative token.
* Without speculative decoding, spawned or Ray workers do not import the plugin: serving such an export then needs
  a vLLM K2-Horizon model that skips the MTP layers, as `k2_horizon/modeling_k2_horizon_vllm.py` does.
* Tested with vLLM 0.24.0 (after `add_xllm_to_vllm.sh`) and 0.31.0, tensor parallel sizes 1 and 2:
  `tests/vllm/check_mtp_draft.py` compares the draft with a reference computed with the Transformers model
  (`logits`) and checks that greedy outputs do not change (`e2e`).

## Call Vllm Server and Prompt the Model
```shell
MODEL="/abs/path/to/hf_ckpts/my-model"
SERVER="http://<vllm_head>"  # printed as vllm_head in slurm.out

QUERY="{
  \"model\": \"${MODEL}\",
  \"prompt\": \"What was 2025's most important film?\",
  \"temperature\": 1.0,
  \"top_p\": 0.6,
  \"max_tokens\": 256,
  \"logprobs\": null,
  \"echo\": false
}"
echo "QUERY=${QUERY}"

curl ${SERVER}/v1/completions -H "Content-Type: application/json" -d "${QUERY}" | jq 
```
* Make sure to update `MODEL` & `SERVER` based on your serving.

## Evaluation
```shell
MODEL="/abs/path/to/hf_ckpts/my-model"
SERVER="http://<vllm_head>"  # printed as vllm_head in slurm.out

MODEL_ARGS="\
model=${MODEL},\
max_length=8192,\
trust_remote_code=True,\
base_url=${SERVER}/v1/completions,\
num_concurrent=256
"

lm_eval \
  --model local-completions \
  --tasks openllm \
  --model_args "${MODEL_ARGS}" \
  --output_path ./results \
  --log_samples \
  --confirm_run_unsafe_code 2>&1 | tee eval_results.txt
```
* Make sure to update `MODEL` & `SERVER` based on your serving.
