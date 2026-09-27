## Environment
```shell
conda create --name vllm024 python==3.12

pip install vllm==0.24.0+cu129 --extra-index-url https://wheels.vllm.ai/0.24.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129
pip install --upgrade transformers==5.13.0 fire ray lm-eval[api]==0.4.10
```

## Add Xllm to Vllm
```shell
bash xbridges/vllm/add_xllm_to_vllm.sh
```

## Launch Vllm Server
```shell
sbatch xbridges/vllm/launch_vllm_server.sh
```
* Make sure to update line 76, `--model`, with the HuggingFace model folder.

## Call Vllm Server and Prompt the Model
```shell
MODEL="/mnt/weka/shrd/k2m/suqi.sun/bbq_image/k2mova-36b-mid4_v2/checkpoint_0010000"
SERVER="http://fs-mbz-gpu-089:6380"

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
MODEL="/mnt/weka/shrd/k2m/suqi.sun/bbq_image/k2mova-36b-mid4_v2/checkpoint_0010000"
SERVER="http://fs-mbz-gpu-089:6380"

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
