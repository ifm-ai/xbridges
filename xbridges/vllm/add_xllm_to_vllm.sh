#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
vllm_dir=$("${PYTHON:-python}" -c 'import vllm ; assert vllm.__version__ == "0.24.0", vllm.__version__ ; print(vllm.__path__[0])')
echo "vllm_dir=${vllm_dir}"

# Break package-cache hardlinks before replacing installed files.
cp --remove-destination "${script_dir}/k2_horizon/modeling_k2_horizon_vllm.py" "${vllm_dir}/model_executor/models/k2_horizon.py"
echo "${vllm_dir}/model_executor/models/k2_horizon.py updated."
cp --remove-destination "${script_dir}/k2_horizon/registry_vllm.py" "${vllm_dir}/model_executor/models/registry.py"
echo "${vllm_dir}/model_executor/models/registry.py updated."
