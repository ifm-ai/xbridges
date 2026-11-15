"""The `vllm` command with the K2-Horizon MTP draft registered (xbridges/vllm/k2_horizon/mtp_plugin.py), e.g.

    python -m xbridges.vllm.vllm_mtp serve /abs/path/to/hf_ckpt \
        --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
"""
import sys

import xbridges.vllm.k2_horizon.mtp_plugin  # noqa: F401  (registers the draft)
from vllm.entrypoints.cli.main import main

if __name__ == '__main__':
    sys.argv[0] = 'vllm'
    sys.exit(main())
