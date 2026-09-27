import fire
from pathlib import Path
import torch
import torch.distributed as dist
from xllm.models import build_model
from xllm.reloading import reload_config_and_tokenizer, init_distributed_mode, reload_fsdp_from_state
from xllm.distributed import get_model_parallel_world_size, get_context_parallel_world_size
from xllm.logger import initialize_logger
from xllm.utils import setup_env, log_host
from xllm.modules.model_parallel import gather_from_model_parallel_region


TEXT = """LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties."""


def main(xllm_dir, tokenizer_dir, model_parallel_size=8):
    initialize_logger()
    setup_env()
    log_host()

    reloaded, xllm_tokenizer, model_cfg = reload_config_and_tokenizer(
        ckpt_dir=Path(xllm_dir), tokenizer_path=tokenizer_dir)
    global_rank, world_size = init_distributed_mode(
        model_parallel_size=model_parallel_size,
        context_parallel_size=1,
        timeout=1800)
    assert get_model_parallel_world_size() == model_parallel_size
    assert get_context_parallel_world_size() == 1

    model_cfg.causal_attn_backend = None
    model_cfg.fused_block = False
    xllm_model = build_model(
        model_cfg,
        dtype='fp32',
        fully_sharded_size=None,
        fp32_reduce_scatter=True,
        reshard_after_forward=True,
        forward_prefetch=False,
        tokenizer=xllm_tokenizer)
    reload_fsdp_from_state(
        ckpt_dir=Path(xllm_dir),
        full_state=False,
        model=xllm_model,
        reloaded=reloaded,
        global_rank=global_rank)
    xllm_model = xllm_model.float()
    sample_param = next(xllm_model.parameters())
    device = sample_param.device
    assert sample_param.dtype == torch.float32

    xllm_token_ids = xllm_tokenizer.encode(TEXT, bos=True, eos=False)
    tokens_tensor = torch.tensor([xllm_token_ids], device=device)
    if global_rank == 0:
        print(f'{xllm_token_ids=}')

    with torch.no_grad():
        xllm_model.eval()
        model_output = xllm_model(tokens=tokens_tensor, multi_segments=False)
        xllm_logits = gather_from_model_parallel_region(
            model_output[0].contiguous())

        if global_rank == 0:
            print(f"Gathered logits shape: {xllm_logits.shape}")

        logprobs = torch.log_softmax(xllm_logits.float(), dim=-1)
        if global_rank == 0:
            print(f'{logprobs=}')

    if dist.is_initialized():
        dist.barrier()


if __name__ == '__main__':
    fire.Fire(main)
