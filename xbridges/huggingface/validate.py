import fire
import json
import torch
from xllm.config import ModelConf, TokenizerConf, SlurmConf
from xllm.data.tokenizer.huggingface import HuggingFaceTokenizer
from xllm.models.transformer import Transformer
from xllm.distributed.slurm import init_torch_distributed
from xllm.distributed import initialize_model_parallel
from transformers import AutoTokenizer
from accelerate import init_empty_weights

from xbridges.huggingface.xllm.modeling_xllm import XllmForCausalLM
from xbridges.huggingface.xllm_to_hf_main import (
    get_total_params,
    load_xllm_state_dict,
    convert_config_xllm_to_hf,
    convert_state_dict_xllm_to_hf)


TEXT = """LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed."""


def check_logits_probs(
        xllm_model, hf_model, hf_tokenizer, xllm_device, hf_device):
    tokens = hf_tokenizer(TEXT)['input_ids']
    print(f'input text: {TEXT}')
    print(f'input ids: {tokens}')
    print('=' * 100)

    hf_model = hf_model.to(hf_device)
    xllm_model = xllm_model.to(xllm_device)

    with torch.no_grad():
        hf_model.eval()
        xllm_model.eval()

        hf_logits = hf_model(
            input_ids=torch.tensor([tokens], device=hf_device),
            output_attentions=True).logits
        hf_logprobs = torch.log_softmax(hf_logits, dim=-1)
        print(f'{hf_logits=}')
        print(f'{hf_logprobs=}')
        print('=' * 100)

        xllm_logits = xllm_model(
            tokens=torch.tensor([tokens], device=xllm_device),
            multi_segments=False)[0].to(hf_device)
        xllm_logprobs = torch.log_softmax(xllm_logits, dim=-1)
        print(f'{xllm_logits=}')
        print(f'{xllm_logprobs=}')
        print('=' * 100)

        max_diff_logits = torch.max(torch.abs(hf_logits - xllm_logits)).item()
        max_diff_logprobs = torch.max(torch.abs(hf_logprobs - xllm_logprobs)).item()
        print(f'{max_diff_logits=}')
        print(f'{max_diff_logprobs=}')


def main(xllm_dir='ckpts/xllm-30b-a3b',
         tokenizer_dir='Qwen/Qwen3-30B-A3B',
         xllm_device='cuda:0',
         hf_device='cuda:1'):
    slurm_cfg_dict = init_torch_distributed()
    slurm_cfg = SlurmConf()
    slurm_cfg.set_values(*slurm_cfg_dict)
    initialize_model_parallel(
        model_parallel_size=1, context_parallel_size=1, timeout=1800)

    xllm_tokenizer = HuggingFaceTokenizer(tokenizer_cfg=TokenizerConf(
        type="huggingface", path=tokenizer_dir))
    xllm_config = json.load(open(f'{xllm_dir}/config.json'))
    xllm_config['model']['causal_attn_backend'] = None
    xllm_config['model'].pop('efficient_attn', None)
    xllm_config['model']['fused_block'] = False

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    hf_config = convert_config_xllm_to_hf(xllm_config, hf_tokenizer)
    with init_empty_weights():
        hf_model = XllmForCausalLM(config=hf_config)
    print(f'{hf_config=}')
    print(f'{hf_model=}')
    print(f'{get_total_params(hf_model)=:,}')
    print('=' * 100)

    with torch.device('meta'):
        xllm_model = Transformer(
            cfg=ModelConf.from_dict(xllm_config['model']),
            tokenizer=xllm_tokenizer)
    xllm_model.load_state_dict(
        state_dict=load_xllm_state_dict(
            xllm_dir=xllm_dir,
            hf_config=hf_config,
            layer_l=0,
            layer_r=hf_config.num_hidden_layers),
        assign=True)
    xllm_state_dict = xllm_model.state_dict()
    print(f'{xllm_model=}')
    print(f'{get_total_params(xllm_model)=:,}')
    print('=' * 100)

    print(f'Converting XLLM state_dict to HF format...')
    state_dict = convert_state_dict_xllm_to_hf(
        hf_state_dict=hf_model.state_dict(),
        hf_config=hf_model.config,
        xllm_state_dict=xllm_state_dict)

    print(f'Loading HF state_dict...')
    hf_model.load_state_dict(state_dict, assign=True)
    for name, param in hf_model.named_parameters():
        assert param.dtype == torch.float32

    check_logits_probs(
        xllm_model=xllm_model,
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        xllm_device=xllm_device,
        hf_device=hf_device)


if __name__ == '__main__':
    fire.Fire(main)
