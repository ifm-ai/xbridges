import os
import json
import fire
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


TEXT = """LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties."""


class PPWrapper(torch.nn.Module):
    def __init__(self, model, stage_index, num_stages):
        super().__init__()

        self.num_layers = model.config.num_hidden_layers
        if stage_index != 0:
            model.model.embed_tokens = None
            # model.model.model.rotary_emb = None
        if stage_index != num_stages - 1:
            model.model.norm = None
            model.lm_head = None

        n_stage_layers = [self.num_layers // num_stages for _ in range(num_stages)]
        for stage_idx in range(self.num_layers % num_stages):
            n_stage_layers[stage_idx] += 1

        self.layer_l = sum(n_stage_layers[:stage_index])
        self.layer_r = sum(n_stage_layers[:stage_index + 1])
        for layer_idx in range(model.config.num_hidden_layers):
            if not self.layer_l <= layer_idx < self.layer_r:
                model.model.layers[layer_idx] = None

        self.model = model

    def load_weights(self, ckpt_dir, device):
        print(f'Loading {device} params -- layers [{self.layer_l}, {self.layer_r})')
        stage_keys = self.model.state_dict().keys()
        if os.path.exists(f'{ckpt_dir}/model.safetensors.index.json'):
            index_name = 'model.safetensors.index.json'
        else:
            index_name = 'pytorch_model.bin.index.json'
        weight_map = json.load(open(f'{ckpt_dir}/{index_name}'))['weight_map']

        state_dict = {}
        for filename in sorted({weight_map[key] for key in stage_keys}):
            if filename.endswith('.safetensors'):
                sd = load_file(f'{ckpt_dir}/{filename}')
            else:
                sd = torch.load(f'{ckpt_dir}/{filename}', weights_only=True)
            for key, value in sd.items():
                if key in stage_keys:
                    state_dict[key] = value

        self.model.load_state_dict(state_dict, assign=True)
        self.model = self.model.to(device).eval()

    def forward(self, input_ids):
        cache_position = torch.arange(
            0, input_ids.shape[1], device=input_ids.device)
        position_ids = cache_position.unsqueeze(0)

        if self.model.model.layers[0] is not None:
            hidden_states = self.model.model.embed_tokens(input_ids)
        else:
            hidden_states = input_ids

        position_embeddings = self.model.model.rotary_emb(
            hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(self.model.model.layers):
            if decoder_layer is not None:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position)
                # print(f'{layer_idx=}, {hidden_states=}, {hidden_states.shape=}')

        if self.model.model.norm is not None:
            hidden_states = self.model.model.norm(hidden_states)
        if self.model.lm_head is not None:
            hidden_states = self.model.lm_head(hidden_states)

        return hidden_states


def main(ckpt_dir):
    dist.init_process_group(backend='nccl')

    stage_index = dist.get_rank()
    num_stages = dist.get_world_size()
    device = torch.device(f'cuda:{dist.get_node_local_rank()}')
    print(f'{stage_index=}; {num_stages=}; {device=}')

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            config=AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True),
            trust_remote_code=True)
    assert model.config.rope_head_dim is not None
    model.model.rotary_emb = model.model.rotary_emb.__class__(
        config=AutoConfig.from_pretrained(
            ckpt_dir,
            head_dim=model.config.rope_head_dim,
            trust_remote_code=True))

    model = PPWrapper(
        model=model, stage_index=stage_index, num_stages=num_stages)
    model.load_weights(ckpt_dir=ckpt_dir, device=device)
    stage = PipelineStage(
        submodule=model,
        stage_index=stage_index,
        num_stages=num_stages,
        device=device)
    schedule = ScheduleGPipe(stage, n_microbatches=num_stages)

    token_ids = tokenizer(TEXT, add_special_tokens=False)['input_ids']
    input_ids = torch.asarray(
        [[tokenizer.bos_token_id] + token_ids] * num_stages, device=device)
    print(f'{input_ids=}', flush=True)

    with torch.inference_mode():
        if stage_index == 0:
            output = schedule.step(input_ids)
        else:
            output = schedule.step()

        if output is not None:
            logits = output[:1]
            logprobs = torch.log_softmax(logits, dim=-1)
            print(f'{logprobs=}', flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    fire.Fire(main)
