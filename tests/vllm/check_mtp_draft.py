"""Checks the K2-Horizon MTP draft of vLLM (xbridges/vllm/k2_horizon/modeling_k2_horizon_mtp_vllm.py) on an xBridges
export with MTP layers, on one GPU, in the vLLM environment with xBridges on PYTHONPATH:

    python tests/vllm/check_mtp_draft.py logits --hf_dir /abs/path/to/hf_ckpt [--eager False]
    python tests/vllm/check_mtp_draft.py e2e --hf_dir /abs/path/to/hf_ckpt

logits: runs vLLM with the draft on a prompt and compares what the draft computes for the first speculative step
(the prompt's prefill: MTP module k at every position p, from the token p + k) with a reference computed with the
Transformers model and the exported MTP weights, by hand: the main model's final-normed hidden states, then enorm,
hnorm, eh_proj, the decoder block, shared_head.norm and lm_head. It compares the modules' outputs at all
positions and the logits vLLM drafts from (last position).

e2e: greedy generation without and with speculative decoding (separate processes), which must give the same tokens
(up to ties), the acceptance rate, and the acceptance the reference modules would have on the same sequences.
"""
import gc
import json
import os
import subprocess
import sys
import tempfile

import fire
import torch

PROMPTS = [
    'The capital of France is',
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    'Once upon a time, in a small village by the sea, there lived',
    '1, 2, 3, 4, 5, 6, 7, 8,',
]


def _llm(hf_dir, dtype, eager, num_speculative_tokens, plugin=True, **kwargs):
    if plugin:
        import xbridges.vllm.k2_horizon.mtp_plugin  # noqa: F401  (registers the draft)
    from vllm import LLM

    if num_speculative_tokens:
        kwargs['speculative_config'] = {'method': 'mtp', 'num_speculative_tokens': num_speculative_tokens}
    return LLM(model=hf_dir, trust_remote_code=True, dtype=dtype, enforce_eager=eager, max_model_len=1024,
               gpu_memory_utilization=0.45, seed=0, **kwargs)


def _stats(name, value, reference):
    diff = (value.double() - reference.double()).abs()
    scale = reference.double().abs().max().item()
    print(f'{name}: max abs diff {diff.max().item():.3e}, mean abs diff {diff.mean().item():.3e}, '
          f'max |ref| {scale:.3e}, max abs diff / max |ref| {diff.max().item() / scale:.3e}', flush=True)
    return diff.max().item() / scale


def _compare_logits(name, value, reference):
    rel = _stats(name, value, reference)
    same_argmax = (value.argmax(-1) == reference.argmax(-1)).float().mean().item()
    top5 = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 5 for a, b in
                         zip(value.topk(5, -1).indices.reshape(-1, 5), reference.topk(5, -1).indices.reshape(-1, 5))])
    print(f'{name}: argmax agreement {same_argmax:.4f}, top-5 overlap {top5.mean().item():.4f}', flush=True)
    return rel, same_argmax


class Reference:
    """The MTP modules computed with the Transformers K2-Horizon model and the exported weights."""

    def __init__(self, hf_dir, device):
        from safetensors.torch import load_file
        from xbridges.huggingface.k2_horizon import modeling_k2_horizon as hf

        self.hf = hf
        self.device = device
        self.model = hf.K2HorizonForCausalLM.from_pretrained(
            hf_dir, dtype=torch.float32, attn_implementation='eager').to(device).eval()
        config = self.config = self.model.config
        n, dim = config.num_hidden_layers, config.hidden_size
        mtp = load_file(os.path.join(hf_dir, 'model-mtp.safetensors'))
        self.modules = []
        for layer in range(n, n + config.num_nextn_predict_layers):
            prefix = f'model.layers.{layer}.'
            sd = {key[len(prefix):]: value.float() for key, value in mtp.items() if key.startswith(prefix)}
            norms = {name: hf.K2HorizonRMSNorm(dim, config.layernorm_num_groups, config.rms_norm_eps)
                     for name in ['enorm', 'hnorm', 'shared_head.norm']}
            for name, norm in norms.items():
                norm.weight.data.copy_(sd.pop(f'{name}.weight'))
            eh_proj = torch.nn.Linear(2 * dim, dim, bias=False)
            eh_proj.weight.data.copy_(sd.pop('eh_proj.weight'))
            block = hf.K2HorizonDecoderLayer(config, layer)
            block.load_state_dict(sd, strict=True)
            self.modules.append(torch.nn.ModuleDict(
                {'enorm': norms['enorm'], 'hnorm': norms['hnorm'], 'norm': norms['shared_head.norm'],
                 'eh_proj': eh_proj, 'block': block}).to(device).eval())

    @torch.no_grad()
    def main(self, ids):
        """The main model's final-normed hidden states and logits (L x D, L x V)."""
        hidden = self.model.model(input_ids=ids[None].to(self.device)).last_hidden_state
        return hidden[0], self.model.lm_head(hidden)[0]

    @torch.no_grad()
    def module(self, k, hidden, ids):
        """MTP module k (1-based) on the previous hidden states (L x D) and the tokens at p + k (L); returns its output
        after its final norm (L x D) and its logits."""
        m = self.modules[k - 1]
        x = m['eh_proj'](torch.cat([m['enorm'](self.model.model.embed_tokens(ids.to(self.device))),
                                    m['hnorm'](hidden)], dim=-1))[None]
        positions = torch.arange(x.shape[1], device=self.device)[None]
        mask = self.hf.create_causal_mask(config=self.config, inputs_embeds=x, attention_mask=None,
                                          past_key_values=None, position_ids=positions)
        x = m['block'](x, position_embeddings=self.model.model.rotary_emb(x, positions), attention_mask=mask,
                       position_ids=positions)
        y = m['norm'](x)[0]
        return y, self.model.lm_head(y)


def _capture_draft(worker, capture_dir):
    """Run in vLLM's workers (collective_rpc): makes the draft save its inputs and outputs (forward) and its logits to
    capture_dir while capture_dir/on exists, from tensor-parallel rank 0."""
    from xbridges.vllm.k2_horizon import modeling_k2_horizon_mtp_vllm as mtp_vllm

    forward, compute_logits = mtp_vllm.K2HorizonMultiTokenPredictor.forward, mtp_vllm.K2HorizonMTP.compute_logits
    count = [0]

    def save(kind, record):
        if not os.path.exists(os.path.join(capture_dir, 'on')):
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            count[0] += 1
            torch.save({key: value.cpu() if torch.is_tensor(value) else value for key, value in record.items()},
                       os.path.join(capture_dir, f'{count[0]:04d}_{kind}.pt'))

    def capture_forward(self, input_ids, positions, previous_hidden_states, inputs_embeds=None, spec_step_idx=0):
        out = forward(self, input_ids, positions, previous_hidden_states, inputs_embeds, spec_step_idx)
        save('forward', {'step': spec_step_idx, 'ids': input_ids, 'positions': positions,
                         'hidden': previous_hidden_states, 'out': out})
        return out

    def capture_compute_logits(self, hidden_states, spec_step_idx=0):
        out = compute_logits(self, hidden_states, spec_step_idx)
        if out is not None:
            save('logits', {'logits': out})
        return out

    mtp_vllm.K2HorizonMultiTokenPredictor.forward = capture_forward
    mtp_vllm.K2HorizonMTP.compute_logits = capture_compute_logits


@torch.no_grad()
def logits(hf_dir, prompt=PROMPTS[0], dtype='float32', eager=True, tp=1, num_speculative_tokens=None, atol=3e-3):
    """Draft vs reference for the first speculative step of `prompt`. Fails above `atol` (max abs diff / max |ref|);
    in float32 the main model's hidden states differ by about 1e-3 as well (vLLM's kernels use TF32)."""
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'  # to send _capture_draft to the workers
    from vllm import SamplingParams

    with open(os.path.join(hf_dir, 'config.json')) as f:
        depth = json.load(f)['num_nextn_predict_layers']
    depth = num_speculative_tokens or depth
    capture_dir = tempfile.mkdtemp()
    llm = _llm(hf_dir, dtype, eager, depth, tensor_parallel_size=tp)
    llm.collective_rpc(_capture_draft, args=(capture_dir,))
    open(os.path.join(capture_dir, 'on'), 'w').close()
    output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=depth + 1))[0]
    os.remove(os.path.join(capture_dir, 'on'))
    records = [(name, torch.load(os.path.join(capture_dir, name))) for name in sorted(os.listdir(capture_dir))]
    calls = [r for name, r in records if name.endswith('_forward.pt')]
    draft_logits = [r['logits'].cuda() for name, r in records if name.endswith('_logits.pt')]
    ids = torch.tensor(output.prompt_token_ids)
    length = len(ids)
    print(f'prompt {prompt!r}: {length} tokens, vLLM greedy continuation {output.outputs[0].token_ids} (TP {tp})',
          flush=True)
    calls, first_logits = calls[:depth], draft_logits[:depth]
    # vLLM runs MTP module k at draft step k (multi-module MTP, e.g. vLLM 0.31), or module 1 at every draft step (e.g.
    # vLLM 0.24), for the request's last position after the prefill
    multi_module = [c['step'] for c in calls] == list(range(depth))
    print(f'vLLM draft steps run modules {[c["step"] + 1 for c in calls]}', flush=True)
    assert multi_module or all(c['step'] == 0 for c in calls), [c['step'] for c in calls]
    assert calls[0]['positions'][:length].tolist() == list(range(length)), calls[0]['positions']
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    ref = Reference(hf_dir, 'cuda')
    hidden, main_logits = ref.main(ids)
    print(f'reference main model next token {main_logits[-1].argmax().item()}, vLLM {output.outputs[0].token_ids[0]}',
          flush=True)
    # the tokens at p + k: the prompt, then the main model's next token, then the drafts
    tokens = torch.cat([ids, torch.tensor(output.outputs[0].token_ids[:1])])
    worst, ref_hidden, ref_y = 0., hidden, None
    for step, c in enumerate(calls, start=1):
        if step > 1:
            tokens = torch.cat([tokens, torch.tensor([first_logits[step - 2][0].argmax().item()])])
        c = {key: value.cuda() if torch.is_tensor(value) else value for key, value in c.items()}
        if multi_module or step == 1:
            # the prefill of module k = step: at every prompt position p, from the token p + k
            k, n, k_ids = step, length, tokens[step:length + step]
            assert c['ids'][:n].tolist() == k_ids.tolist(), (k, c['ids'], k_ids)
            vllm_hidden = c['hidden'][:n].float()
            if k == 1:
                worst = max(worst, _stats('target final-normed hidden states, vLLM vs Transformers', vllm_hidden,
                                          hidden))
                prefill_hidden = vllm_hidden
            what = f'module {k}'
        else:
            # module 1 again, at the next position, from the previous draft step's output
            k, n, k_ids = 1, 1, tokens[1:length + step]
            assert c['ids'][:1].tolist() == k_ids[-1:].tolist() and c['positions'][0].item() == length + step - 2, c
            prefill_hidden = torch.cat([prefill_hidden, c['hidden'][:1].float()])
            vllm_hidden, ref_hidden = prefill_hidden, torch.cat([ref_hidden, ref_y[-1:]])
            what = f'draft step {step} (module 1, position {length + step - 2})'
        # the draft alone: the reference from the hidden states vLLM gave the module
        y, _ = ref.module(k, vllm_hidden, k_ids)
        worst = max(worst, _stats(f'{what} output, from the same input', c['out'][:n], y[-n:]))
        # from the Transformers model's hidden states
        ref_y, k_logits = ref.module(k, ref_hidden, k_ids)
        if multi_module:
            ref_hidden = ref_y
        worst = max(worst, _stats(f'{what} output ({n} positions)', c['out'][:n], ref_y[-n:]))
        rel, _ = _compare_logits(f'{what} logits ({n} positions)', c['out'][:n].float() @ ref.model.lm_head.weight.T,
                                 k_logits[-n:])
        worst = max(worst, rel)
        rel, _ = _compare_logits(f'{what} drafted logits', first_logits[step - 1][0], k_logits[-1])
        worst = max(worst, rel)
        print(f'{what} draft token: vLLM {first_logits[step - 1][0].argmax().item()}, '
              f'reference {k_logits[-1].argmax().item()}', flush=True)
    print(f'worst max abs diff / max |ref|: {worst:.3e} (atol {atol})', flush=True)
    assert worst < atol


def generate(hf_dir, out_json, method='mtp', num_speculative_tokens=0, dtype='float32', eager=False, max_tokens=64,
             tp=1, plugin=True):
    """Greedy generation of PROMPTS, written to out_json with the top-2 logprobs and the speculative decoding
    counters. plugin=False: with vLLM's own K2-Horizon model only (it must skip the MTP layers of the export)."""
    from vllm import SamplingParams

    kwargs = {'speculative_config': None}
    if num_speculative_tokens:
        kwargs['speculative_config'] = {'method': method, 'num_speculative_tokens': num_speculative_tokens}
        if method == 'ngram':
            kwargs['speculative_config'].update(prompt_lookup_min=1, prompt_lookup_max=3)
    llm = _llm(hf_dir, dtype, eager, 0, plugin=plugin, disable_log_stats=False, tensor_parallel_size=tp, **kwargs)
    outputs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, logprobs=2))
    result = {
        'prompt_tokens': [list(o.prompt_token_ids) for o in outputs],
        'tokens': [list(o.outputs[0].token_ids) for o in outputs],
        'text': [o.outputs[0].text for o in outputs],
        'top2': [[sorted(((v.logprob, t) for t, v in lp.items()), reverse=True)[:2] for lp in o.outputs[0].logprobs]
                 for o in outputs],
    }
    for metric in llm.get_metrics():
        if metric.name.startswith('vllm:spec_decode'):
            result[metric.name] = getattr(metric, 'values', None) or getattr(metric, 'value', None)
    with open(out_json, 'w') as f:
        json.dump(result, f)


@torch.no_grad()
def _offline_acceptance(hf_dir, prompt_tokens, tokens, depth):
    """How often the reference MTP modules, run on the greedy sequences, predict them: module k at position p (from
    the tokens up to p + k) against token p + k + 1, and all modules 1..k at p (a draft of k tokens accepted)."""
    ref = Reference(hf_dir, 'cuda')
    hits, all_hits, count = [0] * depth, [0] * depth, 0
    for prompt, generated in zip(prompt_tokens, tokens):
        seq = torch.tensor(prompt + generated)
        hidden, _ = ref.main(seq)
        n = len(seq) - depth - 1  # positions p with tokens up to p + depth + 1
        ok = torch.ones(n, dtype=torch.bool, device=hidden.device)
        for k in range(1, depth + 1):
            hidden, logits = ref.module(k, hidden[:len(seq) - k], seq[k:])
            right = logits[:n].argmax(-1).cpu() == seq[k + 1:k + 1 + n]
            ok &= right.to(ok.device)
            hits[k - 1] += right.sum().item()
            all_hits[k - 1] += ok.sum().item()
        count += n
    print(f'offline (Transformers reference, on the greedy sequences): module k right {[h / count for h in hits]}, '
          f'modules 1..k all right {[h / count for h in all_hits]} over {count} positions', flush=True)


def e2e(hf_dir, method='mtp', num_speculative_tokens=None, dtype='float32', eager=False, max_tokens=64, spawn=False,
        tie_tol=5e-3, tp=1):
    """Greedy outputs without and with speculative decoding (num_speculative_tokens: the model's MTP depth by
    default; method ngram as a control). spawn: the speculative decoding run with spawned workers. The outputs must
    be the same, up to a position where the two best tokens are within `tie_tol` in logprob: the main model's
    numerics differ when it checks several tokens at once (in float32 its logits differ from the Transformers ones by
    about 3e-3), which decides near ties, frequent with a barely trained model."""
    with open(os.path.join(hf_dir, 'config.json')) as f:
        depth = json.load(f)['num_nextn_predict_layers']
    if num_speculative_tokens is None:
        num_speculative_tokens = depth
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in [0, num_speculative_tokens]:
            out = os.path.join(tmp, f'{n}.json')
            env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn') if spawn and n else None
            subprocess.run([sys.executable, __file__, 'generate', '--hf_dir', hf_dir, '--out_json', out,
                            '--method', method, '--num_speculative_tokens', str(n), '--dtype', dtype,
                            '--eager', str(eager), '--max_tokens', str(max_tokens), '--tp', str(tp)], check=True,
                           env=env)
            with open(out) as f:
                results.append(json.load(f))
    base, spec = results
    same = True
    for prompt, a, b, top2, text in zip(PROMPTS, base['tokens'], spec['tokens'], base['top2'], spec['text']):
        if a == b:
            print(f'{prompt!r} -> {text!r}: same tokens', flush=True)
            continue
        i = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        (lp1, t1), (lp2, t2) = top2[i]
        tie = lp1 - lp2 < tie_tol and {a[i], b[i]} == {t1, t2}
        same &= tie
        print(f'{prompt!r}: DIFFERENT from token {i}: {a[i]} without, {b[i]} with speculative decoding; without, the '
              f'two best tokens there are {t1} ({lp1:.6f}) and {t2} ({lp2:.6f}): '
              f'{"a tie" if tie else "NOT a tie"}', flush=True)
    drafts = spec['vllm:spec_decode_num_drafts']
    draft_tokens = spec['vllm:spec_decode_num_draft_tokens']
    accepted = spec['vllm:spec_decode_num_accepted_tokens']
    per_pos = spec.get('vllm:spec_decode_num_accepted_tokens_per_pos')
    print(f'{method}: drafts {drafts}, draft tokens {draft_tokens}, accepted {accepted}: acceptance rate '
          f'{accepted / draft_tokens:.4f}, mean acceptance length {1 + accepted / drafts:.3f}, '
          f'per position {[a / drafts for a in per_pos] if per_pos else None}', flush=True)
    if method == 'mtp':
        _offline_acceptance(hf_dir, base['prompt_tokens'], base['tokens'], depth)
    assert same


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    fire.Fire({'logits': logits, 'generate': generate, 'e2e': e2e})
