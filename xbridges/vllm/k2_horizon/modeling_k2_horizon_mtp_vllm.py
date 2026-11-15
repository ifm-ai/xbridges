# SPDX-License-Identifier: Apache-2.0
"""K2-Horizon multi-token prediction (MTP) layers as a vLLM speculative decoding draft
(`--speculative-config '{"method": "mtp", "num_speculative_tokens": D}'`).

The draft runs the MTP layers of an xBridges export (xbridges/huggingface/mtp.py): MTP module k is layer
num_hidden_layers + k - 1 of the checkpoint, a K2-Horizon decoder block of the last decoder layer's kind with
`enorm`, `hnorm`, `eh_proj` and `shared_head.norm`; the embedding and the output head are the main model's. As in
vLLM's DeepSeekMTP, module k computes at position p

    y = shared_head.norm(block(eh_proj([enorm(embedding of token p + k); hnorm(h)])))

with h the main model's final-normed hidden state at p for k = 1 and module k-1's y after, and the logits
lm_head(y). Unlike DeepSeekMTP, the embedding at position 0 is not masked and each module returns y after its own
norm, which is then both its logits' input and the next module's h, as xLLM trains them (xllm/modules/mtp.py).

vLLM's tables do not know this model; xbridges/vllm/k2_horizon/mtp_plugin.py registers it at runtime.
"""

import re
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import k2_horizon
from vllm.model_executor.models.k2_horizon import (
    K2HorizonDecoderLayer,
    K2HorizonModel,
    K2HorizonRMSNorm,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

# the weights of an MTP layer that are not its decoder block's
MTP_LAYER_WEIGHTS = ("enorm.", "hnorm.", "eh_proj.", "shared_head.")


def mtp_layers(config) -> range:
    """The indices of the checkpoint's MTP layers."""
    num_layers = config.num_hidden_layers
    return range(num_layers, num_layers + getattr(config, "num_nextn_predict_layers", 0))


def is_mtp_layer_weight(config, name: str) -> bool:
    match = re.match(r"model\.layers\.(\d+)\.", name)
    return match is not None and int(match[1]) in mtp_layers(config)


def is_sparse_layer(config, layer_idx: int) -> bool:
    """Whether K2HorizonDecoderLayer builds layer `layer_idx` with a MoE (and MoVA when the model has it)."""
    mlp_only_layers = getattr(config, "mlp_only_layers", [])
    return (layer_idx not in mlp_only_layers) and (
        config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
    )


def make_norm(config) -> K2HorizonRMSNorm:
    return K2HorizonRMSNorm(
        hidden_size=config.hidden_size,
        n_groups=config.layernorm_num_groups,
        eps=config.rms_norm_eps,
    )


class K2HorizonSharedHead(nn.Module):
    """The module's final norm; its output head is the main model's lm_head."""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm = make_norm(config)


# Compiled per MTP layer, not as DeepSeekMTP is (as a whole): the whole draft would be compiled for the first
# spec_step_idx it runs, and with vLLM's guards dropped it would run that layer at every step.
@support_torch_compile
class K2HorizonMultiTokenPredictorLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.enorm = make_norm(config)
        self.hnorm = make_norm(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = K2HorizonSharedHead(config)
        # the prefix names the block's attention (KV cache) and gives its layer index, so its kind
        self.mtp_block = K2HorizonDecoderLayer(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1
            )
        )
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        return hidden_states


class K2HorizonMultiTokenPredictor(nn.Module):
    # the main model's weight loading, for the MTP layers' tensors
    get_expert_mapping = K2HorizonModel.get_expert_mapping
    load_weights = K2HorizonModel.load_weights

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        layers = mtp_layers(config)
        if not layers:
            raise ValueError(
                "The K2-Horizon checkpoint has no MTP layers (num_nextn_predict_layers = 0)"
            )
        self.mtp_start_layer_idx = layers.start
        self.num_mtp_layers = len(layers)
        last = config.num_hidden_layers - 1
        for idx in layers:
            # the block is built from its layer index, as the decoder layers are
            if is_sparse_layer(config, idx) != is_sparse_layer(config, last):
                raise ValueError(
                    f"MTP layer {idx} would not be of the kind of the last decoder layer {last} "
                    "(mlp_only_layers, decoder_sparse_step)"
                )
        self.layers = nn.ModuleDict(
            {
                str(idx): K2HorizonMultiTokenPredictorLayer(
                    vllm_config=vllm_config, prefix=f"{prefix}.layers.{idx}"
                )
                for idx in layers
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        layer = self.layers[
            str(self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers)
        ]
        return layer(positions, previous_hidden_states, inputs_embeds)


class K2HorizonMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = K2HorizonMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # replaced by the main model's embedding and lm_head when the draft is loaded
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """The module's output after its final norm, the input of its logits and of the next module."""
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        # hidden_states are already normed by their module's shared_head.norm
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Loads the MTP layers, the embedding and lm_head of a checkpoint and skips the decoder layers."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def mtp_weights():
            for name, loaded_weight in weights:
                if name == "lm_head.weight":
                    if not self.config.tie_word_embeddings:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                elif name == "model.embed_tokens.weight":
                    yield "embed_tokens.weight", loaded_weight
                elif (
                    is_mtp_layer_weight(self.config, name)
                    and "rotary_emb.inv_freq" not in name
                ):
                    layer, rest = re.match(r"model\.layers\.(\d+)\.(.+)", name).groups()
                    if not rest.startswith(MTP_LAYER_WEIGHTS):
                        rest = "mtp_block." + rest
                    yield f"layers.{layer}.{rest}", loaded_weight

        loaded_params.update(
            "model." + name for name in self.model.load_weights(mtp_weights())
        )
        missing = [
            idx
            for idx in mtp_layers(self.config)
            if f"model.layers.{idx}.eh_proj.weight" not in loaded_params
        ]
        if missing:
            raise ValueError(f"MTP layers {missing} are missing from the checkpoint")
        return loaded_params


class K2HorizonForCausalLM(k2_horizon.K2HorizonForCausalLM):
    """vLLM's K2-Horizon model, which leaves the MTP layers of a checkpoint to the draft. A K2-Horizon model of
    vLLM that does not skip them (other than xbridges/vllm/k2_horizon/modeling_k2_horizon_vllm.py) fails on them."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            (name, weight)
            for name, weight in weights
            if not is_mtp_layer_weight(self.config, name)
        )
