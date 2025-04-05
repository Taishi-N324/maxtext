"""
 Copyright 2023 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

r"""Convert weights from a MaxText model to a HuggingFace model.

Usage:

Get MaxText model weights from a MaxText run

Example cmd:
To save a ckpt
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path <path/to/meta/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size llama2-7b

python3 MaxText/llama_mistral_mixtral_orbax_to_hf.py MaxText/configs/base.yml
            base_output_directory=path/to/saving/intermediate_MaxText_files
            load_parameters_path=/path/to/MaxText/checkpoint run_name=<your run name> model_name=<llama2 or mistral>
            hardware=gpu
            hf_model_path=/local/path/to/save/HF/model/to

Note that we are saving the converted HuggingFace model to a local path. You can write to a GCS location by mounting
the GCS bucket as a local path using `setup_gcsfuse.sh`, but remember to mount as read+write.
"""

from typing import Sequence

import numpy as np
import torch
from absl import app
from jax.sharding import Mesh
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
)

import checkpointing
import llama_or_mistral_ckpt
import max_logging
import max_utils
import pyconfig
from generate_param_only_checkpoint import _read_train_checkpoint


def unpermute_from_match_maxtext_rope(arr, model_size):
    """
    Function to get the RoPE values in correct ordering
    """
    if model_size[:8] != "llama3.1":
        return arr
    evens = arr[..., ::2]
    odds = arr[..., 1::2]
    return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


def pad_embeddings(raw_embed, base_emb_dim, target_size=262208):
    # To align the sizes for mapping, call and evaluate using Gemma3ForConditionalGeneration
    import jax.numpy as jnp
    import torch

    normalizer = jnp.sqrt(base_emb_dim).astype(jnp.bfloat16)
    embedding = raw_embed / normalizer

    embedding_np = np.array(embedding, dtype=np.float32)
    return torch.tensor(embedding_np, dtype=torch.float32)


def reverse_scale(arr, head_dim, model_size):
    """
    MaxText has the scaling factor included into the weights,
    we reverse it when writing out the HuggingFace checkpoint
    """
    if model_size == "gemma2-27b":
        # TODO remove hard-coded
        # https://huggingface.co/google/gemma-2-27b/blob/main/config.json#L15
        # https://huggingface.co/google/gemma-2-27b/blob/main/config.json#L20
        # https://github.com/Taishi-N324/maxtext/blob/c325f606ccc4d44b72fbf08e0173125b4ba54c2b/MaxText/convert_gemma2_chkpt.py#L79-L80
        # 4608 / 32
        scale = 144
    else:
        scale = head_dim
    return arr * np.sqrt(scale)


def scale_rmsnorm_layer_for_hf(arr):
    """Convert MaxText RMSNorm parameters to HuggingFace format

    Args:
        arr: Input array from MaxText

    Returns:
        Scaled array for HuggingFace
    """
    return arr - 1.0


def load_hf_model(model_size):
    """
    Load the model that we are interested in from HuggingFace

    """
    if model_size == "llama2-7b":
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    elif model_size == "mistral-7b":
        model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    elif model_size == "mixtral-8x7b":
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1", device_map="auto"
        )
    elif model_size == "llama3.1-8b":
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
        model = AutoModelForCausalLM.from_config(config)
    elif model_size == "gemma2-2b":
        config = AutoConfig.from_pretrained("google/gemma-2-2b")
        model = AutoModelForCausalLM.from_config(config)
    elif model_size == "gemma2-9b":
        config = AutoConfig.from_pretrained("google/gemma-2-9b")
        model = AutoModelForCausalLM.from_config(config)
    elif model_size == "gemma2-27b":
        config = AutoConfig.from_pretrained("google/gemma-2-27b")
        model = AutoModelForCausalLM.from_config(config)
    elif model_size == "gemma3-4b":
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-4b-it", device_map="auto"
        )
    elif model_size == "gemma3-12b":
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-12b-pt", device_map="auto"
        )
    elif model_size == "gemma3-27b":
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-27b-pt", device_map="auto"
        )
    else:
        raise NotImplementedError
    return model


def load_model_state(config):
    """
    Loads the MaxText model's TrainState from the Orbax checkpoint
    """
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Create a checkpoint manager to load decode checkpoint at config.checkpoint_dir
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
    )

    # Read training state from config.load_paramaters_path
    max_logging.log(f"Read training checkpoint from: {config.load_full_state_path}")
    training_state, _ = _read_train_checkpoint(config, checkpoint_manager, mesh)
    return training_state


def convert_gemma3_state_to_hf(training_state, model_size):
    """
    Port the parameters from the Orbax training_state into the hf_model with correct layer mapping
    """
    model_params = llama_or_mistral_ckpt.MODEL_PARAMS_DICT[model_size]
    base_num_decoder_layers = model_params["num_layers"]
    base_num_query_heads = model_params["num_heads"]
    head_dim = model_params["dims_per_head"]
    base_num_kv_heads = model_params["num_kv_heads"]
    base_emb_dim = model_params["base_emb_dim"]

    hf_model_params = {}

    # ------------------------------------------------------------------------
    # 1) Convert token embedding
    # ------------------------------------------------------------------------
    raw_embed = training_state.params["params"]["token_embedder"]["embedding"]
    hf_model_params["model.embed_tokens.weight"] = pad_embeddings(
        raw_embed, base_emb_dim
    )

    # ------------------------------------------------------------------------
    # 2) Map each decoder layer
    # ------------------------------------------------------------------------
    for hf_layer_idx in tqdm(
        range(base_num_decoder_layers), desc="Porting parameters layerwise"
    ):
        print(f"Converting weights for layer {hf_layer_idx}")

        # --------------------------------------------------------------------
        # Attention mapping
        # --------------------------------------------------------------------
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.q_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    unpermute_from_match_maxtext_rope(
                        training_state.params["params"]["decoder"]["layers"][
                            "self_attention"
                        ]["query"]["kernel"][:, hf_layer_idx, :, :],
                        model_size,
                    )
                    .reshape(base_emb_dim, base_num_query_heads * head_dim)
                    .T
                ),
                dtype=torch.float32,
            )
        )

        # K-proj
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.k_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    unpermute_from_match_maxtext_rope(
                        training_state.params["params"]["decoder"]["layers"][
                            "self_attention"
                        ]["key"]["kernel"][:, hf_layer_idx, :, :],
                        model_size,
                    )
                    .reshape(base_emb_dim, base_num_kv_heads * head_dim)
                    .T
                ),
                dtype=torch.float32,
            )
        )

        # V-proj
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.v_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][
                        "self_attention"
                    ]["value"]["kernel"][:, hf_layer_idx, :, :]
                    .reshape(base_emb_dim, base_num_kv_heads * head_dim)
                    .T
                ),
                dtype=torch.float32,
            )
        )

        # Out-proj
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.o_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][
                        "self_attention"
                    ]["out"]["kernel"][:, hf_layer_idx, :, :]
                    .reshape(base_num_query_heads * head_dim, base_emb_dim)
                    .T
                ),
                dtype=torch.float32,
            )
        )

        # Query-norm and Key-norm (Added for Gemma 3)
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.q_norm.weight"] = (
            torch.tensor(
                np.asarray(
                    scale_rmsnorm_layer_for_hf(
                        training_state.params["params"]["decoder"]["layers"][
                            "self_attention"
                        ]["query_norm"]["scale"][:, hf_layer_idx]
                    )
                ),
                dtype=torch.float32,
            )
        )

        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.k_norm.weight"] = (
            torch.tensor(
                np.asarray(
                    scale_rmsnorm_layer_for_hf(
                        training_state.params["params"]["decoder"]["layers"][
                            "self_attention"
                        ]["key_norm"]["scale"][:, hf_layer_idx]
                    )
                ),
                dtype=torch.float32,
            )
        )

        # --------------------------------------------------------------------
        # MLP mapping
        # --------------------------------------------------------------------
        hf_model_params[f"model.layers.{hf_layer_idx}.mlp.gate_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"]["mlp"]["wi_0"][
                        "kernel"
                    ][:, hf_layer_idx, :].T
                ),
                dtype=torch.float32,
            )
        )

        hf_model_params[f"model.layers.{hf_layer_idx}.mlp.up_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"]["mlp"]["wi_1"][
                        "kernel"
                    ][:, hf_layer_idx, :].T
                ),
                dtype=torch.float32,
            )
        )

        hf_model_params[f"model.layers.{hf_layer_idx}.mlp.down_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"]["mlp"]["wo"][
                        "kernel"
                    ][:, hf_layer_idx, :].T
                ),
                dtype=torch.float32,
            )
        )

        # --------------------------------------------------------------------
        # Norm layers (Gemma2 uses RMSNorm, MaxText's scale param => scale-1.0)
        # --------------------------------------------------------------------
        hf_model_params[f"model.layers.{hf_layer_idx}.input_layernorm.weight"] = (
            torch.tensor(
                np.asarray(
                    scale_rmsnorm_layer_for_hf(
                        training_state.params["params"]["decoder"]["layers"][
                            "pre_self_attention_norm"
                        ]["scale"][:, hf_layer_idx]
                    ).reshape(base_emb_dim)
                ),
                dtype=torch.float32,
            )
        )

        hf_model_params[
            f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight"
        ] = torch.tensor(
            np.asarray(
                scale_rmsnorm_layer_for_hf(
                    training_state.params["params"]["decoder"]["layers"][
                        "post_self_attention_norm"
                    ]["scale"][:, hf_layer_idx]
                ).reshape(base_emb_dim)
            ),
            dtype=torch.float32,
        )

        hf_model_params[
            f"model.layers.{hf_layer_idx}.pre_feedforward_layernorm.weight"
        ] = torch.tensor(
            np.asarray(
                scale_rmsnorm_layer_for_hf(
                    training_state.params["params"]["decoder"]["layers"][
                        "pre_ffw_norm"
                    ]["scale"][:, hf_layer_idx]
                ).reshape(base_emb_dim)
            ),
            dtype=torch.float32,
        )

        hf_model_params[
            f"model.layers.{hf_layer_idx}.post_feedforward_layernorm.weight"
        ] = torch.tensor(
            np.asarray(
                scale_rmsnorm_layer_for_hf(
                    training_state.params["params"]["decoder"]["layers"][
                        "post_ffw_norm"
                    ]["scale"][:, hf_layer_idx]
                ).reshape(base_emb_dim)
            ),
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------------
    # 3) final norm
    # ------------------------------------------------------------------------
    # Final RMSNorm
    norm_arr = np.asarray(
        training_state.params["params"]["decoder"]["decoder_norm"]["scale"].reshape(
            base_emb_dim
        )
    )
    hf_model_params["model.norm.weight"] = torch.tensor(
        scale_rmsnorm_layer_for_hf(norm_arr),
        dtype=torch.float32,
    )

    return hf_model_params


def convert_gemma2_state_to_hf(training_state, model_size):
    """
    Port the parameters from the Orbax training_state into the hf_model with correct layer mapping
    """
    model_params = llama_or_mistral_ckpt.MODEL_PARAMS_DICT[model_size]
    base_num_decoder_layers = model_params["num_layers"]
    base_num_query_heads = model_params["num_heads"]
    head_dim = model_params["dims_per_head"]
    base_num_kv_heads = model_params["num_kv_heads"]
    base_emb_dim = model_params["base_emb_dim"]

    hf_model_params = {}

    # ------------------------------------------------------------------------
    # 1) Convert token embedding
    # ------------------------------------------------------------------------
    # Typically padded from [256000, d_emb] up to [256128, d_emb].
    raw_embed = training_state.params["params"]["token_embedder"]["embedding"]
    import jax.numpy as jnp

    embedding = np.array(raw_embed, copy=True, dtype=np.float32)[:256000, :]
    normalizer = jnp.sqrt(base_emb_dim).astype(jnp.bfloat16)
    embedding = embedding / normalizer
    hf_model_params["model.embed_tokens.weight"] = torch.tensor(
        embedding, dtype=torch.float32
    )

    # ------------------------------------------------------------------------
    # 2) Map each decoder layer
    # ------------------------------------------------------------------------
    for hf_layer_idx in tqdm(
        range(base_num_decoder_layers), desc="Porting parameters layerwise"
    ):
        print(f"Converting weights for layer {hf_layer_idx}")

        # Calculate MaxText layer index and whether this is a local or global layer
        maxtext_layer_idx = hf_layer_idx // 2
        is_local = hf_layer_idx % 2 == 0

        # Select appropriate keys based on layer type
        layer_type = "local" if is_local else "global"
        attention_key = f"self_attention_{layer_type}"
        mlp_key = f"mlp_{layer_type}"

        # Layer norm keys
        pre_attention_norm_key = f"pre_self_attention_norm_{layer_type}"
        post_attention_norm_key = f"post_self_attention_norm_{layer_type}"
        pre_ffw_norm_key = f"pre_ffw_norm_{layer_type}"
        post_ffw_norm_key = f"post_ffw_norm_{layer_type}"

        # --------------------------------------------------------------------
        # Attention mapping
        # --------------------------------------------------------------------
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.q_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    unpermute_from_match_maxtext_rope(
                        reverse_scale(
                            training_state.params["params"]["decoder"]["layers"][
                                attention_key
                            ]["query"]["kernel"][:, maxtext_layer_idx, :, :],
                            head_dim,
                            model_size,
                        ),
                        model_size,
                    )
                    .reshape(base_emb_dim, base_num_query_heads * head_dim)
                    .T
                ),
                dtype=torch.float32,
            )
        )

        # K-proj
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.k_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    unpermute_from_match_maxtext_rope(
                        training_state.params["params"]["decoder"]["layers"][
                            attention_key
                        ]["key"]["kernel"][:, maxtext_layer_idx, :, :],
                        model_size,
                    )
                    .reshape(base_emb_dim, base_num_kv_heads * head_dim)
                    .T
                ),
                dtype=torch.float32,
            )
        )

        # V-proj
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.v_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][attention_key][
                        "value"
                    ]["kernel"][:, maxtext_layer_idx, :, :]
                    .reshape(base_emb_dim, base_num_kv_heads * head_dim)
                    .T
                ),
                dtype=torch.float32,
            )
        )

        # Out-proj
        hf_model_params[f"model.layers.{hf_layer_idx}.self_attn.o_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][attention_key][
                        "out"
                    ]["kernel"][:, maxtext_layer_idx, :, :]
                    .reshape(base_num_query_heads * head_dim, base_emb_dim)
                    .T
                ),
                dtype=torch.float32,
            )
        )

        # --------------------------------------------------------------------
        # MLP mapping
        # --------------------------------------------------------------------
        hf_model_params[f"model.layers.{hf_layer_idx}.mlp.gate_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][mlp_key][
                        "wi_0"
                    ]["kernel"][:, maxtext_layer_idx, :].T
                ),
                dtype=torch.float32,
            )
        )

        hf_model_params[f"model.layers.{hf_layer_idx}.mlp.up_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][mlp_key][
                        "wi_1"
                    ]["kernel"][:, maxtext_layer_idx, :].T
                ),
                dtype=torch.float32,
            )
        )

        hf_model_params[f"model.layers.{hf_layer_idx}.mlp.down_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][mlp_key]["wo"][
                        "kernel"
                    ][:, maxtext_layer_idx, :].T
                ),
                dtype=torch.float32,
            )
        )

        # --------------------------------------------------------------------
        # Norm layers (Gemma2 uses RMSNorm, MaxText's scale param => scale-1.0)
        # --------------------------------------------------------------------
        hf_model_params[f"model.layers.{hf_layer_idx}.input_layernorm.weight"] = (
            torch.tensor(
                np.asarray(
                    scale_rmsnorm_layer_for_hf(
                        training_state.params["params"]["decoder"]["layers"][
                            pre_attention_norm_key
                        ]["scale"][:, maxtext_layer_idx]
                    ).reshape(base_emb_dim)
                ),
                dtype=torch.float32,
            )
        )

        hf_model_params[
            f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight"
        ] = torch.tensor(
            np.asarray(
                scale_rmsnorm_layer_for_hf(
                    training_state.params["params"]["decoder"]["layers"][
                        post_attention_norm_key
                    ]["scale"][:, maxtext_layer_idx]
                ).reshape(base_emb_dim)
            ),
            dtype=torch.float32,
        )

        hf_model_params[
            f"model.layers.{hf_layer_idx}.pre_feedforward_layernorm.weight"
        ] = torch.tensor(
            np.asarray(
                scale_rmsnorm_layer_for_hf(
                    training_state.params["params"]["decoder"]["layers"][
                        pre_ffw_norm_key
                    ]["scale"][:, maxtext_layer_idx]
                ).reshape(base_emb_dim)
            ),
            dtype=torch.float32,
        )

        hf_model_params[
            f"model.layers.{hf_layer_idx}.post_feedforward_layernorm.weight"
        ] = torch.tensor(
            np.asarray(
                scale_rmsnorm_layer_for_hf(
                    training_state.params["params"]["decoder"]["layers"][
                        post_ffw_norm_key
                    ]["scale"][:, maxtext_layer_idx]
                ).reshape(base_emb_dim)
            ),
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------------
    # 3) final norm
    # ------------------------------------------------------------------------
    # Final RMSNorm
    norm_arr = np.asarray(
        training_state.params["params"]["decoder"]["decoder_norm"]["scale"].reshape(
            base_emb_dim
        )
    )
    hf_model_params["model.norm.weight"] = torch.tensor(
        scale_rmsnorm_layer_for_hf(norm_arr),
        dtype=torch.float32,
    )

    return hf_model_params


def convert_state_to_hf(training_state, model_size):
    """
    Port the parameters from the Orbax training_state into the hf_model
    """

    if model_size not in llama_or_mistral_ckpt.MODEL_PARAMS_DICT:
        raise NotImplementedError

    if model_size.startswith("gemma2-"):
        return convert_gemma2_state_to_hf(training_state, model_size)
    elif model_size.startswith("gemma3-"):
        return convert_gemma3_state_to_hf(training_state, model_size)
    # Load the model specific parameters
    model_params = llama_or_mistral_ckpt.MODEL_PARAMS_DICT[model_size]
    base_num_decoder_layers = model_params["num_layers"]
    base_num_query_heads = model_params["num_heads"]
    head_dim = model_params["dims_per_head"]
    base_num_kv_heads = model_params["num_kv_heads"]
    num_experts = model_params["num_experts"] if "num_experts" in model_params else None

    hf_model_params = {}

    # Port the embedding weights
    hf_model_params["model.embed_tokens.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["token_embedder"]["embedding"]),
        dtype=torch.float16,
    )

    for layer_int in tqdm(
        range(base_num_decoder_layers), desc="Porting parameters layerwise"
    ):
        print(f"Converting weights for layer {layer_int}")

        # Attention layers
        hf_model_params[f"model.layers.{layer_int}.self_attn.q_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    unpermute_from_match_maxtext_rope(
                        reverse_scale(
                            training_state.params["params"]["decoder"]["layers"][
                                "self_attention"
                            ]["query"]["kernel"][:, layer_int, :, :],
                            head_dim,
                            model_size,
                        ),
                        model_size,
                    )
                    .reshape(
                        base_num_query_heads * head_dim, base_num_query_heads * head_dim
                    )
                    .T
                ),
                dtype=torch.float16,
            )
        )

        hf_model_params[f"model.layers.{layer_int}.self_attn.k_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    unpermute_from_match_maxtext_rope(
                        training_state.params["params"]["decoder"]["layers"][
                            "self_attention"
                        ]["key"]["kernel"][:, layer_int, :, :],
                        model_size,
                    )
                    .reshape(
                        base_num_query_heads * head_dim, base_num_kv_heads * head_dim
                    )
                    .T
                ),
                dtype=torch.float16,
            )
        )
        hf_model_params[f"model.layers.{layer_int}.self_attn.v_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][
                        "self_attention"
                    ]["value"]["kernel"][:, layer_int, :, :]
                    .reshape(
                        base_num_query_heads * head_dim, base_num_kv_heads * head_dim
                    )
                    .T
                ),
                dtype=torch.float16,
            )
        )
        hf_model_params[f"model.layers.{layer_int}.self_attn.o_proj.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][
                        "self_attention"
                    ]["out"]["kernel"][:, layer_int, :, :]
                    .reshape(
                        base_num_query_heads * head_dim, base_num_query_heads * head_dim
                    )
                    .T
                ),
                dtype=torch.float16,
            )
        )

        # MLP Layers
        if num_experts is None:
            hf_model_params[f"model.layers.{layer_int}.mlp.gate_proj.weight"] = (
                torch.tensor(
                    np.asarray(
                        training_state.params["params"]["decoder"]["layers"]["mlp"][
                            "wi_0"
                        ]["kernel"][:, layer_int, :].T
                    ),
                    dtype=torch.float16,
                )
            )
            hf_model_params[f"model.layers.{layer_int}.mlp.up_proj.weight"] = (
                torch.tensor(
                    np.asarray(
                        training_state.params["params"]["decoder"]["layers"]["mlp"][
                            "wi_1"
                        ]["kernel"][:, layer_int, :].T
                    ),
                    dtype=torch.float16,
                )
            )
            hf_model_params[f"model.layers.{layer_int}.mlp.down_proj.weight"] = (
                torch.tensor(
                    np.asarray(
                        training_state.params["params"]["decoder"]["layers"]["mlp"][
                            "wo"
                        ]["kernel"][:, layer_int, :].T
                    ),
                    dtype=torch.float16,
                )
            )
        else:
            hf_model_params[
                f"model.layers.{layer_int}.block_sparse_moe.gate.weight"
            ] = torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"]["MoeBlock_0"][
                        "gate"
                    ]["kernel"][:, layer_int, :].T
                ),
                dtype=torch.float16,
            )
            for k in range(num_experts):
                hf_model_params[
                    f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w1.weight"
                ] = torch.tensor(
                    np.asarray(
                        training_state.params["params"]["decoder"]["layers"][
                            "MoeBlock_0"
                        ]["wi_0"][k, layer_int, :, :].T
                    ),
                    dtype=torch.float16,
                )
                hf_model_params[
                    f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w2.weight"
                ] = torch.tensor(
                    np.asarray(
                        training_state.params["params"]["decoder"]["layers"][
                            "MoeBlock_0"
                        ]["wo"][k, layer_int, :, :].T
                    ),
                    dtype=torch.float16,
                )
                hf_model_params[
                    f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w3.weight"
                ] = torch.tensor(
                    np.asarray(
                        training_state.params["params"]["decoder"]["layers"][
                            "MoeBlock_0"
                        ]["wi_1"][k, layer_int, :, :].T
                    ),
                    dtype=torch.float16,
                )

        # Pre/post attention layer norm
        hf_model_params[f"model.layers.{layer_int}.input_layernorm.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][
                        "pre_self_attention_layer_norm"
                    ]["scale"][:, layer_int].reshape(base_num_query_heads * head_dim)
                ),
                dtype=torch.float16,
            )
        )
        hf_model_params[f"model.layers.{layer_int}.post_attention_layernorm.weight"] = (
            torch.tensor(
                np.asarray(
                    training_state.params["params"]["decoder"]["layers"][
                        "post_self_attention_layer_norm"
                    ]["scale"][:, layer_int].reshape(base_num_query_heads * head_dim)
                ),
                dtype=torch.float16,
            )
        )

    # LM head and layernorm
    hf_model_params["lm_head.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["logits_dense"]["kernel"].T
        ),
        dtype=torch.float16,
    )
    hf_model_params["model.norm.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["decoder_norm"]["scale"].reshape(
                base_num_query_heads * head_dim
            )
        ),
        dtype=torch.float16,
    )

    return hf_model_params


def convert_orbax_hf(hf_model_path, config):
    """
    Landing function to convert MaxText model's checkpoint to HuggingFace format
    """
    hf_model = load_hf_model(config.model_name)
    training_state = load_model_state(config)
    new_hf_model_params = convert_state_to_hf(training_state, config.model_name)
    print(f"Saving HuggingFace model to path = {hf_model_path}")
    hf_model.save_pretrained(hf_model_path, state_dict=new_hf_model_params)


def main(argv: Sequence[str]):
    config = pyconfig.initialize(argv[:-1])
    hf_model_path = argv[-1].split("=")[1]
    print(f"Will save converted HuggingFace checkpoint to path = {hf_model_path}")

    convert_orbax_hf(hf_model_path, config)


if __name__ == "__main__":
    app.run(main)
