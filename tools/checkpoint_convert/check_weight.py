from transformers import AutoModel
import torch


def compare_models(model_path1, model_path2):
    print(f"Loading models in bf16")
    model1 = AutoModel.from_pretrained(model_path1, torch_dtype=torch.bfloat16)
    model2 = AutoModel.from_pretrained(model_path2, torch_dtype=torch.bfloat16)

    def check_diff(name, param1, param2, threshold, vocab_size=None):
        """
        Compares param1 and param2 up to `vocab_size` rows (if supplied).
        Reports both max difference and mean difference. Pass/fail is based
        on the max difference vs. `threshold`.
        """

        if param1.shape != param2.shape:
            return {
                "status": "error",
                "message": f"Shape mismatch for {name}: {param1.shape} vs {param2.shape}",
            }

        abs_diff = (param1.data - param2.data).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        if max_diff > threshold:
            return {
                "status": "error",
                "message": (
                    f"Error in {name}:\n"
                    f"  Max difference {max_diff:.6f} exceeds threshold {threshold}\n"
                    f"  Mean difference {mean_diff:.6f}"
                ),
            }
        else:
            return {
                "status": "pass",
                "message": (
                    f"{name}:\n"
                    f"  Max difference {max_diff:.6f} <= threshold {threshold}\n"
                    f"  Mean difference {mean_diff:.6f}"
                ),
            }

    results = []

    result = check_diff(
        "embed_tokens",
        model1.embed_tokens.weight,
        model2.embed_tokens.weight,
        0.015626,  # 0.015626 ≈ 1/64
        vocab_size=256000,
    )

    results.append(result)

    # Compare final norm
    result = check_diff(
        "final_norm",
        model1.norm.weight,
        model2.norm.weight,
        0.031251,  # 0.031251 ≈ 1/32
    )
    results.append(result)

    # Compare each layer
    num_layers = len(model1.layers)
    print(f"\nChecking all {num_layers} layers...")

    for layer_idx in range(num_layers):
        layer1 = model1.layers[layer_idx]
        layer2 = model2.layers[layer_idx]

        # Check all LayerNorms
        norm_names = [
            "input_layernorm",
            "post_attention_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm",
        ]
        for norm_name in norm_names:
            if hasattr(layer1, norm_name) and hasattr(layer2, norm_name):
                result = check_diff(
                    f"Layer {layer_idx} - {norm_name}",
                    getattr(layer1, norm_name).weight,
                    getattr(layer2, norm_name).weight,
                    0.031251,  # 0.031251 ≈ 1/32
                )
                results.append(result)

        # Check MLP weights
        for mlp_name in ["down_proj", "gate_proj", "up_proj"]:
            result = check_diff(
                f"Layer {layer_idx} - MLP {mlp_name}",
                getattr(layer1.mlp, mlp_name).weight,
                getattr(layer2.mlp, mlp_name).weight,
                1e-15,
            )
            results.append(result)

        # Check Attention weights
        for attn_name in ["k_proj", "o_proj", "q_proj", "v_proj"]:
            result = check_diff(
                f"Layer {layer_idx} - Attention {attn_name}",
                getattr(layer1.self_attn, attn_name).weight,
                getattr(layer2.self_attn, attn_name).weight,
                1e-15,
            )
            results.append(result)

    # Summarize results
    errors = [r for r in results if r["status"] == "error"]
    passes = [r for r in results if r["status"] == "pass"]

    if errors:
        print("\nThe following errors were found:")
        for result in errors:
            print(f"❌ {result['message']}")

    print("\nThe following checks passed:")
    for result in passes:
        print(f"✅ {result['message']}")


# Example usage:
model_path1 = "google/gemma-2-2b"  # HF model hub path
model_path2 = "/mnt/filestore/checkpoints_maxtext_to_hf/grain_test_2025-02-18-00-16/checkpoints/0"  # local path to compare
compare_models(model_path1, model_path2)
