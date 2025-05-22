import torch
from transformers import AutoModelForCausalLM


def compare_models(model_path1, model_path2):
    print(f"Loading models in bf16")
    model1 = AutoModelForCausalLM.from_pretrained(
        model_path1, torch_dtype=torch.bfloat16
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        model_path2, torch_dtype=torch.bfloat16
    )

    # Helper function to collect all parameters
    def get_parameter_dict(model):
        param_dict = {}
        # Start with embedding
        param_dict["embed_tokens"] = model.model.embed_tokens.weight
        param_dict["norm"] = model.model.norm.weight

        # Add all layer parameters
        for layer_idx, layer in enumerate(model.model.layers):
            # Process each attribute in the layer
            for layer_attr in dir(layer):
                # Skip private attributes and non-parameter attributes
                if layer_attr.startswith("_") or not hasattr(
                    getattr(layer, layer_attr), "weight"
                ):
                    continue

                component = getattr(layer, layer_attr)

                # Handle nested components like MLP or attention
                if hasattr(component, "weight"):
                    param_dict[f"layer_{layer_idx}.{layer_attr}"] = component.weight
                else:
                    # For nested modules like self_attn and mlp
                    for sub_attr in dir(component):
                        if sub_attr.startswith("_") or not hasattr(
                            getattr(component, sub_attr), "weight"
                        ):
                            continue

                        param_dict[f"layer_{layer_idx}.{layer_attr}.{sub_attr}"] = (
                            getattr(component, sub_attr).weight
                        )

        return param_dict

    def check_diff(name, param1, param2, threshold, vocab_size=None):
        """
        Compares param1 and param2 up to `vocab_size` rows (if supplied).
        Reports both max difference and mean difference. Pass/fail is based
        on the max difference vs. `threshold`.
        """
        # Apply vocab_size limit if specified
        if vocab_size is not None and param1.shape[0] > vocab_size:
            param1 = param1[:vocab_size]
            param2 = param2[:vocab_size]

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

    # Get parameters from both models
    model1_params = get_parameter_dict(model1)
    model2_params = get_parameter_dict(model2)

    # Find common parameters and compare them
    results = []
    all_keys = set(model1_params.keys()) | set(model2_params.keys())

    for key in sorted(all_keys):
        if key not in model1_params:
            results.append(
                {"status": "error", "message": f"Parameter {key} missing in model1"}
            )
            continue

        if key not in model2_params:
            results.append(
                {"status": "error", "message": f"Parameter {key} missing in model2"}
            )
            continue

        # Determine appropriate threshold based on parameter type
        # if "norm" in key:
        #     threshold = 1e-15  # 1/32 for norm layers
        if key == "embed_tokens":
            threshold = 1e-15  # 1/64 for embeddings
            result = check_diff(
                key,
                model1_params[key],
                model2_params[key],
                threshold,
                vocab_size=262144,
            )
            results.append(result)
            continue
        else:
            threshold = 1e-15  # Strict comparison for other layers

        result = check_diff(key, model1_params[key], model2_params[key], threshold)
        results.append(result)

    # Summarize results
    errors = [r for r in results if r["status"] == "error"]
    passes = [r for r in results if r["status"] == "pass"]

    if errors:
        print("\nThe following errors were found:")
        for result in errors:
            print(f"❌ {result['message']}")

    print(f"\n{len(passes)} checks passed out of {len(results)} total")
    print("\nSample of passing checks:")
    for result in passes[:10]:  # Show first 10 passing results
        print(f"✅ {result['message']}")

    if len(passes) > 10:
        print(f"... and {len(passes) - 10} more passes")

    return len(errors) == 0


# Example usage:
model_path1 = "google/gemma-3-4b-pt"
model_path2 = "/mnt/filestore/convert_gemma-3-4b-pt"
compare_models(model_path1, model_path2)
