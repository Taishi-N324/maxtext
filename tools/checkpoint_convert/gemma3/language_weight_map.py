import argparse
import glob
import os

import torch
from transformers import AutoModelForCausalLM, Gemma3ForConditionalGeneration


def copy_weights_and_save(
    source_model_path, target_model_path, output_path, torch_dtype, save_mapping=True
):
    """
    Copy weights from a source model to a target model structure and save the result.
    Special handling for embedding layer to only replace first 262,144 entries.
    Skip vision tower and multi-modal projector parameters to maintain original vision capabilities.

    Args:
        source_model_path: Path to the source model (e.g., converted from MaxText)
        target_model_path: Path to the target model structure (e.g., HF Gemma3)
        output_path: Path to save the resulting model
        torch_dtype: PyTorch dtype to use for loading models
        save_mapping: Whether to save the parameter mapping to a file
    """
    print(f"Loading source model from {source_model_path}")
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:  # float32
        dtype = torch.float32

    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_path, torch_dtype=dtype
    )

    print(f"Loading target model from {target_model_path}")
    target_model = Gemma3ForConditionalGeneration.from_pretrained(
        target_model_path, torch_dtype=dtype
    )

    # Create mapping from target model structure to source model structure
    mapping = create_parameter_mapping(target_model, source_model)

    # Copy weights using the mapping
    missing_params = []
    copied_params = []
    skipped_params = []
    partially_copied_params = []
    vision_params = []

    # Get all named parameters from the target model
    target_params = dict(target_model.named_parameters())
    for name, param in target_params.items():
        # Skip vision tower and multi-modal projector parameters completely
        if name.startswith("vision_tower") or name.startswith("multi_modal_projector"):
            vision_params.append(
                (
                    name,
                    "Vision/multimodal component - intentionally preserved",
                    str(tuple(param.shape)),
                )
            )
            continue

        # Special handling for embedding layer
        if name == "language_model.model.embed_tokens.weight":
            if name in mapping:
                source_name = mapping[name]
                if source_name in source_model.state_dict():
                    source_param = source_model.state_dict()[source_name]

                    # Check if this is the embedding layer with size mismatch
                    if param.shape[0] == 262208 and source_param.shape[0] == 262144:
                        print(f"Special handling for embedding layer: {name}")
                        print(
                            f"  Target shape: {param.shape}, Source shape: {source_param.shape}"
                        )

                        # Copy only the first 262,144 entries and keep the rest
                        with torch.no_grad():
                            # Copy the first 262,144 rows
                            param[:262144, :].copy_(source_param)
                            # The remaining rows (262144 to 262208) keep their original values

                        partially_copied_params.append(
                            (
                                name,
                                source_name,
                                f"Partial copy: first {source_param.shape[0]} of {param.shape[0]} rows",
                            )
                        )
                        continue
                    else:
                        exit()

        # Normal parameter handling for all other parameters
        if name in mapping:
            source_name = mapping[name]
            # Get the source parameter through the model's state_dict
            if source_name in source_model.state_dict():
                source_param = source_model.state_dict()[source_name]

                # Check if shapes match
                if param.shape == source_param.shape:
                    # Copy the parameter
                    with torch.no_grad():
                        param.copy_(source_param)
                    copied_params.append((name, source_name, tuple(param.shape)))
                else:
                    print(
                        f"Shape mismatch: {name} {param.shape} vs {source_name} {source_param.shape}"
                    )
                    skipped_params.append(
                        (
                            name,
                            source_name,
                            f"Shape mismatch: {param.shape} vs {source_param.shape}",
                        )
                    )
            else:
                missing_params.append((name, source_name, "Source parameter not found"))
        else:
            missing_params.append((name, "No mapping found", ""))

    # Print summary
    print(f"\nCopied {len(copied_params)} parameters completely")
    print(f"Partially copied {len(partially_copied_params)} parameters")
    print(f"Skipped {len(skipped_params)} parameters due to shape mismatches")
    print(f"Missing {len(missing_params)} parameters")
    print(f"Preserved {len(vision_params)} vision parameters")

    # Show sample of copied parameters
    if copied_params:
        print("\nSample of copied parameters:")
        for target_name, source_name, shape in copied_params[
            :10
        ]:  # Print just 10 examples
            print(f"  {target_name} ({shape}) <- {source_name}")

    # Show partially copied parameters
    if partially_copied_params:
        print("\nPartially copied parameters:")
        for target_name, source_name, details in partially_copied_params:
            print(f"  {target_name} <- {source_name} ({details})")

    # Show sample of vision parameters
    if vision_params:
        print("\nSample of preserved vision parameters:")
        sample_size = min(10, len(vision_params))
        for target_name, reason, shape in vision_params[:sample_size]:
            print(f"  {target_name} ({shape}) - {reason}")

    # Save the mapping to a file if requested
    if save_mapping:
        mapping_data = {
            "copied_params": [(t, s, str(shape)) for t, s, shape in copied_params],
            "partially_copied_params": [
                (t, s, d) for t, s, d in partially_copied_params
            ],
            "skipped_params": [(t, s, r) for t, s, r in skipped_params],
            "missing_params": [(t, s, r) for t, s, r in missing_params],
            "vision_params": [(t, r, s) for t, r, s in vision_params],
        }

    # Save the updated model
    print(f"\nSaving model to {output_path}")
    target_model.save_pretrained(output_path)
    print(f"Model saved successfully to {output_path}")

    # Print mapping statistics summary
    total_params = len(target_params)
    print(f"\nSummary:")
    print(f"Total parameters: {total_params}")
    print(
        f"Successfully copied completely: {len(copied_params)} ({len(copied_params)/total_params*100:.1f}%)"
    )
    print(
        f"Partially copied: {len(partially_copied_params)} ({len(partially_copied_params)/total_params*100:.1f}%)"
    )
    print(
        f"Skipped: {len(skipped_params)} ({len(skipped_params)/total_params*100:.1f}%)"
    )
    print(
        f"Missing: {len(missing_params)} ({len(missing_params)/total_params*100:.1f}%)"
    )
    print(
        f"Preserved vision: {len(vision_params)} ({len(vision_params)/total_params*100:.1f}%)"
    )

    return target_model, {
        "copied": copied_params,
        "partially_copied": partially_copied_params,
        "skipped": skipped_params,
        "missing": missing_params,
        "vision": vision_params,
    }


def create_parameter_mapping(target_model, source_model):
    """
    Create a mapping from target model parameter names to source model parameter names.
    This handles the difference between Gemma3ForConditionalGeneration and standard HF models.
    Vision tower and multi-modal projector parameters are intentionally excluded from mapping.

    Args:
        target_model: The model to copy weights to
        source_model: The model to copy weights from

    Returns:
        Dictionary mapping target parameter names to source parameter names
    """
    # Get all parameter names from both models
    target_params = set(dict(target_model.named_parameters()).keys())
    source_params = set(dict(source_model.named_parameters()).keys())

    # Create a mapping to track successful mappings
    mapping = {}

    # Check if source model has language_model prefix
    source_has_language_model_prefix = any(
        name.startswith("language_model.") for name in source_params
    )

    # For debugging, print some sample parameter names from both models
    print("\nSample target parameter names:")
    print("\n".join(list(target_params)[:5]))

    print("\nSample source parameter names:")
    print("\n".join(list(source_params)[:5]))

    # Count vision and multi-modal projector parameters
    vision_params = [
        p
        for p in target_params
        if p.startswith("vision_tower") or p.startswith("multi_modal_projector")
    ]
    print(
        f"\nFound {len(vision_params)} vision tower and multi-modal projector parameters (will be preserved)"
    )

    # For each target parameter, try to find a corresponding source parameter
    for param_name in target_params:
        # Skip vision tower and multi-modal projector parameters completely - don't try to map them
        if param_name.startswith("vision_tower") or param_name.startswith(
            "multi_modal_projector"
        ):
            continue

        # Extract the non-prefixed part if target has language_model prefix
        param_without_prefix = param_name
        if param_name.startswith("language_model."):
            param_without_prefix = param_name[len("language_model.") :]

        # Generate possible source parameter names
        possible_mappings = []

        # If source has language_model prefix
        if source_has_language_model_prefix:
            possible_mappings.extend(
                [
                    param_name,  # Direct match
                    "language_model."
                    + param_without_prefix,  # Add prefix if not present
                ]
            )

            # Handle model. subpath
            if param_without_prefix.startswith("model."):
                possible_mappings.append("language_model." + param_without_prefix)
                possible_mappings.append(
                    "language_model.model." + param_without_prefix[6:]
                )
            else:
                possible_mappings.append("language_model.model." + param_without_prefix)
        else:
            # If source doesn't have language_model prefix
            possible_mappings.extend(
                [
                    param_name,  # Direct match
                    param_without_prefix,  # Without prefix
                ]
            )

            # Handle model. subpath
            if param_without_prefix.startswith("model."):
                possible_mappings.append(param_without_prefix)
                possible_mappings.append(param_without_prefix[6:])
            else:
                possible_mappings.append("model." + param_without_prefix)

        # Try all possible mappings
        for source_name in possible_mappings:
            if source_name in source_params:
                mapping[param_name] = source_name
                break

    # Count target parameters that are not vision-related or multi-modal
    non_vision_params = [
        p
        for p in target_params
        if not (p.startswith("vision_tower") or p.startswith("multi_modal_projector"))
    ]

    # Print mapping statistics
    print(
        f"\nCreated mapping for {len(mapping)} out of {len(non_vision_params)} non-vision parameters"
    )
    print(f"Source model has {len(source_params)} parameters")

    # Print some sample mappings
    sample_keys = list(mapping.keys())[:5] if mapping else []
    if sample_keys:
        print("\nSample of successful mappings:")
        for key in sample_keys:
            print(f"  {key} -> {mapping[key]}")

    # Print some sample unmapped parameters
    unmapped = [p for p in non_vision_params if p not in mapping]
    if unmapped:
        print("\nSample of unmapped parameters:")
        for param in unmapped[:10]:
            print(f"  {param}")
            # For debugging, print possible source parameter matches
            for src_param in source_params:
                if param.split(".")[-1] in src_param:
                    print(f"    Possible match: {src_param}")

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Copy weights from MaxText checkpoints to HF models"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/mnt/filestore/checkpoints_hf/gemma3_4b_exp1/checkpoints",
        help="Source checkpoints directory",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="google/gemma-3-4b-pt",
        help="Target model (HF model ID)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/filestore/converted_models",
        help="Base output directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint step to convert (e.g., 50000)",
    )
    parser.add_argument("--all", action="store_true", help="Convert all checkpoints")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="PyTorch dtype",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.checkpoint:
        parser.error("Either --checkpoint or --all must be specified")

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Function to process a single checkpoint
    def process_checkpoint(checkpoint_dir, checkpoint_name):
        output_path = os.path.join(args.output_dir, f"{checkpoint_name}")
        os.makedirs(output_path, exist_ok=True)

        print(f"=" * 80)
        print(f"Processing checkpoint: {checkpoint_name}")
        print(f"Source: {checkpoint_dir}")
        print(f"Output: {output_path}")
        print(f"=" * 80)

        copy_weights_and_save(
            source_model_path=checkpoint_dir,
            target_model_path=args.target_model,
            output_path=output_path,
            torch_dtype=args.dtype,
        )

        print(f"Completed processing checkpoint: {checkpoint_name}")
        print(f"=" * 80)
        print("")

    # Process specific checkpoint if requested
    if args.checkpoint:
        checkpoint_dir = os.path.join(args.source_dir, args.checkpoint)
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            return

        process_checkpoint(checkpoint_dir, args.checkpoint)

    # Process all checkpoints if requested
    if args.all:
        # Get all subdirectories in the source directory
        checkpoint_dirs = glob.glob(os.path.join(args.source_dir, "*"))
        checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]

        if not checkpoint_dirs:
            print(f"Error: No checkpoint directories found in {args.source_dir}")
            return

        print(f"Found {len(checkpoint_dirs)} checkpoints to process")

        for checkpoint_dir in sorted(checkpoint_dirs):
            checkpoint_name = os.path.basename(checkpoint_dir)
            process_checkpoint(checkpoint_dir, checkpoint_name)

        print(f"All {len(checkpoint_dirs)} checkpoints processed successfully")


if __name__ == "__main__":
    main()
