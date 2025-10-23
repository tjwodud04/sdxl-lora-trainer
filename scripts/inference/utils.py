"""
Utility functions for inference scripts
"""

from pathlib import Path


def parse_model_path(merged_model_path):
    """
    Parse merged model path to extract dataset_name, model_short_name, and trial_name

    Expected path format:
    output/{dataset_name}/{model_short_name}/{trial_name}/merged/checkpoint-{step}

    Args:
        merged_model_path: Path to merged model checkpoint

    Returns:
        tuple: (dataset_name, model_short_name, trial_name)

    Example:
        >>> parse_model_path("output/jingliu_sdxl_20_ep_dataset/animagine/trial1/merged/checkpoint-final")
        ('jingliu_sdxl_20_ep_dataset', 'animagine', 'trial1')
    """
    path = Path(merged_model_path)
    parts = path.parts

    # Find 'output' in path
    try:
        output_idx = parts.index('output')
        # output/{dataset_name}/{model_short_name}/{trial_name}/merged/checkpoint-{step}
        #   0         1                2                 3         4          5
        dataset_name = parts[output_idx + 1]
        model_short_name = parts[output_idx + 2]
        trial_name = parts[output_idx + 3]

        return dataset_name, model_short_name, trial_name
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Cannot parse model path: {merged_model_path}\n"
            f"Expected format: output/{{dataset}}/{{model}}/{{trial}}/merged/checkpoint-{{step}}\n"
            f"Error: {e}"
        )


if __name__ == "__main__":
    # Test
    test_paths = [
        "output/jingliu_sdxl_20_ep_dataset/animagine/trial1/merged/checkpoint-final",
        "output/jingliu_sdxl_20_ep_dataset/sdxl-base/trial2/merged/checkpoint-500",
        "output/custom_dataset/hidream/trial1/merged/checkpoint-final",
    ]

    print("Testing parse_model_path:")
    for path in test_paths:
        dataset, model, trial = parse_model_path(path)
        print(f"  {path}")
        print(f"    → dataset: {dataset}, model: {model}, trial: {trial}")
