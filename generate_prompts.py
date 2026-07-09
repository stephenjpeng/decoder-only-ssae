import json

from dataset_generation.functions import PromptsGenerator

if __name__ == "__main__":
    pg = PromptsGenerator(
        "dataset_generation/prompts/input/categories_with_properties.json",
        output_folder="dataset_generation_sp/prompts/",
        n_prompts=1500,
    )
