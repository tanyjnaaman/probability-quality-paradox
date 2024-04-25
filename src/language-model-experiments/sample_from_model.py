from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore
from pydantic import BaseModel, Field
from pydantic_argparse import ArgumentParser
from datasets import load_dataset  # type: ignore

import transformers
import pandas as pd
import torch


class ScriptArguments(BaseModel):
    model: str = Field(
        "meta-llama/Llama-2-7b-chat-hf",
        title="Model",
        description="The HF model to use for text generation",
    )
    max_length: int = Field(
        256,
        title="Max Length",
        description="The maximum length of the generated text",
    )
    generation_seed: int = Field(
        42,
        title="Generation Seed",
        description="The seed for generating strings",
    )
    prompt_selection_seed: int = Field(
        42,
        title="Prompt Selection Seed",
        description="The seed for selecting prompts",
    )
    num_prompts: int = Field(
        500,
        title="Number of Prompts",
        description="The number of prompts to use for generation",
    )
    num_generations_per_prompt: int = Field(
        100,
        title="Number of Generations per Prompt",
        description="The number of generations to generate per prompt",
    )
    batch_size: int = Field(
        1,
        title="Batch Size",
        description="The batch size for generation",
    )
    save_path: str = Field(
        title=".csv Save Path",
        description="The path to save the generated outputs",
    )


def main():

    # 1. Parse arguments
    parser = ArgumentParser(model=ScriptArguments)
    args = parser.parse_typed_args()

    # 2. Construct prompts
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")["test"]
    sampled_dataset = dataset.shuffle(seed=args.prompt_selection_seed).select(
        range(args.num_prompts)
    )
    prompts = sampled_dataset.map(
        lambda pair: {
            "prompt": pair["chosen"].split("\n\n")[1].lstrip("Human: ").strip()
        }
    )

    # 3. Set seed for generation
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.use_deterministic_algorithms(True)

    # 4. load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    outputs = dict(prompt=[], generated_text=[])
    for row in tqdm(prompts):
        sequences = pipeline(
            row["prompt"],
            do_sample=True,
            num_return_sequences=args.num_generations_per_prompt,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            batch_size=args.batch_size,
        )["generated_text"]
        outputs["prompt"].extend([row["prompt"]] * len(sequences))
        outputs["generated_text"].extend(sequences)

    # 5. Save outputs
    assert args.save_path is not None, "Save path must be provided"
    assert args.save_path.endswith(".csv"), "Save path must be a .csv file"
    outputs_as_df = pd.DataFrame(outputs)
    outputs_as_df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
