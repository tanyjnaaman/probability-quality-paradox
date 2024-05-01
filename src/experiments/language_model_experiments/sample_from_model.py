from typing import Dict, List, Optional
from typing_extensions import Literal
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
        "meta-llama/Llama-2-7b-hf",
        title="Model",
        description="The HF model to use for text generation",
    )
    max_length: int = Field(
        256,
        title="Max Length",
        description="The maximum length of the generated text",
    )
    prompt_selection_seed: int = Field(
        42,
        title="Prompt Selection Seed",
        description="The seed for selecting prompts",
    )
    num_prompts: int = Field(
        5000,
        title="Number of Prompts",
        description="The number of prompts to use for generation",
    )
    num_generations_per_prompt: int = Field(
        10,
        title="Number of Generations per Prompt",
        description="The number of generations to generate per prompt",
    )
    batch_size: int = Field(
        1,
        title="Batch Size",
        description="The batch size for generation",
    )
    save_path: Optional[str] = Field(
        None,
        title=".csv Save Path",
        description="The path to save the generated outputs",
    )
    device: Literal["auto", "cuda:0", "cuda:1"] = Field(
        "auto",
        title="Device",
        description="The device to use for generation",
    )
    sampling_type: Literal["top_p095", "ancestral", "top_k640"] = Field(
        "top_p095",
        title="Sampling Type",
        description="The sampling type to use for generation",
    )
    sampling_temperature: float = Field(
        1.0,
        title="Sampling Temperature",
        description="The temperature to use for sampling",
    )
    human_assistant_format: bool = Field(
        False,
        title="Human Assistant Format",
        description=(
            "Whether the prompt is in human assistant format, e.g. 'Human: ..."
            " \n\nAssistant:'"
        ),
    )


def main():

    # Parse arguments
    parser = ArgumentParser(model=ScriptArguments)
    args = parser.parse_typed_args()

    # Construct prompts
    dataset = load_dataset("Anthropic/hh-rlhf")["test"]
    prompts = (
        dataset.map(
            lambda pair: {
                "prompt": (
                    pair["chosen"].split("\n\n")[1].lstrip("Human:").strip()
                    if not args.human_assistant_format
                    else pair["chosen"].split("\n\n")[1].strip() + "\n\nAssistant:"
                )
            }
        )
        .filter(lambda row: len(row["prompt"]) < args.max_length)
        .shuffle(seed=args.prompt_selection_seed)
        .select(range(args.num_prompts))
    )
    assert (
        len(prompts) == args.num_prompts
    ), f"Expected {args.num_prompts} prompts, got {len(prompts)}"
    print(f"Example prompts: {prompts['prompt'][:3]}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
    )

    # Generate
    outputs: Dict[str, List[str]] = dict(prompt=[], generated_text=[])
    for row in tqdm(prompts):
        batched_prompts = [row["prompt"]] * args.num_generations_per_prompt
        sequences = pipeline(
            batched_prompts,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            truncation=True,
            top_p=0.95 if args.sampling_type == "top_p095" else 1.0,
            top_k=640 if args.sampling_type == "top_k640" else 0,
            temperature=args.sampling_temperature,
            batch_size=args.batch_size,
        )
        assert len(sequences) == args.num_generations_per_prompt
        generated_texts = [
            sequence[0]["generated_text"][len(row["prompt"].strip()) :].strip()
            for sequence in sequences
        ]
        outputs["prompt"].extend(batched_prompts)
        outputs["generated_text"].extend(generated_texts)
        print(f"Prompt completed: {row['prompt']}\nGenerated: {generated_texts[0]}")

    # Save outputs
    save_path = (
        args.save_path
        or f"{args.model.replace('/', '-')}_l{args.max_length}_promptseed{args.prompt_selection_seed}_numprompt{args.num_prompts}_numgenerations{args.num_generations_per_prompt}_{args.sampling_type}_t{args.sampling_temperature}{'_humanassistant' if args.human_assistant_format else ''}.csv"
    )
    assert save_path.endswith(".csv"), "Save path must be a .csv file"
    outputs_as_df = pd.DataFrame.from_dict(outputs)
    outputs_as_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
