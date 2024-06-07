from typing import List, Optional
import torch
from typing_extensions import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from pydantic import BaseModel, Field
from pydantic_argparse import ArgumentParser

import pandas as pd

from src.experiments.language_model_experiments.utils.prompt_text_processing import (
    transform_prompt,
    transform_prompt_and_text,
)
from src.experiments.language_model_experiments.utils.compute_nll import (
    compute_nll,
    compute_nll_with_decoding_algorithms,
)


class ScriptArguments(BaseModel):
    language_model: str = Field(
        "meta-llama/Llama-2-7b-hf",
        title="Model",
        description="The HF model to use to score string negative log likelihoods",
    )
    max_length: int = Field(
        256,
        title="Max Length",
        description="The maximum length of the generated text",
    )
    csv_file_path: str = Field(
        title="CSV File Path",
        description="The path to the CSV file with the data",
    )
    batch_size: int = Field(
        1,
        title="Batch Size",
        description="The batch size for generation",
    )
    save_path: Optional[str] = Field(
        None,
        title=(
            "path to .csv with generated text. Should have a column called"
            " 'generated_text'."
        ),
        description="The path to save the generated outputs",
    )
    device: Literal["auto", "cuda:0", "cuda:1"] = Field(
        "auto",
        title="Device",
        description="The device to use for generation",
    )
    add_start_token: bool = Field(
        True,
        title="Add Start Token",
        description="Whether to add a start token to the input",
    )
    add_human_assistant_format: bool = Field(
        False,
        title="Human Assistant Format",
        description=(
            "Whether to wrap text with model-specific prompt template, e.g. 'Human:"
            " prompt... \n\nAssistant: text'"
        ),
    )
    include_prompt: bool = Field(
        True,
        title="Include Prompt",
        description="Whether to include the prompt in the text",
    )
    condition_on_prompt: bool = Field(
        False,
        title="Condition on Prompts",
        description="Whether to condition on prompts",
    )


def main():

    # Parse the arguments
    parser = ArgumentParser(model=ScriptArguments)
    args = parser.parse_typed_args()

    # Load csv
    df = pd.read_csv(args.csv_file_path)
    print(df.head())
    print(df.shape)
    raw_texts: List[str] = df["generated_text"].tolist()
    raw_prompts: List[str] = df["prompt"].tolist()
    texts = [
        transform_prompt_and_text(
            prompt,
            text,
            args.add_human_assistant_format,
            args.include_prompt,
            args.language_model,
        )
        for prompt, text in zip(raw_prompts, raw_texts)
    ]
    prompts = [
        transform_prompt(prompt, args.add_human_assistant_format, args.language_model)
        for prompt in raw_prompts
    ]
    print(f"Examples: {texts[:3]}")

    # Load the language model
    assert torch.cuda.is_available(), "CUDA must be available" or args.device == "cpu"
    assert (
        torch.cuda.device_count() > 1 or args.device == "cuda:0" or args.device == "cpu"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.language_model, torch_dtype=torch.float16, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    # Compute negative log likelihoods
    nlls = (
        compute_nll(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            add_start_token=args.add_start_token,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        if not args.condition_on_prompt
        else compute_nll_with_decoding_algorithms(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            add_start_token=args.add_start_token,
            max_length=args.max_length,
            batch_size=args.batch_size,
            condition_on_prompts=prompts,
            top_k=0,
            top_p=1.0,
            typical_p=None,
            eta_cutoff=0.0,
            temperature=1.0,
        )
    )

    # Save
    df["negative_log_probability"] = nlls
    df["negative_log_probability_text"] = texts
    if not args.save_path:
        print(
            "No save path provided. Saving to the input file with '_scorednll'"
            " appended."
        )
        # TODO: clean up naming
        args.save_path = args.csv_file_path.replace(
            ".csv",
            f"_scorednll{'_humanassistant' if args.add_human_assistant_format else ''}{'_includeprompt' if args.include_prompt else ''}{'_conditioned' if args.condition_on_prompt else ''}.csv",
        )
    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
