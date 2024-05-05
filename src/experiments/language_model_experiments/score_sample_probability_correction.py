from typing import List, Optional
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
    compute_nll_with_decoding_algorithms,
)


class ScriptArguments(BaseModel):
    language_model: str = Field(
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
            "Whether to wrap text with human assistant format, e.g. 'Human: prompt..."
            " \n\nAssistant: text'"
        ),
    )
    include_prompt: bool = Field(
        True,
        title="Include Prompt",
        description=(
            "Whether to include the prompt in the text. If false, only the generated"
            " text is used."
        ),
    )
    condition_on_prompt: bool = Field(
        False,
        title="Condition on Prompt",
        description=(
            "Whether to condition on the prompt. Cannot be true if include_prompt is"
            " false."
        ),
    )
    sampling_type: Literal[
        "top_p095",
        "top_p090",
        "top_k50",
        "top_k640",
        "ancestral_strict",
        "ancestral",
    ] = Field(
        title="Sampling Type",
        description="The sampling type to use for scoring negative log likelihoods",
    )
    sampling_temperature: float = Field(
        title="Sampling Temperature",
        description="The temperature to use for scoring negative log likelihoods",
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
            prompt, text, args.add_human_assistant_format, args.include_prompt
        )
        for prompt, text in zip(raw_prompts, raw_texts)
    ]
    print(f"Examples: {texts[:3]}")
    prompts = [
        transform_prompt(prompt, args.add_human_assistant_format)
        for prompt in raw_prompts
    ]

    # Load the language model
    model = AutoModelForCausalLM.from_pretrained(
        args.language_model, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    # Compute negative log likelihoods
    kwargs = dict(
        temperature=args.sampling_temperature,
    )
    if args.sampling_type == "top_p095":
        kwargs["top_p"] = 0.95
    elif args.sampling_type == "top_p090":
        kwargs["top_p"] = 0.90
    elif args.sampling_type == "top_k640":
        kwargs["top_k"] = 640
    elif args.sampling_type in {"top_k50", "ancestral"}:
        kwargs["top_k"] = 50
    elif args.sampling_type == "ancestral_strict":
        kwargs["top_k"] = 0
        kwargs["top_p"] = 1.0
    else:
        raise ValueError(f"Invalid sampling_type: {args.sampling_type}")
    biased_nlls = compute_nll_with_decoding_algorithms(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=args.add_start_token,
        max_length=args.max_length,
        batch_size=args.batch_size,
        condition_on_prompts=prompts if args.condition_on_prompt else None,
        **kwargs,
    )
    nlls = compute_nll_with_decoding_algorithms(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=args.add_start_token,
        max_length=args.max_length,
        batch_size=args.batch_size,
        condition_on_prompts=prompts if args.condition_on_prompt else None,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
    )

    # Save
    df["original_negative_log_probability"] = nlls
    df["samplingbiased_negative_log_probability"] = biased_nlls
    df["nll_correction_texts"] = texts
    if not args.save_path:
        print(
            "No save path provided. Saving to the input file with"
            " '_scorednllcorrection' appended."
        )
        # TODO: clean up naming
        args.save_path = args.csv_file_path.replace(
            ".csv",
            f"_scoredcorrectionnll{'_' + args.sampling_type if args.sampling_type is not None else ''}{'_' + str(args.sampling_temperature) if args.sampling_temperature is not None else ''}{'_humanassistant' if args.add_human_assistant_format else ''}{'_includeprompt' if args.include_prompt else ''}{'_conditioned' if args.condition_on_prompt else ''}.csv",
        )
    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
