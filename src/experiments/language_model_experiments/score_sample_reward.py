from typing import List, Optional
from typing_extensions import Literal
from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore
from pydantic import BaseModel, Field
from pydantic_argparse import ArgumentParser
from src.packages.safe_rlhf import AutoModelForScore
import pandas as pd
import torch


class ScriptArguments(BaseModel):
    reward_model: str = Field(
        "ethz-spylab/reward_model",
        title="Reward Model",
        description="The HF model to use to score strings",
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
    add_human_assistant_format: bool = Field(
        False,
        title="Human Assistant Format",
        description=(
            "Whether to wrap text with human assistant format, e.g. 'Human: prompt..."
            " \n\nAssistant: text'"
        ),
    )


def main():

    # Parse the arguments
    parser = ArgumentParser(model=ScriptArguments)
    args = parser.parse_typed_args()

    # Load csv
    df = pd.read_csv(args.csv_file_path)
    print(df.head())
    print(df.shape)

    # Load the reward model
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    reward_model = AutoModelForScore.from_pretrained(
        args.reward_model, device_map=args.device
    )
    reward_model.eval()

    # Score the data
    raw_texts: List[str] = df["generated_text"].tolist()
    prompts: List[str] = df["prompt"].tolist()
    texts = (
        [
            f"Human: {prompt} Assistant:"
            f" {text[len(prompt):] if text.startswith(prompt) else text}"
            for prompt, text in zip(prompts, raw_texts)
        ]
        if args.add_human_assistant_format
        else raw_texts
    )
    print(f"Examples: {texts[:3]}")

    scores = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i : i + args.batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_length,
        )
        with torch.no_grad():
            outputs = reward_model(**inputs)
            batch_scores = outputs.end_scores.squeeze(dim=-1).cpu().tolist()
            scores.extend(batch_scores)

    # Save the scores
    df["score"] = scores
    if not args.save_path:
        print(
            "No save path provided. Saving to the input file with '_scoredreward'"
            " appended."
        )
        args.save_path = args.csv_file_path.replace(
            ".csv",
            f"_scoredreward{'_humanassistant' if args.add_human_assistant_format else ''}.csv",
        )
    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
