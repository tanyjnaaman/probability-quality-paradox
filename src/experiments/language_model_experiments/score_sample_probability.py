import logging
from typing import List, Optional
from typing_extensions import Literal
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from pydantic import BaseModel, Field
from pydantic_argparse import ArgumentParser
from torch.nn import CrossEntropyLoss

import pandas as pd
import torch


class ScriptArguments(BaseModel):
    language_model: str = Field(
        "meta-llama/Llama-2-7b-hf",
        title="Reward Model",
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


def main():

    # Parse the arguments
    parser = ArgumentParser(model=ScriptArguments)
    args = parser.parse_typed_args()

    # Load csv
    df = pd.read_csv(args.csv_file_path)
    print(df.head())
    print(df.shape)
    raw_texts: List[str] = df["generated_text"].tolist()
    prompts: List[str] = df["prompt"].tolist()
    texts = [
        (
            f"Human: {prompt} Assistant:"
            f" {text[len(prompt):] if text.startswith(prompt) else text}"  # strip the prompt
            if args.add_human_assistant_format
            else f"{text[len(prompt):] if text.startswith(prompt) else text}"
        )
        for prompt, text in zip(prompts, raw_texts)
    ]
    print(f"Examples: {texts[:3]}")

    # Load the language model
    model = AutoModelForCausalLM.from_pretrained(
        args.language_model, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    # Compute negative log likelihoods
    # NOTE: lifted from https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert len(existing_special_tokens) > 0, (
            "If batch_size > 1, model must have at least one special token to use for"
            " padding. Please use a different model or set batch_size=1."
        )
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if args.add_start_token and args.max_length:
        # leave room for <BOS> token to be added:
        assert tokenizer.bos_token is not None, (
            "Input model must already have a BOS token if using add_start_token=True."
            " Please use a different model, or set add_start_token=False"
        )
        max_tokenized_len = args.max_length - 1
    else:
        max_tokenized_len = args.max_length

    # Batch wise tokenization and scoring
    nlls = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i : i + args.batch_size]
        inputs = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        )

        encoded_batch = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        # check that each input is long enough:
        if args.add_start_token:
            assert torch.all(
                torch.ge(attn_mask.sum(1), 1)
            ), "Each input text must be at least one token long."
        else:
            assert torch.all(torch.ge(attn_mask.sum(1), 2)), (
                "When add_start_token=False, each input text must be at least two"
                " tokens long. Run with add_start_token=True if inputting strings of"
                " only one token, and remove all empty input strings."
            )

        loss_fct = CrossEntropyLoss(reduction="none")

        if args.add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            )
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        nll_batch = (
            loss_fct(shift_logits.transpose(1, 2), shift_labels)
            * shift_attention_mask_batch
        ).sum(dim=1)

        nlls += nll_batch.tolist()

    # Save
    df["negative_log_probability"] = nlls
    if not args.save_path:
        print(
            "No save path provided. Saving to the input file with '_scorednll'"
            " appended."
        )
        args.save_path = args.csv_file_path.replace(".csv", "_scorednll.csv")
    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
