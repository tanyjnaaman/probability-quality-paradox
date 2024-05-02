from typing import List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from transformers.generation.utils import LogitsProcessorList  # type: ignore
from transformers.generation.logits_process import (  # type: ignore
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from torch.nn import CrossEntropyLoss

import torch


def compute_nll(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    add_start_token: bool,
    max_length: int,
    batch_size: int,
) -> List[float]:
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

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert tokenizer.bos_token is not None, (
            "Input model must already have a BOS token if using add_start_token=True."
            " Please use a different model, or set add_start_token=False"
        )
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    # Batch wise tokenization and scoring
    nlls = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        if i % 50 * batch_size == 0:
            print(f"Batch example: {batch[0]}")
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
        if add_start_token:
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

        if add_start_token:
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
    return nlls


def compute_nll_with_decoding_algorithms(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    add_start_token: bool,
    max_length: int,
    batch_size: int,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> List[float]:
    # instantiate warpers list
    warpers = LogitsProcessorList()
    min_tokens_to_keep = 1
    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        warpers.append(
            TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep)
        )
    if top_p is not None and top_p < 1.0:
        warpers.append(
            TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)
        )

    return _compute_nll_with_logitswarper(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=add_start_token,
        max_length=max_length,
        batch_size=batch_size,
        logits_warpers=warpers,
    )


def _compute_nll_with_logitswarper(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    add_start_token: bool,
    max_length: int,
    batch_size: int,
    logits_warpers: LogitsProcessorList,
) -> List[float]:
    """
    NOTE: lifted from https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
    NOTE: warpers logic lifted from https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/generation/utils.py#L711

    Note that we would have 0 log probability under some decoding method if topk/whatever method removes the correct next token.
    This is unlikely if you are scoring texts with the same model and decoding method used to generate the text.
    """

    # 1. Set up tokenizer
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

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert tokenizer.bos_token is not None, (
            "Input model must already have a BOS token if using add_start_token=True."
            " Please use a different model, or set add_start_token=False"
        )
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    # 3. Batch wise tokenization and scoring
    nlls = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        if i % 50 * batch_size == 0:
            print(f"Batch example: {batch[0]}")
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

        # check that each input is long enough
        if add_start_token:
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

        if add_start_token:
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
        shift_warped_logits = torch.cat(
            [shift_logits[:, 0, :].unsqueeze(dim=1)]
            + [
                logits_warpers(encoded_batch, shift_logits[:, i, :]).unsqueeze(dim=1)
                for i in range(
                    1, shift_logits.shape[1]
                )  # we don't warp the first token
            ],
            dim=1,
        )

        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        nll_batch = (
            loss_fct(shift_warped_logits.transpose(1, 2), shift_labels)
            * shift_attention_mask_batch
        ).sum(dim=1)

        nlls += nll_batch.tolist()
    return nlls
