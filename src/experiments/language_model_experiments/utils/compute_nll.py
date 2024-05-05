from typing import List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from transformers.generation.utils import LogitsProcessorList  # type: ignore
from transformers.generation.logits_process import (  # type: ignore
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from torch.nn import CrossEntropyLoss
from copy import deepcopy
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
    tokenizer = deepcopy(tokenizer)
    model = model.eval()

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
    condition_on_prompts: Optional[List[str]] = None,
    top_k: int = 0,
    top_p: float = 1.0,
    typical_p: Optional[float] = None,
    temperature: float = 1.0,
) -> List[float]:
    # instantiate warpers list
    assert top_k >= 0, f"top_k must be >= 0, got {top_k}"
    assert 0.0 < top_p <= 1.0, f"top_p must be in (0.0, 1.0], got {top_p}"
    assert temperature > 0.0, f"temperature must be > 0.0, got {temperature}"
    assert (
        typical_p is None or typical_p > 0.0
    ), f"typical_p must be > 0.0, got {typical_p}"
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
    if typical_p is not None:
        warpers.append(
            TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=min_tokens_to_keep)
        )

    return _compute_nll_with_logitswarper(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=add_start_token,
        max_length=max_length,
        batch_size=batch_size,
        condition_on_prompts=condition_on_prompts,
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
    add_special_tokens: bool = False,
    condition_on_prompts: Optional[List[str]] = None,
) -> List[float]:
    """
    NOTE: lifted from https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
    NOTE: warpers logic lifted from https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/generation/utils.py#L711

    Note that we would have 0 log probability under some decoding method if topk/whatever method removes the correct next token.
    This is unlikely if you are scoring texts with the same model and decoding method used to generate the text.
    """

    # 0. validation
    model = model.eval()
    if condition_on_prompts is not None:
        assert len(texts) == len(condition_on_prompts), (
            f"Number of texts ({len(texts)}) != number of prompts"
            f" ({len(condition_on_prompts)})."
        )
        assert all(
            text.startswith(prompt) for text, prompt in zip(texts, condition_on_prompts)
        ), "All texts must start with their respective prompts."

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

    # 1.1 tokenizer should be right padded
    tokenizer = deepcopy(tokenizer)

    # 2. tokenize prompts (if needed)
    if condition_on_prompts is not None:
        encoded_prompts = tokenizer(
            condition_on_prompts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
        )["input_ids"]
        prompt_lengths = [len(encoded_prompt) for encoded_prompt in encoded_prompts]
    else:
        prompt_lengths = [0] * len(texts)

    # 3. Batch wise tokenization and scoring
    nlls = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        batch_prompt_lengths = prompt_lengths[i : i + batch_size]
        if i % 50 * batch_size == 0:
            print(f"Batch example: {batch[0]}")
            print(f"Batch prompt length: {batch_prompt_lengths[0]}")
            print(
                "Batch text being examined:"
                f" {tokenizer.decode(tokenizer(batch[0])['input_ids'][batch_prompt_lengths[0]+1:])}"
            )

        inputs = tokenizer(
            batch,
            add_special_tokens=add_special_tokens,
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

        # 3.2 compute logits
        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        # 3.3 shift logits and apply warpers
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_warped_logits = torch.cat(
            [
                logits_warpers(encoded_batch, shift_logits[:, i, :]).unsqueeze(dim=1)
                for i in range(shift_logits.shape[1])  # we don't warp the first token
            ],
            dim=1,
        )
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        # 3.4 compute (conditional) nll
        nlls_not_summed = (
            CrossEntropyLoss(reduction="none")(
                shift_warped_logits.transpose(1, 2), shift_labels
            )
            * shift_attention_mask_batch
        )
        nlls_not_summed = torch.nan_to_num(
            nlls_not_summed, nan=0.0
        )  # could have nan (padding tokens) so we zero them out
        nlls_not_summed = torch.clamp(
            nlls_not_summed, min=0.0, max=25
        )  # clamp to avoid inf, ln(1e-11) ~= -25

        # 3.5 zero out the nlls for the prompt
        for i, prompt_length in enumerate(batch_prompt_lengths):
            start_idx = attn_mask.nonzero(as_tuple=True)[
                i
            ].min()  # if it's right padded or left padded, we start counting from the first '1'
            print(start_idx)
            nlls_not_summed[i, start_idx : start_idx + prompt_length] = 0.0
            assert not nlls_not_summed[i].isnan().any()
            assert not nlls_not_summed[i].isinf().any()
            assert not nlls_not_summed[i].sum().isinf()
        nll_batch = nlls_not_summed.sum(dim=1)

        nlls += nll_batch.tolist()
    return nlls
