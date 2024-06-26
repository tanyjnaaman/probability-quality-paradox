import pytest
import psutil  # type: ignore
from typing_extensions import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

from src.experiments.language_model_experiments.utils.compute_nll import (
    compute_nll,
    compute_nll_with_decoding_algorithms,
)

_DEVICE = "cpu"
_MODEL = (
    "ethz-spylab/rlhf-7b-harmless"
    if psutil.virtual_memory().total > 32e9
    else "openai-community/gpt2"
)
print(f"Using model: {_MODEL}")


def test_compute_nll() -> None:
    # given
    model = AutoModelForCausalLM.from_pretrained(_MODEL, device_map=_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    texts = ["How are you today?", "How can I help you?"]

    # when
    nll = compute_nll(
        texts, model, tokenizer, add_start_token=True, max_length=256, batch_size=2
    )

    # then
    assert len(nll) == len(texts)
    assert all(isinstance(nll_, float) for nll_ in nll)
    assert all(nll_ >= 0 for nll_ in nll)
    print([(text, nll_) for text, nll_ in zip(texts, nll)])


@pytest.mark.parametrize(
    "sampling_type, temperature",
    [
        pytest.param(
            "top_p095",
            1.0,
            id="top_p095_t1.0",
        ),
        pytest.param(
            "top_p090",
            1.0,
            id="top_p090_t1.0",
        ),
        pytest.param(
            "top_k50",
            1.0,
            id="top_k50_t1.0",
        ),
        pytest.param(
            "top_k30",
            1.0,
            id="top_k30_t1.0",
        ),
        pytest.param(
            "ancestral_strict",
            1.0,
            id="ancestral_strict_t1.0",
        ),
        pytest.param(
            "ancestral",
            1.0,
            id="ancestral_t1.0",
        ),
        pytest.param(
            "typical_p090",
            1.0,
            id="typical_p090_t1.0",
        ),
        pytest.param(
            "eta_n00009",
            1.0,
            id="eta_n00009_t1.0",
        ),
        pytest.param(
            "top_p095",
            1.5,
            id="top_p095_t1.5",
        ),
        pytest.param(
            "top_p090",
            1.5,
            id="top_p090_t1.5",
        ),
        pytest.param(
            "top_k50",
            1.5,
            id="top_k50_t1.5",
        ),
        pytest.param(
            "top_k30",
            1.5,
            id="top_k30_t1.5",
        ),
        pytest.param(
            "ancestral_strict",
            1.5,
            id="ancestral_strict_t1.5",
        ),
        pytest.param(
            "ancestral",
            1.5,
            id="ancestral_t1.5",
        ),
        pytest.param(
            "typical_p090",
            1.5,
            id="typical_p090_t1.5",
        ),
        pytest.param(
            "eta_n00009",
            1.5,
            id="eta_n00009_t.15",
        ),
    ],
)
def test_compute_nll_with_decoding_algorithms(
    sampling_type: Literal[
        "top_p095",
        "top_p090",
        "top_k50",
        "top_k30",
        "ancestral_strict",
        "ancestral",
        "typical_p090",
        "eta_n00009",
    ],
    temperature: float,
) -> None:
    # given
    model = AutoModelForCausalLM.from_pretrained(_MODEL, device_map=_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    texts = ["How are you today?", "How can I help you?"]
    kwargs = dict()
    if sampling_type == "top_p095":
        kwargs["top_p"] = 0.95
    elif sampling_type == "top_p090":
        kwargs["top_p"] = 0.90
    elif sampling_type == "top_k30":
        kwargs["top_k"] = 30
    elif sampling_type in {"top_k50", "ancestral"}:
        kwargs["top_k"] = 50
    elif sampling_type == "ancestral_strict":
        kwargs["top_k"] = 50
        kwargs["top_p"] = 1.0
    elif sampling_type == "typical_p090":
        kwargs["typical_p"] = 0.90
    elif sampling_type == "eta_n00009":
        kwargs["eta_cutoff"] = 0.0009
    else:
        raise ValueError(f"Invalid sampling_type: {sampling_type}")

    # when
    nll = compute_nll_with_decoding_algorithms(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=True,
        max_length=256,
        batch_size=2,
        temperature=temperature,
        **kwargs,  # type: ignore
    )

    # then
    assert len(nll) == len(texts)
    assert all(isinstance(nll_, float) for nll_ in nll)
    for nll_ in nll:
        assert nll_ >= 0, f"nll_={nll_}"
    print([(text, nll_) for text, nll_ in zip(texts, nll)])


def test_compute_nll_with_decoding_parity() -> None:
    # given
    model = AutoModelForCausalLM.from_pretrained(_MODEL, device_map=_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    texts = ["How are you today?", "How can I help you?"]

    # when
    nlls = compute_nll(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=True,
        max_length=256,
        batch_size=2,
    )
    biased_nlls = compute_nll_with_decoding_algorithms(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=True,
        max_length=256,
        batch_size=2,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        eta_cutoff=0.0,
        typical_p=None,
    )

    # then
    assert len(nlls) == len(biased_nlls)
    for nll, biased_nll in zip(nlls, biased_nlls):
        assert nll == biased_nll, f"{nll} != {biased_nll}"


def test_compute_nll_explicit_no_prompt_parity() -> None:
    # given
    model = AutoModelForCausalLM.from_pretrained(_MODEL, device_map=_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    texts = ["How are you today?", "How can I help you?"]

    # when
    nlls = compute_nll(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=True,
        max_length=256,
        batch_size=2,
    )
    biased_nlls = compute_nll_with_decoding_algorithms(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=True,
        max_length=256,
        batch_size=2,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        eta_cutoff=0.0,
        condition_on_prompts=None,
        typical_p=None,
    )

    # then
    assert len(nlls) == len(biased_nlls)
    for nll, biased_nll in zip(nlls, biased_nlls):
        assert nll == biased_nll, f"{nll} != {biased_nll}"


@pytest.mark.parametrize(
    "sampling_type, temperature",
    [
        pytest.param(
            "top_p095",
            1.0,
            id="top_p095_t1.0",
        ),
        pytest.param(
            "top_p090",
            1.0,
            id="top_p090_t1.0",
        ),
        pytest.param(
            "top_k50",
            1.0,
            id="top_k50_t1.0",
        ),
        pytest.param(
            "top_k30",
            1.0,
            id="top_k30_t1.0",
        ),
        pytest.param(
            "ancestral_strict",
            1.0,
            id="ancestral_strict_t1.0",
        ),
        pytest.param(
            "ancestral",
            1.0,
            id="ancestral_t1.0",
        ),
        pytest.param(
            "top_p095",
            1.5,
            id="top_p095_t1.5",
        ),
        pytest.param(
            "top_p090",
            1.5,
            id="top_p090_t1.5",
        ),
        pytest.param(
            "top_k50",
            1.5,
            id="top_k50_t1.5",
        ),
        pytest.param(
            "top_k30",
            1.5,
            id="top_k30_t1.5",
        ),
        pytest.param(
            "ancestral_strict",
            1.5,
            id="ancestral_strict_t1.5",
        ),
        pytest.param(
            "ancestral",
            1.5,
            id="ancestral_t1.5",
        ),
    ],
)
def test_compute_nll_with_prompt_and_decoding_algorithms(
    sampling_type: Literal[
        "top_p095",
        "top_p090",
        "top_k50",
        "top_k30",
        "ancestral_strict",
        "ancestral",
    ],
    temperature: float,
) -> None:
    # given
    model = AutoModelForCausalLM.from_pretrained(_MODEL, device_map=_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL)
    prompts = [
        "Human: Ask me a question.\n\nAssistant:",
        "Human: Be friendly.\n\nAssistant:",
    ]
    texts = [
        "Human: Ask me a question.\n\nAssistant: How are you today?",
        "Human: Be friendly.\n\nAssistant: How can I help you?",
    ]
    kwargs = dict()
    if sampling_type == "top_p095":
        kwargs["top_p"] = 0.95
    elif sampling_type == "top_p090":
        kwargs["top_p"] = 0.90
    elif sampling_type == "top_k30":
        kwargs["top_k"] = 30
    elif sampling_type in {"top_k50", "ancestral"}:
        kwargs["top_k"] = 50
    elif sampling_type == "ancestral_strict":
        kwargs["top_k"] = 50
        kwargs["top_p"] = 1.0
    else:
        raise ValueError(f"Invalid sampling_type: {sampling_type}")

    # when
    nll = compute_nll_with_decoding_algorithms(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        add_start_token=True,
        max_length=256,
        batch_size=2,
        temperature=temperature,
        condition_on_prompts=prompts,
        **kwargs,  # type: ignore
    )

    # then
    assert len(nll) == len(texts)
    assert all(isinstance(nll_, float) for nll_ in nll)
    for nll_ in nll:
        assert nll_ >= 0, f"nll_={nll_}"
    print([(text, nll_) for text, nll_ in zip(texts, nll)])
