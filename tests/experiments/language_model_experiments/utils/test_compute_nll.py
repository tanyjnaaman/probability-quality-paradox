from typing_extensions import Literal
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

from src.experiments.language_model_experiments.utils.compute_nll import (
    compute_nll,
    compute_nll_with_decoding_algorithms,
)

_DEVICE = "cpu"
_MODEL = "ethz-spylab/rlhf-7b-harmless"


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
            "top_k640",
            1.0,
            id="top_k640_t1.0",
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
            "top_k640",
            1.5,
            id="top_k640_t1.5",
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
def test_compute_nll_with_decoding_algorithms(
    sampling_type: Literal[
        "top_p095",
        "top_p090",
        "top_k50",
        "top_k640",
        "ancestral_strict",
        "ancestral",
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
    elif sampling_type == "top_k640":
        kwargs["top_k"] = 640
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
    )

    # then
    assert len(nlls) == len(biased_nlls)
    for nll, biased_nll in zip(nlls, biased_nlls):
        assert nll == biased_nll, f"{nll} != {biased_nll}"
