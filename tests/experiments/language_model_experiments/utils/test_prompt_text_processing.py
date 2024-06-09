import pytest
from src.experiments.language_model_experiments.utils.prompt_text_processing import (
    transform_prompt_and_text,
    transform_prompt,
)


@pytest.mark.parametrize(
    ["rm_name", "rlhf_name"],
    [
        pytest.param(
            "ethz-spylab/reward_model",
            "ethz-spylab/rlhf-7b-harmless",
            id="llama",
        ),
        pytest.param(
            "kaist-ai/janus-rm-7b",
            "kaist-ai/janus-dpo-7b",
            id="janus",
        ),
    ],
)
def test_transform_prompt_and_text_consistency(rm_name: str, rlhf_name: str):
    # ===== given =====
    prompt = "Some prompt?"
    text = "Answer to some prompt."

    # ===== when =====
    prompt_with_template = transform_prompt(
        prompt=prompt, add_human_assistant_format=True, model_type=rlhf_name
    )
    prompt_and_text_with_template = transform_prompt_and_text(
        prompt=prompt,
        text=text,
        add_human_assistant_format=True,
        include_prompt=True,
        model_type=rm_name,
    )

    # ===== then =====
    assert prompt_and_text_with_template.startswith(prompt_with_template)
