def transform_prompt_and_text(
    prompt: str,
    text: str,
    add_human_assistant_format: bool,
    include_prompt: bool,
    model_type: str,
) -> str:
    assert (not include_prompt and not add_human_assistant_format) or include_prompt
    # strip artefacts from hh-rlhf dataset, hardcoded for simplicity
    stripped_prompt = (
        prompt[len("Human: ") :][: -len("Assistant:")]
        if ("Human:" in prompt and "Assistant" in prompt)
        else prompt
    )
    stripped_text = text[len(prompt) :].strip() if text.startswith(prompt) else text
    if not include_prompt:
        return stripped_text
    if add_human_assistant_format:
        if model_type in {"ethz-spylab/rlhf-7b-harmless"}:
            return f"Human: {stripped_prompt} Assistant: {stripped_text}"
        elif model_type in {"kaist-ai/janus-dpo-7b"}:
            return f"[INST] {prompt} [/INST] {stripped_text}"
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    else:
        return f"{stripped_prompt}{stripped_text}"


def transform_prompt(
    prompt: str, add_human_assistant_format: bool, model_type: str
) -> str:
    # strip artefacts from hh-rlhf dataset, hardcoded for simplicity
    stripped_prompt = (
        prompt[len("Human: ") :][: -len("Assistant:")]
        if ("Human:" in prompt and "Assistant" in prompt)
        else prompt
    )
    if add_human_assistant_format:
        if model_type in {"ethz-spylab/rlhf-7b-harmless"}:
            return f"Human: {stripped_prompt} Assistant:"
        elif model_type in {"kaist-ai/janus-dpo-7b"}:
            return f"[INST] {prompt} [/INST]"
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    else:
        return stripped_prompt
