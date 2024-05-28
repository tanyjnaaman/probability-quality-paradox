def transform_prompt_and_text(
    prompt: str, text: str, add_human_assistant_format: bool, include_prompt: bool
) -> str:
    assert (not include_prompt and not add_human_assistant_format) or include_prompt
    stripped_prompt = (
        prompt[len("Human: ") :][: -len("Assistant:")]
        if ("Human:" in prompt and "Assistant" in prompt)
        else prompt
    )
    stripped_text = text[len(prompt) :].strip() if text.startswith(prompt) else text
    if not include_prompt:
        return stripped_text
    return (
        f"Human: {stripped_prompt} Assistant: {stripped_text}"
        if add_human_assistant_format
        else f"{stripped_prompt}{stripped_text}"
    )


def transform_prompt(prompt: str, add_human_assistant_format: bool) -> str:
    stripped_prompt = (
        prompt[len("Human: ") :][: -len("Assistant:")]
        if ("Human:" in prompt and "Assistant" in prompt)
        else prompt
    )
    return (
        f"Human: {stripped_prompt} Assistant:"
        if add_human_assistant_format
        else stripped_prompt
    )
