def transform_prompt_and_text(
    prompt: str,
    text: str,
    add_human_assistant_format: bool,
    include_prompt: bool,
    model_type: str,
) -> str:
    prompt = prompt.strip()
    text = text.strip()
    assert (not include_prompt and not add_human_assistant_format) or include_prompt
    if model_type in {
        "ethz-spylab/reward_model",
        "ethz-spylab/rlhf-7b-harmless",
        "meta-llama/Llama-2-7b-hf",
    }:
        stripped_prompt = (
            prompt[len("Human: ") :][: -len("Assistant:")]
            if ("Human:" in prompt and "Assistant" in prompt)
            else prompt
        )
        stripped_text = text[len(prompt) :] if text.startswith(prompt) else text
        return (
            stripped_text
            if not include_prompt
            else (
                f"Human: {stripped_prompt} Assistant: {stripped_text}"
                if add_human_assistant_format
                else f"{stripped_prompt}{stripped_text}"
            )
        )

    elif model_type in {
        "kaist-ai/janus-dpo-7b",
        "mistral-community/Mistral-7B-v0.2",
        "kaist-ai/janus-rm-7b",
    }:
        stripped_prompt = (
            prompt[len("[INST] Human: ") :][: -len(" Assistant:[/INST]")]
            if prompt.startswith("[INST] Human: ")
            else prompt
        )
        stripped_text = text[len(prompt) :] if text.startswith(prompt) else text
        return (
            stripped_text
            if not include_prompt
            else (
                f"[INST] Human: {stripped_prompt} Assistant:[/INST]{stripped_text}"
                if add_human_assistant_format
                else f"{stripped_prompt}{stripped_text}"
            )
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def transform_prompt(
    prompt: str, add_human_assistant_format: bool, model_type: str
) -> str:
    prompt = prompt.strip()

    if model_type in {
        "ethz-spylab/reward_model",
        "ethz-spylab/rlhf-7b-harmless",
        "meta-llama/Llama-2-7b-hf",
    }:
        # strip artefacts from hh-rlhf dataset, hardcoded for simplicity
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

    elif model_type in {
        "kaist-ai/janus-dpo-7b",
        "mistral-community/Mistral-7B-v0.2",
        "kaist-ai/janus-rm-7b",
    }:
        # strip artefacts from instr. and hh-rlhf dataset, hardcoded for simplicity
        stripped_prompt = (
            prompt[len("[INST]") :][: -len("[/INST]")]
            if ("[INST]" in prompt and "[/INST]" in prompt)
            else prompt
        ).strip()
        stripped_prompt = (
            stripped_prompt[len("Human: ") :][: -len("Assistant:")]
            if ("Human:" in stripped_prompt and "Assistant" in stripped_prompt)
            else stripped_prompt
        ).strip()
        return (
            f"[INST] Human: {stripped_prompt} Assistant:[/INST]"
            if add_human_assistant_format
            else prompt
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
