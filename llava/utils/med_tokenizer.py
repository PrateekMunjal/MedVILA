from llava import conversation as conversation_lib
import torch
from llava.constants import IGNORE_INDEX, SENTINEL_TOKEN
from loguru import logger
from typing import Sequence, Dict, Optional
import transformers

def tokenize_conversation_legacy(
    messages: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    add_generation_prompt: bool = False,
    overrides: Optional[Dict[str, str]] = None,
    no_system_prompt: bool = False,
) -> torch.Tensor:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if no_system_prompt:
        conv.system = ""

    # Skip the first message if it is not from human
    if messages[0]["from"] != "human":
        messages = messages[1:]

    # Add a generation prompt if needed
    if add_generation_prompt:
        messages.append({"from": "gpt", "value": None})

    conv.messages = []
    for turn, message in enumerate(messages):
        role = roles[message["from"]]
        assert role == conv.roles[turn % 2]
        if overrides is not None and message["from"] in overrides:
            conv.append_message(role, overrides[message["from"]])
        else:
            conv.append_message(role, message["value"])

    # return tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt")
    return conv.get_prompt()

def tokenize_conversation(
    messages, tokenizer, include_generation_prompt=False, role_overrides=None, exclude_system_prompt=False
):

    for message in messages:
        message["value"] = message["value"].strip()

    formatted_prompt = None
    if conversation_lib.default_conversation.sep_style != conversation_lib.SeparatorStyle.AUTO:
        # print("\n Running tokenize conversation legacy...... \n")
        formatted_prompt = tokenize_conversation_legacy(
            messages,
            tokenizer,
            add_generation_prompt=include_generation_prompt,
            overrides=role_overrides,
            no_system_prompt=exclude_system_prompt,
        )
    else:
        normalized_conversation = []
        for msg in messages:
            normalized_msg = {}
            if msg["from"] == "human":
                normalized_msg["role"] = "user"
            elif msg["from"] == "gpt":
                normalized_msg["role"] = "assistant"
            else:
                raise ValueError(f"Unexpected sender '{msg['from']}' in conversation entry.")

            normalized_msg["content"] = msg["value"]

            if role_overrides and msg["from"] in role_overrides:
                normalized_msg["content"] = role_overrides[msg["from"]]

            normalized_conversation.append(normalized_msg)

        if exclude_system_prompt:
            normalized_conversation = [{"role": "system", "content": None}] + normalized_conversation

        formatted_prompt = tokenizer.apply_chat_template(
            normalized_conversation, add_generation_prompt=include_generation_prompt, tokenize=False
        )

    assert formatted_prompt is not None
    tokenized_inputids = tokenizer(formatted_prompt, return_tensors="pt").input_ids[0]
    return tokenized_inputids


def add_sentinel_token_if_missing(tokenizer):
    if hasattr(tokenizer, "sentinel_token"):
        tokenizer.add_tokens([SENTINEL_TOKEN], special_tokens=True)
        tokenizer.sentinel_token = SENTINEL_TOKEN
        tokenizer.sentinel_token_id = tokenizer.convert_tokens_to_ids(SENTINEL_TOKEN)


def med_preprocess_conversation(conversation, tokenizer, include_gen_prompt=False, exclude_system_prompt=False, retry_on_failure=False):

    input_tokens = tokenize_conversation(conversation, tokenizer, include_generation_prompt=include_gen_prompt, exclude_system_prompt=exclude_system_prompt)
    label_tokens = torch.ones_like(input_tokens) * IGNORE_INDEX

    if include_gen_prompt:
        # this should be used in inference mode
        return {"input_ids": input_tokens, "labels": label_tokens}

    # masked answer - and get tokens for tempate
    add_sentinel_token_if_missing(tokenizer)
    template_tokens = tokenize_conversation(
        conversation, tokenizer, include_generation_prompt=include_gen_prompt, role_overrides={"gpt": SENTINEL_TOKEN}, exclude_system_prompt=exclude_system_prompt
    )

    # Mask sentinel tokens in the template
    mask = torch.ones_like(template_tokens, dtype=torch.bool)

    for idx in range(len(template_tokens) - 1):
        if template_tokens[idx] == tokenizer.sentinel_token_id:
            mask[idx : idx + 2] = False

            if idx > 0 and retry_on_failure:
                mask[idx - 1] = False

    template_tokens = template_tokens[mask]

    # Match input tokens with the template
    # Unmatched tokens are used as labels for training
    i = 0  # index over template tokens
    j = 0  # index over input tokens

    while j < len(input_tokens):
        if i < len(template_tokens) and input_tokens[j] == template_tokens[i]:
            i += 1
        else:
            label_tokens[j] = input_tokens[j]
        j += 1

    # if the template does not match fully match, retry or mask all tokens
    if i < len(template_tokens):
        if not retry_on_failure:
            return med_preprocess_conversation(
                conversation, tokenizer, include_gen_prompt=include_gen_prompt, exclude_system_prompt=exclude_system_prompt, retry_on_failure=True
            )
        logger.error(f"Failed to preprocess conversation: {conversation}. Masking all tokens.")
        label_tokens[:] = IGNORE_INDEX

    return {"input_ids": input_tokens, "labels": label_tokens}
