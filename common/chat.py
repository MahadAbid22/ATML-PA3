"""Chat-template helpers. Thinking is always disabled (enable_thinking=False)."""
from __future__ import annotations


def render(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Render messages with the model's native chat template (thinking disabled).

    For Qwen3 the generation prompt then ends with an empty think block
    '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'. Templates without the
    enable_thinking switch (e.g. Qwen2.5) ignore the extra keyword.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def end_of_turn_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert tid is not None and tid != tokenizer.unk_token_id, "no <|im_end|> token"
    return tid
