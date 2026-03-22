import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message

from marble.llms.error_handler import api_calling_error_exponential_backoff


def _ensure_non_empty_user_message(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Some chat templates (e.g. vLLM Qwen tool-calling parsers) require at least one
    non-empty user message. Add a minimal user turn when missing.
    """
    has_non_empty_user = any(
        message.get("role") == "user" and bool((message.get("content") or "").strip())
        for message in messages
    )
    if has_non_empty_user:
        return messages

    normalized_messages = list(messages)
    normalized_messages.append(
        {
            "role": "user",
            "content": "Please provide your response based on the instructions above.",
        }
    )
    return normalized_messages


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    """
    # litellm.set_verbose=True
    if "together_ai/TA" in llm_model:
        base_url = "https://api.ohmygpt.com/v1"
    else:
        base_url = None
    normalized_messages = _ensure_non_empty_user_message(messages)
    try:
        completion = litellm.completion(
            model=llm_model,
            messages=normalized_messages,
            max_tokens=max_token_num,
            n=return_num,
            top_p=top_p,
            temperature=temperature,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            base_url=base_url,
        )
    except Exception as exc:
        # Some local Ollama models (e.g. phi3:3.8b) reject tool-calling requests.
        # Fall back to plain chat completion when tools are unsupported.
        exc_text = str(exc).lower()
        tool_calling_unsupported = (
            "does not support tools" in exc_text
            or "tool choice requires" in exc_text
            or "tool-choice" in exc_text
            or "tool-call-parser" in exc_text
            or "enable-auto-tool-choice" in exc_text
        )
        if tools and tool_calling_unsupported:
            print(
                "[WARN] model_prompting: tool-calling request rejected by backend; "
                "retrying without tools."
            )
            completion = litellm.completion(
                model=llm_model,
                messages=normalized_messages,
                max_tokens=max_token_num,
                n=return_num,
                top_p=top_p,
                temperature=temperature,
                stream=stream,
                base_url=base_url,
            )
        else:
            raise
    message_0: Message = completion.choices[0].message
    assert message_0 is not None
    assert isinstance(message_0, Message)
    return [message_0]
