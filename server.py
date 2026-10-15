import tinker
from tinker import SamplingClient, ServiceClient
from transformers import AutoTokenizer, PreTrainedTokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook import renderers
from dotenv import load_dotenv
from uuid import uuid4
from time import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
import uvicorn
from argparse import ArgumentParser
import traceback
import asyncio

app = FastAPI()


# Tokenizer namespaces that select the prompt-building path. `openai/*` uses
# the original tinker_cookbook renderer flow; `moonshotai/*` uses the model's
# HuggingFace chat template directly (the renderer flow has bugs for Kimi K2.5
# tool calls — it crashes on content=None, drops reasoning_content, and adds a
# default system prompt the chat template does not).
_OPENAI_PATH = "openai_renderer"
_MOONSHOT_PATH = "moonshot_chat_template"
_TOKENIZER_PREFIX_TO_PATH = {
    "openai/": _OPENAI_PATH,
    "moonshotai/": _MOONSHOT_PATH,
}


args = None
renderer: renderers.Renderer = None  # type: ignore
tokenizer: PreTrainedTokenizer = None  # type: ignore
service_client: ServiceClient = None  # type: ignore
prompt_path: str = None  # type: ignore
sampler_path_to_sampling_client: dict[str, SamplingClient] = {}
lock: asyncio.Lock = asyncio.Lock()
total_prompt_tokens: int = 0
total_completion_tokens: int = 0


def add_tools_to_messages(
    messages: list[renderers.Message], tools: list[renderers.ToolSpec]
) -> list[dict]:
    has_system_message: bool = messages[0]["role"] == "system"
    system_message_content: str
    if has_system_message:
        system_message_content = messages[0]["content"]  # type: ignore
        assert isinstance(system_message_content, str), (
            "System message content must be a string."
        )
        messages = messages[1:]
    else:
        system_message_content = ""

    system_messages_with_tools: list[renderers.Message] = (
        renderer.create_conversation_prefix_with_tools(
            tools=tools,  # type: ignore
            system_prompt=system_message_content,
        )
    )

    return system_messages_with_tools + messages  # type: ignore


def dict_tool_to_tinker_tool(tool) -> renderers.ToolSpec:
    assert isinstance(tool, dict), (
        f"Each tool should be a dict but got object of type {type(tool)}"
    )
    assert set(tool.keys()) == {"type", "function"}, (
        f'Each tool should be a dictionary with keys "type" and "function", but got one with keys {set(tool.keys())}'
    )
    assert tool["type"] == "function", (
        f'The type field of each tool should be the string "function", but got a tool where it is {tool["type"]}'
    )
    assert isinstance(tool["function"], dict), (
        f'The "function" field of each tool should be a dict, but got an object of type {type(tool["function"])}'
    )
    # Require the core keys but allow extras (e.g. OpenAI/Inspect send "strict").
    assert {"name", "parameters"} <= set(tool["function"].keys()), (
        f'The "function" field of each tool should include "name" and "parameters", but got one with keys {set(tool["function"].keys())}'
    )
    assert isinstance(tool["function"]["name"], str), "tool function name must be a str"
    assert isinstance(tool["function"]["parameters"], dict), "tool function parameters must be a dict"

    return renderers.ToolSpec(
        name=tool["function"]["name"],
        description=tool["function"].get("description", "") or "",
        parameters=tool["function"]["parameters"],
    )


def dict_message_to_tinker_message(message: dict) -> renderers.Message:
    for required_field in ["role", "content"]:
        assert required_field in message.keys(), (
            f'Message does not have a "{required_field}" field. Its keys are {set(message.keys())}'
        )

    if "tool_calls" in message.keys():
        message["tool_calls"] = [
            renderers.ToolCall(**tool_call) for tool_call in message["tool_calls"]
        ]

    return message  # type: ignore


def tinker_tool_call_to_dict_tool_call(tool_call: renderers.ToolCall) -> dict:
    dict_tool_call: dict = tool_call.model_dump(exclude_none=True)
    if "id" not in dict_tool_call.keys():
        dict_tool_call["id"] = "call_" + str(uuid4()).replace("-", "")
    return dict_tool_call


def message_content_to_string_drop_reasoning(content: "renderers.Content") -> str:
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part["type"] == "text")


def split_content_into_text_and_reasoning(
    content,
) -> tuple[str, str | None]:
    """Split parsed renderer content into (text, reasoning_content).

    The renderer's parse_response returns content as a list of TextPart and
    ThinkingPart entries. OpenAI's API exposes these as the `content` and
    `reasoning_content` fields respectively.
    """
    if isinstance(content, str):
        return content, None
    text = "".join(p["text"] for p in content if p["type"] == "text")
    reasoning = "".join(p["thinking"] for p in content if p["type"] == "thinking")
    return text, (reasoning if reasoning else None)


def build_prompt_via_renderer(data: dict) -> tinker.types.ModelInput:
    """Original openai/gpt-oss-* path: build the prompt via the renderer."""
    messages = data["messages"]
    messages = [dict_message_to_tinker_message(message) for message in messages]
    if "tools" in data.keys() and data["tools"] is not None:
        messages = add_tools_to_messages(
            messages=messages,
            tools=[dict_tool_to_tinker_tool(tool) for tool in data["tools"]],
        )
    return renderer.build_generation_prompt(messages)  # type: ignore


def build_prompt_via_chat_template(data: dict) -> tinker.types.ModelInput:
    """moonshotai/* path: delegate to the tokenizer's chat template.

    For Kimi K2.5 this is the only correct option — the tinker_cookbook
    renderer crashes on content=None, drops reasoning_content, and inserts a
    default system prompt the HF chat template does not.
    """
    chat_template_output = tokenizer.apply_chat_template(  # type: ignore
        data["messages"],
        tools=data.get("tools"),
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_tokens: list[int] = chat_template_output["input_ids"]  # type: ignore
    return tinker.types.ModelInput.from_ints(prompt_tokens)  # type: ignore


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def root(data: dict):
    try:
        async with lock:
            if data["model"] not in sampler_path_to_sampling_client.keys():
                _m = data["model"]
                # tinker:// -> a checkpoint/adapter; anything else -> a base model
                # by name (openai/gpt-oss-120b), which samples at native context.
                if _m.startswith("tinker://"):
                    _client = service_client.create_sampling_client(model_path=_m)
                else:
                    _client = service_client.create_sampling_client(base_model=_m)
                sampler_path_to_sampling_client[_m] = _client
            sampling_client = sampler_path_to_sampling_client[data["model"]]

        stream_requested = bool(data.get("stream"))

        known_fields = [
            "model",
            "messages",
            "tools",
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "tool_choice",
            "separate_reasoning",
            "stream",
            "stream_options",
            "reasoning_effort",
        ]
        extra_fields = [field for field in data.keys() if field not in known_fields]
        assert len(extra_fields) == 0, f"Unsupported parameters: {extra_fields}"

        import os as _os
        if _os.environ.get("SHIM_DEBUG"):
            _tok = lambda s: len(tokenizer.encode(s)) if s else 0
            _rows = []
            for _m in data.get("messages", []):
                _c = _m.get("content")
                _txt = _c if isinstance(_c, str) else json.dumps(_c)
                _tc = json.dumps(_m.get("tool_calls")) if _m.get("tool_calls") else ""
                _rows.append((_m.get("role"), _tok(_txt) + _tok(_tc)))
            _tools_tok = _tok(json.dumps(data.get("tools"))) if data.get("tools") else 0
            from collections import Counter as _Ctr
            _byrole = _Ctr()
            for _r, _n in _rows:
                _byrole[_r] += _n
            print(f"[dbg] msgs={len(_rows)} tools={len(data.get('tools') or [])}(~{_tools_tok}tok) "
                  f"byrole={dict(_byrole)} total_msg~{sum(n for _,n in _rows)}", flush=True)

        if prompt_path == _OPENAI_PATH:
            prompt = build_prompt_via_renderer(data)
        elif prompt_path == _MOONSHOT_PATH:
            prompt = build_prompt_via_chat_template(data)
        else:
            # Should be impossible — main() validates prompt_path at startup.
            raise RuntimeError(f"Unknown prompt_path: {prompt_path!r}")

        max_tokens: int | None = data.get("max_tokens")
        max_completion_tokens: int | None = data.get("max_completion_tokens")
        assert max_tokens is None or max_completion_tokens is None, (
            "max_tokens and max_completion_tokens cannot both be not None at the same time"
        )
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else max_completion_tokens,
            seed=data.get("seed"),
            stop=renderer.get_stop_sequences(),
            temperature=data.get("temperature", 1),
            top_k=data.get("top_k", -1),
            top_p=data.get("top_p", 1),
        )

        result = await sampling_client.sample_async(
            prompt=prompt, sampling_params=sampling_params, num_samples=1
        )

        response, parse_success = renderer.parse_response(result.sequences[0].tokens)
        # if not parse_success:
        #     raise HTTPException(status_code=500, detail="Failed parsing LLM response.")

        n_prompt_tokens = len(prompt.to_ints())  # type: ignore
        n_completion_tokens = len(result.sequences[0].tokens)

        global total_prompt_tokens, total_completion_tokens
        total_prompt_tokens += n_prompt_tokens
        total_completion_tokens += n_completion_tokens
        print(
            f"Token usage: {total_prompt_tokens // 1_000_000}M input {total_completion_tokens // 1_000_000}M output"
        )

        if prompt_path == _OPENAI_PATH:
            # Original behavior: flatten content to text only and drop reasoning.
            message_payload: dict = {
                "role": response["role"],
                "content": message_content_to_string_drop_reasoning(
                    response["content"]
                ),
                "tool_calls": [
                    tinker_tool_call_to_dict_tool_call(tool_call)
                    for tool_call in response["tool_calls"]  # type: ignore
                ]
                if "tool_calls" in response.keys()
                and response["tool_calls"] is not None  # type: ignore
                else None,
                "function_call": None,
            }
        else:
            # moonshotai/*: surface reasoning_content so multi-turn tool flows
            # can round-trip thinking back into the next prompt.
            text_content, reasoning_content = split_content_into_text_and_reasoning(
                response["content"]
            )
            message_payload = {
                "role": response["role"],
                "content": text_content,
                "tool_calls": [
                    tinker_tool_call_to_dict_tool_call(tool_call)
                    for tool_call in response["tool_calls"]  # type: ignore
                ]
                if "tool_calls" in response.keys()
                and response["tool_calls"] is not None  # type: ignore
                else None,
                "function_call": None,
            }
            if reasoning_content is not None:
                message_payload["reasoning_content"] = reasoning_content

        completion_id = f"chatcmpl-{uuid4()}"
        created = int(time())
        completion = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": data["model"],
            "system_fingerprint": "my_system_fingerprint",
            "service_tier": "default",
            "choices": [
                {
                    "index": 0,
                    "message": message_payload,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": n_prompt_tokens,
                "completion_tokens": n_completion_tokens,
                "total_tokens": n_prompt_tokens + n_completion_tokens,
            },
        }
        if not stream_requested:
            return completion

        # Minimal SSE streaming for clients (e.g. OpenCode) that require it:
        # the full message is computed above and emitted as a single chunk.
        def _event_stream():
            delta: dict = {
                "role": "assistant",
                "content": message_payload.get("content") or "",
            }
            if message_payload.get("reasoning_content") is not None:
                delta["reasoning_content"] = message_payload["reasoning_content"]
            if message_payload.get("tool_calls"):
                delta["tool_calls"] = [
                    {
                        "index": i,
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for i, tc in enumerate(message_payload["tool_calls"])
                ]
            base = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": data["model"],
            }
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Uncaught exception: {type(e)} {e}\n{traceback.format_exc()}",
        )


def resolve_prompt_path(tokenizer_name: str) -> str:
    """Pick the prompt-building path based on the tokenizer namespace.

    Loudly fails if the tokenizer is not from a supported family — we have
    only verified correctness for `openai/*` (original renderer flow) and
    `moonshotai/*` (HF chat template flow). Other namespaces would silently
    pick one of the two and might be wrong.
    """
    for prefix, path in _TOKENIZER_PREFIX_TO_PATH.items():
        if tokenizer_name.startswith(prefix):
            return path
    supported = ", ".join(sorted(_TOKENIZER_PREFIX_TO_PATH.keys()))
    raise SystemExit(
        f"openai_server.py: tokenizer {tokenizer_name!r} is not supported.\n"
        f"Only tokenizers under the following namespaces are supported: {supported}\n"
        "Add an explicit branch in _TOKENIZER_PREFIX_TO_PATH after verifying "
        "the right prompt-building path for the model family."
    )


def main() -> None:
    global args, renderer, tokenizer, service_client, prompt_path

    parser = ArgumentParser()
    parser.add_argument("--renderer", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    prompt_path = resolve_prompt_path(args.tokenizer)
    print(f"openai_server.py: tokenizer={args.tokenizer!r} → prompt_path={prompt_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    renderer = get_renderer(name=args.renderer, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    load_dotenv()

    main()


# checkpoints that do the same as the non-fine-tuned models:
# openai/gpt-oss-120b: tinker://d5d4218a-e803-5094-90ba-0044afeea523:train:0/sampler_weights/base
