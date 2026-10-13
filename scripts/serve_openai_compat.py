#!/usr/bin/env python3
"""Serve a local CausalLM and optional LoRA adapter over OpenAI chat API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.chat_template_utils import add_response_schema, parse_response


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    stop: str | list[str] | None = None
    stream: bool = False
    response_format: dict[str, Any] | None = None


def normalize_messages_for_template(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI tool history into the Qwen chat-template shape."""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            tool_calls = []
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw_arguments": arguments}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                tool_calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function.get("name"),
                            "arguments": arguments,
                        },
                    }
                )
            normalized.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                }
            )
        elif role == "tool":
            normalized.append(
                {
                    "role": "tool",
                    "name": message.get("name"),
                    "content": str(message.get("content") or ""),
                }
            )
        else:
            normalized.append(
                {
                    "role": role,
                    "content": message.get("content") or "",
                }
            )
    return normalized


def build_prompt(
    tokenizer,
    messages: list[dict[str, Any]],
    enable_thinking: bool,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    normalized_messages = normalize_messages_for_template(messages)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(
            normalized_messages,
            **kwargs,
        )
    except Exception:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(
            normalized_messages,
            **kwargs,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--served-model-name", default="medical-agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = add_response_schema(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            is_trainable=False,
            torch_dtype=torch.bfloat16,
        )
    model.to(args.device)
    model.eval()

    lock = asyncio.Lock()
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": args.served_model_name}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatRequest) -> dict[str, Any] | StreamingResponse:
        prompt = build_prompt(
            tokenizer,
            request.messages,
            args.enable_thinking,
            request.tools,
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        encoded = {
            key: value.to(args.device) for key, value in encoded.items()
        }
        input_length = int(encoded["input_ids"].shape[1])
        do_sample = request.temperature > 0
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": request.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = request.temperature
            generation_kwargs["top_p"] = request.top_p
        if request.stop:
            generation_kwargs["stop_strings"] = request.stop
            generation_kwargs["tokenizer"] = tokenizer
        elif request.tools:
            generation_kwargs["stop_strings"] = [
                "</tool_call>",
                "<|im_end|>",
            ]
            generation_kwargs["tokenizer"] = tokenizer

        async with lock:
            with torch.inference_mode():
                generated = model.generate(**generation_kwargs)
        completion_ids = generated[0, input_length:]
        parsed = parse_response(
            tokenizer,
            completion_ids.tolist(),
            prefix=encoded["input_ids"][0].tolist(),
        )
        content = str(parsed.get("content") or "").strip()
        tool_calls = []
        for index, tool_call in enumerate(parsed.get("tool_calls") or []):
            function = tool_call["function"]
            arguments = function.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "index": index,
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": arguments,
                    },
                }
            )

        completion_tokens = int(completion_ids.shape[0])
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        finish_reason = "tool_calls" if tool_calls else "stop"
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or None,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": input_length + completion_tokens,
            },
        }
        if not request.stream:
            return response

        def event_stream():
            base = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": args.served_model_name,
            }
            first = {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": content if not tool_calls else "",
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
            for tool_call in tool_calls:
                delta_call = {
                    "index": tool_call["index"],
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"],
                    },
                }
                chunk = {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [delta_call]},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            final = {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
