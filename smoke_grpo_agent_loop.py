#!/usr/bin/env python3
"""Replay old grpo-200 tool calls with real sandbox observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medagent_rl.data import (  # noqa: E402
    CaseRegistry,
    build_training_prompt,
    load_cases,
)
from medagent_rl.grpo_sandbox import MedicalToolSandbox  # noqa: E402
from medagent_rl.tools import TOOL_SCHEMAS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8002/v1/chat/completions",
    )
    parser.add_argument("--model", default="grpo-200")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "cases_synthea_grpo.jsonl",
    )
    parser.add_argument(
        "--families",
        default="dental,uri,chest_pain",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "grpo-agent-smoke" / "trajectories.jsonl",
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def allowed_arguments(
    action: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    allowed = {
        "ask": ("question",),
        "lookup": ("query",),
        "check": ("item",),
        "search": ("query",),
        "answer": ("content",),
    }
    return {
        key: arguments[key]
        for key in allowed.get(action, ())
        if key in arguments
    }


def run_case(
    *,
    case,
    registry: CaseRegistry,
    base_url: str,
    model: str,
    max_turns: int,
    timeout: int,
) -> dict[str, Any]:
    sandbox = MedicalToolSandbox(registry, max_turns=max_turns)
    sandbox.reset(case.case_id)
    messages = build_training_prompt(case)
    turns = []
    terminal_reason = "max_turns"

    for turn_index in range(max_turns):
        response = requests.post(
            base_url,
            json={
                "model": model,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                "temperature": 0.0,
                "max_tokens": 256,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            terminal_reason = "text"
            turns.append(
                {
                    "turn": turn_index,
                    "action": "text",
                    "arguments": {},
                    "observation": message.get("content") or "",
                    "tool_error": None,
                }
            )
            break

        call = calls[0]
        function = call["function"]
        action = function["name"]
        call_id = str(call.get("id") or f"clinic-call-{turn_index}")
        arguments = parse_arguments(function.get("arguments"))
        safe_arguments = allowed_arguments(action, arguments)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": action,
                            "arguments": safe_arguments,
                        },
                    }
                ],
            }
        )

        tool_error = None
        if action == "answer":
            try:
                observation = sandbox.answer(
                    safe_arguments.get("content", "")
                )
            except Exception as exc:
                tool_error = str(exc)
                observation = f"ERROR: {exc}"
            terminal_reason = "answer"
        else:
            try:
                observation = getattr(sandbox, action)(**safe_arguments)
            except Exception as exc:
                tool_error = str(exc)
                observation = f"ERROR: {exc}"

        # ``answer`` is the terminal action and must not receive a synthetic
        # tool response. Other actions need the exact id for the next model
        # call, matching the OpenAI tool protocol and the GRPO training data.
        if action != "answer":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": action,
                    "content": str(observation),
                }
            )
        turns.append(
            {
                "turn": turn_index,
                "action": action,
                "raw_arguments": arguments,
                "arguments": safe_arguments,
                "observation": observation,
                "tool_error": tool_error,
            }
        )
        if action == "answer":
            break

    return {
        "case_id": case.case_id,
        "family": case.family,
        "chief_complaint": case.patient.chief_complaint,
        "terminal_reason": terminal_reason,
        "turn_count": len(turns),
        "turns": turns,
        "state": sandbox.result()["state"],
        "reward": sandbox.result()["reward"],
    }


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    registry = CaseRegistry(cases)
    by_family: dict[str, Any] = {}
    for case in cases:
        by_family.setdefault(case.family, case)
    families = [
        item.strip()
        for item in args.families.split(",")
        if item.strip()
    ]
    missing = [family for family in families if family not in by_family]
    if missing:
        raise ValueError(f"Families not found: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for family in families:
            trajectory = run_case(
                case=by_family[family],
                registry=registry,
                base_url=args.base_url,
                model=args.model,
                max_turns=args.max_turns,
                timeout=args.timeout,
            )
            handle.write(
                json.dumps(trajectory, ensure_ascii=False) + "\n"
            )
            print(
                family,
                trajectory["terminal_reason"],
                [
                    turn["action"]
                    for turn in trajectory["turns"]
                ],
                flush=True,
            )
    print(args.output)


if __name__ == "__main__":
    main()
