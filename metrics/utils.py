"""Shared utilities for metric computation."""

import asyncio
import base64
import io
import json
import os
import random
import re
import sys
import time

from PIL import Image

_OPENAI_CLIENTS = {}


def _resolve_api_config(api_config: dict) -> dict:
    """Resolve explicit environment-variable references without modifying config."""
    resolved = dict(api_config)
    for field in ("model", "api_base", "api_key"):
        env_name = resolved.get(f"{field}_env")
        if env_name:
            value = os.environ.get(env_name)
            if not value:
                raise ValueError(f"Required environment variable is not set: {env_name}")
            resolved[field] = value
    if resolved.get("api_key_env"):
        resolved.pop("api_keys", None)
    return resolved


def _pick_api_key(api_config: dict) -> str:
    """Pick one API key. Supports single api_key or api_keys list."""
    keys = api_config.get("api_keys")
    if keys:
        return random.choice(keys)
    return api_config["api_key"]


def _get_openai_client(api_config: dict):
    """Return an AsyncOpenAI client configured like vlm_client.py.

    Supports MoK (mixture-of-keys): randomly picks a key per call and caches the client.
    """
    from openai import AsyncOpenAI

    base_url = api_config.get("api_base") or api_config["url"]
    url = base_url if base_url.rstrip("/").endswith("/chat/completions") else f"{base_url.rstrip('/')}/chat/completions"
    if url.endswith("/chat/completions"):
        base_url = url[: -len("/chat/completions")]
    api_key = _pick_api_key(api_config)
    cache_key = (base_url, api_key)
    if cache_key not in _OPENAI_CLIENTS:
        _OPENAI_CLIENTS[cache_key] = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _OPENAI_CLIENTS[cache_key]


def _image_to_base64_url(image_path: str) -> dict:
    """Convert a local image file to an OpenAI-compatible image_url content part.

    Resizes so that the long edge is at most 1024px before base64 encoding.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    max_size = 1024
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def _build_llm_kwargs(api_config: dict, prompt_text: str, temperature: float, max_tokens: int, image_paths: list = None):
    """Build kwargs for openai chat.completions.create."""
    content = prompt_text
    if image_paths:
        content = [{"type": "text", "text": prompt_text}]
        for path in image_paths:
            content.append(_image_to_base64_url(path))
    kwargs = {
        "model": api_config["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    model_name = api_config["model"]
    if not model_name.lower().startswith("qwen"):
        kwargs["response_format"] = {"type": "json_object"}
    # Pass through extra_body from config (e.g. chat_template_kwargs for Qwen)
    extra_body = api_config.get("extra_body")
    if extra_body:
        kwargs["extra_body"] = extra_body
    if model_name.startswith("gemini"):
        gemini_extra = {
            "google": {
                "thinking_config": {
                    "include_thoughts": True,
                    "thinking_budget": 128,
                },
                "thought_tag_marker": "think",
            }
        }
        if "extra_body" in kwargs:
            kwargs["extra_body"] = {**kwargs["extra_body"], **gemini_extra}
        else:
            kwargs["extra_body"] = gemini_extra
    return kwargs


def _parse_llm_response(resp) -> dict:
    """Extract and parse JSON from an openai completion response.

    Tries direct JSON parsing first, then falls back to extracting the first
    top-level JSON object from free-form text (e.g. when the model outputs
    analysis prose before/after the JSON block).
    """
    choice = resp.choices[0]
    msg = choice.message
    raw = (msg.content or "").strip()
    if not raw:
        finish = getattr(choice, "finish_reason", "unknown")
        raise ValueError(
            f"LLM returned empty content (finish_reason={finish}, model={getattr(resp, 'model', 'unknown')})"
        )

    # Strip thinking blocks (if backend hasn't already removed them)
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Strip markdown fences
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # 1. Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Fallback: extract first top-level {...} from free-form text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 3. Last resort: infer from keywords in free-form prose
    lowered = content.lower()
    key_match = re.search(r'"(coherent|adequate)"\s*:\s*(true|false)', content, re.IGNORECASE)
    if key_match:
        key, val = key_match.group(1), key_match.group(2).lower() == "true"
        print(f"[WARN] Parsed {key}={val} from keyword fallback. Raw: {raw[:200]!r}", file=sys.stderr)
        return {key: val}

    # Heuristic fallback for text_logicality
    if "coherent" in lowered or "consistent" in lowered:
        if "inconsistent" in lowered or "not consistent" in lowered or "contradict" in lowered:
            print(f"[WARN] Inferred coherent=false from keyword fallback. Raw: {raw[:200]!r}", file=sys.stderr)
            return {"coherent": False}
        print(f"[WARN] Inferred coherent=true from keyword fallback. Raw: {raw[:200]!r}", file=sys.stderr)
        return {"coherent": True}

    # Heuristic fallback for text_richness
    if "adequate" in lowered or "sufficient" in lowered:
        if "inadequate" in lowered or "not adequate" in lowered or "insufficient" in lowered:
            print(f"[WARN] Inferred adequate=false from keyword fallback. Raw: {raw[:200]!r}", file=sys.stderr)
            return {"adequate": False}
        print(f"[WARN] Inferred adequate=true from keyword fallback. Raw: {raw[:200]!r}", file=sys.stderr)
        return {"adequate": True}

    raise json.JSONDecodeError(
        f"Failed to parse JSON from LLM response. Raw content starts with: {raw[:300]!r}",
        raw, 0
    )


async def call_llm(api_config: dict, prompt_text: str, temperature: float, max_tokens: int, max_retries: int = 0, image_paths: list = None, format_regex: str | list = None):
    """Async version of call_llm using AsyncOpenAI.

    format_regex: optional regex string or list of strings. The raw LLM output
    text must match at least one pattern, otherwise ValueError is raised
    (triggering a retry within the retry loop).
    """
    api_config = _resolve_api_config(api_config)
    client = _get_openai_client(api_config)
    kwargs = _build_llm_kwargs(api_config, prompt_text, temperature, max_tokens, image_paths)

    if isinstance(format_regex, str):
        patterns = [re.compile(format_regex, re.DOTALL)]
    elif format_regex:
        patterns = [re.compile(p, re.DOTALL) for p in format_regex]
    else:
        patterns = []

    last_resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.chat.completions.create(**kwargs)
            last_resp = resp
            raw = (resp.choices[0].message.content or "").strip()

            if patterns:
                if not any(p.search(raw) for p in patterns):
                    raise ValueError(
                        f"LLM output does not match expected format. "
                        f"Raw (first 300 chars): {raw[:300]!r}"
                    )

            return _parse_llm_response(resp)
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(3)
                continue
            # Final attempt failed — dump diagnostic info
            print(f"[LLM ERROR] type={type(e).__name__}, attempt={attempt}, error={e}", file=sys.stderr)
            if last_resp is not None:
                choice = last_resp.choices[0]
                raw = (choice.message.content or "")
                finish = getattr(choice, "finish_reason", "unknown")
                print(f"[LLM ERROR] finish_reason={finish}, raw_content={raw[:500]!r}", file=sys.stderr)
            else:
                print("[LLM ERROR] no response received (network or API error before response)", file=sys.stderr)
            raise

    raise RuntimeError("call_llm exited retry loop unexpectedly")


def extract_text(blocks):
    return "\n".join(b["content"] for b in blocks if b.get("type") == "text")


def extract_images(blocks):
    return [b["content"] for b in blocks if b.get("type") == "image"]

