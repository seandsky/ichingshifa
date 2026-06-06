# -*- coding: utf-8 -*-
"""AI client wrappers for ichingshifa."""

import json
import urllib.request
from cerebras.cloud.sdk import Cerebras

DEFAULT_MODEL = "gpt-oss-120b"

CEREBRAS_MODEL_OPTIONS = [
    "gpt-oss-120b",
    "zai-glm-4.7",
]

CEREBRAS_MODEL_DESCRIPTIONS = {
    "gpt-oss-120b": "GPT-OSS 120B (推薦)",
    "zai-glm-4.7": "ZAI GLM 4.7",
}


class CerebrasClient:
    """Thin wrapper around the Cerebras SDK for chat completions."""

    def __init__(self, api_key):
        if not api_key:
            raise ValueError("CerebrasClient must be initialized with an API key.")
        self.client = Cerebras(api_key=api_key)

    def get_chat_completion(self, messages, model=DEFAULT_MODEL, **kwargs):
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            **kwargs,
        )


class OpenAICompatibleClient:
    """Simple OpenAI-compatible chat completion client."""

    def __init__(self, api_key, base_url):
        if not api_key:
            raise ValueError("OpenAICompatibleClient must be initialized with an API key.")
        if not base_url:
            raise ValueError("OpenAICompatibleClient must be initialized with a server URL.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def get_chat_completion(self, messages, model, **kwargs):
        if not model:
            raise ValueError("Model is required for OpenAI-compatible API.")

        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        payload = {
            "messages": messages,
            "model": model,
            **kwargs,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
