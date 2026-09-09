"""OpenAI-compatible chat client for judge calls.

Configured entirely from the environment so the same code runs against any
provider (or a local model) without edits:

  LLM_BASE_URL  e.g. https://api.groq.com/openai/v1
  LLM_API_KEY
  JUDGE_MODEL   model id used for judging

Two lessons are baked in, both learned the expensive way:
  * reasoning models spend max_tokens on hidden thinking from the same
    budget, so a tiny cap returns an empty verdict and the default is generous;
  * free tiers rate-limit tokens per minute, so a 429 waits and retries
    instead of raising.
"""

import os
import re
import time

import requests


class JudgeClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: int = 90):
        self.base_url = (base_url or os.environ["LLM_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["LLM_API_KEY"]
        self.model = model or os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int = 600,
             temperature: float = 0.0, retries: int = 6) -> str:
        for _attempt in range(retries):
            r = requests.post(
                self.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self.api_key},
                json={"model": self.model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=self.timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"].get("content") or ""
            if r.status_code == 429:
                wait = 8.0
                m = re.search(r"try again in ([\d.]+)s", r.text or "")
                if m:
                    wait = min(float(m.group(1)) + 0.5, 30.0)
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM {r.status_code}: {r.text[:200]}")
        raise RuntimeError("LLM 429: retries exhausted")
