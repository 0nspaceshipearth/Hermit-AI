"""
API Client Wrapper for OpenAI-Compatible Endpoints.
Allows Hermit to use external servers (LM Studio, Ollama, etc.) instead of embedded llama-cpp-python.
"""

import json
import requests
import sys
from typing import List, Dict, Generator, Any, Union

from chatbot import config


class OpenAIClientWrapper:
    """
    Polymorphic wrapper that mimics llama_cpp.Llama but calls an API.
    """

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        print(f"Initialized API Client: {self.base_url} (Model: {self.model_name})")

    def _is_codex_oauth_mode(self) -> bool:
        # Codex OAuth tokens are not standard sk- API keys and typically target ChatGPT backend APIs.
        key = (self.api_key or "").strip()
        is_oauthish = bool(key) and not key.startswith("sk-")
        return is_oauthish and (
            self.base_url.startswith("https://chatgpt.com/backend-api")
            or self.base_url.startswith("https://api.openai.com")
        )

    def _chat_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        account_id = getattr(config, "API_ACCOUNT_ID", "")
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id
        if self.base_url.startswith("https://chatgpt.com/backend-api"):
            headers["Origin"] = "https://chatgpt.com"
            headers["Referer"] = "https://chatgpt.com/"
        return headers

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Mimics llama_cpp.Llama.create_chat_completion
        """

        # approximate mapping for repeat_penalty if present in kwargs
        presence_penalty = 0.0
        if "repeat_penalty" in kwargs:
            # loose mapping: 1.1 -> 0.1, 1.2 -> 0.2
            presence_penalty = max(0.0, kwargs["repeat_penalty"] - 1.0)

        try:
            if self._is_codex_oauth_mode():
                if stream:
                    return self._stream_responses(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
                return self._blocking_responses(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

            # Default OpenAI-compatible chat/completions path
            url = f"{self.base_url}/chat/completions"
            headers = self._chat_headers()
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
                "presence_penalty": presence_penalty
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            print(f"DEBUG: Requesting URL: {url}")
            if stream:
                return self._stream_request(url, headers, payload)
            else:
                return self._blocking_request(url, headers, payload)

        except Exception as e:
            print(f"API Request Failed: {e}", file=sys.stderr)
            raise RuntimeError(f"API Connection Error: {e}")

    def _blocking_request(self, url, headers, payload):
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code >= 400:
            body = response.text[:500]
            raise RuntimeError(f"HTTP {response.status_code} {response.reason}: {body}")
        return response.json()

    def _stream_request(self, url, headers, payload):
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        if response.status_code >= 400:
            body = response.text[:500]
            raise RuntimeError(f"HTTP {response.status_code} {response.reason}: {body}")

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue

    def _build_codex_responses_payload(self, messages, stream=False, max_tokens=None, temperature=0.7, top_p=0.95):
        instructions = "You are a helpful assistant."
        user_input = []

        for msg in messages or []:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if role == "system" and isinstance(content, str) and content.strip():
                instructions = content.strip()
                continue
            user_input.append({
                "role": role,
                "content": [{"type": "input_text", "text": str(content)}],
            })

        payload = {
            "model": self.model_name,
            "instructions": instructions,
            "input": user_input,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            payload["max_output_tokens"] = max_tokens
        return payload

    def _blocking_responses(self, messages, max_tokens=None, temperature=0.7, top_p=0.95):
        path = "/codex/responses" if self.base_url.startswith("https://chatgpt.com/backend-api") else "/responses"
        url = f"{self.base_url}{path}"
        headers = self._chat_headers()
        payload = self._build_codex_responses_payload(messages, stream=False, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        if max_tokens:
            payload["max_output_tokens"] = max_tokens

        print(f"DEBUG: Requesting URL: {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        if response.status_code >= 400:
            body = response.text[:500]
            raise RuntimeError(f"HTTP {response.status_code} {response.reason}: {body}")

        data = response.json()
        text = self._extract_responses_text(data)
        return {
            "choices": [
                {
                    "message": {
                        "content": text
                    }
                }
            ]
        }

    def _stream_responses(self, messages, max_tokens=None, temperature=0.7, top_p=0.95):
        path = "/codex/responses" if self.base_url.startswith("https://chatgpt.com/backend-api") else "/responses"
        url = f"{self.base_url}{path}"
        headers = self._chat_headers()
        payload = self._build_codex_responses_payload(messages, stream=True, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        print(f"DEBUG: Requesting URL: {url}")
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        if response.status_code >= 400:
            body = response.text[:500]
            raise RuntimeError(f"HTTP {response.status_code} {response.reason}: {body}")

        for line in response.iter_lines():
            if not line:
                continue
            raw = line.decode("utf-8", errors="ignore")
            if not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            etype = (event.get("type") or "").lower()
            if etype == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    yield {"choices": [{"delta": {"content": delta}}]}
            elif etype in {"response.completed", "response.done"}:
                break

    def _extract_responses_text(self, data: Dict[str, Any]) -> str:
        # Newer responses payloads often include output_text directly.
        direct = data.get("output_text")
        if isinstance(direct, str):
            return direct

        # Fallback: walk output[].content[].text
        output = data.get("output", [])
        chunks = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            chunks.append(text)
        return "".join(chunks).strip()
