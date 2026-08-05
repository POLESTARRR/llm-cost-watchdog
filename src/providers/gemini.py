"""Google Gemini adapter.

Uses `google-generativeai`. That package is deprecated in favor of
`google.genai`, but it still works and is what this project was specified
against; the adapter boundary means swapping it later touches only this file.
"""

import os

from src.providers.base import LLMResponse, ProviderError

_configured = False


class GeminiProvider:
    name = "google"

    def is_configured(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def _ensure_configured(self):
        global _configured
        import google.generativeai as genai

        if not self.is_configured():
            raise ProviderError("GEMINI_API_KEY is not set")
        if not _configured:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            _configured = True
        return genai

    def complete(self, prompt: str, model: str, temperature: float) -> LLMResponse:
        genai = self._ensure_configured()
        gen_model = genai.GenerativeModel(model)
        response = gen_model.generate_content(
            prompt, generation_config={"temperature": temperature}
        )
        usage = response.usage_metadata
        # Gemini reports cached tokens only when context caching is in use.
        cached = getattr(usage, "cached_content_token_count", 0) or 0
        return LLMResponse(
            text=response.text,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            cached_input_tokens=cached,
        )
