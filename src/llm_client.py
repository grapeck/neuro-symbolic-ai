"""LLM client for hybrid neuro-symbolic LLM + SymPy math solver.

Provides a unified interface supporting OpenAI-compatible endpoints,
Google Gemini API (via google.genai), and deterministic mock mode for testing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment, misc]

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for generating math solver prompts using LLMs or SymPy mocks.

    Supports OpenAI-compatible APIs (e.g., local vLLM/Ollama or OpenAI),
    Google Gemini API via the google.genai SDK as a fallback, and a deterministic
    mock mode for testing math problem patterns without making API calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        mock_mode: bool | None = None,
    ) -> None:
        """Initialize the LLM client configuration from environment variables or arguments.

        Args:
            api_key: Optional API key override. If not provided, reads LLM_API_KEY,
                OPENAI_API_KEY, or GEMINI_API_KEY / GOOGLE_API_KEY from environment.
            base_url: Optional base URL for OpenAI-compatible endpoint. If not provided,
                reads LLM_BASE_URL from environment.
            model: Optional model name override. If not provided, reads LLM_MODEL
                from environment or defaults based on backend.
            mock_mode: Optional boolean flag to force mock mode. If not provided,
                reads MOCK_MODE from environment.
        """
        # Determine mock mode
        if mock_mode is not None:
            self.mock_mode: bool = mock_mode
        else:
            mock_env = os.getenv("MOCK_MODE", "").strip().lower()
            self.mock_mode = mock_env in ("true", "1", "yes", "t", "on")

        # Determine base URL for OpenAI-compatible API
        base_url_env = os.getenv("LLM_BASE_URL", "").strip()
        self.base_url: str | None = base_url if base_url is not None else (base_url_env or None)

        # Determine API key
        if api_key is not None:
            self.api_key: str | None = api_key
        else:
            api_key_env = os.getenv("LLM_API_KEY", "").strip()
            if not api_key_env:
                if self.base_url:
                    api_key_env = os.getenv("OPENAI_API_KEY", "").strip()
                else:
                    api_key_env = (
                        os.getenv("GEMINI_API_KEY", "").strip()
                        or os.getenv("GOOGLE_API_KEY", "").strip()
                    )
            self.api_key = api_key_env or None

        # Determine model name
        if model is not None:
            self.model: str = model
        else:
            model_env = os.getenv("LLM_MODEL", "").strip()
            if model_env:
                self.model = model_env
            else:
                self.model = "gpt-4o" if self.base_url else "gemini-2.5-pro"

        self.client: Any = None
        if not self.mock_mode:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the underlying API client based on configuration.

        Raises:
            ImportError: If the required backend library is not installed.
            RuntimeError: If client initialization fails.
        """
        if self.base_url:
            if OpenAI is None:
                raise ImportError(
                    "The 'openai' library is required when LLM_BASE_URL is set. "
                    "Install it via: pip install openai"
                )
            try:
                self.client = OpenAI(
                    api_key=self.api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
                    base_url=self.base_url,
                )
                logger.info(
                    "Initialized OpenAI-compatible client with base_url=%s, model=%s",
                    self.base_url,
                    self.model,
                )
            except Exception as exc:
                logger.error("Failed to initialize OpenAI client: %s", exc)
                raise RuntimeError(
                    f"Failed to initialize OpenAI client with base_url '{self.base_url}': {exc}"
                ) from exc
        else:
            if genai is None:
                raise ImportError(
                    "The 'google-genai' library is required for Gemini API requests. "
                    "Install it via: pip install google-genai"
                )
            try:
                effective_key = (
                    self.api_key
                    or os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                )
                if effective_key:
                    self.client = genai.Client(
                        api_key=effective_key,
                        http_options={"api_version": "v1beta"},
                    )
                else:
                    self.client = genai.Client(
                        http_options={"api_version": "v1beta"},
                    )
                logger.info(
                    "Initialized Google Gemini client with model=%s",
                    self.model,
                )
            except Exception as exc:
                logger.error("Failed to initialize Gemini client: %s", exc)
                raise RuntimeError(
                    f"Failed to initialize Gemini client. Ensure LLM_API_KEY or "
                    f"GEMINI_API_KEY is configured: {exc}"
                ) from exc

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Send a request to the LLM or mock solver and return the text response.

        Args:
            system_prompt: System instructions guiding the model's behavior.
            user_message: The current user message or mathematical problem.
            conversation_history: Optional list of previous conversation turns,
                where each turn is a dictionary with 'role' and 'content' keys.

        Returns:
            The generated response string, typically containing SymPy code.

        Raises:
            RuntimeError: If an error occurs during API communication.
        """
        if self.mock_mode:
            return self._mock_response(user_message)

        if self.client is None:
            self._init_client()

        if self.base_url:
            return self._generate_openai(system_prompt, user_message, conversation_history)
        return self._generate_gemini(system_prompt, user_message, conversation_history)

    def _generate_openai(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Execute text generation via OpenAI-compatible API.

        Args:
            system_prompt: System prompt instructions.
            user_message: Current user message.
            conversation_history: Optional previous conversation messages.

        Returns:
            Generated response content from the model.

        Raises:
            RuntimeError: If the OpenAI API request fails.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content") or turn.get("text", "")
                messages.append({"role": str(role), "content": str(content)})
        messages.append({"role": "user", "content": user_message})

        max_rate_retries = 5
        base_delay = 3.0

        for attempt in range(max_rate_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                choice = response.choices[0]
                return choice.message.content or ""
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = any(
                    err in exc_str.lower()
                    for err in ["429", "rate limit", "ratelimit", "resource_exhausted", "quota"]
                )
                if is_rate_limit and attempt < max_rate_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limit reached (OpenAI). Backing off for %.1fs (attempt %d/%d)...",
                        sleep_time, attempt + 1, max_rate_retries
                    )
                    time.sleep(sleep_time)
                    continue
                logger.error("OpenAI generation failed: %s", exc)
                raise RuntimeError(f"OpenAI generation failed: {exc}") from exc
        return ""

    def _generate_gemini(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Execute text generation via Google Gemini API (google.genai).

        Args:
            system_prompt: System prompt instructions.
            user_message: Current user message.
            conversation_history: Optional previous conversation messages.

        Returns:
            Generated text response from Gemini.

        Raises:
            RuntimeError: If the Gemini API request fails.
        """
        config = None
        if system_prompt:
            if types is not None:
                config = types.GenerateContentConfig(system_instruction=system_prompt)
            else:
                config = {"system_instruction": system_prompt}

        if conversation_history:
            contents = []
            for turn in conversation_history:
                role = turn.get("role", "user")
                gemini_role = "model" if role in ("assistant", "model") else "user"
                content_text = str(turn.get("content") or turn.get("text", ""))
                if types is not None:
                    contents.append(
                        types.Content(
                            role=gemini_role,
                            parts=[types.Part.from_text(text=content_text)],
                        )
                    )
                else:
                    contents.append({
                        "role": gemini_role,
                        "parts": [{"text": content_text}],
                    })

            if types is not None:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_message)],
                    )
                )
            else:
                contents.append({
                    "role": "user",
                    "parts": [{"text": user_message}],
                })
        else:
            contents = user_message

        max_rate_retries = 5
        base_delay = 3.0

        for attempt in range(max_rate_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = any(
                    err in exc_str.lower()
                    for err in ["429", "rate limit", "ratelimit", "resource_exhausted", "quota"]
                )
                if is_rate_limit and attempt < max_rate_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limit reached (Gemini model %s). Backing off for %.1fs (attempt %d/%d)...",
                        self.model, sleep_time, attempt + 1, max_rate_retries
                    )
                    time.sleep(sleep_time)
                    continue
                logger.error("Gemini generation failed: %s", exc)
                raise RuntimeError(f"Gemini generation failed: {exc}") from exc
        return ""

    def _mock_response(self, user_message: str) -> str:
        """Return deterministic fake SymPy code for testing without API calls.

        Recognizes math keywords (limit, derivative, integral, series) in the
        user message and returns corresponding valid SymPy code blocks.

        Args:
            user_message: The input mathematical query or instruction.

        Returns:
            A string containing a markdown-formatted SymPy code block.
        """
        text = (user_message or "").lower()

        if "limit" in text or "предел" in text:
            return (
                "```python\n"
                "import sympy as sp\n\n"
                "x = sp.Symbol('x')\n"
                "expr = sp.sin(x) / x\n"
                "result = sp.limit(expr, x, 0)\n"
                "print(result)\n"
                "```"
            )

        if "derivative" in text or "diff" in text or "производн" in text:
            return (
                "```python\n"
                "import sympy as sp\n\n"
                "x = sp.Symbol('x')\n"
                "expr = sp.sin(x) * sp.exp(x)\n"
                "result = sp.diff(expr, x)\n"
                "print(result)\n"
                "```"
            )

        if "integral" in text or "integrate" in text or "интеграл" in text:
            return (
                "```python\n"
                "import sympy as sp\n\n"
                "x = sp.Symbol('x')\n"
                "expr = x**2 * sp.exp(x)\n"
                "result = sp.integrate(expr, x)\n"
                "print(result)\n"
                "```"
            )

        if "series" in text or "taylor" in text or "ряд" in text:
            return (
                "```python\n"
                "import sympy as sp\n\n"
                "x = sp.Symbol('x')\n"
                "expr = sp.cos(x)\n"
                "result = sp.series(expr, x, 0, 6)\n"
                "print(result)\n"
                "```"
            )

        return (
            "```python\n"
            "import sympy as sp\n\n"
            "x = sp.Symbol('x')\n"
            "expr = 2 * x + 5 - 15\n"
            "result = sp.solve(expr, x)\n"
            "print(result)\n"
            "```"
        )
