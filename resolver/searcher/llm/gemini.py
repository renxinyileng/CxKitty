"""Google Gemini 的 OpenAI 兼容端点 https://ai.google.dev/gemini-api/docs/openai"""

from .base import LLMSearcherBase


class GeminiSearcher(LLMSearcherBase):
    """Gemini 答题器"""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_MODEL = "gemini-2.0-flash"
    CONSOLE = "https://ai.google.dev/gemini-api/docs/openai"
    MAX_EFFORT = "high"
    TEMPERATURE_WITH_THINKING = True

    def thinking_params(self) -> tuple[dict, dict]:
        """顶层 reasoning_effort 控制思考深度,
        思考预算走 Google 扩展字段 google.thinking_config.thinking_budget
        """
        params, _ = self.effort_params()
        if not self.thinking_budget:
            return params, {}
        extra = {
            "extra_body": {"google": {"thinking_config": {"thinking_budget": self.thinking_budget}}}
        }
        return params, extra
