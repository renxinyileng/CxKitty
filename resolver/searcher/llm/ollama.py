"""本地 Ollama 推理服务 https://ollama.com/library"""

from .base import LLMSearcherBase


class OllamaSearcher(LLMSearcherBase):
    """Ollama 答题器, 无需 api_key, model 需填写已拉取的模型名"""

    BASE_URL = "http://localhost:11434/v1/"
    DEFAULT_MODEL = None  # 取决于本地拉取了哪些模型, 无法预设
    NEED_KEY = False
    CONSOLE = "https://ollama.com/library"
    TEMPERATURE_WITH_THINKING = True

    def thinking_params(self) -> tuple[dict, dict]:
        """Ollama 的 OpenAI 兼容端点用 reasoning_effort 开启思考,
        旧版本不认该参数, 被拒时自动降级为非思考模式;
        思考模型会把思考写进 <think> 标签, 由 answer 模块剔除
        """
        return self.effort_params()
