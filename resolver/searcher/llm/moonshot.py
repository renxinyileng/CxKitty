"""月之暗面 Kimi https://platform.moonshot.cn"""

from .base import LLMSearcherBase


class MoonshotSearcher(LLMSearcherBase):
    """Kimi 答题器"""

    BASE_URL = "https://api.moonshot.cn/v1/"
    DEFAULT_MODEL = "moonshot-v1-8k"
    CONSOLE = "https://platform.moonshot.cn"
    # K3 起用顶层 reasoning_effort, 档位最高为 max
    MAX_EFFORT = "max"
    TEMPERATURE_WITH_THINKING = False

    def thinking_params(self) -> tuple[dict, dict]:
        """只下发 reasoning_effort: 接口不允许 thinking 与 reasoning_effort 同时出现"""
        return self.effort_params()
