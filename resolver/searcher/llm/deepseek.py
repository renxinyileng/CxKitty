"""深度求索 DeepSeek https://platform.deepseek.com"""

from .base import LLMSearcherBase


class DeepSeekSearcher(LLMSearcherBase):
    """DeepSeek 答题器"""

    BASE_URL = "https://api.deepseek.com/v1/"
    DEFAULT_MODEL = "deepseek-v4-flash"
    CONSOLE = "https://platform.deepseek.com"
    # V4 系列的思考档位为 low/high/max
    MAX_EFFORT = "max"
    # 官方说明: 思考模式不支持 temperature / top_p / presence_penalty / frequency_penalty
    TEMPERATURE_WITH_THINKING = False

    def thinking_params(self) -> tuple[dict, dict]:
        """顶层 reasoning_effort 控制思考档位, 无 token 级思考预算 (靠 timeout 兜底)"""
        return self.effort_params()
