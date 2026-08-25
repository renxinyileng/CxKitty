"""硅基流动 SiliconFlow https://cloud.siliconflow.cn"""

from .base import LLMSearcherBase


class SiliconFlowSearcher(LLMSearcherBase):
    """硅基流动答题器"""

    BASE_URL = "https://api.siliconflow.cn/v1/"
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    CONSOLE = "https://cloud.siliconflow.cn"
    TEMPERATURE_WITH_THINKING = True

    def thinking_params(self) -> tuple[dict, dict]:
        """与百炼一致: enable_thinking 开启思考, thinking_budget 限制思考链 token 数
        平台上的非思考模型会忽略/拒绝该参数, 被拒时自动降级为非思考模式
        """
        extra = {"enable_thinking": True}
        if self.thinking_budget:
            extra["thinking_budget"] = self.thinking_budget
        return {}, extra
