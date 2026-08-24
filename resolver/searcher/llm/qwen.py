"""阿里通义千问 (百炼) https://bailian.console.aliyun.com"""

from .base import LLMSearcherBase


class QwenSearcher(LLMSearcherBase):
    """通义千问答题器"""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
    DEFAULT_MODEL = "qwen-plus"
    CONSOLE = "https://bailian.console.aliyun.com"
    TEMPERATURE_WITH_THINKING = True

    def thinking_params(self) -> tuple[dict, dict]:
        """百炼用 extra_body 的 enable_thinking 开启思考,
        thinking_budget 限制思考链 token 数 (取值 1~32768)
        注: 部分商业版模型的思考模式仅支持流式调用, 接口报错时会自动降级为非思考模式
        """
        extra = {"enable_thinking": True}
        if self.thinking_budget:
            extra["thinking_budget"] = self.thinking_budget
        return {}, extra
