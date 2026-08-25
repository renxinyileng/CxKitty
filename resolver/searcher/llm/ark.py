"""火山方舟 (豆包) https://console.volcengine.com/ark"""

from .base import LLMSearcherBase


class ArkSearcher(LLMSearcherBase):
    """火山方舟答题器, model 需填写推理接入点 id (ep-xxx)"""

    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/"
    DEFAULT_MODEL = None  # 方舟以推理接入点 id 作为 model, 无法预设
    CONSOLE = "https://console.volcengine.com/ark"
    TEMPERATURE_WITH_THINKING = True

    def thinking_params(self) -> tuple[dict, dict]:
        """方舟用 extra_body 的 thinking.type 开启思考 (enabled/disabled/auto)"""
        return {}, {"thinking": {"type": "enabled"}}
