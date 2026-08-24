"""OpenAI 及任意 OpenAI 兼容接口 (中转站、私有部署等)

不填 base_url / model 时按 OpenAI 官方接口使用; 需要对接其他服务时自行填写这两项即可。
"""

from typing import Optional

from .base import LLMSearcherBase


class OpenAISearcher(LLMSearcherBase):
    """OpenAI 答题器 https://platform.openai.com/docs/models"""

    BASE_URL = "https://api.openai.com/v1/"
    DEFAULT_MODEL = "gpt-4o-mini"
    CONSOLE = "https://platform.openai.com/docs/models"
    # 推理模型档位为 minimal/low/medium/high, 无 max 档
    MAX_EFFORT = "high"
    # o 系列等推理模型只接受默认 temperature, 思考时不下发
    TEMPERATURE_WITH_THINKING = False

    def __new__(cls, *args, provider: Optional[str] = None, **kwargs):
        """兼容 `type: OpenAISearcher` + `provider: deepseek` 的配置写法,
        直接构造对应服务商的答题器
        """
        if cls is not OpenAISearcher or not provider:
            return super().__new__(cls)
        from . import PROVIDERS  # 延迟导入避免循环引用

        if (target := PROVIDERS.get(provider.lower())) is None:
            raise ValueError(f"未知的大模型服务商 {provider}, 可用: {', '.join(PROVIDERS)}")
        if target is cls:
            return super().__new__(cls)
        return target(*args, **kwargs)  # 目标类自行完成初始化

    def thinking_params(self) -> tuple[dict, dict]:
        """顶层 reasoning_effort 控制思考深度, 不支持按 token 限制思考链"""
        return self.effort_params()
