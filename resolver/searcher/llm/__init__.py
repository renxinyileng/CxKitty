"""大模型答题器

一个服务商一个模块, 各自声明接口地址、默认模型与思考参数;
公共逻辑在 `base.py` (请求/重试/降级/缓存)、`prompt.py` (提问) 与 `answer.py` (答案归一化)。

新增服务商 = 复制一份 `deepseek.py` 改三个类字段和 `thinking_params()`
→ 注册进本文件的 `PROVIDERS` 与 `resolver/question.py` 的 `SEARCHERS`
→ config.yml 与 README 补说明。
"""

from .ark import ArkSearcher
from .base import LLMSearcherBase
from .deepseek import DeepSeekSearcher
from .gemini import GeminiSearcher
from .moonshot import MoonshotSearcher
from .ollama import OllamaSearcher
from .openai import OpenAISearcher
from .qwen import QwenSearcher
from .siliconflow import SiliconFlowSearcher
from .zhipu import ZhipuSearcher

# 服务商名 -> 答题器类, 供 `type: OpenAISearcher` + `provider: xxx` 的写法查表
PROVIDERS: dict[str, type[LLMSearcherBase]] = {
    "openai": OpenAISearcher,
    "deepseek": DeepSeekSearcher,
    "moonshot": MoonshotSearcher,
    "kimi": MoonshotSearcher,
    "qwen": QwenSearcher,
    "dashscope": QwenSearcher,
    "bailian": QwenSearcher,
    "zhipu": ZhipuSearcher,
    "glm": ZhipuSearcher,
    "siliconflow": SiliconFlowSearcher,
    "ark": ArkSearcher,
    "doubao": ArkSearcher,
    "gemini": GeminiSearcher,
    "ollama": OllamaSearcher,
}

__all__ = [
    "LLMSearcherBase",
    "PROVIDERS",
    "OpenAISearcher",
    "DeepSeekSearcher",
    "MoonshotSearcher",
    "QwenSearcher",
    "ZhipuSearcher",
    "SiliconFlowSearcher",
    "ArkSearcher",
    "GeminiSearcher",
    "OllamaSearcher",
]
