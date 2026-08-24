"""智谱 GLM https://open.bigmodel.cn"""

from .base import LLMSearcherBase


class ZhipuSearcher(LLMSearcherBase):
    """智谱 GLM 答题器"""

    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    DEFAULT_MODEL = "glm-4-flash"
    CONSOLE = "https://open.bigmodel.cn"
    # GLM-5 系列改用 reasoning_effort 且最高档为 max, 需要时配置 thinking_effort 覆盖
    MAX_EFFORT = "max"
    TEMPERATURE_WITH_THINKING = True

    def thinking_params(self) -> tuple[dict, dict]:
        """GLM-4.5/4.6 用 extra_body 的 thinking.type 开启思考, 无 token 级思考预算
        (GLM-5 起默认常开思考, 该参数会被忽略)
        """
        return {}, {"thinking": {"type": "enabled"}}
