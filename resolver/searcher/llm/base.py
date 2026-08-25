"""大模型答题器基类

各服务商的答题器只需继承本类, 在自己的模块里声明接口地址、默认模型与思考参数,
公共的提问渲染、请求重试、降级、答案归一化与缓存都在这里。
"""

import time
from typing import Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from cxapi.schema import QuestionModel
from logger import Logger

from .. import SearcherBase, SearcherResp
from . import answer as answer_util
from .prompt import DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT, render_example, render_prompt

# 未配置 temperature 时的取值: 思考模式下按服务商决定是否下发, 非思考模式下用 0 保证作答稳定
AUTO = "auto"

# 缓存条数上限, 长时间运行时避免无限增长
CACHE_LIMIT = 1024


class LLMSearcherBase(SearcherBase):
    """大模型答题器基类 (OpenAI 兼容接口)

    子类需要覆盖的类字段:
        BASE_URL: 接口地址
        DEFAULT_MODEL: 默认模型, None 表示必须由用户指定 (如火山方舟的推理接入点 id)
        NEED_KEY: 是否需要 api_key
        CONSOLE: 控制台/文档地址, 用于报错提示
        MAX_EFFORT: 该服务商支持的最高思考档位
        TEMPERATURE_WITH_THINKING: 思考模式下是否可以同时下发 temperature
    子类可覆盖的方法:
        thinking_params(): 构造开启深度思考的参数, 各服务商互不兼容
    """

    IS_AI = True  # 归入 AI 组, 题库查不到时才调用

    BASE_URL: str = ""
    DEFAULT_MODEL: Optional[str] = None
    NEED_KEY: bool = True
    CONSOLE: str = ""
    MAX_EFFORT: str = "high"
    TEMPERATURE_WITH_THINKING: bool = False

    client: OpenAI
    model: str
    system_prompt: str
    prompt: str
    temperature: Optional[float] | str
    max_tokens: Optional[int]
    timeout: float
    max_retries: int
    few_shot: bool
    thinking: bool
    thinking_effort: str
    thinking_budget: Optional[int]
    extra_body: dict
    cache: Optional[dict[str, str]]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # 留空则取本服务商的接口地址
        model: Optional[str] = None,  # 留空则取本服务商的默认模型
        system_prompt: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] | str = AUTO,  # 留空自动, 置 null 则不下发该参数
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,  # 单次请求超时, 同时也是思考时长的硬上限
        max_retries: int = 2,  # 网络错误/限速时的重试次数
        few_shot: bool = True,  # 是否附带同题型的单样本示例
        thinking: bool = True,  # 是否开启深度思考 (默认开到服务商支持的最高档位)
        thinking_effort: Optional[str] = None,  # 思考档位, 留空取服务商最高档
        thinking_budget: Optional[int] = 2048,  # 思考链最大 token 数, null 为不限制
        extra_body: Optional[dict] = None,  # 透传给接口的额外参数, 优先级最高
        cache: bool = True,  # 是否缓存同一题目的作答结果
        **kwargs,  # 兼容 provider 等已被上层消化的字段
    ) -> None:
        super().__init__()
        if self.NEED_KEY and not api_key:
            raise ValueError(f"{self.__class__.__name__} 需要配置 api_key, 请前往 {self.CONSOLE} 获取")
        if not (model := model or self.DEFAULT_MODEL):
            raise ValueError(
                f"{self.__class__.__name__} 未预设默认模型, 请在 config.yml 指定 model ({self.CONSOLE})"
            )

        self.logger = Logger(self.__class__.__name__)
        # 关闭 SDK 自带重试, 由本类统一控制重试与退避
        self.client = OpenAI(
            api_key=api_key or "EMPTY",  # 本地推理服务不校验 key, 但 SDK 要求非空
            base_url=base_url or self.BASE_URL,
            max_retries=0,
        )
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.prompt = prompt or DEFAULT_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.few_shot = few_shot
        self.thinking = thinking
        self.thinking_effort = thinking_effort or self.MAX_EFFORT
        self.thinking_budget = thinking_budget
        self.extra_body = extra_body or {}
        # 接口拒绝思考参数 (模型不支持) 时置否, 之后不再重复下发
        self._thinking_available = True
        self.cache = {} if cache else None

    # ---- 服务商适配点 ----

    def thinking_params(self) -> tuple[dict, dict]:
        """构造开启深度思考的参数, 由各服务商模块覆盖
        Returns:
            dict, dict: 顶层参数, extra_body 参数
        """
        return {}, {}

    def effort_params(self) -> tuple[dict, dict]:
        """顶层 reasoning_effort 风格的思考参数 (多数服务商适用)"""
        return {"reasoning_effort": self.thinking_effort}, {}

    # ---- 答题流程 ----

    def invoke(self, question: QuestionModel) -> SearcherResp:
        cache_key = self.__cache_key(question)
        if self.cache is not None and (cached := self.cache.get(cache_key)) is not None:
            self.logger.debug(f"命中本地缓存 {question.value} -> {cached}")
            return SearcherResp(0, "", self, question.value, cached)

        content = render_prompt(self.prompt, question)
        self.logger.debug(f"提问内容:\n{content}")
        try:
            raw_answer = self.__request(question, content)
        except Exception as err:
            self.logger.error(f"请求大模型失败 {err.__class__.__name__}: {err}")
            return SearcherResp(
                -500, f"{err.__class__.__name__}: {err}", self, question.value, None
            )

        self.logger.debug(f"模型原始返回: {raw_answer}")
        # 归一化为 QuestionResolver 可直接匹配的形式
        answer = answer_util.normalize(question, raw_answer)
        if not answer:
            self.logger.warning(f"模型返回无法解析: {raw_answer}")
            return SearcherResp(-404, f"无法解析模型返回: {raw_answer}", self, question.value, None)

        if self.cache is not None:
            if len(self.cache) >= CACHE_LIMIT:
                self.cache.clear()
            self.cache[cache_key] = answer
        self.logger.info(f"作答成功 ({question.type.name}) {question.value} -> {answer}")
        return SearcherResp(0, "", self, question.value, answer)

    @staticmethod
    def __cache_key(question: QuestionModel) -> str:
        """构造缓存 key, 题干相同但选项不同的题目不应复用"""
        if isinstance(question.options, dict):
            options = "|".join(f"{k}={v}" for k, v in question.options.items())
        elif isinstance(question.options, list):
            options = "|".join(question.options)
        else:
            options = ""
        return f"{question.type.value}#{question.value}#{options}"

    def _build_messages(self, question: QuestionModel, content: str) -> list[dict[str, str]]:
        """构造对话消息, 可选附带同题型的单样本示例"""
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.few_shot:
            example_question, example_answer = render_example(self.prompt, question)
            if example_question:
                messages.append({"role": "user", "content": example_question})
                messages.append({"role": "assistant", "content": example_answer})
        messages.append({"role": "user", "content": content})
        return messages

    def _build_params(self, question: QuestionModel, content: str, thinking: bool) -> dict:
        """构造请求参数
        Args:
            question: 题目数据模型
            content: 提问内容
            thinking: 本次请求是否开启深度思考
        """
        params = {
            "model": self.model,
            "messages": self._build_messages(question, content),
            "timeout": self.timeout,  # 兜住思考时长, 超时即中断请求
        }
        extra_body = {}
        if thinking:
            thinking_params, thinking_extra = self.thinking_params()
            params.update(thinking_params)
            extra_body.update(thinking_extra)

        # temperature 留空时自动决定: 思考模式下是否可下发由各服务商声明
        temperature = self.temperature
        if temperature == AUTO:
            temperature = None if thinking and not self.TEMPERATURE_WITH_THINKING else 0.0
        if temperature is not None:
            params["temperature"] = temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        extra_body.update(self.extra_body)  # 用户显式配置的参数优先级最高
        if extra_body:
            params["extra_body"] = extra_body
        return params

    def __request(self, question: QuestionModel, content: str) -> str:
        """请求大模型, 对网络错误与限速做指数退避重试
        模型不支持思考参数或思考超时时, 自动降级为非思考模式重试
        """
        retry = 0
        degrade = 0  # 降级重试次数, 不计入 max_retries
        degrade_thinking = False
        while True:
            thinking = self.thinking and self._thinking_available and not degrade_thinking
            params = self._build_params(question, content, thinking)
            try:
                resp = self.client.chat.completions.create(**params)
                # 思考内容内联在正文里时需要剔除
                answer = answer_util.strip_thinking(resp.choices[0].message.content)
                if not answer and thinking and degrade < 2:
                    # 思考链占满了输出预算, 没留下答案
                    degrade += 1
                    degrade_thinking = True
                    self.logger.warning("思考未产出答案, 改用非思考模式重试")
                    continue
                return answer
            except APITimeoutError:
                # 思考耗时超过 timeout, 先降级为非思考模式再走常规重试
                if thinking and degrade < 2:
                    degrade += 1
                    degrade_thinking = True
                    self.logger.warning(f"思考超时 (>{self.timeout}s), 改用非思考模式重试")
                    continue
                if retry >= self.max_retries:
                    raise
                retry += 1
                self.__backoff(retry, "APITimeoutError")
            except BadRequestError as err:
                # 模型不支持思考参数, 关闭后重试, 并记住不再下发
                if thinking and degrade < 2:
                    degrade += 1
                    self._thinking_available = False
                    self.logger.warning(f"模型不支持思考参数, 已关闭深度思考: {err}")
                    continue
                raise
            except (APIConnectionError, RateLimitError, InternalServerError) as err:
                if retry >= self.max_retries:
                    raise
                retry += 1
                self.__backoff(retry, err.__class__.__name__)
            except APIStatusError as err:
                # 4xx 多为鉴权/参数错误, 重试无意义
                if err.status_code < 500 or retry >= self.max_retries:
                    raise
                retry += 1
                self.__backoff(retry, err.__class__.__name__)

    def __backoff(self, retry: int, reason: str) -> None:
        """重试前的指数退避"""
        delay = 2.0 ** (retry - 1)
        self.logger.warning(f"请求失败 ({reason}), {delay}s 后重试 {retry}/{self.max_retries}")
        time.sleep(delay)
