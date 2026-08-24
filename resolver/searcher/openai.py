import difflib
import re
import time
from dataclasses import dataclass
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

from cxapi.schema import QuestionModel, QuestionType
from logger import Logger

from . import SearcherBase, SearcherResp


@dataclass
class Provider:
    """大模型服务商预设"""

    base_url: str  # OpenAI 兼容接口地址
    model: Optional[str] = None  # 默认模型, 为 None 表示必须由用户指定
    need_key: bool = True  # 是否需要 api_key (本地推理服务不需要)
    console: str = ""  # 控制台/文档地址, 用于错误提示
    # 开启深度思考的参数风格, 各服务商互不兼容:
    #   effort         顶层 reasoning_effort 字段 (OpenAI / DeepSeek / Kimi / Ollama 等)
    #   gemini         顶层 reasoning_effort + google.thinking_config 思考预算
    #   enable_thinking  extra_body 的 enable_thinking + thinking_budget (百炼 / 硅基流动 等)
    #   thinking_type  extra_body 的 thinking.type (智谱 / 火山方舟 等)
    thinking_style: Optional[str] = "effort"
    max_effort: str = "high"  # 该服务商支持的最高思考档位


# 常见大模型服务商预设, 均为 OpenAI 兼容接口
# 默认模型仅为开箱可用的建议值, 服务商下线模型后按 config.yml 的 model 字段覆盖即可
PROVIDERS: dict[str, Provider] = {
    "openai": Provider(
        "https://api.openai.com/v1/",
        "gpt-4o-mini",
        console="https://platform.openai.com/docs/models",
    ),
    "deepseek": Provider(
        "https://api.deepseek.com/v1/",
        "deepseek-v4-flash",
        console="https://platform.deepseek.com",
        max_effort="max",
    ),
    "moonshot": Provider(
        "https://api.moonshot.cn/v1/",
        "moonshot-v1-8k",
        console="https://platform.moonshot.cn",
        max_effort="max",
    ),
    "qwen": Provider(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/",
        "qwen-plus",
        console="https://bailian.console.aliyun.com",
        thinking_style="enable_thinking",
    ),
    "zhipu": Provider(
        "https://open.bigmodel.cn/api/paas/v4/",
        "glm-4-flash",
        console="https://open.bigmodel.cn",
        thinking_style="thinking_type",
    ),
    "siliconflow": Provider(
        "https://api.siliconflow.cn/v1/",
        "Qwen/Qwen2.5-7B-Instruct",
        console="https://cloud.siliconflow.cn",
        thinking_style="enable_thinking",
    ),
    "ark": Provider(
        # 火山方舟 (豆包) 的 model 为推理接入点 id (ep-xxx), 无法预设
        "https://ark.cn-beijing.volces.com/api/v3/",
        console="https://console.volcengine.com/ark",
        thinking_style="thinking_type",
    ),
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.0-flash",
        console="https://ai.google.dev/gemini-api/docs/openai",
        thinking_style="gemini",
    ),
    "ollama": Provider(
        "http://localhost:11434/v1/",
        need_key=False,
        console="https://ollama.com/library",
    ),
}

# 未配置 temperature 时的取值: 思考模式下不下发 (部分服务商的思考模式不接受该参数),
# 非思考模式下用 0 以保证作答稳定
AUTO = "auto"

# 默认提示词
DEFAULT_SYSTEM_PROMPT = """你是一位答题专家, 只输出答案本身, 不输出解析、推理过程和多余的标点。
严格按照用户给出的作答格式要求回答, 不确定时也必须给出最可能的答案, 不允许拒答。"""
DEFAULT_PROMPT = "请回答这个{type}：\n{value}\n{options}"

# 各题型的作答格式要求, 追加在提问末尾, 引导模型输出可被解析的答案
FORMAT_REQUIREMENTS = {
    QuestionType.单选题: "作答格式: 只回复唯一正确选项的字母, 例如: B",
    QuestionType.多选题: "作答格式: 只回复全部正确选项的字母, 按字母顺序连写, 例如: ABD",
    QuestionType.判断题: "作答格式: 只回复 正确 或 错误",
    QuestionType.填空题: "作答格式: 按空的顺序作答, 每空独占一行, 每行只写该空的答案, 不写空号",
}
DEFAULT_FORMAT_REQUIREMENT = "作答格式: 直接给出答案正文, 不要解释"

# 各题型的单样本示例 (question, answer), 用于引导模型输出格式
FEW_SHOT_EXAMPLES = {
    QuestionType.单选题: (
        {
            "type": "单选题",
            "value": "We didn't have health____ at the time and my parents couldn't pay for the treatment.",
            "options": "选项：\nA. assurance\nB. insurance\nC. requirement\nD. issure\n",
        },
        "B",
    ),
    QuestionType.多选题: (
        {
            "type": "多选题",
            "value": "下列属于计算机输入设备的有?",
            "options": "选项：\nA. 键盘\nB. 鼠标\nC. 显示器\nD. 扫描仪\n",
        },
        "ABD",
    ),
    QuestionType.判断题: (
        {
            "type": "判断题",
            "value": "计算机的运算速度通常用 MIPS 来衡量。",
            "options": "",
        },
        "正确",
    ),
    QuestionType.填空题: (
        {
            "type": "填空题",
            "value": "中国的首都是____, 最大的岛屿是____。",
            "options": "本题共 2 个空\n",
        },
        "北京\n台湾岛",
    ),
}

# 选项字母前缀, 如 "A." "(B)" "C、"
PATT_OPTION_KEY = re.compile(r"^\s*[（(\[【]?([A-Za-z])[)）\]】.、,，:：。;；]?(?=\s|$)")

# 填空题答案行首的空号, 如 "1." "第2空:" "③"
PATT_BLANK_PREFIX = re.compile(r"^\s*(?:第\s*\d+\s*[空题]?\s*[).、:：]?|\d+\s*[).、:：]|[①-⑳])\s*")

# 内联思考块, 部分本地模型/中转站不走 reasoning_content 字段而是直接混在正文里
PATT_THINK_BLOCK = re.compile(r"<(think|thinking|thought)>.*?(</\1>|$)", re.S | re.I)

# 判断题否定/肯定表述
PATT_FALSE = re.compile(r"(错误|不对|不正确|错|否|false|×|✗)", re.I)
PATT_TRUE = re.compile(r"(正确|对|是|true|√|✓)", re.I)


class SafeFormatDict(dict):
    """format 用字典, 模板中出现未知字段时以空串填充, 避免用户模板写错直接抛 KeyError"""

    def __missing__(self, key: str) -> str:
        return ""


class OpenAISearcher(SearcherBase):
    """大模型在线答题器 (OpenAI 兼容接口)"""

    PROVIDER = "openai"  # 子类通过覆盖该字段即可派生出各服务商的答题器

    client: OpenAI
    preset: Provider
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
        provider: Optional[str] = None,  # 服务商预设, 见 PROVIDERS
        base_url: Optional[str] = None,  # 留空则取服务商预设地址
        model: Optional[str] = None,  # 留空则取服务商预设模型
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
    ) -> None:
        super().__init__()
        name = (provider or self.PROVIDER).lower()
        if name not in PROVIDERS:
            raise ValueError(f"未知的大模型服务商 {name}, 可用: {', '.join(PROVIDERS)}")
        preset = PROVIDERS[name]
        if preset.need_key and not api_key:
            raise ValueError(f"{name} 需要配置 api_key, 请前往 {preset.console} 获取")
        if not (model := model or preset.model):
            raise ValueError(f"{name} 未预设默认模型, 请在 config.yml 指定 model ({preset.console})")

        self.logger = Logger(f"{self.__class__.__name__}")
        # 关闭 SDK 自带重试, 由本类统一控制重试与退避
        self.client = OpenAI(
            api_key=api_key or "EMPTY",  # 本地推理服务不校验 key, 但 SDK 要求非空
            base_url=base_url or preset.base_url,
            max_retries=0,
        )
        self.preset = preset
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.prompt = prompt or DEFAULT_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.few_shot = few_shot
        self.thinking = thinking and preset.thinking_style is not None
        self.thinking_effort = thinking_effort or preset.max_effort
        self.thinking_budget = thinking_budget
        self.extra_body = extra_body or {}
        # 接口拒绝思考参数 (模型不支持) 时置否, 之后不再重复下发
        self.__thinking_available = True
        self.cache = {} if cache else None

    def invoke(self, question: QuestionModel) -> SearcherResp:
        cache_key = self.__cache_key(question)
        if self.cache is not None and (cached := self.cache.get(cache_key)) is not None:
            self.logger.debug(f"命中本地缓存 {question.value} -> {cached}")
            return SearcherResp(0, "", self, question.value, cached)

        content = self.__render_prompt(question)
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
        answer = self.__normalize_answer(question, raw_answer)
        if not answer:
            self.logger.warning(f"模型返回无法解析: {raw_answer}")
            return SearcherResp(-404, f"无法解析模型返回: {raw_answer}", self, question.value, None)

        if self.cache is not None:
            if len(self.cache) >= 1024:  # 长时间运行时避免缓存无限增长
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

    @staticmethod
    def __format_options(question: QuestionModel) -> str:
        """将选项转换为模型易读的形式"""
        # 选择题: dict 形式的选项
        if isinstance(question.options, dict):
            return "选项：\n" + "".join(f"{k}. {v}\n" for k, v in question.options.items())
        # 填空题: list 形式的填空项, 只需告知空的数量
        if isinstance(question.options, list) and question.options:
            return f"本题共 {len(question.options)} 个空\n"
        # 判断题等无选项题型
        return ""

    def __render_prompt(self, question: QuestionModel) -> str:
        """渲染提问内容"""
        rendered = self.prompt.format_map(
            SafeFormatDict(
                type=question.type.name,
                value=question.value,
                options=self.__format_options(question),
            )
        )
        requirement = FORMAT_REQUIREMENTS.get(question.type, DEFAULT_FORMAT_REQUIREMENT)
        return f"{rendered.rstrip()}\n{requirement}"

    def __build_messages(self, question: QuestionModel, content: str) -> list[dict[str, str]]:
        """构造对话消息, 可选附带同题型的单样本示例"""
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.few_shot and (example := FEW_SHOT_EXAMPLES.get(question.type)):
            example_question, example_answer = example
            requirement = FORMAT_REQUIREMENTS.get(question.type, DEFAULT_FORMAT_REQUIREMENT)
            messages.append(
                {
                    "role": "user",
                    "content": self.prompt.format_map(SafeFormatDict(**example_question)).rstrip()
                    + f"\n{requirement}",
                }
            )
            messages.append({"role": "assistant", "content": example_answer})
        messages.append({"role": "user", "content": content})
        return messages

    def __thinking_params(self) -> tuple[dict, dict]:
        """按服务商风格构造开启深度思考的参数
        Returns:
            dict, dict: 顶层参数, extra_body 参数
        """
        budget = self.thinking_budget
        match self.preset.thinking_style:
            case "effort":
                return {"reasoning_effort": self.thinking_effort}, {}
            case "gemini":
                extra = (
                    {"extra_body": {"google": {"thinking_config": {"thinking_budget": budget}}}}
                    if budget
                    else {}
                )
                return {"reasoning_effort": self.thinking_effort}, extra
            case "enable_thinking":
                extra = {"enable_thinking": True}
                if budget:
                    extra["thinking_budget"] = budget
                return {}, extra
            case "thinking_type":
                return {}, {"thinking": {"type": "enabled"}}
            case _:
                return {}, {}

    def __build_params(self, question: QuestionModel, content: str, thinking: bool) -> dict:
        """构造请求参数
        Args:
            question: 题目数据模型
            content: 提问内容
            thinking: 本次请求是否开启深度思考
        """
        params = {
            "model": self.model,
            "messages": self.__build_messages(question, content),
            "timeout": self.timeout,  # 兜住思考时长, 超时即中断请求
        }
        extra_body = {}
        if thinking:
            thinking_params, thinking_extra = self.__thinking_params()
            params.update(thinking_params)
            extra_body.update(thinking_extra)

        # temperature 留空时自动决定: 思考模式下不下发 (部分服务商的思考模式不接受该参数)
        temperature = self.temperature
        if temperature == AUTO:
            temperature = None if thinking else 0.0
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
            thinking = self.thinking and self.__thinking_available and not degrade_thinking
            params = self.__build_params(question, content, thinking)
            try:
                resp = self.client.chat.completions.create(**params)
                answer = (resp.choices[0].message.content or "").strip()
                # 思考内容内联在正文里时需要剔除
                answer = PATT_THINK_BLOCK.sub("", answer).strip()
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
                    self.__thinking_available = False
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

    def __normalize_answer(self, question: QuestionModel, raw_answer: str) -> str:
        """将模型返回归一化为 QuestionResolver 能匹配的答案形式
        Args:
            question: 题目数据模型
            raw_answer: 模型返回的原始文本
        Returns:
            str: 单选题为选项原文, 多选题为 `#` 分隔的选项原文,
                 判断题为 `正确`/`错误`, 填空题为 `#` 分隔的各空答案
        """
        raw_answer = raw_answer.strip()
        if not raw_answer:
            return ""
        match question.type:
            case QuestionType.单选题:
                return self.__match_option(question.options, raw_answer) or raw_answer
            case QuestionType.多选题:
                return self.__match_multi_options(question.options, raw_answer)
            case QuestionType.判断题:
                # 先判否定表述, 避免 "不正确" 被识别为 "正确"
                if PATT_FALSE.search(raw_answer):
                    return "错误"
                if PATT_TRUE.search(raw_answer):
                    return "正确"
                return ""
            case QuestionType.填空题:
                return self.__split_blanks(question, raw_answer)
            case _:
                return raw_answer

    @staticmethod
    def __match_option(options: dict[str, str], text: str) -> Optional[str]:
        """将一段作答文本匹配到唯一选项, 返回选项原文
        依次尝试: 选项字母 -> 选项原文包含 -> 编辑距离最相似的选项
        """
        if not isinstance(options, dict) or not options:
            return None
        text = text.strip()

        # 以选项字母作答, 如 "B" "B." "(B) insurance"
        if key_match := PATT_OPTION_KEY.match(text):
            key = key_match.group(1).upper()
            if key in options:
                return options[key]

        # 直接给出选项原文
        for value in options.values():
            if value and value in text:
                return value

        # 兜底: 取最相似的选项
        best_value, best_ratio = None, 0.0
        for value in options.values():
            ratio = difflib.SequenceMatcher(a=value, b=text).ratio()
            if ratio > best_ratio:
                best_value, best_ratio = value, ratio
        return best_value if best_ratio >= 0.6 else None

    def __match_multi_options(self, options: dict[str, str], text: str) -> str:
        """解析多选题作答, 返回 `#` 分隔的选项原文 (按选项顺序)"""
        if not isinstance(options, dict) or not options:
            return ""
        hit_keys = set()

        # 纯字母作答, 如 "ABD" "A、B、D" "A,B,D"
        letters = text.strip()
        if letters and not re.search(r"[^A-Za-z\s,，、;；#/和]", letters):
            hit_keys = {c.upper() for c in re.findall(r"[A-Za-z]", letters)} & options.keys()

        # 逐行/逐段作答, 如 "A. 键盘\nB. 鼠标"
        if not hit_keys:
            value2key = {v: k for k, v in options.items()}
            for part in re.split(r"[\n;；#]+", text):
                if not (part := part.strip()):
                    continue
                if value := self.__match_option(options, part):
                    hit_keys.add(value2key[value])

        # 兜底: 整段文本中出现过的选项原文
        if not hit_keys:
            hit_keys = {k for k, v in options.items() if v and v in text}

        return "#".join(options[k] for k in options if k in hit_keys)

    @staticmethod
    def __split_blanks(question: QuestionModel, text: str) -> str:
        """解析填空题作答, 返回 `#` 分隔的各空答案"""
        blanks = [
            stripped
            for line in text.splitlines()
            if (stripped := PATT_BLANK_PREFIX.sub("", line).strip())
        ]
        # 模型把多个空写在一行时, 尝试按分隔符再切一次
        blank_amount = len(question.options) if isinstance(question.options, list) else 0
        if len(blanks) == 1 and blank_amount > 1:
            parts = [part.strip() for part in re.split(r"[#;；、,，]", blanks[0]) if part.strip()]
            if len(parts) == blank_amount:
                blanks = parts
        return "#".join(blanks)


# ---- 常见服务商答题器 ----
# 均为 OpenAI 兼容接口, 仅预设了 base_url 与默认模型,
# 其余配置项与 OpenAISearcher 完全一致, 也可用 OpenAISearcher + provider 字段等价配置


class DeepSeekSearcher(OpenAISearcher):
    """DeepSeek 答题器 https://platform.deepseek.com"""

    PROVIDER = "deepseek"


class MoonshotSearcher(OpenAISearcher):
    """月之暗面 Kimi 答题器 https://platform.moonshot.cn"""

    PROVIDER = "moonshot"


class QwenSearcher(OpenAISearcher):
    """阿里通义千问 (百炼) 答题器 https://bailian.console.aliyun.com"""

    PROVIDER = "qwen"


class ZhipuSearcher(OpenAISearcher):
    """智谱 GLM 答题器 https://open.bigmodel.cn"""

    PROVIDER = "zhipu"


class SiliconFlowSearcher(OpenAISearcher):
    """硅基流动答题器 https://cloud.siliconflow.cn"""

    PROVIDER = "siliconflow"


class ArkSearcher(OpenAISearcher):
    """火山方舟 (豆包) 答题器, model 需填写推理接入点 id https://console.volcengine.com/ark"""

    PROVIDER = "ark"


class GeminiSearcher(OpenAISearcher):
    """Google Gemini 答题器 (OpenAI 兼容端点) https://ai.google.dev/gemini-api/docs/openai"""

    PROVIDER = "gemini"


class OllamaSearcher(OpenAISearcher):
    """本地 Ollama 答题器, 无需 api_key, model 需填写已拉取的模型名 https://ollama.com/library"""

    PROVIDER = "ollama"
