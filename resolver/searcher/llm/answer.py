"""作答侧: 把模型返回归一化为 QuestionResolver 能直接匹配的答案形式

与服务商无关, 只关心学习通这边的题型约定:
单选题为选项原文, 多选题为 `#` 分隔且按选项顺序的选项原文,
判断题为 `正确`/`错误`, 填空题为 `#` 分隔的各空答案。
"""

import difflib
import re
from typing import Optional

from cxapi.schema import QuestionModel, QuestionType

# 选项字母前缀, 如 "A." "(B)" "C、"
PATT_OPTION_KEY = re.compile(r"^\s*[（(\[【]?([A-Za-z])[)）\]】.、,，:：。;；]?(?=\s|$)")

# 填空题答案行首的空号, 如 "1." "第2空:" "③"
PATT_BLANK_PREFIX = re.compile(r"^\s*(?:第\s*\d+\s*[空题]?\s*[).、:：]?|\d+\s*[).、:：]|[①-⑳])\s*")

# 内联思考块, 部分本地模型/中转站不走 reasoning_content 字段而是直接混在正文里
PATT_THINK_BLOCK = re.compile(r"<(think|thinking|thought)>.*?(</\1>|$)", re.S | re.I)

# 判断题否定/肯定表述
PATT_FALSE = re.compile(r"(错误|不对|不正确|错|否|false|×|✗)", re.I)
PATT_TRUE = re.compile(r"(正确|对|是|true|√|✓)", re.I)


def strip_thinking(text: str) -> str:
    """剔除正文中内联的思考块 (含被思考预算截断的未闭合标签)"""
    return PATT_THINK_BLOCK.sub("", text or "").strip()


def normalize(question: QuestionModel, raw_answer: str) -> str:
    """将模型返回归一化为 QuestionResolver 能匹配的答案形式
    Args:
        question: 题目数据模型
        raw_answer: 模型返回的原始文本
    Returns:
        str: 无法解析时返回空串
    """
    raw_answer = raw_answer.strip()
    if not raw_answer:
        return ""
    match question.type:
        case QuestionType.单选题:
            return match_option(question.options, raw_answer) or raw_answer
        case QuestionType.多选题:
            return match_multi_options(question.options, raw_answer)
        case QuestionType.判断题:
            # 先判否定表述, 避免 "不正确" 被识别为 "正确"
            if PATT_FALSE.search(raw_answer):
                return "错误"
            if PATT_TRUE.search(raw_answer):
                return "正确"
            return ""
        case QuestionType.填空题:
            return split_blanks(question, raw_answer)
        case _:
            return raw_answer


def answerable(question: QuestionModel, text: Optional[str]) -> bool:
    """判断一段答案文本是否真的能用于作答 (能与该题的选项/题型对上)
    比 QuestionResolver.fill() 的匹配更宽松, 因此这里判否时 fill() 必然也填不上,
    用于识别"题库虽然返回了内容, 但根本用不了"的情况
    """
    if not text or not text.strip():
        return False
    match question.type:
        case QuestionType.单选题:
            return match_option(question.options, text) is not None
        case QuestionType.多选题:
            return bool(match_multi_options(question.options, text))
        case QuestionType.判断题:
            return normalize(question, text) in ("正确", "错误")
        case QuestionType.填空题:
            return bool(split_blanks(question, text))
        case _:
            return True


def match_option(options: dict[str, str], text: str) -> Optional[str]:
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


def match_multi_options(options: dict[str, str], text: str) -> str:
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
            if value := match_option(options, part):
                hit_keys.add(value2key[value])

    # 兜底: 整段文本中出现过的选项原文
    if not hit_keys:
        hit_keys = {k for k, v in options.items() if v and v in text}

    return "#".join(options[k] for k in options if k in hit_keys)


def split_blanks(question: QuestionModel, text: str) -> str:
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
