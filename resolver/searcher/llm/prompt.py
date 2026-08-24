"""提问侧: 提示词模板、各题型的作答格式要求与单样本示例

与服务商无关, 只关心怎么把题目问清楚、怎么让模型输出可解析的答案。
"""

from cxapi.schema import QuestionModel, QuestionType

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


class SafeFormatDict(dict):
    """format 用字典, 模板中出现未知字段时以空串填充, 避免用户模板写错直接抛 KeyError"""

    def __missing__(self, key: str) -> str:
        return ""


def format_requirement(question: QuestionModel) -> str:
    """取该题型的作答格式要求"""
    return FORMAT_REQUIREMENTS.get(question.type, DEFAULT_FORMAT_REQUIREMENT)


def format_options(question: QuestionModel) -> str:
    """将选项转换为模型易读的形式"""
    # 选择题: dict 形式的选项
    if isinstance(question.options, dict):
        return "选项：\n" + "".join(f"{k}. {v}\n" for k, v in question.options.items())
    # 填空题: list 形式的填空项, 只需告知空的数量
    if isinstance(question.options, list) and question.options:
        return f"本题共 {len(question.options)} 个空\n"
    # 判断题等无选项题型
    return ""


def render_prompt(template: str, question: QuestionModel) -> str:
    """渲染提问内容 (题干 + 选项 + 该题型的作答格式要求)"""
    rendered = template.format_map(
        SafeFormatDict(
            type=question.type.name,
            value=question.value,
            options=format_options(question),
        )
    )
    return f"{rendered.rstrip()}\n{format_requirement(question)}"


def render_example(template: str, question: QuestionModel) -> tuple[str, str]:
    """渲染该题型的单样本示例, 返回 (提问, 作答); 无示例时返回 (None, None)"""
    if not (example := FEW_SHOT_EXAMPLES.get(question.type)):
        return None, None
    example_question, example_answer = example
    content = template.format_map(SafeFormatDict(**example_question)).rstrip()
    return f"{content}\n{format_requirement(question)}", example_answer
