"""config.yml 可视化配置编辑器

在控制台里以菜单方式增删改 config.yml, 保存时保留原文件的注释与格式。

入口:
    poetry run python3 main.py --config   # 从主程序进入
    poetry run python3 config_editor.py   # 直接运行 (config.yml 缺失或写坏时也能用)
"""

import inspect
import shutil
import sys
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

CONFIG_PATH = Path("config.yml")

# 需要打码显示的字段 (含以下子串的 key)
SECRET_KEYS = ("api_key", "token", "passwd", "password", "secret")


@dataclass
class Field:
    """一个可编辑的配置项"""

    key: str  # 点分路径, 如 "work.wait"
    name: str  # 中文名
    kind: str  # bool / int / float / str / choice
    help: str = ""  # 说明
    choices: tuple[str, ...] = ()  # kind 为 choice 时的可选值
    nullable: bool = False  # 是否允许留空 (null)
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class Section:
    """一组配置项"""

    name: str
    fields: list[Field] = field(default_factory=list)


SECTIONS: list[Section] = [
    Section(
        "基本配置",
        [
            Field("multi_session", "多会话模式", "bool", "是否允许保存并切换多个账号"),
            Field("mask_acc", "账号打码", "bool", "TUI 中隐藏姓名与手机号"),
            Field("tui_max_height", "TUI 最大高度", "int", "留空为自适应高度", nullable=True, minimum=10),
            Field("fetch_uploaded_face", "拉取已上传人脸", "bool", "人脸识别时尝试复用已上传的图片"),
        ],
    ),
    Section(
        "路径配置",
        [
            Field("session_path", "会话存档路径", "str"),
            Field("log_path", "日志文件路径", "str"),
            Field("export_path", "试题导出路径", "str"),
            Field("face_image_path", "人脸图片路径", "str"),
        ],
    ),
    Section(
        "视频任务",
        [
            Field("video.enable", "使能", "bool"),
            Field("video.wait", "完成等待时间", "int", "单位秒, 防风控", minimum=0),
            Field("video.speed", "播放倍速", "float", "过高易触发风控", minimum=0.1, maximum=16.0),
            Field("video.report_rate", "播放汇报率", "int", "没事别改", minimum=1),
        ],
    ),
    Section(
        "章节测验",
        [
            Field("work.enable", "使能", "bool"),
            Field("work.export", "导出试题", "bool", "配合 enable=false 可只导出不作答"),
            Field("work.wait", "完成等待时间", "int", "单位秒", minimum=0),
            Field("work.fallback_fuzzer", "未匹配随机选", "bool", "答案匹配失败时随机作答"),
            Field("work.fallback_save", "失败时保存", "bool", "作答失败后暂存已答题目"),
        ],
    ),
    Section(
        "文档任务",
        [
            Field("document.enable", "使能", "bool"),
            Field("document.wait", "完成等待时间", "int", "单位秒", minimum=0),
        ],
    ),
    Section(
        "课程考试",
        [
            Field("exam.fallback_fuzzer", "未匹配随机选", "bool"),
            Field("exam.persubmit_delay", "提交前延迟", "int", "单位秒", minimum=0),
            Field("exam.confirm_submit", "交卷需确认", "bool", "false 为自动交卷"),
        ],
    ),
    Section(
        "搜索器调度",
        [
            Field("searcher_policy.parallel", "并行请求", "bool", "同一组内的搜索器并行调用"),
            Field("searcher_policy.max_workers", "并行线程上限", "int", minimum=1, maximum=64),
            Field("searcher_policy.prefer_bank", "题库优先", "bool", "题库命中则不请求 AI"),
            Field("searcher_policy.ai_consensus", "AI 交叉对比", "bool", "多个 AI 的答案投票"),
            Field("searcher_policy.ai_min_votes", "共识票数", "int", "几个 AI 一致才采信", minimum=1),
            Field(
                "searcher_policy.ai_fallback",
                "无共识处理",
                "choice",
                "first=用第一个结果, none=放弃作答",
                choices=("first", "none"),
            ),
        ],
    ),
]

# 搜索器类型 -> 说明, 用于添加搜索器时的选单
SEARCHER_HINTS = {
    "JsonFileSearcher": "本地 JSON 题库",
    "SqliteSearcher": "本地 SQLite 题库",
    "RestApiSearcher": "通用 REST API 题库",
    "JsonApiSearcher": "JSON API 题库",
    "EnncySearcher": "Enncy 题库",
    "CxSearcher": "网课小工具 (Go题)",
    "TiKuHaiSearcher": "题库海",
    "LyCk6Searcher": "冷月题库",
    "MukeSearcher": "Muke 题库",
    "LemonSearcher": "柠檬题库",
    "OpenAISearcher": "OpenAI / 任意兼容接口",
    "DeepSeekSearcher": "DeepSeek",
    "MoonshotSearcher": "月之暗面 Kimi",
    "QwenSearcher": "通义千问 (百炼)",
    "ZhipuSearcher": "智谱 GLM",
    "SiliconFlowSearcher": "硅基流动",
    "ArkSearcher": "火山方舟 (豆包)",
    "GeminiSearcher": "Google Gemini",
    "OllamaSearcher": "本地 Ollama",
}


def load_searcher_types() -> dict[str, Optional[type]]:
    """取全部搜索器类型 (类不可导入时值为 None, 仍可手工填字段)"""
    try:
        from resolver.question import SEARCHERS

        return dict(SEARCHERS)
    except Exception:  # config.yml 有问题时 resolver 无法导入, 退化为仅按名称配置
        return {name: None for name in SEARCHER_HINTS}


def is_secret(key: str) -> bool:
    """该字段是否为密钥类 (输入时隐藏回显, 展示时打码)"""
    return any(secret in key.lower() for secret in SECRET_KEYS)


def mask_secret(key: str, value: Any) -> str:
    """打码展示密钥类字段"""
    text = str(value)
    if is_secret(key) and len(text) > 4:
        return f"{text[:4]}{'*' * min(len(text) - 4, 12)}"
    return text


class ConfigEditor:
    """config.yml 配置编辑器"""

    console: Console
    path: Path
    doc: CommentedMap
    dirty: bool  # 是否有未保存的改动
    crlf: bool  # 原文件是否为 CRLF 换行

    def __init__(self, path: Path = CONFIG_PATH, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self.path = Path(path)
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.width = 4096  # 避免长字符串被折行
        self.yaml.indent(mapping=2, sequence=4, offset=2)  # 与配置文件里的示例缩进保持一致
        self.dirty = False
        self.crlf = False
        self.doc = self.__load()

    # ---- 读写 ----

    def __load(self) -> CommentedMap:
        """载入配置文件, 不存在时以空配置开始"""
        if not self.path.is_file():
            self.console.print(f"[yellow]{self.path} 不存在, 将新建配置")
            return CommentedMap()
        raw = self.path.read_bytes().decode("utf8")
        self.crlf = "\r\n" in raw
        # 统一按 LF 解析: 留着 \r 会被 ruamel 当成注释内容, 回写时多出空行
        doc = self.yaml.load(raw.replace("\r\n", "\n"))
        return doc if isinstance(doc, CommentedMap) else CommentedMap()

    def dumps(self) -> str:
        """序列化当前配置 (保留注释与原换行风格)"""
        buf = StringIO()
        self.yaml.dump(self.doc, buf)
        text = buf.getvalue()
        return text.replace("\n", "\r\n") if self.crlf else text

    def save(self) -> None:
        """写回配置文件, 覆盖前备份为 config.yml.bak"""
        if self.path.is_file():
            backup = self.path.with_suffix(self.path.suffix + ".bak")
            shutil.copyfile(self.path, backup)
            self.console.print(f"[dim]原配置已备份至 {backup}")
        self.path.write_bytes(self.dumps().encode("utf8"))
        self.dirty = False
        self.console.print(f"[green]已保存到 {self.path}")

    # ---- 取值/赋值 ----

    def get(self, key: str, default: Any = None) -> Any:
        """按点分路径取值"""
        node = self.doc
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """按点分路径赋值, 中间层级不存在时自动创建"""
        parts = key.split(".")
        node = self.doc
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = CommentedMap()
            node = node[part]
        node[parts[-1]] = value
        self.dirty = True

    # ---- 交互 ----

    def run(self) -> int:
        """编辑器主循环"""
        self.console.print(
            Panel(
                "[bold]CxKitty 配置编辑器[/]\n"
                f"配置文件: [cyan]{self.path.absolute()}[/]\n"
                "[dim]保存时会保留原文件的注释, 并自动备份为 config.yml.bak",
                border_style="green",
            )
        )
        while True:
            self.__show_menu()
            choices = [str(i) for i in range(1, len(SECTIONS) + 2)] + ["s", "p", "v", "q"]
            action = Prompt.ask(
                "请选择", choices=choices, show_choices=False, default="q", console=self.console
            )
            if action == "q":
                if self.dirty and Confirm.ask(
                    "[yellow]有未保存的改动, 是否保存?", default=True, console=self.console
                ):
                    self.save()
                self.console.print("[green]已退出配置编辑器")
                return 0
            if action == "s":
                self.save()
            elif action == "v":
                self.__validate()
            elif action == "p":
                self.console.print(Panel(self.dumps(), title="config.yml", border_style="blue"))
            elif action == str(len(SECTIONS) + 1):
                self.__edit_searchers()
            else:
                self.__edit_section(SECTIONS[int(action) - 1])

    def __show_menu(self) -> None:
        """展示主菜单"""
        table = Table(title="配置分区", title_style="bold", border_style="blue")
        table.add_column("序号", justify="right")
        table.add_column("分区")
        table.add_column("包含配置项")
        for index, section in enumerate(SECTIONS, 1):
            table.add_row(str(index), section.name, "、".join(f.name for f in section.fields))
        searchers = self.get("searchers") or []
        table.add_row(
            str(len(SECTIONS) + 1),
            "题库 / AI 答题器",
            f"当前已配置 {len(searchers)} 个" if searchers else "[red]尚未配置",
        )
        self.console.print(table)
        self.console.print(
            "[dim]输入序号编辑分区, [/][cyan]s[/][dim]=保存, [/][cyan]p[/][dim]=预览, [/]"
            "[cyan]v[/][dim]=校验, [/]"
            f"[cyan]q[/][dim]=退出{' [yellow](有未保存改动)' if self.dirty else ''}"
        )

    def __validate(self) -> None:
        """校验当前配置: 调度策略是否合法、搜索器能否按当前参数实例化"""
        problems: list[str] = []

        # 调度策略
        try:
            from resolver.searcher import SearcherPolicy

            SearcherPolicy(**(self.get("searcher_policy") or {}))
            self.console.print("[green]√[/] 搜索器调度策略合法")
        except ImportError:
            self.console.print("[yellow]![/] 未能导入 resolver, 跳过调度策略校验")
        except Exception as err:
            problems.append(f"searcher_policy: {err}")

        # 必填路径
        for key in ("export_path", "face_image_path"):
            if not self.get(key):
                problems.append(f"{key} 未配置 (程序启动时会直接报错)")

        # 搜索器
        searchers = self.get("searchers") or []
        if not searchers:
            problems.append("尚未配置任何题库或 AI 答题器, 自动答题无法工作")
        types = load_searcher_types()
        for index, item in enumerate(searchers, 1):
            typename = item.get("type", "")
            typename = typename[:1].upper() + typename[1:]
            if typename not in types:
                problems.append(f"第 {index} 个搜索器: 未知类型 {item.get('type')}")
                continue
            cls = types[typename]
            if cls is None:
                continue
            params = {k: v for k, v in item.items() if k != "type"}
            try:
                cls(**params)
                self.console.print(f"[green]√[/] 第 {index} 个搜索器 {typename} 配置可用")
            except Exception as err:
                problems.append(f"第 {index} 个搜索器 {typename}: {err}")

        if problems:
            self.console.print(
                Panel(
                    "\n".join(f"[red]×[/] {p}" for p in problems), title="发现问题", border_style="red"
                )
            )
        else:
            self.console.print("[green]配置校验通过")

    def __edit_section(self, section: Section) -> None:
        """编辑一个分区"""
        while True:
            table = Table(title=section.name, title_style="bold", border_style="blue")
            table.add_column("序号", justify="right")
            table.add_column("配置项")
            table.add_column("当前值", style="cyan")
            table.add_column("说明", style="dim")
            for index, item in enumerate(section.fields, 1):
                value = self.get(item.key)
                shown = "[red]未配置" if value is None and not item.nullable else str(value)
                table.add_row(str(index), item.name, shown, item.help)
            self.console.print(table)

            choices = [str(i) for i in range(1, len(section.fields) + 1)] + ["b"]
            action = Prompt.ask(
                "选择要修改的配置项 (b=返回)",
                choices=choices,
                show_choices=False,
                default="b",
                console=self.console,
            )
            if action == "b":
                return
            item = section.fields[int(action) - 1]
            if (value := self.__ask_value(item)) is not ...:
                self.set(item.key, value)
                self.console.print(f"[green]{item.name} -> {value}")

    def __ask_value(self, item: Field) -> Any:
        """按类型询问新值, 返回 ... 表示放弃修改"""
        current = self.get(item.key)
        match item.kind:
            case "bool":
                return Confirm.ask(f"{item.name}", default=bool(current), console=self.console)
            case "choice":
                return Prompt.ask(
                    f"{item.name}",
                    choices=list(item.choices),
                    default=str(current or item.choices[0]),
                    console=self.console,
                )
            case "int" | "float":
                cast: Callable[[str], Any] = int if item.kind == "int" else float
                while True:
                    tip = f"{item.name}" + (" (输入 null 置空)" if item.nullable else "")
                    raw = Prompt.ask(
                        tip, default="" if current is None else str(current), console=self.console
                    ).strip()
                    if item.nullable and raw.lower() in ("", "null", "none", "~"):
                        return None
                    if not raw:
                        self.console.print("[red]该项不能为空")
                        continue
                    try:
                        value = cast(raw)
                    except ValueError:
                        self.console.print(f"[red]请输入合法的{'整数' if item.kind == 'int' else '数字'}")
                        continue
                    if item.minimum is not None and value < item.minimum:
                        self.console.print(f"[red]不能小于 {item.minimum}")
                        continue
                    if item.maximum is not None and value > item.maximum:
                        self.console.print(f"[red]不能大于 {item.maximum}")
                        continue
                    return value
            case _:
                return Prompt.ask(
                    f"{item.name}",
                    default="" if current is None else str(current),
                    console=self.console,
                )

    # ---- 搜索器 ----

    def __searchers(self) -> CommentedSeq:
        """取搜索器列表, 不存在时创建"""
        if not isinstance(self.doc.get("searchers"), list):
            self.doc["searchers"] = CommentedSeq()
        return self.doc["searchers"]

    def __edit_searchers(self) -> None:
        """题库 / AI 答题器的增删改"""
        while True:
            searchers = self.__searchers()
            table = Table(title="题库 / AI 答题器", title_style="bold", border_style="blue")
            table.add_column("序号", justify="right")
            table.add_column("类型")
            table.add_column("配置", style="cyan")
            for index, item in enumerate(searchers, 1):
                typename = item.get("type", "?")
                params = "  ".join(
                    f"{k}={mask_secret(k, v)}" for k, v in item.items() if k != "type"
                )
                table.add_row(
                    str(index), f"{typename} [dim]{SEARCHER_HINTS.get(typename, '')}", params
                )
            if not searchers:
                table.add_row("-", "[red]尚未配置", "[red]至少需要一个题库或 AI 答题器才能自动答题")
            self.console.print(table)

            action = Prompt.ask(
                "a=添加  e=编辑  d=删除  b=返回",
                choices=["a", "e", "d", "b"],
                show_choices=False,
                default="b",
                console=self.console,
            )
            if action == "b":
                return
            if action == "a":
                self.__add_searcher()
            elif not searchers:
                self.console.print("[red]当前没有可操作的搜索器")
            elif action == "e":
                index = self.__ask_index(len(searchers), "编辑")
                if index is not None:
                    self.__edit_searcher(searchers[index])
            elif action == "d":
                index = self.__ask_index(len(searchers), "删除")
                if index is not None and Confirm.ask(
                    f"确认删除 {searchers[index].get('type')}?", default=False, console=self.console
                ):
                    searchers.pop(index)
                    self.dirty = True
                    self.console.print("[green]已删除")

    def __ask_index(self, total: int, action: str) -> Optional[int]:
        """询问要操作的搜索器序号"""
        raw = Prompt.ask(f"要{action}的序号 (留空取消)", default="", console=self.console)
        if not raw.strip():
            return None
        if not raw.strip().isdigit() or not 1 <= int(raw) <= total:
            self.console.print("[red]序号无效")
            return None
        return int(raw) - 1

    def __add_searcher(self) -> None:
        """添加一个搜索器: 选类型 -> 按构造参数逐项填写"""
        types = load_searcher_types()
        names = list(types)
        table = Table(title="可用的题库 / AI 答题器", title_style="bold", border_style="blue")
        table.add_column("序号", justify="right")
        table.add_column("类型")
        table.add_column("说明", style="dim")
        for index, name in enumerate(names, 1):
            table.add_row(str(index), name, SEARCHER_HINTS.get(name, ""))
        self.console.print(table)

        raw = Prompt.ask("选择类型的序号 (留空取消)", default="", console=self.console)
        if not raw.strip().isdigit() or not 1 <= int(raw) <= len(names):
            return
        typename = names[int(raw) - 1]
        item = CommentedMap({"type": typename})
        self.console.print(f"[dim]逐项填写 {typename} 的参数, 留空即使用默认值 (标 * 的为必填)")
        for name, param in self.__params(types[typename]).items():
            required = param.default is inspect.Parameter.empty
            hint = f"{'*' if required else ''}{name}"
            if not required and param.default is not None:
                hint += f" (默认 {param.default})"
            while True:
                raw = Prompt.ask(hint, default="", password=is_secret(name), console=self.console)
                if not raw.strip():
                    if required:
                        self.console.print("[red]该参数必填")
                        continue
                    break
                item[name] = self.__parse_scalar(raw.strip())
                break
        self.__searchers().append(item)
        self.dirty = True
        self.console.print(f"[green]已添加 {typename}")

    def __edit_searcher(self, item: CommentedMap) -> None:
        """编辑单个搜索器的字段"""
        while True:
            keys = [k for k in item if k != "type"]
            table = Table(title=f"{item.get('type')} 的参数", title_style="bold", border_style="blue")
            table.add_column("序号", justify="right")
            table.add_column("参数")
            table.add_column("值", style="cyan")
            for index, key in enumerate(keys, 1):
                table.add_row(str(index), key, mask_secret(key, item[key]))
            self.console.print(table)

            action = Prompt.ask(
                "输入序号修改  a=新增参数  d=删除参数  b=返回",
                choices=[*(str(i) for i in range(1, len(keys) + 1)), "a", "d", "b"],
                show_choices=False,
                default="b",
                console=self.console,
            )
            if action == "b":
                return
            if action == "a":
                params = self.__params(load_searcher_types().get(item.get("type")))
                if params:
                    self.console.print(f"[dim]可用参数: {', '.join(params)}")
                name = Prompt.ask("参数名 (留空取消)", default="", console=self.console).strip()
                if not name:
                    continue
                value = Prompt.ask(
                    f"{name} 的值", default="", password=is_secret(name), console=self.console
                )
                item[name] = self.__parse_scalar(value.strip())
                self.dirty = True
            elif action == "d":
                index = self.__ask_index(len(keys), "删除")
                if index is not None:
                    del item[keys[index]]
                    self.dirty = True
            else:
                key = keys[int(action) - 1]
                value = Prompt.ask(
                    f"{key} 的值",
                    default="" if is_secret(key) else str(item[key]),
                    password=is_secret(key),
                    console=self.console,
                )
                if value.strip():
                    item[key] = self.__parse_scalar(value.strip())
                    self.dirty = True

    @staticmethod
    def __params(cls: Optional[type]) -> dict[str, inspect.Parameter]:
        """取搜索器构造函数的参数表 (类不可用时返回空)"""
        if cls is None:
            return {}
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            return {}
        return {
            name: param
            for name, param in signature.parameters.items()
            if name != "self"
            and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        }

    @staticmethod
    def __parse_scalar(raw: str) -> Any:
        """把输入的字符串转成合适的 YAML 标量"""
        if raw.lower() in ("null", "none", "~"):
            return None
        if raw.lower() in ("true", "yes", "on"):
            return True
        if raw.lower() in ("false", "no", "off"):
            return False
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw


def main(path: Path = CONFIG_PATH) -> int:
    """配置编辑器入口"""
    console = Console()
    try:
        return ConfigEditor(path, console).run()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]已取消, 未保存的改动已丢弃")
        return 1
    except Exception as err:
        console.print(f"[red]配置编辑器异常: {err.__class__.__name__}: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
