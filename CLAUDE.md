# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概况

CxKitty 是超星学习通的自动化答题/刷课工具, 基于协议模拟 (requests) 而非浏览器自动化, 交互界面用 rich 构建 TUI。
本仓库 fork 自 `SocialSisterYi/CxKitty` (上游仓库现已不可访问, 同类 fork 可作为上游参考)。

## 常用命令

```bash
poetry install                 # 安装依赖 (会创建 venv)
poetry run python3 main.py     # 运行主程序 (必须在仓库根目录运行)

poetry run black .             # 格式化, line-length = 100 (见 pyproject.toml)
poetry run isort .             # import 排序

docker build --tag socialsisteryi/cx-kitty .
```

仓库没有测试套件、CI 测试与 lint 配置, 只有 black / isort 两个开发期工具。改动后的最低验证手段是编译检查与导入自检:

```bash
poetry run python -W error::SyntaxWarning -c "import compileall,re; print(compileall.compile_dir('.', quiet=1, force=True, rx=re.compile(r'(\.venv|\.git)')))"
poetry run python -c "import main, cxapi.exam, cxapi.captcha.image, resolver.question"
```

### 依赖变更流程

依赖三处必须同步, 只改 `pyproject.toml` 会让 `poetry install` 因 lock 不一致而失败:

```bash
poetry lock                    # 或 poetry lock --regenerate 重新解析全部版本
poetry export -f requirements.txt --without-hashes -o requirements.txt
```

- 默认包源是清华镜像 (`[[tool.poetry.source]]`, `priority = "primary"`), `requirements.txt` 首行的 `--index-url` 由 export 自动带出。
- Python 约束为 `>=3.10,<3.14`。`onnxruntime` 被显式限制 `<1.24`: 1.24 起不再提供 cp310 wheel, 放开会让 3.10 装不上 (它由 ddddocr 间接引入)。改动版本上下界时, 应确认锁定版本在 3.10~3.13 全区间都有 wheel。

### 运行前置条件

- `config.yml` 必须存在于当前工作目录: `config.py` 在 import 期读取它, 且 `export_path` / `face_image_path` 无默认值, 缺失会直接抛错。
- `utils.__version__` 通过读取 `pyproject.toml` 文本解析版本号, 因此程序只能从仓库根目录启动。
- 至少配置一个题库后端 (`config.yml` 的 `searchers`), 否则 `load_searcher()` 抛 `AttributeError`。

## 架构

三层结构: **协议层 `cxapi/`** → **决策层 `resolver/`** → **编排与 TUI `main.py` + `dialog.py`**。

### 协议层 cxapi/

- `session.py` `SessionWraper(requests.Session)` 是所有网络请求的唯一出口。它在 `request()` 里做请求重放式风控处理: 响应命中 `SpecialPageType.CAPTCHA` 时用 ddddocr + OpenCV 预处理识别图形验证码并重新发起原请求; 命中 `SpecialPageType.FACE` 时走人脸识别流程 (`face_detection.py`)。TUI 通过 `reg_captcha_before/after`、`reg_face_before/after` 注册回调, 由 `main.py` 提供渲染实现——协议层不直接依赖 rich。
- `utils.py` 生成移动端环境参数与签名: `get_ua()` 产出带 `schild` 签名的客户端 UA (`mobile_ua_sign`), `inf_enc_sign()` 为表单追加 `inf_enc` 签名。学习通接口一旦改签名算法, 改这里。
- `api.py` `ChaoXingAPI`: 登录 (密码 / 二维码)、账号信息、课程列表, 是会话与业务对象的入口。
- `classes.py` → `ClassSelector` 迭代产出待处理对象: `ChapterContainer` (章节) / `ExamDto` (考试) / 考试列表。
- `chapters.py` + `task_point/`: 章节容器按 index 取任务点, 每个任务点 Dto (`PointVideoDto` / `PointDocumentDto` / `PointWorkDto`) 遵循 `fetch_attachment()` → `parse_attachment()` → 执行 的固定生命周期。
- `base.py` `QAQDtoBase` 定义"题目-作答-提交"的公共 trait (`fetch` / `fetch_all` / `submit` / `final_submit` / `fallback_save` / `export`, 以及 `__iter__` / `__next__`)。章节测验 `task_point/work.py` 与课程考试 `exam.py` 都实现它, 这是 resolver 能同时处理两者的原因。
- `exam.py` 额外处理考试特有的准入条件: 考试码、人脸识别、人机验证码 (`need_captcha` → `captcha/image.py` 用 `cv2.matchTemplate` 算滑块位移)、动态 `enc` 校验。
- `schema.py` 用 dataclasses-json 定义题目/账号/导出数据模型; `exception.py` 把接口错误细分成具体异常 (章节未开放、交卷过早、IP 限制等), 上层据此分类处理。

### 决策层 resolver/

- `question.py` `QuestionResolver` 消费任意 `QAQDtoBase` 迭代器: 取题 → 调搜索器 → 用 `difflib` 做选项模糊匹配 → 回填提交, 匹配失败时可按配置走 fuzzer 兜底并把未完成题目导出为 json。
- 搜索器是插件式的: `SearcherBase.invoke(question) -> SearcherResp` 是唯一契约, `question.py` 顶部的 `SEARCHERS` 字典把类名映射到实现, `load_searcher()` 依据 `config.yml` 的 `searchers[].type` 动态实例化 (其余 key 直接作为构造参数)。**新增题库后端 = 实现 `SearcherBase` → 注册进 `SEARCHERS` → 在 `config.yml` 补注释示例 → README 补说明**, 四处缺一会导致配置项无法被识别。
- 现有实现: `json.py` (本地 JSON)、`sqlite.py` (本地库)、`restapi.py` (通用 REST + Enncy/网课小工具/题库海/冷月/Muke/柠檬等第三方)、`openai.py` (OpenAI 兼容大模型在线答题, 只有 `api_key` 必填, 其余配置项均有默认值)。
- `openai.py` 顶部的 `PROVIDERS` 是服务商预设表 (base_url / 默认模型 / 是否需要 key / 思考参数风格 `thinking_style` 与最高档位 `max_effort`), 文件末尾的 `DeepSeekSearcher` 等子类只覆盖一个 `PROVIDER` 字段。**新增服务商 = 往 `PROVIDERS` 加一条 → 加一个子类 → 注册进 `SEARCHERS` → config.yml 与 README 补说明**; 服务商换了默认模型名时只需改 `PROVIDERS`。
- `openai.py` 的关键点是**答案归一化**: 提问时按题型追加作答格式要求, 返回后把选项字母 / `不正确` 之类的表述还原成 `fill()` 能直接匹配的形式 (单选=选项原文, 多选=`#` 连接的选项原文且按选项顺序, 判断=`正确`/`错误`, 填空=`#` 连接各空)。改动 `fill()` 的匹配规则时要同步这里, 否则大模型作答会命中不了。
- 深度思考默认开到各服务商最高档 (`thinking_budget` 限思考 token, `timeout` 限思考时长)。各家参数互不兼容, 因此 `__request` 带**降级重试**: 接口拒绝思考参数 (`BadRequestError`) 时永久关闭思考, 思考超时或思考占满输出预算 (正文为空) 时本次改用非思考模式重试。新增服务商时若拿不准参数, 让降级逻辑兜底即可。
- `media.py` / `document.py` 分别模拟视频播放心跳与文档阅读进度。

### 编排与 TUI

- `main.py` 组织 rich `Layout`, 串起"选会话 → 选课程 → 遍历章节任务点 / 执行考试"的主流程, 并负责任务间的课间等待 (`config.yml` 各任务的 `wait`, 防风控) 与顶层异常兜底。
- 各 Dto 自带 `refresh_tui()` / rich 渲染协议, 由 `main.py` 塞进对应 Layout 分区, 因此协议层对象也持有自己的 `tui_ctx`。
- `dialog.py` 提供交互式选择 (登录、会话、班级、考试)。
- `logger.py` 每个模块一个 `Logger(name)`, 日志按会话手机号写入 `config.LOGS_PATH`, 与 TUI 输出分离——排查问题优先看日志文件而非终端。
- 会话 cookie 以 json 形式持久化在 `config.SESSIONS_PATH` (`utils.sessions_load` / `ck2dict` / `dict2ck`)。
