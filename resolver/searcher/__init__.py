from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Optional

from cxapi.schema import QuestionModel
from logger import Logger

# 交叉对比无共识时的错误码
CODE_NO_CONSENSUS = -409


@dataclass
class SearcherResp:
    """搜索器返回协议"""

    code: int  # 错误代码
    message: str  # 错误信息
    searcher: "SearcherBase"  # 搜索器对象
    question: str  # 源题干信息
    answer: Optional[str]  # 答案
    note: str = ""  # 附加说明 (如交叉对比的票数), 仅用于展示

    def __repr__(self) -> str:
        return f"SearchResp(code={self.code}, message={self.message}, searcher={self.searcher}, question={self.question}, answer={self.answer})"


class SearcherBase:
    """搜索器基类"""

    IS_AI: bool = False  # 是否为大模型答题器, 决定其被分到题库组还是 AI 组

    def invoke(self, question: QuestionModel) -> SearcherResp:
        """搜题器调用接口
        >>> SearchResp(
        >>>     code=0,  # 错误码为 0 表示成功, 否则为失败
        >>>     message=ok,  # 错误信息, 默认为 ok
        >>>     question=题目,
        >>>     answer=答案
        >>> )
        """
        raise NotImplementedError


@dataclass
class SearcherPolicy:
    """搜索器调度策略, 对应 config.yml 的 `searcher_policy`"""

    parallel: bool = True  # 同一组内的搜索器是否并行请求
    max_workers: int = 8  # 并行线程数上限
    prefer_bank: bool = True  # 题库优先: 题库出答案就不再请求 AI
    ai_consensus: bool = True  # 多个 AI 的作答是否交叉对比
    ai_min_votes: int = 2  # 至少多少个 AI 给出相同答案才采信
    ai_fallback: str = "first"  # 无共识时: first=用第一个成功结果, none=放弃作答

    def __post_init__(self) -> None:
        if self.ai_fallback not in ("first", "none"):
            raise ValueError(f'ai_fallback 只能为 "first" 或 "none", 当前为 {self.ai_fallback}')
        if self.ai_min_votes < 1:
            raise ValueError(f"ai_min_votes 至少为 1, 当前为 {self.ai_min_votes}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers 至少为 1, 当前为 {self.max_workers}")


def canonical_answer(question: QuestionModel, answer: Optional[str]) -> str:
    """把答案归一化成可比较的形式, 用于交叉对比与题库命中判定
    (与大模型答题器共用一套归一化规则: 单选=选项原文, 多选=`#` 连接的选项原文,
    判断=`正确`/`错误`, 填空=`#` 连接各空)
    """
    if not answer:
        return ""
    from .llm.answer import normalize  # 延迟导入避免包初始化时的循环引用

    try:
        return normalize(question, answer)
    except Exception:
        return answer.strip()


class MultiSearcherWraper:
    """搜索器封装

    搜索器分为两组: 题库组 (本地库 / 第三方题库 API) 与 AI 组 (大模型答题器)。
    先并行请求题库组, 题库能出答案时直接采用; 题库查不到才并行请求 AI 组,
    并对多个 AI 的作答做交叉对比, 取达到票数要求的答案。
    """

    bank_slot: list[SearcherBase]  # 题库搜索器槽位
    ai_slot: list[SearcherBase]  # 大模型答题器槽位
    policy: SearcherPolicy  # 调度策略
    logger: Logger  # 日志记录器

    def __init__(self, policy: Optional[SearcherPolicy] = None) -> None:
        self.logger = Logger("Searcher")
        self.bank_slot = []
        self.ai_slot = []
        self.policy = policy or SearcherPolicy()

    @property
    def slot(self) -> list[SearcherBase]:
        """全部搜索器 (题库组在前)"""
        return [*self.bank_slot, *self.ai_slot]

    def add(self, searcher: SearcherBase) -> None:
        """添加搜索器, 按类型分流到题库组/AI 组
        Args:
            searcher: 欲添加的搜索器对象
        """
        if not isinstance(searcher, SearcherBase):
            raise TypeError
        if searcher.IS_AI:
            self.ai_slot.append(searcher)
        else:
            self.bank_slot.append(searcher)

    def invoke(self, question: QuestionModel) -> list[SearcherResp]:
        """调用搜索器
        Args:
            question: 题目数据模型
        Returns:
            list[SearchResp]: 搜索器响应列表, 排在前面的结果优先被采用
        """
        if not self.slot:
            raise RuntimeError("至少需要加载一个搜索器")

        results: list[SearcherResp] = []
        if self.bank_slot:
            bank_resp = self.__invoke_group(self.bank_slot, question, "题库")
            results += bank_resp
            # 题库出了可用答案就不再请求 AI, 省下 token
            if self.policy.prefer_bank and self.__has_usable(question, bank_resp):
                self.logger.info("题库已命中, 跳过 AI 答题器")
                self.logger.debug(f"搜索器 Req={question} Rsp={results}")
                return results
            if self.ai_slot:
                self.logger.info("题库未查到可用答案, 转由 AI 答题器作答")

        if self.ai_slot:
            results += self.__cross_check(
                question, self.__invoke_group(self.ai_slot, question, "AI")
            )

        self.logger.info(f"搜索器调用完毕 (共 {len(results)} 个结果)")
        self.logger.debug(f"搜索器 Req={question} Rsp={results}")
        return results

    def __invoke_group(
        self, searchers: list[SearcherBase], question: QuestionModel, group: str
    ) -> list[SearcherResp]:
        """请求一组搜索器, 返回结果 (顺序与配置顺序一致)"""

        def _invoke(searcher: SearcherBase) -> SearcherResp:
            try:
                return searcher.invoke(question)
            except Exception as err:  # 单个搜索器异常不影响其他搜索器
                self.logger.error(f"{searcher.__class__.__name__} 调用异常: {err}")
                return SearcherResp(
                    -500, f"{err.__class__.__name__}: {err}", searcher, question.value, None
                )

        if self.policy.parallel and len(searchers) > 1:
            self.logger.debug(f"并行请求{group}搜索器 (共 {len(searchers)} 个)")
            with ThreadPoolExecutor(
                max_workers=min(self.policy.max_workers, len(searchers))
            ) as pool:
                return list(pool.map(_invoke, searchers))
        return [_invoke(searcher) for searcher in searchers]

    @staticmethod
    def __has_usable(question: QuestionModel, results: list[SearcherResp]) -> bool:
        """判断一组结果里是否有能用于作答的答案
        (题库返回了无法与选项对上的内容时, 视为没查到, 继续问 AI)
        """
        from .llm.answer import answerable  # 延迟导入避免包初始化时的循环引用

        return any(result.code == 0 and answerable(question, result.answer) for result in results)

    def __cross_check(
        self, question: QuestionModel, results: list[SearcherResp]
    ) -> list[SearcherResp]:
        """对多个 AI 的作答做交叉对比, 把达成共识的结果排到前面"""
        success = [r for r in results if r.code == 0 and r.answer]
        if not self.policy.ai_consensus or len(success) < 2:
            return results

        # 按归一化后的答案分组投票
        groups: dict[str, list[SearcherResp]] = {}
        for result in success:
            key = canonical_answer(question, result.answer) or result.answer.strip()
            groups.setdefault(key, []).append(result)
        best_key, best = max(groups.items(), key=lambda item: len(item[1]))

        if len(best) >= self.policy.ai_min_votes:
            votes = f"{len(best)}/{len(success)}"
            self.logger.info(f"AI 交叉对比: {votes} 一致 -> {best_key}")
            hit_ids = {id(r) for r in best}
            # 达成共识的结果排前面, 供 QuestionResolver 优先采用
            return [replace(r, note=f"共识 {votes}") for r in best] + [
                r for r in results if id(r) not in hit_ids
            ]

        answers = " | ".join(f"{r.searcher.__class__.__name__}={r.answer}" for r in success)
        self.logger.warning(
            f"AI 交叉对比无共识 (需 {self.policy.ai_min_votes} 票, 最高 {len(best)} 票): {answers}"
        )
        if self.policy.ai_fallback == "none":
            return [
                replace(r, code=CODE_NO_CONSENSUS, message="AI 交叉对比无共识", answer=None)
                for r in results
            ]
        return results


__all__ = ["SearcherResp", "SearcherBase", "SearcherPolicy", "MultiSearcherWraper"]
