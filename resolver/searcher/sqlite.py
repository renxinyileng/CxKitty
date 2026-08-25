import sqlite3
from pathlib import Path

from cxapi.schema import QuestionModel

from . import SearcherBase, SearcherResp


class SqliteSearcher(SearcherBase):
    "SQLite 数据库搜索器"
    db: sqlite3.Connection
    table: str
    rsp_field: str

    def __init__(
        self,
        file_path: Path | str,
        req_field: str = "question",
        rsp_field: str = "answer",
        table: str = "question",
    ) -> None:
        # 并行请求时搜索器会在工作线程中被调用, 故放开线程校验 (同一搜索器不会被并发调用)
        self.db = sqlite3.connect(file_path, check_same_thread=False)
        self.table = table
        self.req_field = req_field
        self.rsp_field = rsp_field

    def invoke(self, question: QuestionModel) -> SearcherResp:
        try:
            cur = self.db.execute(
                f"SELECT {self.req_field},{self.rsp_field} FROM {self.table} WHERE {self.req_field}=(?)",
                (question.value,),
            )
            q, a = cur.fetchone()
            return SearcherResp(0, "ok", self, q, a)
        except Exception as err:
            return SearcherResp(-500, err.__str__(), self, question.value, None)


__all__ = ["SqliteSearcher"]
