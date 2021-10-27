from jina import Executor, DocumentArray, requests


class PaddlepaddleOCR(Executor):
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
