from executor import PaddlepaddleOCR

from pathlib import Path
from jina import Document, DocumentArray


def test_exec():
    data_fn = str(Path(__file__).parents[1] / 'toy-data' / 'test.png')

    docs = DocumentArray()
    doc = Document(uri=data_fn)
    docs.append(doc)
    exec = PaddlepaddleOCR()
    exec.extract(docs=docs)
    for d in docs:
        assert len(d.chunks) == 3

def test_no_documents():
    docs = DocumentArray()
    exec = PaddlepaddleOCR()
    exec.extract(docs=docs)
    assert len(docs) == 0 


def test_docs_no_uris():
    docs = DocumentArray([Document()])
    exec = PaddlepaddleOCR()
    exec.extract(docs=docs)
    assert len(docs) == 1
    assert len(docs[0].chunks) == 0