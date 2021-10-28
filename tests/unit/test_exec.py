from executor import PaddlepaddleOCR
import pytest
from pathlib import Path
from jina import Document, DocumentArray

@pytest.fixture(scope="module")
def ocr() -> PaddlepaddleOCR:
    return PaddlepaddleOCR()


def test_one_doc(ocr : PaddlepaddleOCR):
    data_fn = str(Path(__file__).parents[1] / 'toy-data' / 'test1.png')

    docs = DocumentArray()
    doc = Document(uri=data_fn)
    docs.append(doc)
    ocr.extract(docs=docs)
    for d in docs:
        assert len(d.chunks) == 3

def test_two_docs(ocr : PaddlepaddleOCR):

    doc1 = Document(uri=str(Path(__file__).parents[1] / 'toy-data' / 'test1.png'))
    doc2 = Document(uri=str(Path(__file__).parents[1] / 'toy-data' / 'test2.png'))
    docs = DocumentArray([doc1,doc2])
    ocr.extract(docs=docs)
    assert len(docs) == 2 


def test_no_documents(ocr : PaddlepaddleOCR):
    docs = DocumentArray()
    ocr.extract(docs=docs)
    assert len(docs) == 0 


def test_docs_no_uris(ocr : PaddlepaddleOCR):
    docs = DocumentArray([Document()])
    ocr.extract(docs=docs)
    assert len(docs) == 1
    assert len(docs[0].chunks) == 0

def test_none_docs(ocr: PaddlepaddleOCR):
    ocr.extract(docs=None)

def test_lang_ru():

    data_fn = str(Path(__file__).parents[1] / 'toy-data' / 'test3.png')
    docs = DocumentArray()
    doc = Document(uri=data_fn)
    docs.append(doc)
    ocr = PaddlepaddleOCR(lang='ru')
    ocr.extract(docs=docs)
    for d in docs : 
        assert len(d.chunks) == 2  