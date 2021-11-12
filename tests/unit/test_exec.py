from executor import PaddlepaddleOCR
import pytest
from pathlib import Path
from jina import Document, DocumentArray
import numpy as np

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
    assert docs[0].chunks[0].text == 'Multimodal Document'
    assert docs[0].chunks[1].text == 'SearchinJina'

def test_two_docs(ocr : PaddlepaddleOCR):

    doc1 = Document(uri=str(Path(__file__).parents[1] / 'toy-data' / 'test1.png'))
    doc2 = Document(uri=str(Path(__file__).parents[1] / 'toy-data' / 'test2.png'))
    docs = DocumentArray([doc1,doc2])
    ocr.extract(docs=docs)
    assert len(docs) == 2 
    assert docs[0].chunks[0].text == 'Multimodal Document'
    assert docs[0].chunks[0].tags['coordinates'] == [[336.0, 310.0], [1370.0, 316.0], [1369.0, 389.0], [336.0, 383.0]]
    assert docs[1].chunks[0].text == 'Support'
    assert docs[1].chunks[0].tags['coordinates'] == [[20.0, 9.0], [134.0, 9.0], [134.0, 42.0], [20.0, 42.0]]

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


def test_clip_any_image_shape(ocr : PaddlepaddleOCR):
    docs = DocumentArray([Document(blob=np.ones((224, 224, 3), dtype=np.uint8))])

    ocr.extract(docs=docs)
    assert len(docs) == 1

    docs = DocumentArray([Document(blob=np.ones((100, 100, 3), dtype=np.uint8))])
    ocr.extract(docs=docs)
    assert len(docs) == 1

def test_lang_ru():

    data_fn = str(Path(__file__).parents[1] / 'toy-data' / 'test3.png')
    docs = DocumentArray()
    doc = Document(uri=data_fn)
    docs.append(doc)
    ocr = PaddlepaddleOCR(lang='ru')
    ocr.extract(docs=docs)
    assert docs[0].chunks[0].text == 'LOM' 
    assert docs[0].chunks[0].tags['coordinates'] == [[142.0, 49.0], [300.0, 45.0], [301.0, 105.0], [143.0, 109.0]]
    assert docs[0].chunks[1].text == 'MPOLAETC8'
    assert docs[0].chunks[1].tags['coordinates'] == [[29.0, 134.0], [433.0, 131.0], [433.0, 183.0], [29.0, 186.0]]

@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(ocr : PaddlepaddleOCR, batch_size: int):
    docs = DocumentArray([Document(uri=str(Path(__file__).parents[1] / 'toy-data' / 'test1.png')) for _ in range(32)])
    ocr.extract(docs=docs)
    for doc in docs :
        assert doc.chunks[0].text == 'Multimodal Document'
