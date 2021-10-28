from executor import PaddlepaddleOCR
import pytest
from pathlib import Path
from jina import Document, DocumentArray, Flow


@pytest.mark.parametrize("request_size", [1, 10, 50, 100])
def test_integration(request_size: int):
    doc1 = Document(uri=str(Path(__file__).parents[1] / 'toy-data' / 'test1.png'))
    docs = DocumentArray([doc1])

    with Flow(return_results=True).add(uses=PaddlepaddleOCR) as flow:
            resp = flow.post(
                on="/index",
                inputs=docs,
                request_size=request_size,
                return_results=True,
            )
    
    assert sum(len(resp_batch.docs) for resp_batch in resp) == 1
    for r in resp:
        assert r.docs[0].chunks[0].text == 'Multimodal Document'
        assert r.docs[0].chunks[0].tags['coordinates'] == [[336.0, 310.0], [1370.0, 316.0], [1369.0, 389.0], [336.0, 383.0]]