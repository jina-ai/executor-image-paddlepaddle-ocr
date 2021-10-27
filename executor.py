from jina import Executor, DocumentArray, Document, requests
from paddleocr import PaddleOCR


class PaddlepaddleOCR(Executor):
    def __init__(self, *args, **kwargs):
        super(PaddlepaddleOCR, self).__init__(*args, **kwargs)
        self.model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

    @requests(on='/extract')
    def extract(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            for r in self.model.ocr(doc.uri, cls=True):
                cord, (text, score) = r
                c = Document(text=text, weight=score)
                c.tags['cordinates'] = cord
                doc.chunks.append(c)
                print(f'{text}')
