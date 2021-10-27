from jina import Executor, DocumentArray, Document, requests
from paddleocr import PaddleOCR
from typing import Optional, Dict
from jina.logging.logger import JinaLogger

class PaddlepaddleOCR(Executor):
    """
    An executor to extract text from images using paddlepaddleOCR
    """
    def __init__(
        self,
        paddleocr_args : Optional[Dict] = None,
        **kwargs
        ):
        """
        :param paddleocr_args: the arguments for `paddleocr` for extracting text. By default
        `use_angle_cls=True`, `lang='en'`, `use_gpu=False` 
        """
        self._paddleocr_args = paddleocr_args or {}
        self._paddleocr_args.setdefault('use_angle_cls', True) 
        self._paddleocr_args.setdefault('lang', 'en')
        self._paddleocr_args.setdefault('use_gpu', False)
        super(PaddlepaddleOCR, self).__init__(**kwargs)
        self.model = PaddleOCR(**self._paddleocr_args)
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests(on='/extract')
    def extract(self, docs: Optional[DocumentArray] = None, **kwargs):
        """
        Load the image from Document.uri extract text and bounding boxes. The result is stored
        it into chunks
        :param docs: the input Documents with image URI in the `uri` field
        """
        if docs is None:
            return
        for doc in docs:
            self.logger.info(f'received {doc.id}')

            if doc.uri == '':
                self.logger.error(f'No uri passed for the Document: {doc.id}')
                continue
            for r in self.model.ocr(doc.uri, cls=True):
                coord, (text, score) = r
                c = Document(text=text, weight=score)
                c.tags['coordinates'] = coord
                doc.chunks.append(c)
                print(f'{text}')
