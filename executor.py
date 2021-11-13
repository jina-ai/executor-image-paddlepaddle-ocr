from jina import Executor, DocumentArray, Document, requests
from paddleocr import PaddleOCR
from typing import Optional, Dict
from jina.logging.logger import JinaLogger
import urllib
import random 
import string
import tempfile
import os 
import io

class PaddlepaddleOCR(Executor):
    """
    An executor to extract text from images using paddlepaddleOCR
    """
    def __init__(
        self,
        paddleocr_args : Optional[Dict] = None,
        copy_uri: bool = True,
        **kwargs
        ):
        """
        :param paddleocr_args: the arguments for `paddleocr` for extracting text. By default
        `use_angle_cls=True`,
        `lang='en'` means the language you want to extract, 
        `use_gpu=False` whether you want to use gpu or not.
        Other params can be found in `paddleocr --help`. More information can be found here https://github.com/PaddlePaddle/PaddleOCR
        :param copy_uri: Set to `True` to store the image `uri` at the `.tags['image_uri']` of the chunks that are extracted from the image.
        By default, `copy_uri=True`. Set this to `False` when you don't want to store umage `uri` with the chunks 
        """
        self._paddleocr_args = paddleocr_args or {}
        self._paddleocr_args.setdefault('use_angle_cls', True) 
        self._paddleocr_args.setdefault('lang', 'en')
        self._paddleocr_args.setdefault('use_gpu', False)
        super(PaddlepaddleOCR, self).__init__(**kwargs)
        self.model = PaddleOCR(**self._paddleocr_args)
        self.copy_uri = copy_uri
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests()
    def extract(self, docs: Optional[DocumentArray] = None, **kwargs):
        """
        Load the image from `uri`, extract text and bounding boxes. The text is stored in the  
        `text` attribute of the chunks and the coordinates are stored in the `tags['coordinates']` as a list. 
        The `tags['coordinates']`  contains four lists, each of which corresponds to the `(x, y)` coordinates one corner of the bounding box. 
        :param docs: the input Documents with image URI in the `uri` field
        """
        missing_doc_ids = []
        if docs is None:
            return
        for doc in docs:
            if not doc.uri :
                missing_doc_ids.append(doc.id)
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                source_fn = (
                    self._save_uri_to_tmp_file(doc.uri, tmpdir)
                    if self._is_datauri(doc.uri)
                    else doc.uri
                )
                for r in self.model.ocr(source_fn, cls=True):
                    coord, (text, score) = r
                    c = Document(text=text, weight=score)
                    c.tags['coordinates'] = coord
                    if self.copy_uri:
                        c.tags['img_uri'] = doc.uri
                    doc.chunks.append(c)
        if missing_doc_ids  :
            self.logger.warning(f'No uri passed for the following Documents:{", ".join(missing_doc_ids)}')

    def _is_datauri(self, uri):
        scheme = urllib.parse.urlparse(uri).scheme
        return scheme in {'data'}

    def _save_uri_to_tmp_file(self, uri, tmpdir):
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        tmp_fn = os.path.join(
            tmpdir,
            ''.join([random.choice(string.ascii_lowercase) for i in range(10)])
            + '.png',
        )
        with urllib.request.urlopen(req) as fp:
            buffer = fp.read()
            binary_fn = io.BytesIO(buffer)
            with open(tmp_fn, 'wb') as f:
                f.write(binary_fn.read())
        return tmp_fn