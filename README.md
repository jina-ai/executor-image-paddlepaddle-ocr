# PaddlepaddleOCR

**PaddlepaddleOCR** wraps the models from [paddlepaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 

**PaddlepaddleOCR** executor receives `Documents` with `uri` attribute. Each `Document`'s `uri` represents the path of an image.
This executor will read the image and extract bounding boxes surrouding text, the text itself and the confidence score.

Every result will be saved inside a `document` where `text` attribute will be the extracted text, `weight` attribute will be the confidence score 
and we add a `coordinates` tag containing the bounding box coordinates. The result is finally saved as a `chunk` in the original document containing the image.  

## Usage

Use the prebuilt images from Jina Hub in your Python codes, add it to your Flow and extract text from an image:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://PaddlepaddleOCR')

docs = DocumentArray()
doc = Document(uri='/your/image/path')
docs.append(doc)

with f:
    f.post(on='/extract', inputs=docs, on_done=lambda resp: print(resp.docs[0].text))
```
## Returns 

`Document` with `text` field filled with an `str`, `weight` field filled with `float32` and `coordinates` filled with list of lists containing `float32` representing the bounding box.

## GPU usage 

## Reference

[PaddlepaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR) 
