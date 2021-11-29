# PaddlepaddleOCR

**PaddlepaddleOCR** wraps the models from [paddlepaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 

**PaddlepaddleOCR** executor receives `Documents` with `uri` attribute. Each `Document`'s `uri` represents the path of an image.
This executor will read the image and extract bounding boxes surrouding text, the text itself and the confidence score.

Every result will be saved inside a `document` where `text` attribute will be the extracted text, `weight` attribute will be the confidence score 
and we add a `coordinates` tag containing the bounding box coordinates. The result is finally saved as a `chunk` in the original document containing the image.  

## Usage

Use the prebuilt images from Jina Hub in your Python code, add it to your Flow and extract text from an image:

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://PaddlepaddleOCR')

docs = DocumentArray()
doc = Document(uri='/your/image/path')
docs.append(doc)

def print_results(resp):
    """
    Function to print the extracted text from the response after
    applying OCR to the input images

    :resp: response resulting for the executor
    """

    for doc in resp.docs:
        for chunk in doc.chunks:
            print(chunk.text)
            
with f:
    f.post(on='/extract', inputs=docs, on_done=print_results)
```
## Returns 

`Document` with `text` field filled with an `str`, `weight` field filled with `float32` and `coordinates` filled with list of lists containing `float32` representing the bounding box.

## GPU usage 

This executor also offers a GPU version. To use it, make sure to pass `'use_gpu'=True`, as the initialization parameter, and `gpus='all'` when adding the containerized Executor to the Flow. See the [Executor on GPU](https://docs.jina.ai/tutorials/gpu-executor/) section of Jina documentation for more details.

Here's how you would modify the example above to use a GPU

```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://PaddlepaddleOCR',
    uses_with={'use_gpu':True},
    gpus='all
    )
```
## Reference

[PaddlepaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR) 
