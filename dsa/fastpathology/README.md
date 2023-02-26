# FastPathology support module

## Install

To install the package, simply run:

```
git clone https://github.com/andreped/FP-dsa-plugin.git
cd FP-dsa-plugin/dsa/fastpathology/
pip install -e .
```

## Usage

The fastpathology package can be used either as a command line tool (CLI) or as a python library.

### CLI

Generic API to run a FAST pipeline (FPL) on any whole slide image (WSI).

```
fastpathology --fpl /path/to/pipeline.fpl --input /path/to/wsi.svs --output /path/to/store/output/ --model /path/to/model_name
```

### Python

For nuclei segmentation, you can run this:

```
from fastpathology.utils import download_models, run_pipeline

download_models("nuclei-segmentation-model")
run_pipeline(
    "./fastpathology/pipelines/nuclei_segmentation.fpl",
    "/opt/WSI/image.svs",
    "/opt/Preds/nuclei_segmentations",
    "~/FAST/datahub/"
)
```
