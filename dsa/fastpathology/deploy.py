import fast
import os
from utils import force_run_exporters
from argparse import ArgumentParser


def enable_fast_verbosity():
    fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)


def download_models(model_name):
    model_name = model_name.replace("_", "_") + "-model"
    fast.DataHub().download(model_name)


def run_pipeline(fpl, input_, output, model):
    pipeline = fast.Pipeline(fpl, {'input': input_, 'output': output, 'model': model})
    pipeline.parse()
    force_run_exporters(pipeline)


def main(fpl, input_, output, model):
    download_models(fpl.split("/")[-1].split(".")[0])
    run_pipeline(input, output, model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--fpl", type=str, help="Which pipeline to use, e.g., '/opt/pipelines/tumor_seg.fpl'.")
    parser.add_argument("-i", "--input", type=str, help="Full path to which whole slide image (WSI) to run pipeline on, e.g., '/opt/images/image.tiff'.")
    parser.add_argument("-o", "--output", type=str, help="Path to where to store the result of the pipeline, e.g., '/opt/results/'.")
    parser.add_argument("-m", "--model", type=str, help="Path to where models are stored on disk, e.g., '/opt/models/'.")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="Whether to enable FAST verbosity or not. Default: 1 (enabled).")
    args = parser.parse_args()

    print("Arguments used:", args)

    if args.verbose:
        enable_fast_verbosity()

    main()

    '''
    fast.DataHub().download('breast-tumour-segmentation-model')
    fpl_file = "/opt/pipelines/breast_tumour_segmentation.fpl"
    output = "/opt/pipelines/prediction"
    model = "/root/FAST/datahub/"

    pipeline = fast.Pipeline(
        fpl_file,
        {
            'wsi': '/opt/pipelines/A05.svs',
            'output': output,
            'model': model,
        }
    )
    pipeline.parse()
    force_run_exporters(pipeline)
    '''
