import fast
import os
from .utils import enable_fast_verbosity, download_models, run_pipeline
from argparse import ArgumentParser


def main():
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

    download_models(args.fpl.split("/")[-1].split(".")[0])
    run_pipeline(args.fpl, args.input, args.output, args.model)


if __name__ == "__main__":
    main()
