import json
import logging
import os
import time

import large_image
import numpy as np
import tempfile

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.label as htk_seg_label
import histomicstk.segmentation.nuclear as htk_nuclear
import histomicstk.utils as htk_utils
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser


logging.basicConfig(level=logging.CRITICAL)


def main(args):
    import dask

    print("\n>> trying to import FAST...\n")
    import fast  # <- @TODO: Does this work within the plugin? Will likely fail

    print("\n>> Sucessfully imported FAST ...\n")

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print(args)

    if not os.path.isfile(args.inputImageFile):
        raise OSError('Input image file does not exist.')

    if len(args.reference_mu_lab) != 3:
        raise ValueError('Reference Mean LAB should be a 3 element vector.')

    if len(args.reference_std_lab) != 3:
        raise ValueError('Reference Stddev LAB should be a 3 element vector.')

    if len(args.analysis_roi) != 4:
        raise ValueError('Analysis ROI must be a vector of 4 elements.')

    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True
    else:
        process_whole_image = False

    #
    # Initiate Dask client
    #
    print('\n>> Creating Dask client ...\n')

    start_time = time.time()

    c = cli_utils.create_dask_client(args)

    print(c)

    dask_setup_time = time.time() - start_time
    print('Dask setup time = {}'.format(
        cli_utils.disp_time_hms(dask_setup_time)))

    # create temporary directory to save result
    fast_output_dir = tempfile.TemporaryDirectory()
    print("\nTemporary directory to save FPL result:", fast_output_dir.name)

    # get current home directory and get path to FAST DataHub
    from pathlib import Path
    home = str(Path.home())
    datahub_dir = home + "/FAST/datahub/"

    # run nuclei segmentation FPL
    import subprocess as sp
    sp.check_call([
        "fastpathology", "-f", "/opt/fastpathology/dsa/cli/fastpathology/pipelines/breast_tumour_segmentation.fpl",
        "-i", args.inputImageFile, "-o", fast_output_dir.name, "-m", datahub_dir
    ])

    # when prediction file has been converted to an annotation file, the temporary dir can be closed (and deleted)
    fast_output_dir.cleanup()

    # convert pyramidal TIFF output from pyFAST to JSON annotation file (*.ann)



    #
    # Write annotation file
    #
    print('\n>> Writing annotation file ...\n')

    annot_fname = os.path.splitext(
        os.path.basename(args.outputNucleiAnnotationFile))[0]

    annotation = {
        "name": annot_fname + '-nuclei-' + args.nuclei_annotation_format,
        "elements": nuclei_list
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)

    total_time_taken = time.time() - total_start_time

    print('Total analysis time = {}'.format(
        cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
