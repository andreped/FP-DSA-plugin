import json
import logging
import os
import time

import large_image
import numpy as np
import tempfile

from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

from pathlib import Path
from urllib.error import HTTPError
import subprocess as sp
import h5py

from fastpathology.annot_utils import Timeout, get_annot_from_tiff_tile


logging.basicConfig(level=logging.CRITICAL)


def main(args):

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')
    print(args)

    if not os.path.isfile(args.inputImageFile):
        raise OSError('Input image file does not exist.')

    if len(args.analysis_roi) != 4:
        raise ValueError('Analysis ROI must be a vector of 4 elements.')
    
    process_roi = False if np.all(np.array(args.analysis_roi) == -1) else True

    # create temporary directory to save result
    fast_output_dir = tempfile.TemporaryDirectory()

    # get current home directory and get path to FAST DataHub
    home = str(Path.home())
    datahub_dir = home + "/FAST/datahub/"

    # run nuclei segmentation FPL in a subprocess
    sp.check_call([
        "fastpathology", "-f", "/opt/fastpathology/dsa/cli/fastpathology/pipelines/bach.fpl",
        "-i", args.inputImageFile, "-o", fast_output_dir.name, "-m", datahub_dir, "-v", "0"
    ])

    pred_output_path = fast_output_dir.name + ".hd5"

    print('\n>> Converting HDF5 annotations to JSON ...\n')

    # get slide tile source
    ts = large_image.getTileSource(pred_output_path)

    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
    }

    # whether to only convert annotations within ROI (FAST will still run inference on entire WSI)
    # @TODO: Right now we scale the region to work with FAST TIFF segmentation WSIs using a hardcoded magn_frac

    magn_frac = int(40 / 40)  # BACH model is applied on 40x resolution image plane
    patch_size = 512

    if process_roi:
        it_kwargs['region'] = {
            'left': int(args.analysis_roi[0] / magn_frac),
            'top': int(args.analysis_roi[1] / magn_frac),
            'width': int(args.analysis_roi[2] / magn_frac),
            'height': int(args.analysis_roi[3] / magn_frac),
            'units': 'base_pixels'
        }


    # load classification result as HDF5 from FAST
    with h5py.File(pred_output_path, "r") as f:
        data = f["tensor"]
        spacing = f["spacing"]
    
    print("hdf5 keys:", list(data.keys()))
    data = np.asarray(data)
    spacing = np.asarray(spacing)
    print("hdf5 extracted numpy:", data.shape, data.dtype, np.unique(data))
    print("spacing:", spacing)

    # iterate over all values in data array and store each as rectangle objects
    annot_list = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # scale left-up patch coordinate
            curr_y = int(i * spacing[0])
            curr_x = int(j * spacing[1])

            # scale patch height width of rectangle
            patch_height = int(patch_size * spacing[0])
            patch_width = int(patch_size * spacing[1])

            # create annotation json
            cur_bbox = {
                "type": "rectangle",
                "center": [curr_x, curr_y, 0],
                "width": patch_width,
                "height": patch_height,
                "rotation": 0,
                "fillColor": "rgba(0,255,0,0)",
                "lineColor": "rgb(0, 255, 0)"
            }

            annot_list.append(cur_bbox)

    # get magnification information from full WSI, to know how to scale HDF5 heatmap preds
    # - magn_defined above, as well as patch_size

    # need to take into account that heatmap can be larger than full WSI after rescaling
    # - use patch size to check modulo to handle edge cases

    # convert HDF5 to JSON -> store each pw-prediction as rectangle object
    
    print("\n>> Done iterating tiles. Total number of tiles were:", len(annot_list))
    print("\n>> Writing annotation file ...")

    annot_fname = os.path.splitext(os.path.basename(args.outputNucleiAnnotationFile))[0]
    annotation = {
        "name": annot_fname + '-nuclei-' + args.nuclei_annotation_format,
        "elements": annot_list
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)

    total_time_taken = time.time() - total_start_time

    file_exist = os.path.exists(args.outputNucleiAnnotationFile)
    print("\n>> Does JSON file exist:", file_exist)

    if file_exist:
        print("\n>> JSON file size on disk [MB]:", os.path.getsize(args.outputNucleiAnnotationFile) / (1024.0 ** 2))

    # when analysis is over, the temporary dir can be closed (and deleted)
    fast_output_dir.cleanup()

    print('\n Total analysis time = {}'.format(cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
