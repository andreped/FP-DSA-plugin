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
        "fastpathology", "-f", "/opt/fastpathology/dsa/cli/fastpathology/pipelines/nuclei_segmentation.fpl",
        "-i", args.inputImageFile, "-o", fast_output_dir.name, "-m", datahub_dir, "-v", "0"
    ])

    pred_output_path = fast_output_dir.name + ".tiff"

    print('\n>> Converting Pyramidal TIFF annotations to JSON ...\n')

    # get slide tile source
    ts = large_image.getTileSource(pred_output_path)

    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
    }

    # whether to only convert annotations within ROI (FAST will still run inference on entire WSI)
    # @TODO: Right now we scale the region to work with FAST TIFF segmentation WSIs using a hardcoded magn_frac
    magn_frac = int(40 / 10)
    if process_roi:
        it_kwargs['region'] = {
            'left': int(args.analysis_roi[0] / magn_frac),
            'top': int(args.analysis_roi[1] / magn_frac),
            'width': int(args.analysis_roi[2] / magn_frac),
            'height': int(args.analysis_roi[3] / magn_frac),
            'units': 'base_pixels'
        }

    annot_list = []
    generator = ts.tileIterator(**it_kwargs)

    iter = 0
    while True:
        iter += 1
        print("\n>> Iter:", iter)

        with Timeout(seconds=3):
            try:
                tile = next(generator)
                tile_position = tile['tile_position']['position']

                # detect nuclei
                curr_annot_list = get_annot_from_tiff_tile(
                    pred_output_path,
                    tile_position,
                    magn_frac,
                    args,
                    it_kwargs
                )

                print("\n>> Objects found in tile:", len(curr_annot_list))

                # append result to list
                annot_list.extend(curr_annot_list)

            except StopIteration:
                print("\n>> Iterator is empty. Stopped iterator ...")
                break
            except TimeoutError:
                print("\n>> Operation timed out ...")
                pass
            except HTTPError as e:
                print(e)
                pass
            except Exception:
                print("\n>> Something wrong with tile ...")
                pass
    
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
