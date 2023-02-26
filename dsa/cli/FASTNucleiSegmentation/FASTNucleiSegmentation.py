import json
import logging
import os
import time

import large_image
import numpy as np
import tempfile
from tqdm import tqdm

from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

from pathlib import Path
import cv2
from urllib.error import HTTPError
import signal
import subprocess as sp


logging.basicConfig(level=logging.CRITICAL)


# @TODO: Move this to a utils
class Timeout:
    
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def create_tile_boundary_annotations(im_seg_mask, tile_info, magn_frac=4):
    gx = tile_info['gx']
    gy = tile_info['gy']
    wfrac = tile_info['gwidth'] / np.double(tile_info['width'])
    hfrac = tile_info['gheight'] / np.double(tile_info['height'])

    # make binary image (if not already)
    im_seg_mask = (im_seg_mask > 0).astype("uint8")
    
    contours, hierarchy = cv2.findContours(image=im_seg_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_NONE)

    object_annot_list = []
    for i in range(len(contours)):

        # get boundary points and convert to base pixel space
        curr_contour = np.asarray(contours[i])
        num_points = len(curr_contour)

        # remove redundant axis in the middle
        curr_contour = np.squeeze(curr_contour, axis=1)

        if num_points < 3:
            continue
        
        # need to scale all coordinates to match full resolution
        cur_points = np.zeros((num_points, 3))
        cur_points[:, 0] = np.round(gx + curr_contour[:, 0] * wfrac, 2) * magn_frac
        cur_points[:, 1] = np.round(gy + curr_contour[:, 1] * hfrac, 2) * magn_frac
        cur_points = cur_points.tolist()

        # create annotation json
        cur_annot = {
            "type": "polyline",
            "points": cur_points,
            "closed": True,
            "fillColor": "rgba(0,0,0,0)",
            "lineColor": "rgb(0,255,0)"
        }

        object_annot_list.append(cur_annot)

    return object_annot_list


def get_annot_from_tiff_tile(slide_path, tile_position, magn_frac, args, it_kwargs):
    annot_list = []
    try:
        # get slide tile source
        ts = large_image.getTileSource(slide_path)

        # get requested tile
        tile_info = ts.getSingleTile(
            tile_position=tile_position,
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
            **it_kwargs)

        # get tile uint image (assumed it is a segmentation image)
        im_seg_mask = tile_info['tile'][:, :, 0]  # :3

        # generate annotations
        flag_object_found = np.any(im_seg_mask)

        # if counts > 50000:  # uncomment to avoid memory leak - however then dense nuclei regions will not be segmented/annotated/drawn
        #    return annot_list  # for now, skip if annotation structure is TOO large (mongodb limitations...)

        if flag_object_found:
            # @TODO: Should automatically calculate magn_frac based on WSI magnification and which level FAST has run inference on
            annot_list = create_tile_boundary_annotations(im_seg_mask, tile_info, magn_frac)
        
    except Exception as e:
        print(e)
        return annot_list

    return annot_list


def main(args):
    import dask
    import fast  # needs to be imported here to not break Dockerfile test

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')
    print(args)

    if not os.path.isfile(args.inputImageFile):
        raise OSError('Input image file does not exist.')

    if len(args.analysis_roi) != 4:
        raise ValueError('Analysis ROI must be a vector of 4 elements.')
    
    if np.all(np.array(args.analysis_roi) == -1):
        process_roi = False
    else:
        process_roi = True

    print('\n>> Creating Dask client ...')

    start_time = time.time()
    c = cli_utils.create_dask_client(args)
    print(c)

    dask_setup_time = time.time() - start_time
    print('Dask setup time = {}'.format(cli_utils.disp_time_hms(dask_setup_time)))

    # create temporary directory to save result
    fast_output_dir = tempfile.TemporaryDirectory()
    print("\nTemporary directory to save FPL result:", fast_output_dir.name)

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

    start_time = time.time()
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
                #curr_annot_list = dask.delayed(get_annot_from_tiff_tile)(
                curr_annot_list = get_annot_from_tiff_tile(
                    pred_output_path,
                    tile_position,
                    magn_frac,
                    args,
                    it_kwargs
                )
                #.compute()

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

    print("\n>> Does JSON file exist:", os.path.exists(args.outputNucleiAnnotationFile))

    # when analysis is over, the temporary dir can be closed (and deleted)
    fast_output_dir.cleanup()

    print('\n Total analysis time = {}'.format(cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
