import json
import logging
import os
import time

import large_image
import numpy as np
import tempfile
from tqdm import tqdm

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.label as htk_seg_label
import histomicstk.segmentation.nuclear as htk_nuclear
import histomicstk.segmentation as htk_seg
import histomicstk.utils as htk_utils
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

import cv2
import skimage
import scipy
from urllib.error import HTTPError


logging.basicConfig(level=logging.CRITICAL)


import signal

class timeout:
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


def create_tile_boundary_annotations(im_seg_mask, tile_info):

    gx = tile_info['gx']
    gy = tile_info['gy']
    wfrac = tile_info['gwidth'] / np.double(tile_info['width'])
    hfrac = tile_info['gheight'] / np.double(tile_info['height'])

    # make binary image (if not already)
    im_seg_mask = (im_seg_mask > 0).astype("uint8")
    
    contours, hierarchy = cv2.findContours(image=im_seg_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_NONE)

    object_annot_list = []

    #print("number of objects (len(contours)):", len(contours))

    #for i in range(len(bx)):
    for i in range(len(contours)):
        # get boundary points and convert to base pixel space
        curr_contour = np.asarray(contours[i])
        num_points = len(curr_contour)

        #print("shape curr contour:", curr_contour.shape)

        # remove redundant axis in the middle
        curr_contour = np.squeeze(curr_contour, axis=1)

        #print("UPDATED shape curr contour:", curr_contour.shape)

        if num_points < 3:
            continue

        cur_points = np.zeros((num_points, 3))
        cur_points[:, 0] = np.round(gx + curr_contour[:, 0] * wfrac, 2) * 4
        cur_points[:, 1] = np.round(gy + curr_contour[:, 1] * hfrac, 2) * 4
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
    
    #print("final output from tile (number of unique objects - nested list):", len(object_annot_list))

    return object_annot_list


def get_annot_from_tiff_tile(slide_path, tile_position, args, it_kwargs):
    annot_list = []
    try:
        # get slide tile source
        ts = large_image.getTileSource(slide_path)

        # get requested tile
        tile_info = ts.getSingleTile(
            tile_position=tile_position,
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
            **it_kwargs)

        # get tile image
        im_tile = tile_info['tile'][:, :, :3]

        # make segmentation image
        im_seg_mask = im_tile[:, :, 0]

        # Delete border nuclei
        #if args.ignore_border_nuclei is True:
        #    im_seg_mask = htk_seg_label.delete_border(im_seg_mask)

        # generate annotations
        flag_object_found = np.any(im_seg_mask)

        counts = np.count_nonzero(im_seg_mask)
        #print("count nonzero seg mask tile:", counts)
        if counts == 0:
            return annot_list

        #elif counts > 50000:
        #    return annot_list  # for now, skip if annotation structure is TOO large (mongodb limitations...)

        if flag_object_found:
            annot_list = create_tile_boundary_annotations(im_seg_mask, tile_info)

    except Exception as e:
        print(e)
        return annot_list

    return annot_list


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
        "fastpathology", "-f", "/opt/fastpathology/dsa/cli/fastpathology/pipelines/nuclei_segmentation.fpl",
        "-i", args.inputImageFile, "-o", fast_output_dir.name, "-m", datahub_dir, "-v", "0"
    ])

    pred_output_path = fast_output_dir.name + ".tiff"

    # convert pyramidal TIFF output from pyFAST to JSON annotation file (*.anot)
    # iterate over annotation image in tiled fashion, get unique elements, and save coordinates from each in JSON file

    print('\n>> Converting Pyramidal TIFF annotations to JSON ...\n')

    print("loading pyramidal tiff seg:", pred_output_path)

    # get slide tile source
    ts = large_image.getTileSource(pred_output_path)

    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},  # args.analysis_mag},
    }

    start_time = time.time()

    annot_list = []

    generator = ts.tileIterator(**it_kwargs)

    #iter = 0
    #for tile in tqdm(ts.tileIterator(**it_kwargs), "Tile"):
    iter = 0
    while True:
        iter += 1
        print("\nIter:", iter)

        with timeout(seconds=3):
            try:
                tile = next(generator)

                tile_position = tile['tile_position']['position']

                print(tile_position)

                #if tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
                #    continue 

                # detect nuclei
                #curr_annot_list = get_annot_from_tiff_tile(
                #curr_annot_list = dask.delayed(get_annot_from_tiff_tile)(
                curr_annot_list = get_annot_from_tiff_tile(
                    pred_output_path,
                    tile_position,
                    args,
                    it_kwargs
                )
                
                #.compute()

                # print(curr_annot_list)

                print("objects found in tile:", len(curr_annot_list))

                # append result to list
                annot_list.append(curr_annot_list)

                # append to JSON for each tile (to avoid memory leakage for millions of objects)
                #annot_fname = os.path.splitext(os.path.basename(args.outputNucleiAnnotationFile))[0]

                #annotation = {
                #    "name": annot_fname + '-nuclei-' + str(iter) + "-" + args.nuclei_annotation_format,
                #    "elements": curr_annot_list
                #}

                #with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
                #    json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)


                print("Finished before timeout!")

            except StopIteration:
                print("\n Iterator is empty. Stopped iterator ...")
                break
            except TimeoutError:
                print("Operation timed out...")
                pass
            except HTTPError as e:
                print(e)
                pass
            except Exception:
                print("\n Something wrong with tile ...")
                pass

        #iter += 1

        #if iter == 50:
        #    break
    
    print("\n\n\n\n\n ... Done iterating tiles. Total number of tiles were:", len(annot_list))
    
    #from dask.diagnostics import ProgressBar

    #with ProgressBar():
    #tile_list = dask.delayed(tile_list).compute()

    print("\n flatten gigantic list ... ")
    annot_list = [anot for anot_list in annot_list for anot in anot_list]

    print("\n Done flattening... Attempts to write large array to JSON file ...")

    #nuclei_detection_time = time.time() - start_time

    #print('Number of nuclei = {}'.format(len(annot_list)))

    #print('Nuclei detection time = {}'.format(cli_utils.disp_time_hms(nuclei_detection_time)))

    #print('\n>> Writing annotation file ...\n')

    #print('\n outputNucleiAnnotationFile:', args.outputNucleiAnnotationFile)

    annot_fname = os.path.splitext(os.path.basename(args.outputNucleiAnnotationFile))[0]

    annotation = {
        "name": annot_fname + '-nuclei-' + args.nuclei_annotation_format,
        "elements": annot_list
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)


    total_time_taken = time.time() - total_start_time

    print("\n Does JSON file exist:", os.path.exists(args.outputNucleiAnnotationFile))

    # when analysis is over, the temporary dir can be closed (and deleted)
    fast_output_dir.cleanup()

    print('\n Total analysis time = {}'.format(
        cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
