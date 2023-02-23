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


logging.basicConfig(level=logging.CRITICAL)


def create_tile_boundary_annotations(im_seg_mask, tile_info):
    nuclei_annot_list = []

    gx = tile_info['gx']
    gy = tile_info['gy']
    wfrac = tile_info['gwidth'] / np.double(tile_info['width'])
    hfrac = tile_info['gheight'] / np.double(tile_info['height'])

    by, bx = htk_seg.label.trace_object_boundaries(im_seg_mask,
                                                   trace_all=True)

    for i in range(len(bx)):
        # get boundary points and convert to base pixel space
        num_points = len(bx[i])

        if num_points < 3:
            continue

        cur_points = np.zeros((num_points, 3))
        cur_points[:, 0] = np.round(gx + bx[i] * wfrac, 2)
        cur_points[:, 1] = np.round(gy + by[i] * hfrac, 2)
        cur_points = cur_points.tolist()

        # create annotation json
        cur_annot = {
            "type": "polyline",
            "points": cur_points,
            "closed": True,
            "fillColor": "rgba(0,0,0,0)",
            "lineColor": "rgb(0,255,0)"
        }

        nuclei_annot_list.append(cur_annot)

    return nuclei_annot_list


def get_annot_from_tiff_tile(slide_path, tile_position, args, it_kwargs):
    print("\nprocessing tile ...")
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
    if args.ignore_border_nuclei is True:
        im_seg_mask = htk_seg_label.delete_border(im_seg_mask)

    # generate annotations
    annot_list = []
    
    flag_object_found = np.any(im_seg_mask)

    if flag_object_found:
        annot_list = create_tile_boundary_annotations(im_seg_mask, tile_info)

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
        "fastpathology", "-f", "/opt/fastpathology/dsa/cli/fastpathology/pipelines/breast_tumour_segmentation.fpl",
        "-i", args.inputImageFile, "-o", fast_output_dir.name, "-m", datahub_dir, "-v", "0"
    ])

    # when prediction file has been converted to an annotation file, the temporary dir can be closed (and deleted)
    fast_output_dir.cleanup()

    # convert pyramidal TIFF output from pyFAST to JSON annotation file (*.anot)
    # iterate over annotation image in tiled fashion, get unique elements, and save coordinates from each in JSON file

    print('\n>> Converting Pyramidal TIFF annotations to JSON ...\n')

    # get slide tile source
    ts = large_image.getTileSource(args.inputImageFile)

    #tile_fgnd_frac_list = [1.0]

    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
    }

    start_time = time.time()

    tile_list = []

    for tile in tqdm(ts.tileIterator(**it_kwargs), "Tile"):

        tile_position = tile['tile_position']['position']

        #if tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
        #    continue

        # detect nuclei
        #curr_annot_list = dask.delayed(get_annot_from_tiff_tile)(
        curr_annot_list = get_annot_from_tiff_tile(
            args.inputImageFile,
            tile_position,
            args,
            it_kwargs
        )

        # append result to list
        tile_list.append(curr_annot_list)
    
    print("Done iterating tiles. Total number of tiles were:", len(tile_list))
    
    #from dask.diagnostics import ProgressBar

    #with ProgressBar():
    #    tile_list = dask.delayed(tile_list).compute()

    annot_list = [anot for anot_list in annot_list for anot in anot_list]

    nuclei_detection_time = time.time() - start_time

    print('Number of nuclei = {}'.format(len(annot_list)))

    print('Nuclei detection time = {}'.format(
        cli_utils.disp_time_hms(nuclei_detection_time)))

    print('\n>> Writing annotation file ...\n')

    annot_fname = os.path.splitext(
        os.path.basename(args.outputNucleiAnnotationFile))[0]

    annotation = {
        "name": annot_fname + '-nuclei-' + args.nuclei_annotation_format,
        "elements": annot_list
    }

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)

    total_time_taken = time.time() - total_start_time

    print('Total analysis time = {}'.format(
        cli_utils.disp_time_hms(total_time_taken)))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
