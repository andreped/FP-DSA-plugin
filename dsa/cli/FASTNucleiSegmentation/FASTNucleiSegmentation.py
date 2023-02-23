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
    # get slide tile source
    ts = large_image.getTileSource(slide_path)

    # get requested tile
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **it_kwargs)

    # get tile image
    im_tile = tile_info['tile'][:, :, :3]

    # Delete border nuclei
    if args.ignore_border_nuclei is True:
        im_seg_mask = htk_seg_label.delete_border(im_tile)

    # generate annotations
    annot_list = []

    flag_nuclei_found = np.any(im_seg_mask)

    if flag_nuclei_found:
        annot_list = create_tile_boundary_annotations(
            im_seg_mask, tile_info, args.annotation_format
        )

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
        "-i", args.inputImageFile, "-o", fast_output_dir.name, "-m", datahub_dir
    ])

    # when prediction file has been converted to an annotation file, the temporary dir can be closed (and deleted)
    fast_output_dir.cleanup()

    # convert pyramidal TIFF output from pyFAST to JSON annotation file (*.anot)
    # iterate over annotation image in tiled fashion, get unique elements, and save coordinates from each in JSON file
    
    importer = fast.TIFFImagePyramidExporter.create(args.inputImageFile)
    patchgen = fast.PatchGenerator.create(width=256, height=256, level=0).connect(importer)
    streamer = fast.DataStream(patchgen)

    annot_list = []
    for patch in streamer:
        print(patch)
        coords = [x[0] for x in patch.getTransform().getTranslation()]
        patch_image = np.asarray(patch)  # convert from fast image to numpy array
        print(coords)
        print("---")

    
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
