import json
import logging
import os
import time
from pathlib import Path

from cli.client import MONAILabelClient
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig(level=logging.INFO)


def fetch_annotations(args, tiles=None):
    total_start_time = time.time()

    location = [args.analysis_roi[0], args.analysis_roi[1]]
    size = [args.analysis_roi[2], args.analysis_roi[3]]

    start_time = time.time()

    res = "test.txt"
    logging.info(f"Annotation Detection Time = {round(time.time() - start_time, 2)}")

    res = str(res, encoding="utf-8")
    with open(args.outputAnnotationFile, "w") as fp:
        fp.write(res)

    # print last but one line of result (description)
    try:
        d = json.loads(json.loads(res[-5000:].split("\n")[-2:][0].replace('"description": ', "")))
        logging.info(f"\n{json.dumps(d, indent=2)}")
    except Exception:
        pass

    total_time_taken = time.time() - total_start_time
    logging.info(f"Total Annotation Fetch time = {round(total_time_taken, 2)}")


def main(args):
    total_start_time = time.time()
    logging.info("CLI Parameters ...\n")
    for arg in vars(args):
        logging.info(f"USING:: {arg} = {getattr(args, arg)}")

    if not os.path.isfile(args.inputImageFile):
        raise OSError("Input image file does not exist.")

    if len(args.analysis_roi) != 4:
        raise ValueError("Analysis ROI must be a vector of 4 elements.")

    logging.info(">> Reading input image ... \n")
    tiles = []

    if args.min_fgnd_frac >= 0:
        import large_image
        import numpy as np
        from histomicstk.cli import utils as cli_utils
        from histomicstk.utils import compute_tile_foreground_fraction

        ts = large_image.getTileSource(args.inputImageFile)
        ts_metadata = ts.getMetadata()
        logging.info(json.dumps(ts_metadata, indent=2))

        it_kwargs = {
            "tile_size": {"width": args.analysis_tile_size},
            "scale": {"magnification": 0},
        }
        if not np.all(np.array(args.analysis_roi) <= 0):
            it_kwargs["region"] = {
                "left": args.analysis_roi[0],
                "top": args.analysis_roi[1],
                "width": args.analysis_roi[2],
                "height": args.analysis_roi[3],
                "units": "base_pixels",
            }

        num_tiles = ts.getSingleTile(**it_kwargs)["iterator_range"]["position"]
        logging.info(f"Number of tiles = {num_tiles}")

        logging.info(">> Computing tissue/foreground mask at low-res ...\n")
        start_time = time.time()

        im_fgnd_mask_lres, fgnd_seg_scale = cli_utils.segment_wsi_foreground_at_low_res(ts)
        logging.info(f"low-res foreground mask computation time = {round(time.time() - start_time, 2)}")

        logging.info(">> Computing foreground fraction of all tiles ...\n")
        start_time = time.time()

        tile_fgnd_frac_list = compute_tile_foreground_fraction(
            args.inputImageFile,
            im_fgnd_mask_lres,
            fgnd_seg_scale,
            it_kwargs,
        )

        num_fgnd_tiles = np.count_nonzero(tile_fgnd_frac_list >= args.min_fgnd_frac)
        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

        logging.info(f"Number of foreground tiles = {num_fgnd_tiles:d} ({percent_fgnd_tiles:2f}%%)")
        logging.info(f"Tile foreground fraction computation time = {round(time.time() - start_time, 2)}")

        skip_count = 0
        for tile in ts.tileIterator(**it_kwargs):
            tile_position = tile["tile_position"]["position"]
            location = [int(tile["x"]), int(tile["y"])]
            size = [int(tile["width"]), int(tile["height"])]
            frac = tile_fgnd_frac_list[tile_position]

            if frac <= args.min_fgnd_frac:
                # logging.info(f"Skip:: {tile_position} => {location}, {size} ({frac} <= {args.min_fgnd_frac})")
                skip_count += 1
                continue

            # logging.info(f"Add:: {tile_position} => {location}, {size}")
            tiles.append(
                {
                    "location": location,
                    "size": size,
                }
            )

        logging.info(f"Total Tiles skipped: {skip_count}")
        logging.info(f"Total Tiles To Annotate: {len(tiles)}")

    # fetch_annotations(args, tiles)
    logging.info(f"Total Job time = {round(time.time() - total_start_time)}")
    print("All done!")


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
