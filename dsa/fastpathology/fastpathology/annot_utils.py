import signal
import numpy as np
import cv2
import large_image


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
