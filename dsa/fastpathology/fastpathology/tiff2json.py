import fast
import numpy as np
import json


def convert(image_path):
    # Data to be written
    dictionary = {
        "_accessLevel": "sathiyajith",
        "_id": 56,
        "_modelType": 8.6,
        "_version": "9976770500",
        "annotation": {
            "elements": 1
        },
        "name": 1,
        "created": 1,
        "creatorId": 1,
        "groups": "[null]",
        "itemId": 1,
        "public": "false",
        "updated": 1,
        "updatedId": 1,
        "_elementQuery": {
            "count": 1,
            "details": 1,
            "filter": {
                "_version": 1,
                "annotationId": 1
            },
            "offset": 0,
            "returend": 1,
            "sort": ["_id", 1]
        },
    }

    
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # loop over pyramidal TIFF segmentation image and export annotations in JSON format
    annot_list = []

    any_gt_found = np.any(im_nuclei_seg_mask)

    if flag_nuclei_found:
        nuclei_annot_list = cli_utils.create_tile_nuclei_annotations(
            im_nuclei_seg_mask, tile_info, args.nuclei_annotation_format)
    
    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)

    pass


if __name__ == "__main__":
    #convert()
    pass
