import numpy as np
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility


from ..catalog import DatasetCatalog, MetadataCatalog
from models.structures.boxes import BoxMode

import os
from alfred.utils.log import logger as logging
import pickle


'''
if your dataset not coco format,
try convert it into detectron2 standard format first
the register it by adding all these dict

we have nuscenes for example

'''


def get_nuscenes_dicts(path="./", version='v1.0-mini', categories=None):
    """
    This is a helper fuction that create dicts from nuscenes to detectron2 format.
    Nuscenes annotation use 3d bounding box, but for detectron we need 2d bounding box.
    The simplest solution is get max x, min x, max y and min y coordinates from 3d bb and
    create 2d box. So we lost accuracy, but this is not critical.
    :param path: <string>. Path to Nuscenes dataset.
    :param version: <string>. Nuscenes dataset version name.
    :param categories <list<string>>. List of selected categories for detection.
        Get from https://www.nuscenes.org/data-annotation
        Categories names:
            ['human.pedestrian.adult',
             'human.pedestrian.child',
             'human.pedestrian.wheelchair',
             'human.pedestrian.stroller',
             'human.pedestrian.personal_mobility',
             'human.pedestrian.police_officer',
             'human.pedestrian.construction_worker',
             'animal',
             'vehicle.car',
             'vehicle.motorcycle',
             'vehicle.bicycle',
             'vehicle.bus.bendy',
             'vehicle.bus.rigid',
             'vehicle.truck',
             'vehicle.construction',
             'vehicle.emergency.ambulance',
             'vehicle.emergency.police',
             'vehicle.trailer',
             'movable_object.barrier',
             'movable_object.trafficcone',
             'movable_object.pushable_pullable',
             'movable_object.debris',
             'static_object.bicycle_rack']
    :return: <dict>. Return dict with data annotation in detectron2 format.
    """
    logging.info('your nuscenes root: {}'.format(path))
    # assert(path[-1] == "/"), "Insert '/' in the end of path"
    path = os.path.abspath(path) + '/'
    nusc = NuScenes(version=version, dataroot=path, verbose=False)
    logging.info('nuScenes object loaded done.')

    # Select all catecategories if not set
    if categories == None:
        categories = [data["name"] for data in nusc.category]
    assert(isinstance(categories, list)), "Categories type must be list"

    dataset_dicts = []
    cache_f = os.path.join(path, 'ct_nuscenes.cache')
    if os.path.exists(cache_f):
        dataset_dicts = pickle.load(open(cache_f, 'rb'))
        logging.info('resumed nuscenes cache file: {}'.format(cache_f))
        return dataset_dicts
    idx = 0
    logging.info('start parsing nuscenes data to train dict format...')
    for i in tqdm(range(0, len(nusc.scene))):
        scene = nusc.scene[i]
        scene_rec = nusc.get('scene', scene['token'])
        sample_rec_cur = nusc.get('sample', scene_rec['first_sample_token'])

        # Go through all frame in current scene
        while True:
            # we maybe add CAM_LEFT as well
            data = nusc.get('sample_data', sample_rec_cur['data']["CAM_FRONT"])

            record = {}
            record["file_name"] = path + data["filename"]
            if os.path.exists(record['file_name']):
                record["image_id"] = idx
                record["height"] = data["height"]
                record["width"] = data["width"]
                idx += 1

                # Get boxes from front camera
                _, boxes, camera_intrinsic = nusc.get_sample_data(
                    sample_rec_cur['data']["CAM_FRONT"], BoxVisibility.ANY)
                # Get only necessary boxes
                boxes = [box for box in boxes if box.name in categories]
                # Go through all bounding boxes
                objs = []
                for box in boxes:
                    corners = view_points(
                        box.corners(), camera_intrinsic, normalize=True)[:2, :]
                    max_x = int(max(corners[:][0]))
                    min_x = int(min(corners[:][0]))
                    max_y = int(max(corners[:][1]))
                    min_y = int(min(corners[:][1]))
                    obj = {
                        "bbox": [min_x, min_y, max_x, max_y],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": categories.index(box.name),
                        "iscrowd": 0
                    }
                    objs.append(obj)

                record["annotations"] = objs
                dataset_dicts.append(record)

                # Get next frame
                sample_rec_cur = nusc.get('sample', sample_rec_cur['next'])
                # End of scene condition
                if (sample_rec_cur["next"] == ""):
                    break
            else:
                logging.info('pass record: {} since image not found!'.format(record['file_name']))
                sample_rec_cur = nusc.get('sample', sample_rec_cur['next'])
                if (sample_rec_cur["next"] == ""):
                    break
                continue
    logging.info('data convert done! all samples: {}'.format(len(dataset_dicts)))
    if not os.path.exists(cache_f):
        os.makedirs(os.path.dirname(cache_f), exist_ok=True)
        pickle.dump(dataset_dicts, open(cache_f, 'wb'))
        logging.info('nuscenes cache saved into: {}'.format(cache_f))
    return dataset_dicts
