import numpy as np
from PIL import Image
import pickle
import os.path
import math
import random
import time
from collections import defaultdict

import m_hough
import m_airplaneprops
import common

# config
testone = False
augmentable_folder_swath_sols = 'E:/workspaces/LIDAR_WORKSPACE/augmentation/augmentables_scantraces_solutions'
augmentable_folder_airplane_sols = 'E:/workspaces/LIDAR_WORKSPACE/augmentation/augmentables_scantraces_solutions'
lidar_folder = 'E:/workspaces/LIDAR_WORKSPACE/lidar'
prediction_dump_path = 'E:/workspaces/LIDAR_WORKSPACE/preds.bin'
swathspan = [3000, 3500, 4000, 4500, 5000, 5500, 6000]

class Result:
    def __init__(self, datasetname, index, position, airplane_dir, derivdirs, houghdirs, bmpspan):
        self.datasetname = datasetname
        self.index = index
        self.position = position
        self.airplane_dir = airplane_dir
        self.derivdirs = derivdirs
        self.houghdirs = houghdirs
        self.bmpspan = bmpspan

def angle_between_vectors(u, v): # should be [xu,yu], [xv, yv]
    u = np.array(u)
    v = np.array(v)
    dotprod = u.dot(v) / np.sqrt(u.dot(v)) * np.sqrt(u.dot(v))
    angle = math.acos(dotprod)
    if (angle > math.pi / 2):
        u = -u
        dotprod = u.dot(v) / np.sqrt(u.dot(u)) * np.sqrt(v.dot(v))
        angle = math.acos(dotprod)
        return angle
    return angle


# first compute the predictions
predictions = defaultdict(list)
try:
    predictions = pickle.load(open(prediction_dump_path, 'rb'))
except:
    pass

names = common.get_dataset_names(lidar_folder)
for name in names:

    print('processing ' + name)
    dataset = common.LidarDatasetNormXYZRGBAngle(lidar_folder, name)
    augs_swath = common.AugmentableSet(augmentable_folder_swath_sols, name)
    augs_plane = common.AugmentableSet(augmentable_folder_airplane_sols, name)

    method = m_airplaneprops.AirplanePropertiesEstimation(bmpsize_full_dataset=4000, bmpswathspan=swathspan)



    results = []
    for i, aug in enumerate(augs_swath.augmentables):

        start = time.time()

        if len(predictions[name]) >= i + 1 and predictions[name][i].position != []:
            continue # case already solved

        try:
            pos, ortho_dir, derivdirs, houghdirs = method.run(dataset=dataset, augmentable=aug)
            res = Result(name, i, pos, ortho_dir, derivdirs, houghdirs, swathspan)
            results.append(res)
            print('OK')
        except:
            # store index of augmentable to later view where something went wrong
            print('something went wrong')
            res = Result(name, i, [], [], [], [], [])
            results.append(res)

        end = time.time()
        print("Time taken: " + str(end - start))
    predictions[name] = results
    pickle.dump(predictions, open(prediction_dump_path, 'wb'))

# predictions[dataset] -> list(Result) ->

# compare predictions with ground truth
for name in names:

    augs_swath = common.AugmentableSet(augmentable_folder_swath_sols, name)
    augs_plane = common.AugmentableSet(augmentable_folder_airplane_sols, name)
    preds = predictions[name]

    for i, x in enumerate(augs_swath.augmentables):
        D_swath = x.directions
        D_plane = augs_plane.augmentables[i].directions
        D_pred = preds[i]

        # hough
        preds1 = []
        for p_dir in D_pred.houghdirs:
            minangle = min([angle_between_vectors(d, p_dir) for d in D_swath])
            preds1.append(minangle)
        D_pred.houghpreds = preds1

        # deriv
        preds2 = []
        for p_dir in D_pred.derivdirs:
            minangle = min([angle_between_vectors(d, p_dir) for d in D_swath])
            preds2.append(minangle)
        D_pred.derivpreds = preds2

        # plane
        minangle = min([angle_between_vectors(d, D_pred.airplane_dir) for d in D_plane])
        D_pred.airplanepred = minangle



