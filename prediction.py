import numpy as np
from PIL import Image
import pickle
import os.path
import math
import random
import time

import m_hough
import m_airplaneprops
import common

# config
testone = False
augmentable_folder_swath_sols = 'E:/workspaces/LIDAR_WORKSPACE/augmentation/augmentables_scantraces_solutions'
augmentable_folder_airplane_sols = 'E:/workspaces/LIDAR_WORKSPACE/augmentation/augmentables_scantraces_solutions'
lidar_folder = 'E:/workspaces/LIDAR_WORKSPACE/lidar'
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



# first compute the predictions

names = common.get_dataset_names(lidar_folder)
for name in names:

    dataset = common.LidarDatasetNormXYZRGBAngle(lidar_folder, name)
    augs_swath = common.AugmentableSet(augmentable_folder_swath_sols, name)
    augs_plane = common.AugmentableSet(augmentable_folder_airplane_sols, name)

    method = m_airplaneprops.AirplanePropertiesEstimation(bmpsize_full_dataset=4000, bmpswathspan=swathspan)


    results = []
    for i, aug in enumerate(augs_swath.augmentables):

        try:
            pos, ortho_dir, derivdirs, houghdirs = method.run(dataset=dataset, augmentable=aug)
            res = Result(name, i, pos, ortho_dir, derivdirs, houghdirs, swathspan)
            results.append(res)
        except:
            # store index of augmentable to later view where something went wrong
            res = Result(name, i, [], [], [], [], [])
            results.append(res)

    # pickle predictions
