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





names = common.get_dataset_names(lidar_folder)
for name in names:

    name = '391_38'
    dataset = common.LidarDatasetNormXYZRGBAngle(lidar_folder, name)
    augset = common.AugmentableSet(augmentable_folder_swath_sols, name)

    for aug in augset.augmentables:
        method = m_airplaneprops.AirplanePropertiesEstimation(bmpsize_full_dataset=4000, filter=True)
        pos, ortho_dir, scandir_deriv, scandir_hough = method.run(dataset=dataset, augmentable=aug)
        aug.directions