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
augmentable_folder = 'E:/workspaces/LIDAR_WORKSPACE/augmentation/augmentables'
lidar_folder = 'E:/workspaces/LIDAR_WORKSPACE/lidar'

# augmentable reading
def read_augmentables(filepath):
    lines = open(filepath, 'r').readlines()
    return [common.Augmentable.fromLine(line) for line in lines]

def read_lidar(filepath):
    lines = open(filepath, 'r').readlines()
    return [common.LidarPointXYZRGBAngle(line) for line in lines]

def testonemethod():
    # load all lidar file names
    names = common.get_dataset_names(lidar_folder)
    for name in names:
        name = '391_38'
        method = m_hough.HoughMethod(20, 3500)
        start = time.time()
        dataset = common.LidarDatasetNormXYZRGBAngle(lidar_folder, name)
        end = time.time()
        method.run(dataset, 50)
        print(end - start)

if testone:
    testonemethod()

names = common.get_dataset_names(lidar_folder)
for name in names:

    name = '391_38'
    dataset = common.LidarDatasetNormXYZRGBAngle(lidar_folder, name)
    augset = common.AugmentableSet(augmentable_folder, name)

    for aug in augset.augmentables:
        method = m_airplaneprops.AirplanePropertiesEstimation(R=30, bmpsize_full_dataset=4000, filter=True)
        method.run(dataset=dataset, augmentable=aug)






# Prerequisites:
# Lidar files need to ne xyzad - a is for scan angle, d is for scan direction
# make a class to hold and augmentable, will be more readable that way!
# define functions for comparison metrics: divide angles into bins, average angle error + std dev
# save pickles frequently

# load all lidar txt files
# load all solved augmentables

# R = working radius around augmentable

# resultsA[DATASET][METHOD][COMPARISON_METRIC] = [list of predictions] -> for R independent methods
# resultsB[DATASET][METHOD][COMPARISON_METRIC][R] = [list of predictions] -> for R dependent methods

# sort LIDAR_CHUNK, AUGS so they are the same
# DATA = zip(LIDAR_CHUNKS, AUGS)

############################################################################################################
## RESULT ACCUMULATION
############################################################################################################
# foreach D in DATA:

    # foreach aug in D:

        #real_direction = realdirection of aug


        # foreach radius R in [select some sensible ones]:

            # filtered = find neighbor points in radius R
            # filtered = filter by scan angle

            # foreach comparison_metric in (function pointers):

                # B = fit the hough transform filtered by scan angle in radius R
                # C = fit lines to nearby indices by scan angle
                # r1 = compare(real_direction, B, metric=comparison_metric)
                # r2 = compare (real_direction, C, metric=comparsion_metric)

                # resultsB[DATASET][METHOD=HOUGHSCANANGLE][COMPARISON_METRIC=comparison_metric][R].append(r1)
                # resultsB[DATASET][METHOD=LINEFIT][COMPARISON_METRIC=comparison_metric][R].append(r2)



        # B = fit the hough transform global
        # C = fit hough transform local
        # r1 = compare(real_direction, B)
        # r2 = compare(real_direction, C)

        # resultsA[DATASET][METHOD=HOUGHLOCAL][COMPARISON_METRIC=comparison_metric][R].append(r1)
        # resultsA[DATASET][METHOD=HOUGHGLOBAL][COMPARISON_METRIC=comparison_metric][R].append(r2)


############################################################################################################
## RESULT AGGREGATION
############################################################################################################
# Make graphs for each method and each comparison metric
# For R independent methods:
#     problem because you still have the thing about inner hyperparameters: how much area around are you gonna see in local?
#     how finely grained are gonna be the global chunks?
# For R dependent methods:
#     they really do depend only on R so this is ok.
#
#
#
#







# 1. find neighbor points in radius R (radius R):
    # for augmentable a in augmentables:
    # for pt in points:
        # all points that are less than R away by l2dist
    # return filtered points

# 2. filter by scan angle
#    # Find 10 nearest points to the augmentable
     # angle = randomly select on and pick scan angle
     # filter all others whose scan angle is different by +-1
     # return points

# 3. fit hough transform
#
