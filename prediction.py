import numpy as np
from PIL import Image
import pickle
import os.path
import math
import random
import time
from collections import defaultdict
import multiprocessing

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

    def printresult(self):
        print(self.datasetname + ' ' + str(self.index) + ' ' + str(self.position) + ' airplanedir: ' + str(self.airplane_dir))
        print('derivdirs: ' + str(self.derivdirs))
        print('houghdirs: ' + str(self.houghdirs))

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

class OneAugmentableWork:

    def __init__(self, name, dataset, aug, i):
        self.method = m_airplaneprops.AirplanePropertiesEstimation(bmpsize_full_dataset=4000, bmpswathspan=swathspan)
        self.result = []
        self.name = name
        self.dataset = dataset
        self.aug = aug
        self.i = i

    def work(self, threadindex=-1):
        print('start work' + str(threadindex))
        if len(predictions[name]) >= self.i + 1 and predictions[name][self.i].position != []:
            self.result = None

        try:
            pos, ortho_dir, derivdirs, houghdirs = self.method.run(dataset=self.dataset, augmentable=self.aug)
            self.result = Result(self.name, self.i, pos, ortho_dir, derivdirs, houghdirs, swathspan)
        except:
            # store index of augmentable to later view where something went wrong
            print('something went wrong')
            self.result = Result(name, i, [], [], [], [], [])
        print('finish work' + str(threadindex))

# first compute the predictions
predictions = defaultdict(list)
try:
    predictions = pickle.load(open(prediction_dump_path, 'rb'))
except:
    pass

names = common.get_dataset_names(lidar_folder)
names.sort()

for name in names:

    print('processing ' + name)
    dataset = common.RawLidarDatasetNormXYZRGBAngle(lidar_folder, name)
    augs = common.AugmentableSet(augmentable_folder_swath_sols, name)

    partition = list(common.partition_list(augs.augmentables, 5))

    results = []
    for chunk in partition:

        start = time.time()
        w0 = OneAugmentableWork(name=name, dataset=dataset, aug=chunk[0], i=chunk[0].idx)
        w1 = OneAugmentableWork(name=name, dataset=dataset, aug=chunk[1], i=chunk[1].idx)
        w2 = OneAugmentableWork(name=name, dataset=dataset, aug=chunk[2], i=chunk[2].idx)
        w3 = OneAugmentableWork(name=name, dataset=dataset, aug=chunk[3], i=chunk[3].idx)
        w4 = OneAugmentableWork(name=name, dataset=dataset, aug=chunk[4], i=chunk[4].idx)

        t0 = multiprocessing.Process(target=w0.work, args=(0,))
        t1 = multiprocessing.Process(target=w1.work, args=(1,))
        t2 = multiprocessing.Process(target=w2.work, args=(2,))
        t3 = multiprocessing.Process(target=w3.work, args=(3,))
        t4 = multiprocessing.Process(target=w4.work, args=(4,))

        t0.start()
        t1.start()
        t2.start()
        t3.start()
        t4.start()

        t0.join()
        t1.join()
        t2.join()
        t3.join()
        t4.join()

        results.append(w0.result)
        results.append(w1.result)
        results.append(w2.result)
        results.append(w3.result)
        results.append(w4.result)

        for r in results:
            r.printresult()

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



