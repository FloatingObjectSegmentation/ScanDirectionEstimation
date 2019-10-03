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
swathspan = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

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

    def __init__(self, name, dataset, aug, i, do_work=True):
        self.method = m_airplaneprops.AirplanePropertiesEstimation(bmpsize_full_dataset=4000, bmpswathspan=swathspan)
        self.result = []
        self.name = name
        self.dataset = dataset
        self.aug = aug
        self.i = i
        self.do_work = do_work

    def work(self, threadindex=-1):
        print('start work' + str(threadindex))
        if self.do_work == False:
            self.result = Result(name, i, [], [], [], [], [])
            return
        try:
            pos, ortho_dir, derivdirs, houghdirs = self.method.run(dataset=self.dataset, augmentable=self.aug)
            self.result = Result(self.name, self.i, pos, ortho_dir, derivdirs, houghdirs, swathspan)
        except:
            # store index of augmentable to later view where something went wrong
            print('something went wrong')
            self.result = Result(self.name, i, [], [], [], [], [])
        print('finish work' + str(threadindex))


def predict(name, predictions):
    pass
def predict_parallel(name ,predictions):

    print('processing ' + name)
    dataset = common.RawLidarDatasetNormXYZRGBAngle(lidar_folder, name)
    augs = common.AugmentableSet(augmentable_folder_swath_sols, name)

    partition = list(common.partition_list(augs.augmentables, 4))

    results = []
    for chunk in partition:

        start = time.time()

        w, p = [], []
        for i in range(len(chunk)):

            # prepare
            x, y = chunk[i].location[0] - dataset.minx, chunk[i].location[1] - dataset.miny
            nbrs = dataset.find_neighbours_pointset((x, y), 50.0)
            data = common.LidarDatasetNormXYZRGBAngle(nbrs, dataset.minx, dataset.miny)

            # verify if it is already processed
            do_work = True
            if len(predictions[name]) >= i + 1 and predictions[name][i].position != []:
                do_work = False

            wrk = OneAugmentableWork(name=name, dataset=data, aug=chunk[i], i=chunk[i].idx, do_work=do_work)
            w.append(wrk)
            thr = multiprocessing.Process(target=wrk.work, args=(i,))
            p.append(thr)

        print('starting procs')
        for i in range(len(p)):
            p[i].start()

        print('joining procs')
        p[0].join()
        p[1].join()
        p[2].join()
        p[3].join()

        for i in range(len(w)):
            results.append(w[i].result)

        # for r in results:
        #    r.printresult()

        end = time.time()
        print("Time taken: " + str(end - start))

        exit(0)

    predictions[name] = results
    pickle.dump(predictions, open(prediction_dump_path, 'wb'))

if __name__ == '__main__':
    # first compute the predictions
    predictions = defaultdict(list)
    try:
        predictions = pickle.load(open(prediction_dump_path, 'rb'))
    except:
        pass

    names = common.get_dataset_names(lidar_folder)
    names.sort()

    for name in names:
        predict_parallel(name, predictions)















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



