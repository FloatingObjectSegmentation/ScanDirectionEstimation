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
augmentable_folder_airplane_sols = 'E:/workspaces/LIDAR_WORKSPACE/augmentation/augmentables_airways_solutions'
lidar_folder = 'E:/workspaces/LIDAR_WORKSPACE/lidar'
prediction_dump_path = 'E:/workspaces/LIDAR_WORKSPACE/preds.bin'
swathspan = [2000, 3000, 4000, 5000, 6000] #, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

############################################################################
## AUGMENTABLE LEVEL LOGIC
############################################################################

class Result:
    def __init__(self, datasetname, index, position, airplane_dir, derivdirs, houghdirs, bmpspan):
        self.datasetname = datasetname
        self.index = index
        self.position = position
        self.airplane_dir = airplane_dir
        self.derivdirs = derivdirs
        self.houghdirs = houghdirs
        self.bmpspan = bmpspan

        self.houghpreds = []
        self.derivpreds = []
        self.airplanepred = -1
        self.airplanepred_swath = -1

    def printresult(self):
        print(self.datasetname + ' ' + str(self.index) + ' ' + str(self.position) + ' airplanedir: ' + str(self.airplane_dir))
        print('derivdirs: ' + str(self.derivdirs))
        print('houghdirs: ' + str(self.houghdirs))

def angle_between_vectors(u, v): # should be [xu,yu], [xv, yv]
    u = np.array(u)
    v = np.array(v)
    dotprod = u.dot(v) / (np.sqrt(u.dot(u)) * np.sqrt(v.dot(v)))
    angle = math.acos(dotprod)
    if (angle > math.pi / 2):
        u = -u
        dotprod = u.dot(v) / (np.sqrt(u.dot(u)) * np.sqrt(v.dot(v)))
        angle = math.acos(dotprod)
        return angle * 180 / math.pi
    return angle * 180 / math.pi


class OneAugmentableWork:

    def __init__(self, name, dataset, aug, i, do_work=True):
        self.method = m_airplaneprops.AirplanePropertiesEstimation(bmpsize_full_dataset=4000, bmpswathspan=swathspan)
        self.result = []
        self.name = name
        self.dataset = dataset
        self.aug = aug
        self.i = i
        self.do_work = do_work

    def work(self, threadindex=-1, queue=multiprocessing.Queue()):
        print('start work' + str(threadindex))
        if self.do_work == False:
            queue.put(Result(self.name, self.i, [], [], [], [], []))
            return
        try:
            pos, ortho_dir, derivdirs, houghdirs = self.method.run(dataset=self.dataset, augmentable=self.aug)
            queue.put(Result(self.name, self.i, pos, ortho_dir, derivdirs, houghdirs, swathspan))
        except:
            # store index of augmentable to later view where something went wrong
            print('something went wrong')
            queue.put(Result(self.name, self.i, [], [], [], [], []))
        print('finish work' + str(threadindex))

############################################################################
## CHUNK LEVEL LOGIC
############################################################################

def predict(name, predictions):

    print('processing ' + name)
    dataset = common.RawLidarDatasetNormXYZRGBAngle(lidar_folder, name)
    augs = common.AugmentableSet(augmentable_folder_swath_sols, name)

    for aug in augs.augmentables:
        x, y = aug.location[0] - dataset.minx, aug.location[1] - dataset.miny
        nbrs = dataset.find_neighbours_pointset((x, y), 50.0)
        data = common.LidarDatasetNormXYZRGBAngle(nbrs, dataset.minx, dataset.miny)

        do_work = True
        if len(predictions[name]) >= aug.idx + 1 and predictions[name][aug.idx].position != []:
            do_work = False

        wrk = OneAugmentableWork(name=name, dataset=data, aug=aug, i=aug.idx, do_work=do_work)
        wrk.work()

def predict_parallel(name ,predictions):

    print('processing ' + name)
    dataset = common.RawLidarDatasetNormXYZRGBAngle(lidar_folder, name)
    augs = common.AugmentableSet(augmentable_folder_swath_sols, name)

    partition = list(common.partition_list(augs.augmentables, 6))

    results = []
    for chunk in partition:

        start = time.time()

        w, p, qs = [], [], []
        for i in range(len(chunk)):

            # prepare
            x, y = chunk[i].location[0] - dataset.minx, chunk[i].location[1] - dataset.miny
            nbrs = dataset.find_neighbours_pointset((x, y), 50.0)
            data = common.LidarDatasetNormXYZRGBAngle(nbrs, dataset.minx, dataset.miny)

            # verify if it is already processed
            do_work = True
            if name in predictions.keys() and len(predictions[name]) >= i + 1 and predictions[name][i] != [] and predictions[name][i].position != []:
                print('will not do work')
                do_work = False


            q = multiprocessing.Queue()
            wrk = OneAugmentableWork(name=name, dataset=data, aug=chunk[i], i=chunk[i].idx, do_work=do_work)
            w.append(wrk)
            thr = multiprocessing.Process(target=wrk.work, args=(i,q))
            p.append(thr)
            qs.append(q)

        print('starting procs')
        for i in range(len(p)):
            p[i].start()

        print('joining procs')
        for i in range(len(p)):
            p[i].join()

        for i in range(len(w)):
            results.append(qs[i].get())

        # for r in results:
        #    r.printresult()

        end = time.time()
        print("Time taken: " + str(end - start))

    # only add successful predictions
    if name in predictions.keys():
        for i in range(len(results)):
            if results[i].position == []:
                continue
            if predictions[name][i].position != []:
                continue
            predictions[name][i] = results[i]
    else:
        predictions[name] = results

    pickle.dump(predictions, open(prediction_dump_path, 'wb'))

############################################################################
## DATASET LEVEL LOGIC
############################################################################

if __name__ == '__main__':
    # first compute the predictions
    predictions = defaultdict(list)
    try:
        predictions = pickle.load(open(prediction_dump_path, 'rb'))
    except:
        pass

    names = common.get_dataset_names(lidar_folder)
    names.sort()

    #for name in names:
     #   predict_parallel(name, predictions)






    # compare predictions with ground truth
    nameidx = 0
    for name in names:
        nameidx += 1
        print("processing " + str(nameidx) + " " + name)

        if name not in predictions.keys():
            break

        preds = predictions[name]
        augs_swath = common.AugmentableSet(augmentable_folder_swath_sols, name)

        # UNWRAP PREDICTIONS CORRECTLY
        for i, x in enumerate(augs_swath.augmentables):

            D_pred = preds[i]
            if D_pred.airplane_dir == []:
                continue

            # change hough predictions into actual directions
            D_pred.houghdirs = [d[0] for d in D_pred.houghdirs]
            D_pred.houghdirs = [np.array(d[0]) - np.array(d[1]) for d in D_pred.houghdirs]

            # change deriv predictions into actual directions
            D_pred.derivdirs = [d[0] for d in D_pred.derivdirs]
            D_pred.derivdirs = [np.array(d[0]) - np.array(d[1]) for d in D_pred.derivdirs]

            # change airplane predictions into actual direction
            try:
                D_pred.airplane_dir = np.array(D_pred.airplane_dir[0]) - np.array(D_pred.airplane_dir[1])
            except:
                pass


        # SWATH ANGLES
        for i, x in enumerate(augs_swath.augmentables):

            D_swath = x.directions
            D_pred = preds[i]
            if D_pred.airplane_dir == []:
                continue

            # change swath labels to actual directions
            D_swath = [np.array(a[0]) - np.array(a[1]) for a in common.partition_list(D_swath, 2)]

            # hough
            preds1 = []
            for p_dir in D_pred.houghdirs:
                minangle = min([angle_between_vectors(d, p_dir) for d in D_swath])
                preds1.append(minangle)
            D_pred.houghpreds = preds1
            print(preds1)

            # deriv
            preds2 = []
            for p_dir in D_pred.derivdirs:
                minangle = min([angle_between_vectors(d, p_dir) for d in D_swath])
                preds2.append(minangle)
            D_pred.derivpreds = preds2
            print(preds2)
            print()



        # AIRPLANE ANGLES SWATH
        #augs_plane = common.AugmentableSet(augmentable_folder_airplane_sols, name,
        #                                  appendix='augmentation_result_transformed.txt')
        for i, x in enumerate(augs_swath.augmentables):

            D_plane = x.directions
            D_pred = preds[i]
            if D_pred.airplane_dir == []:
                continue

            # change airplane labels to actual directions
            D_plane = [np.array(a[0]) - np.array(a[1]) for a in common.partition_list(D_plane, 2)]

            # plane
            minangle = min([angle_between_vectors(d, D_pred.airplane_dir) for d in D_plane])
            D_pred.airplanepred_swath = minangle
            print(minangle)


        # #AIRPLANE ANGLES
        # try:
        #     augs_plane = common.AugmentableSet(augmentable_folder_airplane_sols, name,
        #                                   appendix='augmentation_result_transformed.txt')
        # except:
        #     continue
        #
        # for i in range(len(augs_swath.augmentables)):
        #
        #     D_pred = preds[i]
        #     if D_pred.airplane_dir == []:
        #         continue
        #
        #     D_plane = augs_swath.augmentables[i].directions
        #     D_plane = [np.array(a[0]) - np.array(a[1]) for a in common.partition_list(D_plane, 2)]
        #
        #     if i > 0:
        #
        #         D_planeprev = augs_swath.augmentables[i - 1].directions
        #         D_planeprev = [np.array(a[0]) - np.array(a[1]) for a in common.partition_list(D_planeprev, 2)]
        #
        #         if all([x[0] == y[0] and x[1] == y[1] for x,y in zip(D_plane, D_planeprev)]):
        #             continue
        #
        #     # plane
        #     minangle = min([angle_between_vectors(d, D_pred.airplane_dir) for d in D_plane])
        #     D_pred.airplanepred = minangle
        #     print(minangle)


    # PREDICTIONS TO LATEX TABLES
    mainlines = ''

    avghoughbynames = []
    stdhoughbynames = []
    avgderivbynames = []
    stdderivbynames = []
    avgairplanebynames = []
    stdairplanebynames = []

    for name in names:

        if name not in predictions.keys():
            break

        swathpreds = []
        derivpreds = []
        houghpreds = []

        for pred in predictions[name]:

            if pred.airplanepred_swath != []:
                swathpreds.append(pred.airplanepred_swath)
            if pred.derivpreds != []:
                derivpreds.append(pred.derivpreds)
            if pred.houghpreds != []:
                houghpreds.append(pred.houghpreds)

        avgpred = np.average(np.array(swathpreds))
        stddevpred = np.std(np.array(swathpreds))


        avgderiv = []
        stdderiv = []
        for i in range(len(swathspan)):
            x = np.average(np.array([p[i] for p in derivpreds]))
            y = np.std(np.array([p[i] for p in derivpreds]))
            avgderiv.append(x)
            stdderiv.append(y)



        avghough = []
        stdhough = []
        for i in range(len(swathspan)):
            x = np.average(np.array([p[i] for p in houghpreds]))
            y = np.std(np.array([p[i] for p in houghpreds]))
            avghough.append(x)
            stdhough.append(y)


        def format(num):
            return '{0:.2f}'.format(num)


        a = name.split('_')

        avgairplanebynames.append(avgpred)
        stdairplanebynames.append(stddevpred)
        avghoughbynames.append(np.min(np.array(avghough)))
        stdhoughbynames.append(np.min(np.array(stdhough)))
        avgderivbynames.append(np.min(np.array(avgderiv)))
        stdderivbynames.append(np.min(np.array(stdderiv)))

        # main table
        # line = a[0] + '\\_' + a[1] + ' & '
        # line += format(avgpred) + ' & '
        # line += format(stddevpred) + ' & '
        # line += format(np.min(np.array(avghough))) + ' & '
        # line += format(np.min(np.array(stdhough))) + ' & '
        # line += format(np.min(np.array(avgderiv))) + ' & '
        # line += format(np.min(np.array(stdderiv))) + '\\\\ \n'
        # line += '\\hline'
        # print(line)


        # hough table by bmpsize
        line = a[0] + '\\_' + a[1] + ' & '
        for i in range(len(avghough)):
            line += format(avghough[i]) + ' & '
        line = line[:-3] + '\\\\ \n'
        line += '\\hline'
        print(line)

        # deriv table by bmpsize
        # line = a[0] + '\\_' + a[1] + ' & '
        # for i in range(len(avgderiv)):
        #     line += format(avgderiv[i]) + ' & '
        # line = line[:-3] + '\\\\ \n'
        # line += '\\hline'
        # print(line)

    # aggregate by names
    def avgbynames(array):
        return '\\textbf{' + format(np.average(np.array(array))) + "}"
    print('AVG BY NAMES')
    line = " & "
    line += avgbynames(avgairplanebynames) + " & "
    line += avgbynames(stdairplanebynames) + " & "
    line += avgbynames(avghoughbynames) + " & "
    line += avgbynames(stdhoughbynames) + " & "
    line += avgbynames(avgderivbynames) + " & "
    line += avgbynames(stdderivbynames) + "\\\\ \n"

    print(line)



