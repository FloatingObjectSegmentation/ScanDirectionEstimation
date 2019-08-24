import numpy as np
from PIL import Image
import pickle
import os.path
import math
import random

tempfolder = 'E:\\workspaces\\LIDAR_WORKSPACE\\temp'

def load_points(lasfile, width, height, do_pickle=True):


    # load lidar file and project it to 2D bmp. Return the projection.
    filename = tempfolder + '\\' + lasfile.split('\\')[-1] + str(width) + "_" + str(height) + ".bin"
    if (os.path.isfile(filename)):
        A, minx, miny = pickle.load(open(filename, 'rb'))
        return A, minx, miny


    # load the las file
    lines = open(lasfile, 'r').readlines()
    points = []
    for line in lines:
        parts = line.split(' ')
        x = float(parts[0])
        y = float(parts[1])
        points.append((x,y))


    # find mins
    minx, miny = 10000000, 10000000
    for i in range(len(points)):
        if points[i][0] < minx:
            minx = points[i][0]
        if points[i][1] < miny:
            miny = points[i][1]


    # normalize points
    for i in range(len(points)):
        points[i] = (points[i][0] - minx, points[i][1] - miny)


    # fill bitmap
    X = np.zeros((width, height))
    for i in range(len(points)):
        try:
            x = (float(width) / 1000.0) * points[i][0]
            y = (float(height) / 1000.0) * points[i][1]
            X[int(x), int(y)] = 1
        except:
            pass


    if do_pickle:
        pickle.dump((X, minx, miny), open(filename, 'wb'))

    return X, minx, miny