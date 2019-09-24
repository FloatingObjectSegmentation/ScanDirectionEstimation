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


## AUGMENTABLE DEFINITION
class Augmentable:

    def __init__(self, idx, location, scale, tp, airplane_pos, dff, directions):
        self.idx = idx
        self.location = location
        self.scale = scale
        self.type = tp
        self.airplane_pos = airplane_pos
        self.distance_from_floor = dff
        self.directions = directions

    @staticmethod
    def vecStringToFloatLst(line):
        return [float(a) for a in line.split(',')]

    @staticmethod
    def fromLine(line):
        parts = line.split(' ')
        idx = int(parts[0])
        location = Augmentable.vecStringToFloatLst(parts[1])
        scale = Augmentable.vecStringToFloatLst(parts[2])
        type = parts[3]
        airplane_pos = Augmentable.vecStringToFloatLst(parts[4])
        distance_from_floor = float(parts[5])
        directions = [Augmentable.vecStringToFloatLst(part) for part in parts[6:]]
        return Augmentable(idx, location, scale, type, airplane_pos, distance_from_floor, directions)


## LIDAR DEFINITION
class LidarPointXYZRGBAngle:

    def __init__(self, line):
        parts = line.split(' ')
        self.X = float(parts[0])
        self.Y = float(parts[1])
        self.Z = float(parts[2])
        self.R = int(parts[3])
        self.G = int(parts[4])
        self.B = int(parts[5])
        self.scan_angle = int(parts[6])

