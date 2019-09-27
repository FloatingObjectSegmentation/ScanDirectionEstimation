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


def transform_points_to_bmp(points, bmpsize):

    # find minimum and maximum point within points
    minx, miny = 100000000000, 10000000000
    for pt in points:
        if pt.x < minx: minx = pt.x
        if pt.y < miny: miny = pt.y

    # normalize points
    for i in range(len(points)):
        points[i].x -= minx
        points[i].y -= miny

    # fill to bmp
    X = np.zeros((bmpsize, bmpsize))
    for i in range(len(points)):
        try:
            x = (float(bmpsize) / 1000.0) * points[i].x
            y = (float(bmpsize) / 1000.0) * points[i].y
            X[int(x), int(y)] = 1
        except:
            pass

    return X


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


## HOUGH TRANSFORM WITH HELPER METHODS
class HoughTransform:

    @staticmethod
    def hough_line(img):
      #Rho and Theta ranges
      thetas = np.deg2rad(np.arange(-90.0, 90.0))
      width, height = img.shape
      diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
      rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

      # Cache some reusable values
      cos_t = np.cos(thetas)
      sin_t = np.sin(thetas)
      num_thetas = len(thetas)

      # Hough accumulator array of theta vs rho
      accumulator = np.zeros((2 * int(diag_len), num_thetas), dtype=np.uint64)
      y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

      # Vote in the hough accumulator
      for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        if i % 10000 == 0:
            print(i)
        for t_idx in range(num_thetas):
          # Calculate rho. diag_len is added for a positive index
          rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
          accumulator[int(rho), int(t_idx)] += 1

      return accumulator, thetas, rhos

    @staticmethod
    def visualize_matrix(Y):
        img = Image.new('RGB', (Y.shape[0], Y.shape[1]), 'white')  # Create a new black image
        pixels = img.load()  # create the pixel map
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if Y[i, j] == 1:
                    pixels[i, j] = (255, 0, 0)
                if Y[i, j] == 2:
                    pixels[i, j] = (0, 255, 0)
        img.show()

    @staticmethod
    def visualize_accumulator(accumulator):
        img = Image.new('RGB', (accumulator.shape[0], accumulator.shape[1]), 'white')
        pixels = img.load()
        for i in range(accumulator.shape[0]):
            for j in range(accumulator.shape[1]):
                pixels[i, j] = (int(accumulator[i, j] / np.max(accumulator) * 255), 0, 0)
        img.show()

    @staticmethod
    def insert_resulting_lines(Y, accumulator, rhos, thetas):

        idx0 = np.argpartition(accumulator.ravel(), -3)[-3:]
        idxs = idx0[accumulator.ravel()[idx0].argsort()][::-1]
        for idx in idxs:
            rho = rhos[int(idx / accumulator.shape[1])]
            theta = thetas[idx % accumulator.shape[1]]

            for i in range(Y.shape[0]):
                try:
                    x = i
                    y = rho / math.sin(theta) - x * math.cos(theta) / math.sin(theta)
                    Y[int(y), int(x)] = 2
                except:
                    pass
        return Y
