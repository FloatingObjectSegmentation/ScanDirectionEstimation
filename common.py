import numpy as np
from PIL import Image
import pickle
import os.path
import math
import random
import re
import time
from sklearn.neighbors import KDTree

tempfolder = 'E:\\workspaces\\LIDAR_WORKSPACE\\temp'

############################################################################################################
############################################################################################################
############################################################################################################
# DATA HOLDERS
############################################################################################################
############################################################################################################
############################################################################################################
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

class AugmentableSet:

    def __init__(self, folder, name):
        self.name = name
        self.folder = folder
        self.path = self.folder + '//' + self.name + 'augmentation_result_transformed_airplane_heights.txt'
        self.augmentables = []
        lines = open(self.path, 'r').readlines()
        for line in lines:
            aug = Augmentable.fromLine(line)
            self.augmentables.append(aug)



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
        self.origidx = 0



class LidarDatasetNormXYZRGBAngle:

    def __init__(self, folder, name, do_pickle=True):

        self.name = name
        self.do_pickle = do_pickle
        self.path = folder + '/' + name

        if not self.load_pickled():

            lines = open(self.path + '.txt', 'r').readlines()
            lines_scan_angles = open(self.path + 'angle.txt', 'r').readlines()

            self.points = [LidarPointXYZRGBAngle(line[0].rstrip() + ' ' + line[1].rstrip()) for line in zip(lines, lines_scan_angles)]
            for i in range(len(self.points)):
                self.points[i].origidx = i

            self.minx, self.miny, self.points = self.normalize_points(self.points)

            pts2d = [(p.X, p.Y) for p in self.points]
            self.kdtree = KDTree(np.array(pts2d))

            if do_pickle:
                self.store_pickled()

    # public
    def find_neighbours(self, point, R): # point = (x,y) tuple
        all_nn_indices = self.kdtree.query_radius([point], r=R)
        nn_indices = [[idx for idx in nn_indices] for nn_indices in all_nn_indices]
        return nn_indices[0]

    def find_closest_neighbour(self, point): # point = (x,y) tuple
        nearest_dist, nearest_ind = self.kdtree.query([point], k=1)
        return nearest_ind[0][0]


    # auxiliary (private)
    def normalize_points(self, points):
        minx, miny = 10000000000000, 100000000000

        for i in range(len(points)):
            if points[i].X < minx: minx = points[i].X
            if points[i].Y < miny: miny = points[i].Y

        for i in range(len(points)):
            points[i].X -= minx
            points[i].Y -= miny
        return minx, miny, points

    def store_pickled(self):
        filename = LidarDatasetNormXYZRGBAngle.getserializedfilename(self.path)
        print(filename)
        pickle.dump((self.path, self.minx, self.miny, self.points, self.kdtree), open(filename, 'wb'))

    def load_pickled(self):
        filename = LidarDatasetNormXYZRGBAngle.getserializedfilename(self.path)
        if os.path.isfile(filename) and self.do_pickle:
            self.path, self.minx, self.miny, self.points, self.kdtree = pickle.load(open(filename, 'rb'))
            return True
        return False

    @staticmethod
    def getserializedfilename(path):
        return tempfolder + "\\" + path.replace('/', '_k_k_').replace('\\', '_k_k_').replace(':', '___')

    @staticmethod
    def getpath(serialized_filename):
        fname = os.path.basename(serialized_filename)
        return fname.replace('_k_k_', '/').replace('___', ':')

class PointSetNormalized:

    def __init__(self, points): # should be list(LidarPointXYZRGBAngle)
        self.minx, self.miny, self.maxx, self.maxy, self.points = self.normalize_points(points)

    def normalize_points(self, points):
        minx, miny = 10000000000000, 100000000000
        maxx, maxy = 0, 0

        self.points = points[:]
        for i in range(len(points)):
            if self.points[i].X < minx: minx = self.points[i].X
            if self.points[i].Y < miny: miny = self.points[i].Y
            if self.points[i].X > maxx: maxx = self.points[i].X
            if self.points[i].Y > maxy: maxy = self.points[i].Y

        for i in range(len(points)):
            self.points[i].X -= minx
            self.points[i].Y -= miny

        return minx, miny, maxx, maxy, self.points

class PointSet:

    def __init__(self, points): # should be list(LidarPointXYZRGBAngle)
        self.points = points[:]
        self.minx, self.miny, self.maxx, self.maxy = self.find_extremes(self.points)

    def find_extremes(self, points):
        minx, miny = 10000000000000, 100000000000
        maxx, maxy = 0, 0

        for i in range(len(points)):
            if self.points[i].X < minx: minx = self.points[i].X
            if self.points[i].Y < miny: miny = self.points[i].Y
            if self.points[i].X > maxx: maxx = self.points[i].X
            if self.points[i].Y > maxy: maxy = self.points[i].Y

        return minx, miny, maxx, maxy



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
    def visualize_scananglematrix(Y):
        img = Image.new('RGB', (Y.shape[0], Y.shape[1]), 'white')  # Create a new black image
        pixels = img.load()  # create the pixel map
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                red, green, blue = 0,0,0
                angle = Y[i, j]
                if angle < 8:
                    red = 0
                    green = 0
                    blue = 0
                elif angle < 15:
                    red = 255
                    green = 0
                    blue = 0
                elif angle < 22:
                    red = 0
                    green = 255
                    blue = 0
                elif angle < 29:
                    red = 0
                    green = 0
                    blue = 255
                elif angle < 37:
                    red = 255
                    green = 255
                    blue = 0
                elif angle < 45:
                    red = 255
                    green = 0
                    blue = 255
                elif angle < 53:
                    red = 0
                    green = 255
                    blue = 255
                elif angle < 61:
                    red = 255
                    green = 255
                    blue = 255
                pixels[i,j] = (red, green, blue)
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

class PointOperations:

    @staticmethod
    def find_points_within_R(dataset: LidarDatasetNormXYZRGBAngle, augmentable: Augmentable, R):
        result = []

        aug = [augmentable.location[0] - dataset.minx, augmentable.location[1] - dataset.miny]

        minptidx, minptval = 0, 10000000000
        for idx, pt in enumerate(dataset.points):
            dist = np.linalg.norm(np.array(aug) - np.array([pt.X, pt.Y]))
            if dist > R:
                result.append(pt)
            if dist < minptval:
                minptidx = idx
                minptval = dist
        return result, minptidx

    @staticmethod
    def filter_points_by_angle(nbrs: PointSet, scan_angle):
        nbrlist = nbrs.points
        filtered_nbrs = []
        for nbr in nbrlist:
            if nbr.scan_angle == scan_angle:
                filtered_nbrs.append(nbr)
        return PointSet(filtered_nbrs)

    @staticmethod
    def filter_points_by_angle_range(nbrs: PointSet, minangle, maxangle):
        nbrlist = nbrs.points
        filtered_nbrs = []
        for nbr in nbrlist:
            if nbr.scan_angle >= minangle and nbr.scan_angle <= maxangle:
                filtered_nbrs.append(nbr)
        return PointSet(filtered_nbrs)

class Visualization:

    @staticmethod
    def visualize(points:PointSet, minx, miny, maxx, maxy, bmpsize):
        G = Visualization.transform_points_to_bmp_with_bounds(points, bmpsize, minx, maxx, miny, maxy)
        HoughTransform.visualize_matrix(G)

    @staticmethod
    def visualize_points(points, minx, miny, maxx, maxy, bmpsize):
        G = Visualization.transform_rawpoints_to_bmp_with_bounds(points, bmpsize, minx, maxx, miny, maxy)
        HoughTransform.visualize_matrix(G)


    @staticmethod
    def transform_dataset_to_scananglebmp(dataset: LidarDatasetNormXYZRGBAngle, bmpsize):

        X = np.zeros((bmpsize, bmpsize))
        for i in range(len(dataset.points)):
            try:
                x = (float(bmpsize) / 1000.0) * dataset.points[i].X
                y = (float(bmpsize) / 1000.0) * dataset.points[i].Y
                X[int(x), int(y)] = dataset.points[i].scan_angle + 30
            except:
                pass

        return X

    @staticmethod
    def transform_dataset_to_bmp(dataset: LidarDatasetNormXYZRGBAngle, bmpsize, do_pickle=True):

        picklepath = tempfolder + "\\" + dataset.name + ".txt" + str(bmpsize) + "_" + str(bmpsize) + ".bin"

        if do_pickle and os.path.isfile(picklepath):
            return pickle.load(open(picklepath, 'rb'))

        # fill to bmp
        X = np.zeros((bmpsize, bmpsize))
        for i in range(len(dataset.points)):
            try:
                x = (float(bmpsize) / 1000.0) * dataset.points[i].X
                y = (float(bmpsize) / 1000.0) * dataset.points[i].Y
                X[int(x), int(y)] = 1
            except:
                pass

        if do_pickle:
            pickle.dump(X, open(picklepath, 'wb'))

        return X

    @staticmethod
    def transform_points_to_bmp(points: PointSet, bmpsize: int):

        # fill to bmp
        X = np.zeros((bmpsize, bmpsize))
        for i in range(len(points.points)):
            try:
                x = (float(bmpsize) / (points.maxx - points.minx)) * (points.points[i].X - points.minx)
                y = (float(bmpsize) / (points.maxy - points.miny)) * (points.points[i].Y - points.miny)
                X[int(x), int(y)] = 1
            except:
                pass
        return X

    @staticmethod
    def transform_points_to_bmp_with_bounds(points: PointSet, bmpsize: int, minx, maxx, miny, maxy):

        # fill to bmp
        X = np.zeros((bmpsize, bmpsize))
        for i in range(len(points.points)):
            try:
                x = (float(bmpsize) / (maxx - minx)) * (points.points[i].X - minx)
                y = (float(bmpsize) / (maxy - miny)) * (points.points[i].Y - miny)
                X[int(x), int(y)] = 1
            except:
                pass
        return X

    @staticmethod
    def transform_rawpoints_to_bmp_with_bounds(points, bmpsize: int, minx, maxx, miny, maxy): # points = (x,y)

        # fill to bmp
        X = np.zeros((bmpsize, bmpsize))
        for i in range(len(points)):
            try:
                x = (float(bmpsize) / (maxx - minx)) * (points[i][0] - minx)
                y = (float(bmpsize) / (maxy - miny)) * (points[i][1] - miny)
                X[int(x), int(y)] = 1
            except:
                pass
        return X



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

def get_dataset_names(lidar_folder):
    files = [lidar_folder + '\\' + f for f in os.listdir(lidar_folder)]
    pattern = '[0-9]{3}[_]{1}[0-9]{2,3}'
    dataset_names = list(set([x.group(0) for x in [re.search(pattern, match, flags=0) for match in files] if x != None]))
    return dataset_names