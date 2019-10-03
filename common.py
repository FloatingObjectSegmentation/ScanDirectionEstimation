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


class RawLidarDatasetNormXYZRGBAngle:

    def __init__(self, folder, name, do_pickle=True):

        self.name = name
        self.do_pickle = do_pickle
        self.path = folder + '/' + name

        if not self.load_pickled():

            lines = open(self.path + '.txt', 'r').readlines()
            lines_scan_angles = open(self.path + 'angle.txt', 'r').readlines()

            self.points = [LidarPointXYZRGBAngle(line[0].rstrip() + ' ' + line[1].rstrip()) for line in
                           zip(lines, lines_scan_angles)]
            for i in range(len(self.points)):
                self.points[i].origidx = i
            lines = None
            lines_scan_angles = None

            self.minx, self.miny, self.points = self.normalize_points(self.points)

            pts2d = [(p.X, p.Y) for p in self.points]
            self.kdtree = KDTree(np.array(pts2d))

            if do_pickle:
                self.store_pickled()

    def find_neighbours(self, point, R): # point = (x,y)
        all_nn_indices = self.kdtree.query_radius([point], r=R)
        nn_indices = [[idx for idx in nn_indices] for nn_indices in all_nn_indices]
        return nn_indices[0]

    def find_neighbours_pointset(self, point, R):
        indices = self.find_neighbours(point, R)
        S_whole = PointSet([self.points[i] for i in indices])
        return S_whole.points

    def find_closest_neighbour(self, point):
        nearest_dist, nearest_ind = self.kdtree.query([point], k=1)
        return nearest_ind[0][0]

    def store_pickled(self):
        filename = RawLidarDatasetNormXYZRGBAngle.getserializedfilename(self.path)
        print(filename)
        pickle.dump((self.path, self.minx, self.miny, self.points, self.kdtree), open(filename, 'wb'))

    def load_pickled(self):
        filename = RawLidarDatasetNormXYZRGBAngle.getserializedfilename(self.path)
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

    def normalize_points(self, points):
        minx, miny = 10000000000000, 100000000000

        for i in range(len(points)):
            if points[i].X < minx: minx = points[i].X
            if points[i].Y < miny: miny = points[i].Y

        for i in range(len(points)):
            points[i].X -= minx
            points[i].Y -= miny
        return minx, miny, points


class LidarDatasetNormXYZRGBAngle:

    def __init__(self, points, minx, miny):
        '''
        points = (list of LidarPointXYZRGBAngle),
        minx, miny are min coords in original dataset
        '''

        self.points = points
        pts2d = [(p.X, p.Y) for p in self.points]
        self.kdtree = KDTree(np.array(pts2d))
        self.minx = minx
        self.miny = miny

    def find_neighbours(self, point, R): # point = (x,y) tuple
        all_nn_indices = self.kdtree.query_radius([point], r=R)
        nn_indices = [[idx for idx in nn_indices] for nn_indices in all_nn_indices]
        return nn_indices[0]

    def find_closest_neighbour(self, point): # point = (x,y) tuple
        nearest_dist, nearest_ind = self.kdtree.query([point], k=1)
        return nearest_ind[0][0]



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

    def __init__(self):
        pass


    def hough_line(self, img):
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


    def visualize_matrix(self, Y):
        img = Image.new('RGB', (Y.shape[0], Y.shape[1]), 'white')  # Create a new black image
        pixels = img.load()  # create the pixel map
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if Y[i, j] == 1:
                    pixels[i, j] = (255, 0, 0)
                if Y[i, j] == 2:
                    pixels[i, j] = (0, 255, 0)
        img.show()


    def visualize_scananglematrix(self, Y): # spectrum from 0 to 60
        img = Image.new('RGB', (Y.shape[0], Y.shape[1]), 'white')  # Create a new black image
        pixels = img.load()  # create the pixel map
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                red, green, blue = 0, 0, 0
                angle = Y[i, j]
                if angle % 8 == 0:
                    red = 0
                    green = 0
                    blue = 0
                elif angle % 8 == 1:
                    red = 255
                    green = 0
                    blue = 0
                elif angle % 8 == 2:
                    red = 0
                    green = 255
                    blue = 0
                elif angle % 8 == 3:
                    red = 0
                    green = 0
                    blue = 255
                elif angle % 8 == 4:
                    red = 255
                    green = 255
                    blue = 0
                elif angle % 8 == 5:
                    red = 255
                    green = 0
                    blue = 255
                elif angle % 8 == 6:
                    red = 0
                    green = 255
                    blue = 255
                elif angle % 8 == 7:
                    red = 255
                    green = 255
                    blue = 255
                pixels[i,j] = (red, green, blue)
        img.show()

class PointOperations:

    def __init__(self):
        pass


    def find_points_within_R(self, dataset: LidarDatasetNormXYZRGBAngle, augmentable: Augmentable, R):
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


    def filter_points_by_angle(self, nbrs: PointSet, scan_angle):
        nbrlist = nbrs.points
        filtered_nbrs = []
        for nbr in nbrlist:
            if nbr.scan_angle == scan_angle:
                filtered_nbrs.append(nbr)
        return PointSet(filtered_nbrs)


    def filter_points_by_angle_range(self, nbrs: PointSet, minangle, maxangle):
        nbrlist = nbrs.points
        filtered_nbrs = []
        for nbr in nbrlist:
            if nbr.scan_angle >= minangle and nbr.scan_angle <= maxangle:
                filtered_nbrs.append(nbr)
        return PointSet(filtered_nbrs)

class Visualization:

    def __init__(self):
        pass


    def visualize(self, points:PointSet, minx, miny, maxx, maxy, bmpsize):
        G = Visualization().transform_points_to_bmp_with_bounds(points, bmpsize, minx, maxx, miny, maxy)
        HoughTransform().visualize_matrix(G)


    def visualize_points(self, points, minx, miny, maxx, maxy, bmpsize):
        G = Visualization().transform_rawpoints_to_bmp_with_bounds(points, bmpsize, minx, maxx, miny, maxy)
        HoughTransform().visualize_matrix(G)


    def transform_dataset_to_scananglebmp(self, dataset: LidarDatasetNormXYZRGBAngle, bmpsize):

        X = np.zeros((bmpsize, bmpsize))
        for i in range(len(dataset.points)):
            try:
                x = (float(bmpsize) / 1000.0) * dataset.points[i].X
                y = (float(bmpsize) / 1000.0) * dataset.points[i].Y
                X[int(x), int(y)] = dataset.points[i].scan_angle + 30
            except:
                pass

        return X


    def transform_dataset_to_bmp(self, dataset: LidarDatasetNormXYZRGBAngle, bmpsize, do_pickle=True):

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


    def transform_points_to_bmp(self, points: PointSet, bmpsize: int):

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


    def transform_points_to_bmp_with_bounds(self, points: PointSet, bmpsize: int, minx, maxx, miny, maxy):

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


    def transform_rawpoints_to_bmp_with_bounds(self, points, bmpsize: int, minx, maxx, miny, maxy): # points = (x,y)

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



def load_points(lasfile, lasfilescanangles, width, height, do_pickle=True):

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

    # load the scan angles files
    lines = open(lasfilescanangles, 'r').readlines()
    angles = []
    for line in lines:
        angles.append(float(line.rstrip()))

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
            X[int(x), int(y)] = angles[i] + 30
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

def partition_list(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
