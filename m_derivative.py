import numpy as np
import common
import math
from PIL import Image
import random

class DerivativeMethod:

    def __init__(self, R, bmpsize_full_dataset, filter=True):
        self.bmpsize_full_dataset = bmpsize_full_dataset
        self.R = R # in meters
        self.filter = filter

    # not that dataset is already in normalized space and augmentable in in original point cloud space!
    def run(self, dataset: common.LidarDatasetNormXYZRGBAngle, augmentable: common.Augmentable):

        # find alpha of nearest neighbour
        aug_loc = (augmentable.location[0] - dataset.minx, augmentable.location[1] - dataset.miny)
        minptidx = dataset.find_closest_neighbour(aug_loc)
        minpt = dataset.points[minptidx]
        minptalpha = minpt.scan_angle

        # S_whole = assume x = 1000, take 2.5 degrees of range for it, divide by 2 because it's polmer!
        R = 2.5 * DerivativeMethod.length_of_one_degree(minptalpha, 1000.0) / 2
        bmpsizenbrs = int(self.bmpsize_full_dataset / 1000 * R)

        nbr_indices = dataset.find_neighbours(aug_loc, R=R)
        S_whole = common.PointSet([dataset.points[i] for i in nbr_indices])

        # S_small = from S_big take only the points that are the same degree
        S_small = common.PointOperations.filter_points_by_angle(S_whole, minptalpha)

        # scan direction = Fit by derivative method or hough transform to get scan direction
        bmpsizenbrs = int(self.bmpsize_full_dataset / 1000 * R) # size of 2.5 degs
        Y = common.PointOperations.transform_points_to_bmp(S_small, bmpsizenbrs)
        Y = self.circular_mask(Y)
        minangle = self.compute_bestangle(Y)
        self.visualize_at_derivative(Y, angle=minangle, padding=1)

        # FROM DEGREE RANGE ESTIMATE HEIGHT
        # find nearest points with the neighboring angle
        angle_nbrs = self.nearest_angle_neighbors(dataset=dataset, S_small=S_small)
        G = common.PointOperations.transform_points_to_bmp(angle_nbrs, bmpsizenbrs)
        common.HoughTransform.visualize_matrix(G)

        # average points
        p1, p2 = self.average_points_in_clusters(points=angle_nbrs)

        # average distance is the distance of one degree - scan_direction and height are now computable
        dist = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        scan_direction = (p1[0] - p2[0], p1[1] - p2[1])
        height = self.height_at_degree(angle=minangle, dist=dist)

        # Interpolate the true scan angle
            # take scan direction
            # in this direction find alpha - 1 and alpha + 1 points
            # linear interpolation to find out true value of angle

        # airplane way
            #@at mag at the end

    def compute_bestangle(self, Y):
        minangle = 0
        minscore = 1000000
        for x in np.linspace(0, 3.14, 180):
            score = self.derivative(Y, angle=x, padding=1)
            print(score)
            if score < minscore:
                minscore = score
                minangle = x
        return minangle

    def average_points_in_clusters(self, points: common.PointSet):

        angles = list(set([a.scan_angle for a in points.points]))
        list1x = []
        list1y = []
        list2x = []
        list2y = []

        for point in points.points:
            if point.scan_angle == angles[0]:
                list1x.append(point.X)
                list1y.append(point.Y)
            if point.scan_angle == angles[1]:
                list2x.append(point.X)
                list2y.append(point.Y)
        x1 = np.average(np.array(list1x))
        y1 = np.average(np.array(list1y))
        x2 = np.average(np.array(list2x))
        y2 = np.average(np.array(list2y))

        return (x1, y1), (x2, y2)





    def nearest_angle_neighbors(self, dataset: common.LidarDatasetNormXYZRGBAngle, S_small: common.PointSetNormalized):

        angle = S_small.points[0].scan_angle
        neighbours = []
        for point in S_small.points:
            nbridxs = dataset.find_neighbours((point.X, point.Y), R=1.0)
            for idx in nbridxs:
                if dataset.points[idx].scan_angle == angle - 1 or dataset.points[idx].scan_angle == angle + 1:
                    neighbours.append(dataset.points[idx])
        return common.PointSetNormalized(neighbours)

    def circular_mask(self, Y):
        y, x = np.ogrid[-Y.shape[0] / 2: Y.shape[0] / 2, -Y.shape[0] / 2: Y.shape[0] / 2]
        mask = x ** 2 + y ** 2 <= (Y.shape[0] / 2) ** 2
        Y = mask * Y
        return Y

    def derivative(self, Y, angle, padding):
        Y_trans = self.rotate_matrix(Y, angle)
        Y_trans_padded = np.hstack((np.zeros((padding, Y.shape[0])).T, Y_trans[:, :-padding]))
        deriv = np.abs(Y_trans - Y_trans_padded)
        score = np.sum(deriv)
        return score

    def get_line(self, angle):
        point = [0, 1]
        point_rotated = DerivativeMethod.rotate_origin_only(point, angle)
        return point_rotated

    def visualize_at_derivative(self, Y, angle, padding):

        G = np.zeros(Y.shape)
        y = Y.shape[0] / 2
        xs = np.linspace(Y.shape[0] / 4, 3 * Y.shape[0] / 4, 50)

        for i in range(xs.shape[0]):
            G[int(y), int(xs[i])] = 1
        G = self.rotate_matrix(G, angle)

        Y_trans = self.rotate_matrix(Y, angle)
        Y_trans_padded = np.hstack((np.zeros((padding, Y.shape[0])).T, Y_trans[:, :-padding]))
        deriv = np.abs(Y_trans - Y_trans_padded)

        #point = [0, 50]
        #point_rotated = HoughGlobal.rotate_origin_only(point, -angle)
        #point_rotated = np.array(point_rotated) + 80
        #xs = np.linspace(0, point_rotated[0], 50)
        #ys = np.linspace(0, point_rotated[1], 50)
        #for i in range(ys.shape[0]):
            #Y[int(xs[i]), int(ys[i])] = 2



        Y_sides = np.zeros((Y.shape[0], Y.shape[1] * 5))
        Y_sides[:, :Y.shape[0]] = Y
        Y_sides[:, Y.shape[0]:2 * Y.shape[0]] = Y_trans
        Y_sides[:, 2 * Y.shape[0]:3 * Y.shape[0]] = Y_trans_padded
        Y_sides[:, 3 * Y.shape[0]: 4 * Y.shape[0]] = deriv
        Y_sides[:, 4 * Y.shape[0]:] = G

        common.HoughTransform.visualize_matrix(Y_sides)

    @staticmethod
    def rotate_origin_only(xy, radians):
        """Only rotate a point around the origin (0, 0)."""
        x, y = xy
        xx = x * math.cos(radians) + y * math.sin(radians)
        yy = -x * math.sin(radians) + y * math.cos(radians)

        return xx, yy

    def rotate_matrix(self, A, radians):

        pos = []
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i,j] == 1:
                    pos.append([ i - A.shape[0] / 2, j - A.shape[1] / 2])

        A_trans = np.zeros(A.shape)
        for i in range(len(pos)):
            x, y = DerivativeMethod.rotate_origin_only((pos[i][0], pos[i][1]), radians)
            x, y = x + A.shape[0] / 2, y + A.shape[1] / 2
            A_trans[int(x),int(y)] = 1

        return A_trans

    @staticmethod
    def length_of_one_degree(angle, height):
        alpha1 = (angle + 1) * math.pi / 180.0
        alpha2 = angle * math.pi / 180.0
        dist = (height * math.tan(alpha1)) - height * math.tan(alpha2)
        return dist

    @staticmethod
    def height_at_degree(angle, dist):
        # we know that X * tg(alpha) = y, y and alpha are knowns
        # airplane height = X
        alpha1 = (angle + 1) * math.pi / 180.0
        alpha2 = angle * math.pi / 180.0
        height = dist / (math.tan(alpha1) - math.tan(alpha2))
        return height