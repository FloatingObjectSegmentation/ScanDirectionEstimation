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

    def run(self, dataset: common.LidarDatasetNormXYZRGBAngle, augmentable: common.Augmentable):

        print('finding points within R')
        nbrs, minptidx = common.PointOperations.find_points_within_R(dataset, augmentable, self.R)

        print('perform filtering')
        if self.filter:
            nbrs = common.PointOperations.filter_points_by_angle(nbrs, dataset[minptidx].scan_angle, self.R)

        print('to bmp')
        bmpsizenbrs = self.bmpsize_full_dataset / 1000 * self.R
        Y = common.PointOperations.transform_points_to_bmp(nbrs, bmpsizenbrs)

        # circular mask
        y, x = np.ogrid[-Y.shape[0] / 2: Y.shape[0] / 2, -Y.shape[0] / 2: Y.shape[0] / 2]
        mask = x ** 2 + y ** 2 <= (Y.shape[0] / 2) ** 2
        Y = mask * Y

        print('compute fit')
        # compute best fitting angle
        minangle = 0
        minscore = 1000000
        for x in np.linspace(0, 3.14, 180):
            score = self.derivative(Y, angle=x, padding=1)
            print(score)
            if score < minscore:
                minscore = score
                minangle = x

        self.visualize_at_derivative(Y, angle=minangle, padding=1)

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