import numpy as np
import common
import math
from PIL import Image
import random


class DerivativeMethod:

    def __init__(self, do_visualization=False):
        self.do_visualization = do_visualization
        pass

    def scan_direction_derivative(self, S_whole, angle, bmpsize):
        pass
        # filter by neighboring scan angles S_whole
        minx, maxx, miny, maxy = S_whole.minx, S_whole.maxx, S_whole.miny, S_whole.maxy
        S_whole = common.PointOperations.filter_points_by_angle_range(S_whole, angle - 1, angle + 1)
        Y = common.Visualization.transform_points_to_bmp_with_bounds(S_whole, bmpsize, minx, maxx, miny, maxy)
        Y = self.circular_mask(Y)
        bestangle = self.compute_bestangle(Y)
        if self.do_visualization:
            self.visualize_at_derivative(Y, angle=bestangle, padding=1)

        G = np.zeros(Y.shape)
        y = Y.shape[0] / 2
        xs = np.linspace(Y.shape[0] / 4, 3 * Y.shape[0] / 4, 50)
        for i in range(xs.shape[0]):
            G[int(y), int(xs[i])] = 1
        G = self.rotate_matrix(G, angle)
        xs, ys = np.where(G == 1)

        scan_direction = [[[xs[0], ys[0]], [xs[len(xs) - 1], ys[len(ys) - 1]]]]
        return scan_direction

    def compute_bestangle(self, Y):
        minangle = 0
        minscore = 1000000
        for x in np.linspace(0, 3.14, 180):
            score = self.derivative(Y, angle=x, padding=1)

            if score < minscore:
                minscore = score
                minangle = x
        return minangle

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
                if A[i, j] == 1:
                    pos.append([i - A.shape[0] / 2, j - A.shape[1] / 2])

        A_trans = np.zeros(A.shape)
        for i in range(len(pos)):
            x, y = DerivativeMethod.rotate_origin_only((pos[i][0], pos[i][1]), radians)
            x, y = x + A.shape[0] / 2, y + A.shape[1] / 2
            A_trans[int(x), int(y)] = 1

        return A_trans

    def visualize_at_derivative(self, Y, angle, padding):

        G = np.zeros(Y.shape)
        y = Y.shape[0] / 2
        xs = np.linspace(Y.shape[0] / 4, 3 * Y.shape[0] / 4, 50)

        for i in range(xs.shape[0]):
            G[int(y), int(xs[i])] = 1
        G = self.rotate_matrix(G, angle)

        # take all points where G = 1
        xs, ys = np.where(G == 1)
        # make array of points from it
        points = [[x, y] for x, y in zip(list(xs), list(ys))]
        # again project to bmp
        D = common.Visualization.transform_rawpoints_to_bmp_with_bounds(points, Y.shape[0], 0, Y.shape[0], 0, Y.shape[0])


        Y_trans = self.rotate_matrix(Y, angle)
        Y_trans_padded = np.hstack((np.zeros((padding, Y.shape[0])).T, Y_trans[:, :-padding]))
        deriv = np.abs(Y_trans - Y_trans_padded)
        tmp = deriv[5:-5,5:-5]
        tmp = self.circular_mask(tmp)
        deriv = np.zeros(Y.shape)
        deriv[5:-5,5:-5] = tmp




        Y_sides = np.zeros((Y.shape[0], Y.shape[1] * 6))
        Y_sides[:, :Y.shape[0]] = Y
        Y_sides[:, Y.shape[0]:2 * Y.shape[0]] = Y_trans
        Y_sides[:, 2 * Y.shape[0]:3 * Y.shape[0]] = Y_trans_padded
        Y_sides[:, 3 * Y.shape[0]: 4 * Y.shape[0]] = deriv
        Y_sides[:, 4 * Y.shape[0]: 5 * Y.shape[0]] = G
        Y_sides[:, 5 * Y.shape[0]:] = D

        common.HoughTransform.visualize_matrix(Y_sides)