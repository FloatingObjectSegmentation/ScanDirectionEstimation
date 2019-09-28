import numpy as np
import common
import math
from PIL import Image
import random

class DerivativeMethod:
    def __init__(self, chunk_size, bmpsize):
        self.bmpsize = bmpsize
        self.chunk_size = chunk_size # in meters

    def run(self, dataset: common.LidarDatasetNormXYZRGBAngle, randattempts): # hough global is augmentable independent

        # the difference in chunk size wrt the bmp
        chunkdiff = self.bmpsize / 1000.0 * self.chunk_size
        chunks = int(1000.0 / self.chunk_size)

        X = common.transform_points_to_bmp(dataset, self.bmpsize)
        X_result = np.zeros(X.shape)
        for i in range(randattempts):
            i = random.random() * (chunks - 1)
            j = random.random() * (chunks - 1)

            # take chunks
            Y = X[int(i * chunkdiff):int((i + 1) * chunkdiff), int(j * chunkdiff):int((j + 1) * chunkdiff)]

            # circular mask
            y, x = np.ogrid[-Y.shape[0] / 2: Y.shape[0] / 2, -Y.shape[0] / 2: Y.shape[0] / 2]
            mask = x ** 2 + y ** 2 <= (Y.shape[0] / 2) ** 2
            Y = mask * Y

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







            # BY HOUGH TRANSFORM
            # common.HoughTransform.visualize_matrix(Y)
            # accumulator, thetas, rhos = common.HoughTransform.hough_line(Y)
            # Y = common.HoughTransform.insert_resulting_lines(Y, accumulator, rhos, thetas)







            X_result[int(i * chunkdiff):int((i + 1) * chunkdiff), int(j * chunkdiff):int((j + 1) * chunkdiff)] = Y
        common.HoughTransform.visualize_matrix(X_result)

    def derivative(self, Y, angle, padding):
        Y_trans = self.rotate_matrix(Y, angle)
        Y_trans_padded = np.hstack((np.zeros((padding, Y.shape[0])).T, Y_trans[:, :-padding]))
        deriv = np.abs(Y_trans - Y_trans_padded)
        score = np.sum(deriv)
        return score

    def get_line(self, angle):
        point = [0, 1]
        point_rotated = HoughGlobal.rotate_origin_only(point, minangle)
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
            x, y = HoughGlobal.rotate_origin_only((pos[i][0], pos[i][1]), radians)
            x, y = x + A.shape[0] / 2, y + A.shape[1] / 2
            A_trans[int(x),int(y)] = 1

        return A_trans