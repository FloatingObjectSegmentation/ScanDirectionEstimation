import numpy as np
import common
import math
from PIL import Image
import random


class HoughMethod:

    def __init__(self, do_visualization=False):
        self.do_visualization = do_visualization

    def scan_direction_hough(self, S_whole, angle, bmpsize):
        pass
        # filter by neighboring scan angles S_whole
        minx, maxx, miny, maxy = S_whole.minx, S_whole.maxx, S_whole.miny, S_whole.maxy
        S_whole = common.PointOperations.filter_points_by_angle_range(S_whole, angle - 1, angle + 1)
        Y = common.Visualization.transform_points_to_bmp_with_bounds(S_whole, bmpsize, minx, maxx, miny, maxy)
        Y = self.circular_mask(Y)

        accumulator, thetas, rhos = self.hough_line(Y)
        if self.do_visualization:
            self.visualize_accumulator(accumulator)
        Y = self.insert_resulting_lines(Y, accumulator, rhos, thetas)
        if self.do_visualization:
            common.HoughTransform.visualize_matrix(Y)


        xs, ys = np.where(Y == 2)
        scan_direction = [[[xs[0], ys[0]], [xs[len(xs) - 1], ys[len(ys) - 1]]]]

        return scan_direction


    def circular_mask(self, Y):
        y, x = np.ogrid[-Y.shape[0] / 2: Y.shape[0] / 2, -Y.shape[0] / 2: Y.shape[0] / 2]
        mask = x ** 2 + y ** 2 <= (Y.shape[0] / 2) ** 2
        Y = mask * Y
        return Y


    def hough_line(self, img):
        # Rho and Theta ranges
        thetas = np.deg2rad(np.arange(-90.0, 90.0))
        width, height = img.shape
        diag_len = np.ceil(np.sqrt(width * width + height * height))  # max_dist
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

            for t_idx in range(num_thetas):
                # Calculate rho. diag_len is added for a positive index
                rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
                accumulator[int(rho), int(t_idx)] += 1

        return accumulator, thetas, rhos


    def visualize_accumulator(self, accumulator):
        img = Image.new('RGB', (accumulator.shape[0], accumulator.shape[1]), 'white')
        pixels = img.load()
        for i in range(accumulator.shape[0]):
            for j in range(accumulator.shape[1]):
                pixels[i, j] = (int(accumulator[i, j] / np.max(accumulator) * 255), 0, 0)
        img.show()

    def insert_resulting_lines(self, Y, accumulator, rhos, thetas):

        idx0 = np.argpartition(accumulator.ravel(), -1)[-1:]
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