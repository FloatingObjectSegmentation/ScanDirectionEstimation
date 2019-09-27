import numpy as np
import common
import math
from PIL import Image
import random


class HoughGlobal:

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

            accumulator, thetas, rhos = common.HoughTransform.hough_line(Y)
            Y = common.HoughTransform.insert_resulting_lines(Y, accumulator, rhos, thetas)
            X_result[int(i * chunkdiff):int((i + 1) * chunkdiff), int(j * chunkdiff):int((j + 1) * chunkdiff)] = Y
        common.HoughTransform.visualize_matrix(X_result)
