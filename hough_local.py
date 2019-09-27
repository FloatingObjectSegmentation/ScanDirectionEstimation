import numpy as np
import common
import math
from PIL import Image
import random
import itertools


class HoughGlobal:

    def __init__(self, chunk_size, bmpsize, neighbor_distance):
        self.bmpsize = bmpsize
        self.chunk_size = chunk_size # in meters
        self.neighbor_distance = neighbor_distance


    def run(self, dataset: common.LidarDatasetNormXYZRGBAngle, augmentable):
        '''
        dataset_name = name in the form [0-9]{3}[_]{1}[0-9]{3}, to help with pickling
        dataset_norm = location normalized to 1000 meters, so find minimums and subtract them
        minx, miny = min values for x,y coordinates in original dataset
        augmentable - the augmentable we are direction detecting around
        '''

        # the difference in chunk size wrt the bmp
        chunkdiff = self.bmpsize / 1000.0 * self.chunk_size
        chunks = int(1000.0 / self.chunk_size)

        # find augmentable's chunk
        idx_x = (augmentable.location[0] - dataset.minx) / self.chunk_size
        idx_y = (augmentable.location[1] - dataset.miny) / self.chunk_size

        xidxs = list(range(max(idx_x - self.neighbor_distance, 0), min(chunks, idx_x + self.neighbor_distance + 1)))
        yidxs = list(range(max(idx_y - self.neighbor_distance, 0), min(chunks, idx_y + self.neighbor_distance + 1)))

        X = common.transform_points_to_bmp(dataset, self.bmpsize)
        X_result = np.zeros(X.shape)

        for i, j in itertools.product(idx_x, idx_y):

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
