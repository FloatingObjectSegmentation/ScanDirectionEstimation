import numpy as np
import common
import math
from PIL import Image


class HoughGlobal:

    def __init__(self, dataset, chunk_size, bmpsize):
        self.R = R
        self.bmpsize = bmpsize
        self.chunk_size = chunk_size

    def run(self, dataset, augmentable):

        nbrs, minptidx = self.find_points_within_R(dataset, augmentable)

        chunkdiff = X.

    def transform_points_into_bmp(self):


    xparts = 50
    yparts = 50

    xdiff = X.shape[0] / xparts
    ydiff = X.shape[1] / yparts

    X_result = np.zeros(X.shape)
    rand_attempts = 200

    for i in range(rand_attempts):
        i = random.random() * (xparts - 1)
        j = random.random() * (yparts - 1)

        Y = X[int(i * xdiff):int((i + 1) * xdiff), int(j * ydiff):int((j + 1) * ydiff)]
        y, x = np.ogrid[-Y.shape[0] / 2: Y.shape[0] / 2, -Y.shape[0] / 2: Y.shape[0] / 2]
        mask = x ** 2 + y ** 2 <= (Y.shape[0] / 2) ** 2
        Y = mask * Y

        # visualize_matrix(Y)

        accumulator, thetas, rhos = hough_line(Y)
        Y = insert_resulting_lines(Y, accumulator, rhos, thetas)
        X_result[int(i * xdiff):int((i + 1) * xdiff), int(j * ydiff):int((j + 1) * ydiff)] = Y
