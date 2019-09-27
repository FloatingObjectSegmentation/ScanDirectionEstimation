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

        if (filter):
            nbrs = self.filter_points(nbrs, dataset[minptidx].scan_angle)

        X = self.transform_points_to_bmp(nbrs)
        self.hough_line(X)