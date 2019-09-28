import numpy as np
import common
import math
from PIL import Image


class HoughFiltered:

    def __init__(self, dataset: common.LidarDatasetNormXYZRGBAngle, R, bmpsize, filter=True):
        self.R = R
        self.bmpsize = bmpsize # this is how big the whole file should be, but in this case only the neighbors will be bmpized
        self.filter = filter

    def run(self, dataset, augmentable):

        nbrs, minptidx = common.PointOperations.find_points_within_R(dataset, augmentable, self.R)

        if (filter):
            nbrs = common.PointOperations.filter_points(nbrs, dataset[minptidx].scan_angle)

        bmpsizenbrs = self.bmpsize / 1000 * self.R
        X = common.transform_points_to_bmp(nbrs, bmpsizenbrs)

        accumulator, thetas, rhos = common.HoughTransform.hough_line(X)
        return accumulator, thetas, rhos