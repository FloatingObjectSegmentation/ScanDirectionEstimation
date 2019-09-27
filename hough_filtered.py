import numpy as np
import common
import math
from PIL import Image


class HoughFiltered:

    def __init__(self, dataset, R, bmpsize, filter=True):
        self.R = R
        self.bmpsize = bmpsize # this is how big the whole file should be, but in this case only the neighbors will be bmpized
        self.filter = filter

    def run(self, dataset, augmentable):
        nbrs, minptidx = self.find_points_within_R(dataset, augmentable)

        if (filter):
            nbrs = self.filter_points(nbrs, dataset[minptidx].scan_angle)

        bmpsizenbrs = self.bmpsize / 1000 * self.R
        X = common.transform_points_to_bmp(nbrs, bmpsizenbrs)

        accumulator, thetas, rhos = common.HoughTransform.hough_line(X)
        return accumulator, thetas, rhos

    def find_points_within_R(self, dataset: list([common.LidarPointXYZRGBAngle]), augmentable: common.Augmentable):
        result = []
        minptidx, minptval = 0, 10000000000
        for idx, pt in enumerate(dataset):
            dist = np.linalg.norm(np.array(augmentable.location) - np.array([pt.X, pt.Y, pt.Z]))
            if dist > self.R:
                result.append(pt)
            if dist < minptval:
                minptidx = idx
                minptval = dist
        return result, minptidx

    def filter_points(self, nbrlist: list([common.LidarPointXYZRGBAngle]), scan_angle):
        # compute translation between R taken and scan angle we need to take
        distance_per_scan_degree = 2 * math.tan(30) * 1000 / 60
        degs = self.R / distance_per_scan_degree
        filtered_nbrs = []
        for nbr in nbrlist:
            if nbr.scan_angle >= scan_angle - degs / 2 and nbr.scan_angle <= scan_angle + degs / 2:
                filtered_nbrs.append(nbr)
        return filtered_nbrs