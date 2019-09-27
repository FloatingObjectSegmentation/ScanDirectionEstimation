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

        X = self.transform_points_to_bmp(nbrs)

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

    def transform_points_to_bmp(self, points):
        bmpsizenbrs = self.bmpsize / 1000 * self.R

        # find minimum and maximum point within nbrs
        minx, miny = 100000000000, 10000000000
        for pt in points:
            if pt.x < minx: minx = pt.x
            if pt.y < miny: miny = pt.y

        # do the subtraction of minimums to make minimum (0,0)
        for i in range(len(points)):
            points[i].x -= minx
            points[i].y -= miny

        # fill to bmp
        X = np.zeros((bmpsizenbrs, bmpsizenbrs))
        for i in range(len(points)):
            try:
                x = (float(bmpsizenbrs) / 1000.0) * points[i].x
                y = (float(bmpsizenbrs) / 1000.0) * points[i].y
                X[int(x), int(y)] = 1
            except:
                pass

        return X