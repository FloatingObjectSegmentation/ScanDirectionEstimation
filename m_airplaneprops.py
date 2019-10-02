import numpy as np
import common
import math
import m_derivative
import m_hough

from PIL import Image
import random

class AirplanePropertiesEstimation:

    def __init__(self, bmpsize_full_dataset, bmpswathspan, do_visualization=False):
        self.bmpsize_full_dataset = bmpsize_full_dataset
        self.bmpswathspan = bmpswathspan
        self.do_visualization = do_visualization

    # not that dataset is already in normalized space and augmentable in in original point cloud space!
    def run(self, dataset: common.LidarDatasetNormXYZRGBAngle, augmentable: common.Augmentable):

        ########################
        ## PREP DATA
        ########################

        # find alpha of nearest neighbour
        aug_loc = (augmentable.location[0] - dataset.minx, augmentable.location[1] - dataset.miny)
        minptidx = dataset.find_closest_neighbour(aug_loc)
        minpt = dataset.points[minptidx]
        minptalpha = minpt.scan_angle


        # compute params
        R = 2.5 * AirplanePropertiesEstimation.length_of_one_degree(minptalpha, 1000.0) / 2
        bmpsizenbrs = int(self.bmpsize_full_dataset / 1000 * R)


        # S_whole = assume x = 1000, take 2.5 degrees of range for it, divide by 2 because it's polmer!
        nbr_indices = dataset.find_neighbours(aug_loc, R=R)
        S_whole = common.PointSet([dataset.points[i] for i in nbr_indices])
        if self.do_visualization:
            common.Visualization.visualize(S_whole, S_whole.minx, S_whole.miny, S_whole.maxx, S_whole.maxy, bmpsizenbrs)


        # Find out whether both neighboring degrees are present
        angles = set([p.scan_angle for p in S_whole.points])
        hasbothangles = minptalpha - 1 in angles and minptalpha + 1 in angles
        if not hasbothangles:
            # select new point from the neighboring angle
            selected = 0
            for p in S_whole.points:
                if p.scan_angle == minptalpha - 1 or p.scan_angle == minptalpha + 1:
                    selected = p
                    break

            minptalpha = p.scan_angle

            # compute new S_whole around it and params (because neighboring degree to alpha-1 may not be contained in old S_whole)
            R = 2.5 * AirplanePropertiesEstimation.length_of_one_degree(minptalpha, 1000.0) / 2
            bmpsizenbrs = int(self.bmpsize_full_dataset / 1000 * R)
            nbr_indices = dataset.find_neighbours(aug_loc, R=R)
            S_whole = common.PointSet([dataset.points[i] for i in nbr_indices])

        # S_small = from S_big take only the points that are the same degree
        S_small = common.PointOperations.filter_points_by_angle(S_whole, minptalpha)
        if self.do_visualization:
            common.Visualization.visualize(S_small, S_whole.minx, S_whole.miny, S_whole.maxx, S_whole.maxy, bmpsizenbrs)

        ########################
        ## FROM DEGREE RANGE ESTIMATE HEIGHT
        ########################

        # find nearest points with the neighboring angle
        angle_nbrs = self.nearest_angle_neighbors(dataset=dataset, S_small=S_small)
        common.Visualization.visualize(angle_nbrs, S_whole.minx, S_whole.miny, S_whole.maxx, S_whole.maxy, bmpsizenbrs)


        # average points
        p_min, p_max = self.average_points_in_clusters(points=angle_nbrs)
        if self.do_visualization:
            common.Visualization.visualize_points([p_min, p_max], S_whole.minx, S_whole.miny, S_whole.maxx, S_whole.maxy, bmpsizenbrs)


        # from p_min and p_max now compute dist, scan_direction and height x
        dist = math.sqrt(((p_max[0] - p_min[0]) ** 2) + ((p_max[1] - p_min[1]) ** 2))
        scan_direction = np.array([p_max[0] - p_min[0], p_max[1] - p_min[1]])
        scan_direction = scan_direction / np.linalg.norm(scan_direction)
        height = self.height_at_degree(angle=minptalpha, dist=dist)


        # Interpolate the true scan angle
        aug = np.array([minpt.X, minpt.Y]) # because we're working with minpt not with aug directly!
        p1 = 0
        p2 = 0
        for i in range(1, 50):
            p = aug + i * scan_direction
            nbridxs = dataset.find_neighbours(p, R=1.0)
            for i in nbridxs:
                if (dataset.points[i].scan_angle == minptalpha + 1):
                    p2 = p
                    break
        for i in range(1, 50):
            p = aug - i * scan_direction
            nbridxs = dataset.find_neighbours(p, R=1.0)
            for i in nbridxs:
                if (dataset.points[i].scan_angle == minptalpha - 1):
                    p1 = p
                    break
        a1 = minptalpha
        a2 = minptalpha + 1

        x_m = 0
        x_n = np.linalg.norm(p1 - p2)
        x_A = np.linalg.norm(p2 - aug)
        a_m = a1
        a_n = a2
        a_aug = a_m + (x_A - x_m) * (a_n - a_m) / (x_n  - x_m)


        # Compute airplane_position
        x = np.array([0,0,height])
        xtana = scan_direction * height * math.tan(a_aug)
        xtana = np.array([xtana[0], xtana[1], 0])
        A = [minpt.X, minpt.Y, minpt.Z]
        airplane_position = A + (x + xtana)

        airplane_ortho_direction = [[0,0], [scan_direction[0], scan_direction[1]]]

        # find scan direction
        derivdirs = []
        houghdirs = []
        for bmpsize in self.bmpswathspan:
            scan_direction_derivative = m_derivative.DerivativeMethod().scan_direction_derivative(S_whole, minptalpha, bmpsize)
            derivdirs.append(scan_direction_derivative)
            scan_direction_hough = m_hough.HoughMethod().scan_direction_hough(S_whole, minptalpha, bmpsize)
            houghdirs.append(scan_direction_hough)

        return airplane_position, airplane_ortho_direction, derivdirs, houghdirs


    def average_points_in_clusters(self, points: common.PointSet):
        # will be in ascending order
        # p_max is at higher angle, p_min is at smaller angle

        angles = list(set([a.scan_angle for a in points.points]))
        angles.sort()
        list1x = []
        list1y = []
        list2x = []
        list2y = []

        for point in points.points:
            if point.scan_angle == angles[0]:
                list1x.append(point.X)
                list1y.append(point.Y)
            if point.scan_angle == angles[1]:
                list2x.append(point.X)
                list2y.append(point.Y)
        x_min = np.average(np.array(list1x))
        y_min = np.average(np.array(list1y))
        x_max = np.average(np.array(list2x))
        y_max = np.average(np.array(list2y))

        return (x_min, y_min), (x_max, y_max)


    def nearest_angle_neighbors(self, dataset: common.LidarDatasetNormXYZRGBAngle, S_small: common.PointSetNormalized):

        angle = S_small.points[0].scan_angle
        neighbours = []
        for point in S_small.points:
            nbridxs = dataset.find_neighbours((point.X, point.Y), R=1.0)
            for idx in nbridxs:
                if dataset.points[idx].scan_angle == angle - 1 or dataset.points[idx].scan_angle == angle + 1:
                    neighbours.append(dataset.points[idx])
        return common.PointSet(neighbours)

    @staticmethod
    def length_of_one_degree(angle, height):
        angle = math.fabs(angle)
        alpha1 = (angle + 1) * math.pi / 180.0
        alpha2 = angle * math.pi / 180.0
        dist = (height * math.tan(alpha1)) - height * math.tan(alpha2)
        return dist

    @staticmethod
    def height_at_degree(angle, dist):
        # we know that X * tg(alpha) = y, y and alpha are knowns
        # airplane height = X
        angle = math.fabs(angle)
        alpha1 = (angle + 1) * math.pi / 180.0
        alpha2 = angle * math.pi / 180.0
        height = dist / (math.tan(alpha1) - math.tan(alpha2))
        return height