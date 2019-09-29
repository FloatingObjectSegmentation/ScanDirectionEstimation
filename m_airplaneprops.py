import numpy as np
import common
import math
from PIL import Image
import random

class AirplanePropertiesEstimation:

    def __init__(self, R, bmpsize_full_dataset, filter=True):
        self.bmpsize_full_dataset = bmpsize_full_dataset
        self.R = R # in meters
        self.filter = filter

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


        # find scan direction
        scan_direction = self.scan_direction_derivative(S_whole, minptalpha, bmpsizenbrs)


        # S_small = from S_big take only the points that are the same degree
        S_small = common.PointOperations.filter_points_by_angle(S_whole, minptalpha)

        ########################
        ## FROM DEGREE RANGE ESTIMATE HEIGHT
        ########################

        # find nearest points with the neighboring angle
        angle_nbrs = self.nearest_angle_neighbors(dataset=dataset, S_small=S_small)
        common.Visualization.visualize(angle_nbrs, S_whole.minx, S_whole.miny, S_whole.maxx, S_whole.maxy, bmpsizenbrs)


        # average points
        p_min, p_max = self.average_points_in_clusters(points=angle_nbrs)
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
        a_aug = a_m + (x_A - x_m) * (a_n - a_m) / (x_n - x_m)


        # Compute airplane_position
        x = np.array([0,0,height])
        xtana = scan_direction * height * math.tan(a_aug)
        xtana = np.array([xtana[0], xtana[1], 0])
        A = [minpt.X, minpt.Y, minpt.Z]
        airplane_position = A + (x + xtana)

        airplane_direction = [[0,0], [scan_direction[0], scan_direction[1]]]


        return airplane_position, airplane_direction, scan_direction
    

    def scan_direction_derivative(self, S_whole, angle, bmpsize):
        pass
        # filter by neighboring scan angles S_whole
        minx, maxx, miny, maxy = S_whole.minx, S_whole.maxx, S_whole.miny, S_whole.maxy
        S_whole = common.PointOperations.filter_points_by_angle_range(S_whole, angle - 1, angle + 1)
        Y = common.Visualization.transform_points_to_bmp_with_bounds(S_whole, bmpsize, minx, maxx, miny, maxy)
        Y = self.circular_mask(Y)
        bestangle = self.compute_bestangle(Y)
        self.visualize_at_derivative(Y, angle=bestangle, padding=1)

        G = np.zeros(Y.shape)
        y = Y.shape[0] / 2
        xs = np.linspace(Y.shape[0] / 4, 3 * Y.shape[0] / 4, 50)
        for i in range(xs.shape[0]):
            G[int(y), int(xs[i])] = 1
        G = self.rotate_matrix(G, angle)
        xs, ys = np.where(G == 1)

        scan_direction = [[[xs[0], ys[0]], [xs[len(xs) - 1], ys[len(ys) - 1]]]]
        return scan_direction




    def compute_bestangle(self, Y):
        minangle = 0
        minscore = 1000000
        for x in np.linspace(0, 3.14, 180):
            score = self.derivative(Y, angle=x, padding=1)
            print(score)
            if score < minscore:
                minscore = score
                minangle = x
        return minangle

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

    def circular_mask(self, Y):
        y, x = np.ogrid[-Y.shape[0] / 2: Y.shape[0] / 2, -Y.shape[0] / 2: Y.shape[0] / 2]
        mask = x ** 2 + y ** 2 <= (Y.shape[0] / 2) ** 2
        Y = mask * Y
        return Y

    def derivative(self, Y, angle, padding):
        Y_trans = self.rotate_matrix(Y, angle)
        Y_trans_padded = np.hstack((np.zeros((padding, Y.shape[0])).T, Y_trans[:, :-padding]))
        deriv = np.abs(Y_trans - Y_trans_padded)
        score = np.sum(deriv)
        return score

    def get_line(self, angle):
        point = [0, 1]
        point_rotated = AirplanePropertiesEstimation.rotate_origin_only(point, angle)
        return point_rotated

    def visualize_at_derivative(self, Y, angle, padding):

        G = np.zeros(Y.shape)
        y = Y.shape[0] / 2
        xs = np.linspace(Y.shape[0] / 4, 3 * Y.shape[0] / 4, 50)

        for i in range(xs.shape[0]):
            G[int(y), int(xs[i])] = 1
        G = self.rotate_matrix(G, angle)

        # take all points where G = 1
        xs, ys = np.where(G == 1)
        # make array of points from it
        points = [[x, y] for x, y in zip(list(xs), list(ys))]
        # again project to bmp
        D = common.Visualization.transform_rawpoints_to_bmp_with_bounds(points, Y.shape[0], 0, Y.shape[0], 0, Y.shape[0])


        Y_trans = self.rotate_matrix(Y, angle)
        Y_trans_padded = np.hstack((np.zeros((padding, Y.shape[0])).T, Y_trans[:, :-padding]))
        deriv = np.abs(Y_trans - Y_trans_padded)
        tmp = deriv[5:-5,5:-5]
        tmp = self.circular_mask(tmp)
        deriv = np.zeros(Y.shape)
        deriv[5:-5,5:-5] = tmp




        Y_sides = np.zeros((Y.shape[0], Y.shape[1] * 6))
        Y_sides[:, :Y.shape[0]] = Y
        Y_sides[:, Y.shape[0]:2 * Y.shape[0]] = Y_trans
        Y_sides[:, 2 * Y.shape[0]:3 * Y.shape[0]] = Y_trans_padded
        Y_sides[:, 3 * Y.shape[0]: 4 * Y.shape[0]] = deriv
        Y_sides[:, 4 * Y.shape[0]: 5 * Y.shape[0]] = G
        Y_sides[:, 5 * Y.shape[0]:] = D

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
            x, y = AirplanePropertiesEstimation.rotate_origin_only((pos[i][0], pos[i][1]), radians)
            x, y = x + A.shape[0] / 2, y + A.shape[1] / 2
            A_trans[int(x),int(y)] = 1

        return A_trans

    @staticmethod
    def length_of_one_degree(angle, height):
        alpha1 = (angle + 1) * math.pi / 180.0
        alpha2 = angle * math.pi / 180.0
        dist = (height * math.tan(alpha1)) - height * math.tan(alpha2)
        return dist

    @staticmethod
    def height_at_degree(angle, dist):
        # we know that X * tg(alpha) = y, y and alpha are knowns
        # airplane height = X
        alpha1 = (angle + 1) * math.pi / 180.0
        alpha2 = angle * math.pi / 180.0
        height = dist / (math.tan(alpha1) - math.tan(alpha2))
        return height