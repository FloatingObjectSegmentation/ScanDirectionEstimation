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
        self.hough_line(X)



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

    def hough_line(self, img):
      #Rho and Theta ranges
      thetas = np.deg2rad(np.arange(-90.0, 90.0))
      width, height = img.shape
      diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
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

        if i % 10000 == 0:
            print(i)
        for t_idx in range(num_thetas):
          # Calculate rho. diag_len is added for a positive index
          rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
          accumulator[int(rho), int(t_idx)] += 1

      return accumulator, thetas, rhos

    # auxiliary methods
    def visualize_matrix(self, Y):
        img = Image.new('RGB', (Y.shape[0], Y.shape[1]), 'white')  # Create a new black image
        pixels = img.load()  # create the pixel map
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if Y[i, j] == 1:
                    pixels[i, j] = (255, 0, 0)
                if Y[i, j] == 2:
                    pixels[i, j] = (0, 255, 0)
        img.show()

    def visualize_accumulator(self, accumulator):
        img = Image.new('RGB', (accumulator.shape[0], accumulator.shape[1]), 'white')
        pixels = img.load()
        for i in range(accumulator.shape[0]):
            for j in range(accumulator.shape[1]):
                pixels[i, j] = (int(accumulator[i, j] / np.max(accumulator) * 255), 0, 0)
        img.show()

    def insert_resulting_lines(self, Y, accumulator, rhos, thetas):

        idx0 = np.argpartition(accumulator.ravel(), -3)[-3:]
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