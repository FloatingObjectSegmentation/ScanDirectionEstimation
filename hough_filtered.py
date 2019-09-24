import numpy as np
import common
import math


class HoughFiltered:

    def __init__(self, dataset, R, bmpsize, filter=True):
        self.R = R
        self.bmpsize = bmpsize # this is how big the whole file should be, but in this case only the neighbors will be bmpized
        self.filter = filter

    def run(self, dataset, augmentable):

        nbrs, minptidx = self.find_points_within_R(dataset, augmentable)

        if (filter):
            nbrs = self.filter_points(nbrs, dataset[minptidx].scan_angle)

        # need to transform this into a bmp so that something may be fitted
        bmpsizenbrs = self.R / 1000 * self.bmpsize

        # find minx, miny of nbrs
        minx, miny = 100000000000, 1000000000000
        for nbr in nbrs:
            if (nbr.X < minx): minx = nbr.X
            if (nbr.Y < miny): miny = nbr.Y

        for i in range(len(nbrs)):
            nbr[i].X = nbr[i].X - minx
            nbr[i].Y = nbr[i].Y - miny


        # fit into bmp


        labels = .split(';')
        labels = [l.split(',') for l in labels]
        labels = [(float(l[0][1:-1]) - minx, float(l[1][1:-1]) - miny) for l in labels]
        labels = [(x[0] * width / 1000.0, x[1] * height / 1000.0) for x in labels]



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
        pass

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