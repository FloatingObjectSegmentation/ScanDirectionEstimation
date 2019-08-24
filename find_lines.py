import numpy as np
from PIL import Image
import pickle
import os.path
import math
import random
import common


def hough_line(img):
  # Rho and Theta ranges
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # Cache some resuable values
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

def visualize_accumulator(accumulator):

    img = Image.new('RGB', (accumulator.shape[0], accumulator.shape[1]), 'white')
    pixels = img.load()
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            pixels[i,j] = (int(accumulator[i,j] / np.max(accumulator) * 255), 0, 0)
    img.show()

def insert_resulting_lines(Y, accumulator, rhos, thetas):

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

def visualize_matrix(Y):
    img = Image.new('RGB', (Y.shape[0], Y.shape[1]), 'white') # Create a new black image
    pixels = img.load() # create the pixel map
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i, j] == 1:
                pixels[i, j] = (255, 0, 0)
            if Y[i, j] == 2:
                pixels[i, j] = (0, 255, 0)
    img.show()




lasfile = "E:\\workspaces\\LIDAR_WORKSPACE\\lidar\\521_126.txt"
X = common.load_points(lasfile, 10000, 10000)
visualize_matrix(X)

# PARTITION INTO MULTIPLE SPACE
xparts = 50
yparts = 50

xdiff = X.shape[0] / xparts
ydiff = X.shape[1] / yparts

X_result = np.zeros(X.shape)
rand_attempts = 200


for i in range(rand_attempts):

        i = random.random() * (xparts - 1)
        j = random.random() * (yparts - 1)


        Y = X[int(i * xdiff):int((i + 1) * xdiff),int(j * ydiff):int((j + 1) * ydiff)]
        y,x = np.ogrid[-Y.shape[0] / 2: Y.shape[0] / 2,  -Y.shape[0] / 2: Y.shape[0] / 2]
        mask = x**2 + y**2 <= (Y.shape[0] / 2)**2
        Y = mask * Y

        #visualize_matrix(Y)

        accumulator, thetas, rhos = hough_line(Y)
        Y = insert_resulting_lines(Y, accumulator, rhos, thetas)
        X_result[int(i * xdiff):int((i + 1) * xdiff),int(j * ydiff):int((j + 1) * ydiff)] = Y

visualize_matrix(X_result)
