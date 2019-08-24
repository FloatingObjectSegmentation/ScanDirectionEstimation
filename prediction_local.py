import numpy as np
from PIL import Image
import pickle
import os.path
import math
import random

# load all lidar txt files
# foreach lidar txt file:
    # load bmp
    # load points
    # predict at each point
    # save each prediction