import math
import numpy as np
import pylab as pl

# a = np.linspace(0, 6.28, 20)
# xlst, ylst = [], []
# for el in a:
#     x = 50 * math.cos(el)
#     y = 50 * math.sin(el)
#     z = math.atan2(y, x)
#     if (z < 0):
#         z += math.pi
#     print(z)
#     xlst.append(x)
#     ylst.append(y)
# pl.plot(xlst, ylst, 'ro')
# pl.show()

x = 500.0
for alpha in np.linspace(1, 30, 30):
    pass
    alpha1 = alpha * math.pi / 180.0
    alpha2 = (alpha - 1) * math.pi / 180.0
    dist = (x * math.tan(alpha1)) - x * math.tan(alpha2)
    print(dist)
