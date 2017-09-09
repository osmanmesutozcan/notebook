import importlib
import numpy as np
import hough_module

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import canny

im = canny(rgb2gray(imread('../data/cat.jpg')), 2)
zero_arr = np.zeros(im.shape)

importlib.reload(hough_module)
hough_module.hough_line(im, zero_arr)
