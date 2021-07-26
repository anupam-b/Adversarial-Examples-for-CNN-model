import cv2
import numpy as np
from skimage.util import random_noise

def noise(X):
    X = random_noise(X, mode='gaussian',mean=0.0)
    return X