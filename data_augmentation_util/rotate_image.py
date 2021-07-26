from skimage.transform import rotate
import random

def rotate_image(X, intensity=0.5):
	delta = 30. * intensity
	X = rotate(X, random.uniform(-delta, delta))
	return X