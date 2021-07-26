from skimage.exposure import adjust_gamma
import random

def brightness(X, intensity=0.5):
	delta = 1. * intensity
	X = adjust_gamma(X, random.uniform(1 - delta, 1 + delta))
	return X