import numpy as np

def flip(X, y):
	flip_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
	flip_vertically = np.array([1, 5, 12, 15, 17])

	if(y in flip_horizontally):
		X = np.fliplr(X)
	elif(y in flip_vertically):
		X = np.flipud(X)
	return X