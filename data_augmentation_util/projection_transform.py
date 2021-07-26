import numpy as np
import random
from skimage.transform import ProjectiveTransform
from skimage.transform import warp

def projection_transform(X, intensity=0.5):
    image_size = X.shape[1]
    d = image_size * 0.3 * intensity

    tl_top = random.uniform(-d, d)
    tl_left = random.uniform(-d, d)
    bl_bottom = random.uniform(-d, d)
    bl_left = random.uniform(-d, d)
    tr_top = random.uniform(-d, d)
    tr_right = random.uniform(-d, d)
    br_bottom = random.uniform(-d, d)
    br_right = random.uniform(-d, d)

    transform = ProjectiveTransform()
    transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))

    X = warp(X, transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')

    return X