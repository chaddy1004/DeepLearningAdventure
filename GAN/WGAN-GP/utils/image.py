import numpy as np

def normalize_img(img):
    """ Normalize image from 0 ~ 255 to -1 ~ 1
    Return: np array with dtype float32
    """
    return img.astype(np.float32) / 127.5 - 1
