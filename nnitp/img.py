#
# Copyright (c) Microsoft Corporation.
#

import numpy as np

# Prepare a list of images for display by matplotlib. Normalizes the
# images between 0.0 and 1.0., and also converts single-channel images
# to gray-scale images.

def prepare_images(imgs):
    if len(imgs) == 0:
        return imgs
    imin,imax = min(map(np.min,imgs)),max(map(np.max,imgs))
    iptp = imax - imin
    def to_gray(img):
        shp = img.shape
        if shp[2] == 1:
            img = img.reshape(shp[0],shp[1])
            img = np.stack((img,)*3, axis=-1)
        return img
    def normalize(img):
        if iptp != 0:
            img = (img - imin)/iptp
        return img
    return [to_gray(normalize(img)) for img in imgs]
