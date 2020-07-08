#loading Tiff images and CSV Data
import numpy as np
from skimage import io


def read_stack_from_tiff(tifFilename):
    # load the tiff data
    Images = io.imread(tifFilename)

    list_of_loaded_frames = []
    for mat in Images:
        list_of_loaded_frames.append(np.mat(mat))
    return list_of_loaded_frames
