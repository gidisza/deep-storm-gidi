import numpy as np
import cv2


def gaussian_kernel_2d(size, sigma):
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    return kernel_1d @ kernel_1d.T
