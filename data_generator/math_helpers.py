import numpy as np
from sklearn.preprocessing import normalize

# normalize image given mean and std
def normalize_im(im, dmean, dstd):
    im = np.squeeze(im)
    im_norm = np.zeros(im.shape,dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm

def normalize_data(data):
    mean_train = np.zeros(data.shape[0], dtype=np.float32)
    std_train = np.zeros(data.shape[0], dtype=np.float32)
    for i in range(data.shape[0]):
        data[i, :, :] = normalize(data[i, :, :])
        mean_train[i] = data[i, :, :].mean()
        std_train[i] = data[i, :, :].std()

    # resulting normalized training images
    mean_val_train = mean_train.mean()
    std_val_train = std_train.mean()
    data_normalized = np.zeros(data.shape, dtype=np.float32)
    for i in range(data.shape[0]):
        data_normalized[i, :, :] = normalize_im(data[i, :, :], mean_val_train, std_val_train)

    return data_normalized,mean_val_train,std_val_train
