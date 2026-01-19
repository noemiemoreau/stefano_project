import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import scipy
import src.utils as utils


train_df = pd.read_csv("train_cleaned.csv")
test_df = pd.read_csv("test_cleaned.csv")

new_directory = "/projects/ag-bozek/nmoreau/dlbcl/data/normalized/"

aligned_channel = np.bool([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0])

for idx in range(0, train_df.shape[0]):
    index, filename, label = train_df.iloc[idx][0], train_df.iloc[idx][1], train_df.iloc[idx][2]
    print(filename)
    new_filename = "/projects/ag-bozek/sugliano/dlbcl/data/interim/aligned/" + os.path.basename(filename)[
                                                                               0:3] + "_aligned.npy"
    image = np.load(new_filename)
    image = np.reshape(image, [20, *np.shape(image)[-2:]])
    image = image[aligned_channel]
    # mask the background
    mask = image[0] > 400
    mask = scipy.ndimage.binary_dilation(mask, structure=np.ones((20, 20)))
    counts = np.bincount(labeled.ravel())
    small_labels = np.where(counts < 100000)[0]
    # never remove background (label 0)
    small_labels = small_labels[small_labels != 0]

    # create output mask
    mask_clean = labeled.copy()
    for lbl in small_labels:
        mask_clean[mask_clean == lbl] = 0

    # binarize for convenience
    mask_clean = (mask_clean > 0).astype(mask.dtype)
    mask_clean = scipy.ndimage.binary_dilation(mask_clean, structure=np.ones((50, 50)))
    for c in range(0, image.shape[0]):
        image_c_temp = image[c]
        image_c_temp[mask_clean == 0] = 0
        image[c] = image_c_temp

    # z-score normalization
    image = utils.normalize_quantile(image)

    np.save(new_directory + os.path.basename(filename)[0:3] + "_normalized.npy", image)