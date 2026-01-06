import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from torch import tensor, float32

train_df = pd.read_csv("train_bis.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test_bis.csv")

# os.makedirs("/projects/ag-bozek/nmoreau/dapi_bis/train")
size_max_x = 9400
size_max_y= 9400

for idx in range(0, test_df.shape[0]):
    index, filename, label = test_df.iloc[idx][0], test_df.iloc[idx][1], test_df.iloc[idx][2]
    image = np.load(filename)
    print(image.shape)
    print(image.dtype)
    image = tensor(image, dtype=float32)[0]
    print(image.shape)
    print(image.dtype)





#     size_x = image.shape[0]
#     size_y = image.shape[1]
#     if size_x > size_max_x:
#         size_max_x = size_x
#         size_max_y = size_y
#     print(index, label, size_x, size_y)
# print(size_max_x, size_max_y)
    # plt.imshow(image)
    # plt.savefig('/projects/ag-bozek/nmoreau/dapi_bis/test/' + str(index) + "_" +str(label) + ".png")



# image_path = "/projects/ag-bozek/sugliano/dlbcl/data/interim/resnet_imgs/"
# onlyfiles = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
#
# for f in onlyfiles:
#     image = np.load(image_path + f)
#     image = image[0, :, :]
#     plt.imshow(image)
#     plt.savefig('dapi/'+f[:-3]+"png")