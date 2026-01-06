import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from torch import tensor, float32
import torchvision.transforms as transforms

train_df = pd.read_csv("train_bis.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test_bis.csv")

# os.makedirs("/projects/ag-bozek/nmoreau/dapi_bis/train")
size_max_x = 9400
size_max_y= 9400
which_channels = [list(range(14))]

for idx in range(0, test_df.shape[0]):
    index, filename, label = test_df.iloc[idx][0], test_df.iloc[idx][1], test_df.iloc[idx][2]
    image_np = np.load(filename)
    image_np = image_np[which_channels, :, :]
    print(image_np.shape)
    print(image_np.dtype)
    image = tensor(image_np, dtype=float32)[0]
    diff_x = (size_max_x - image_np.shape[0]) // 2
    diff_y = (size_max_y - image_np.shape[1]) // 2
    print(diff_x, diff_y)
    transform = transforms.Pad((diff_x, diff_y))
    image = transform(image)
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