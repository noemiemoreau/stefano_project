import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

os.makedirs("/projects/ag-bozek/nmoreau/dapi_bis")

for idx in range(0, train_df.shape[0]):
    index, filename, label = train_df.iloc[idx][0], train_df.iloc[idx][1], train_df.iloc[idx][2]
    image = np.load(filename)
    image = image[0, :, :]
    print(index, label)
    plt.imshow(image)
    plt.savefig('/projects/ag-bozek/nmoreau/dapi_bis/' + str(index) + "_" +str(label) + ".png")



# image_path = "/projects/ag-bozek/sugliano/dlbcl/data/interim/resnet_imgs/"
# onlyfiles = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
#
# for f in onlyfiles:
#     image = np.load(image_path + f)
#     image = image[0, :, :]
#     plt.imshow(image)
#     plt.savefig('dapi/'+f[:-3]+"png")