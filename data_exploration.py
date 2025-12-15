import matplotlib.pyplot as plt
import os
import numpy as np

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

os.makedirs("/projects/ag-bozek/nmoreau/dapi_bis")

for sample in train_df:
    index, filename, label = sample[0], sample[1], sample[2]
    image = np.load(filename)
    image = image[0, :, :]
    print(index)
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