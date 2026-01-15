import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from torch import tensor, float32
import torchvision.transforms as transforms
import src.utils as utils
from src.dataset import DistributedWeightedSampler, ImageDataset
from torch.utils.data import DataLoader

# train_df = pd.read_csv("train_bis.csv")
# val_df = pd.read_csv("val.csv")
# test_df = pd.read_csv("test_bis.csv")


train_transform = transforms.Compose([
        transforms.CenterCrop(7000),
        transforms.Resize((4700, 4700)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
])

which_channels = [list(range(14))]

train_df = pd.read_csv("train_cleaned.csv")
train_dataset = ImageDataset(train_df, fn_col = 'filename', lbl_col = "relapse", transform = train_transform, return_filename=True)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=True)

for i, batch in enumerate(train_loader):
    img_tensor, target, filename = batch[0], batch[1], batch[2]
    x, valid_ind = utils.get_valid_patches(img_tensor, self.patch_size, self.patch_size, rand_offset=False)
    print(x.shape)

# size_max_x = 9400
# size_max_y= 9400
# which_channels = [list(range(14))]
#
# for idx in range(0, train_df.shape[0]):
#     index, filename, label = train_df.iloc[idx][0], train_df.iloc[idx][1], train_df.iloc[idx][2]
#     image_np = np.load(filename)
#     image_np = image_np[which_channels, :, :]
#     print(filename, image_np.shape)
#     train_transform = transforms.Compose([
#         transforms.CenterCrop(7000),
#         transforms.Resize((1024, 1024)),
#         # transforms.ToTensor(),
#     ])
#     image = tensor(image_np, dtype=float32)[0]
#     # diff_x = (size_max_x - image.shape[1]) // 2
#     # diff_y = (size_max_y - image.shape[2]) // 2
#     # transform = transforms.Pad((diff_y, diff_x))
#     image = train_transform(image)
#     print(image.shape)
#     file_name = os.path.basename(filename)
#     plt.imshow(image[0, :, :])
#     plt.savefig('dapi_croped/train//' + file_name[:-3] + "png")





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