from torch.utils.data import Dataset
import numpy as np
from torch import tensor, float32
import torchvision.transforms as torch_transforms
import matplotlib.pyplot as plt
import os
import src.utils as utils

# dict_channel = { "DAPI" : 0, "CD8A" : 1905, "CD31" : 1651, "LAMIN_B" : -1, "CD11B" : 2802, "CD3D" : 1322, "CD20" : 3200,
# "CD163" : 5032, "CD68" : 2527, "CD204" : 1277, "CD4" : 4282, "FOXP3" : 6051, "LAMIN_AC" : -1, "PDL\_1" : 2372 }

class ImageDataset(Dataset):
    """Dataset class for a .csv containing paths to images
    
    Arguments:
    df -- dataframe
    fn_col -- name of column with the filenames
    lbl_col -- name of column with the class labels
    transform -- transform to apply
    return_filename -- if True, __getitem__ also returns the filename of the sample
    """
    #todo pad all images to have the same size before resizing to keep good resolution (max size train: 9340 8232, test: 8428 7824)
    def __init__(self, df, fn_col = None, lbl_col = None, transform = None, return_filename = False
                 , which_channels = [list(range(14))]):
        self.df = df
        self.fn_col = fn_col if fn_col != None else df.columns[0]
        self.lbl_col = lbl_col if lbl_col != None else df.columns[1]
        self.transform = transform
        self.return_filename = return_filename
        self.which_channels = which_channels
        self.aligned_channel = np.bool([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        fn = self.df.iloc[idx][self.fn_col]
        new_filename = "/projects/ag-bozek/sugliano/dlbcl/data/interim/aligned/" + os.path.basename(fn)[0:3] + "_aligned.npy"
        image = np.load(new_filename)
        image = np.reshape(image, [20, *np.shape(image)[-2:]])
        image = image[self.aligned_channel]

        #mask the background
        mask = image[0] > 400
        mask = scipy.ndimage.binary_dilation(mask, structure=np.ones((20, 20)))
        mask_connected_components, ncomponents = scipy.ndimage.label(
            mask)
        largestCC = mask_connected_components == np.argmax(
                np.bincount(mask_connected_components.flat)[1:]) + 1
        mask_connected_components[largestCC == False] = 0
        mask_connected_components = scipy.ndimage.binary_dilation(mask_connected_components, structure=np.ones((50, 50)))
        for c in range(0, image.shape[0]):
            image_c_temp = image[c]
            image_c_temp[mask_connected_components == 0] = 0
            image[c] = image_c_temp

        #z-score normalization
        image = utils.normalize_quantile(image)

        image = tensor(image, dtype=float32)
        if self.transform != None:
            image = self.transform(image)
        lbl = self.df.iloc[idx][self.lbl_col]
        out_tuple = (image, lbl, fn) if self.return_filename else (image, lbl)
        return out_tuple

    # def __getitem__(self, idx):
    #     fn = self.df.iloc[idx][self.fn_col]
    #     image = np.load(fn)
    #     image = image[self.which_channels, :, :]
    #     image = tensor(image, dtype=float32)[0]
    #     # diff_x = (self.size_max_x - image.shape[1]) // 2
    #     # diff_y = (self.size_max_y - image.shape[2]) // 2
    #     # transform_padding = torch_transforms.Pad((diff_y, diff_x))
    #     # image = transform_padding(image)
    #     if self.transform != None:
    #         image = self.transform(image)
    #     # file_name = os.path.basename(fn)
    #     # plt.imshow(image[0, :, :])
    #     # plt.savefig('dapi/' + file_name[:-3] + "png")
    #     lbl = self.df.iloc[idx][self.lbl_col]
    #     out_tuple = (image, lbl, fn) if self.return_filename else (image, lbl)
    #     return out_tuple
    
    def df(self):
        return self.df






