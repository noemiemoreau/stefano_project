from torch.utils.data import Dataset
import numpy as np
from torch import tensor, float32
import torchvision.transforms as torch_transforms
import matplotlib.pyplot as plt
import os
import src.utils as utils
import scipy
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
    def __init__(self, df, fn_col = None, lbl_col = None, transform = None, return_filename = False
                 , which_channels = [list(range(14))]):
        self.df = df
        self.fn_col = fn_col if fn_col != None else df.columns[0]
        self.lbl_col = lbl_col if lbl_col != None else df.columns[1]
        self.transform = transform
        self.return_filename = return_filename
        self.which_channels = which_channels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        fn = self.df.iloc[idx][self.fn_col]
        #new_filename = "/projects/ag-bozek/nmoreau/dlbcl/data/clipped/" + os.path.basename(fn)[0:3] + "_clipped.npy"
        image = np.load(new_filename)
        image = image[self.which_channels, :, :]
        image = tensor(image, dtype=float32)[0]
        if self.transform != None:
            image = self.transform(image)
        lbl = self.df.iloc[idx][self.lbl_col]
        out_tuple = (image, lbl, fn) if self.return_filename else (image, lbl)
        return out_tuple
    
    def df(self):
        return self.df






