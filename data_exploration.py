import matplotlib.pyplot as plt
import os
import numpy as np

image_path = "/projects/ag-bozek/sugliano/dlbcl/data/interim/resnet_imgs/"
onlyfiles = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

image = np.load(image_path + onlyfiles[103])
image = image[0, :, :]
plt.imshow(image)
plt.savefig('foo_bis.png')