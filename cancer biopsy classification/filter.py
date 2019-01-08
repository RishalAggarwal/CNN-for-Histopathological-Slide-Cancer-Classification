import numpy as np
import skimage
from skimage import io
import os
benign_path_final='D:\\Rishal\\PycharmProjects\\sop\\binary_classification'
for roots,dirs,files in os.walk(benign_path_final):
    for file in files:
        img = skimage.io.imread(os.path.join(roots,file))
        if np.asarray(img).shape[0] != 460:
            print(file)