"""
STAIN.NORM: various methods for stain normalization.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'vlad'
__version__ = 0.1

import numpy as np
import scipy
from scipy import signal
from skimage import io
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from sklearn import preprocessing
from matplotlib.colors import LinearSegmentedColormap
import os



def compute_macenko_norm_matrix(im, alpha=1.0, beta=0.15):
    """
    Implements the staining normalization method from
      Macenko M. et al. "A method for normalizing histology slides for
      quantitative analysis". ISBI 2009
    :param im:
    :param alpha:
    :param beta:
    :return:
    """
    if im.ndim != 3:
        raise ValueError('Input image must be RGB')
    h, w, _ = im.shape

    im = (im + 1.0) / 255.0 # img_as_float(im)
    # im = rescale_intensity(im, out_range=(0.001, 1.0))  # we'll take log...
    im = im.reshape((h*w, 3), order='F')
    od = -np.log(im)                 # optical density
    odhat = od[~np.any(od < beta, axis=1), ]
    _, V = np.linalg.eigh(np.cov(odhat, rowvar=0))  # eigenvectors of a symmetric matrix
    theta = np.dot(odhat,V[:, 1:3])
    phi = np.arctan2(theta[:,1], theta[:,0])
    minPhi, maxPhi = np.percentile(phi, [alpha, 100-alpha])
    vec1 = np.dot(V[:,1:3] , np.array([[np.cos(minPhi)],[np.sin(minPhi)]]))
    vec2 = np.dot(V[:,1:3] , np.array([[np.cos(maxPhi)],[np.sin(maxPhi)]]))
    stain_matrix = np.zeros((3,3))
    if vec1[0] > vec2[0]:
        stain_matrix[:, :2] = np.hstack((vec1, vec2))
    else:
        stain_matrix[:, :2] = np.hstack((vec2, vec1))

    stain_matrix[:, 2] = np.cross(stain_matrix[:, 0], stain_matrix[:, 1])

    #he1_from_rgb = linalg.inv(rgb_from_he1)
    return stain_matrix.transpose()
path='D:\\Rishal\\crchistophenotypes_2016_04_28\\CRCHistoPhenotypes_2016_04_28'
path1=os.path.join(path,os.listdir(path)[0])
path2=os.path.join(path,os.listdir(path)[0])
for i in os.listdir(path1):
    
img=io.imread('C:\\Users\\risha\\Downloads\\crchistophenotypes_2016_04_28\\CRCHistoPhenotypes_2016_04_28\\Detection\\img20\\img20.bmp')
norm_img_kernel=compute_macenko_norm_matrix(img, alpha=1.0, beta=0.15)
norm_img_kernel=preprocessing.normalize(norm_img_kernel)
decon_kernel=np.linalg.inv(norm_img_kernel)
img1=img
img1[img1==0]=1
print(np.unique(img1))
OD=-np.log(img1)
OD1=OD
rOD = np.reshape(OD,(-1,3))
rC = np.dot(rOD,decon_kernel)
C = np.reshape(rC,OD.shape)
C[OD1>0.15]=0
C=(C-np.min(C))/(np.max(C)-np.min(C))
#C[:,:,0]=(C[:,:,0]-np.min(C[:,:,0]))/(np.max(C[:,:,0])-np.min(C[:,:,0]))
#C[:,:,1]=(C[:,:,1]-np.min(C[:,:,1]))/(np.max(C[:,:,1])-np.min(C[:,:,1]))
#C[:,:,2]=(C[:,:,2]-np.min(C[:,:,2]))/(np.max(C[:,:,2])-np.min(C[:,:,2]))
print(np.unique(C))
cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white','navy'])
f,(ax1,ax2)=plt.subplots(1,2)
ax1.imshow(img)
ax2.imshow(C[:,:,0],cmap=cmap_hema)
plt.show()