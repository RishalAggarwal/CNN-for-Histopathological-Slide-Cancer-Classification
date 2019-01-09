import matplotlib.pyplot as plt
from skimage import io
from matplotlib.colors import LinearSegmentedColormap
#reading and visualising hematoxylin stain vector
im=io.imread('D:\Rishal\BreaKHis_v1\BreaKHis_v1\histology_slides\\breast\\benign\SOB\\tubular_adenoma\SOB_B_TA_14-3411F\\40X\SOB_B_TA-14-3411F-40-011_norm.tif')
cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white','navy'])
plt.imshow(im[:,:,0],cmap=cmap_hema)
plt.show()
