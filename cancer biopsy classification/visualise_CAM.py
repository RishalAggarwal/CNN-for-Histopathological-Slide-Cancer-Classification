import keras.backend as K
from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras import callbacks
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Lambda
from keras.layers import Dropout
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm
from keras.layers import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal
from keras.utils import np_utils
from keras import backend as K
from keras.models import *
from keras.callbacks import *
import cv2
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import color
def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer
def visualize_class_activation_map(model_path, img_path_norm,img_path_rgb, output_path): #model path, normalised image path, original image path
    model = load_model(model_path)
    original_img = io.imread(img_path_norm)
    original_img=np.expand_dims(original_img,axis=0)
    rgb_img=cv2.imread(img_path_rgb,1)
    width, height, _ = rgb_img.shape

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-2].get_weights()[0]
    #final_conv_layer = get_output_layer()
    get_output = K.function([model.layers[0].input], [model.layers[-4].output,model.layers[-1].output])
    [conv_outputs, predictions] = get_output([original_img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    target_class = np.argmax(predictions)
    for i, w in enumerate(class_weights[:, target_class]):
        cam += w * conv_outputs[:, :, i]
    print ("predictions", predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    cam=(cam-np.min(cam))/(np.max(cam)-np.min(cam))*255
    cam=cv2.equalizeHist(cam.astype(np.uint8))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))*255
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.25 + rgb_img
    img=np.hstack((rgb_img,img))
    cv2.imwrite(output_path,img)



visualize_class_activation_map('D:\\Rishal\\PycharmProjects\\sop\\vgg_models\\40X\\weights.02_CAM.hdf5','D:\\Rishal\\PycharmProjects\sop\\binary_classification\\40x\\validation\\malignant\\SOB_M_PC-14-19440-40-004_norm.tif' ,'D:\Rishal\BreaKHis_v1\BreaKHis_v1\histology_slides\\breast\\malignant\SOB\\papillary_carcinoma\\SOB_M_PC_14-19440\\40X\\SOB_M_PC-14-19440-40-004.png','D:\Rishal\PycharmProjects\sop\CAM imgs\\40x\SOB_M_PC-14-19440-40-004_cam.png')
