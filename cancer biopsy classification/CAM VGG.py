from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras import callbacks
import cv2
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


num_classes = 2
#data generators and augmentation
train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        vertical_flip=True,)

test_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
        'D:\\Rishal\\PycharmProjects\\sop\\binary_classification\\40x\\training',
        batch_size=15,
        target_size=(460, 700),
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        'D:\\Rishal\\PycharmProjects\\sop\\binary_classification\\40x\\test',
        batch_size=15,
        target_size=(460, 700),
        class_mode='categorical')
val_generator = test_datagen.flow_from_directory(
        'D:\\Rishal\\PycharmProjects\\sop\\binary_classification\\40x\\validation',
        batch_size=15,
        target_size=(460, 700),
        class_mode='categorical')

class_names = ['benign','malignant']


def global_average_pooling(x):
    return K.mean(x, axis=(1, 2))

def global_average_pooling_shape(input_shape):
    return (input_shape[0],input_shape[3])
#model definition
def base_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(460,700,3)))
    model.add(Conv2D(64, (3, 3)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    #global average pooling layer for CAMs
    model.add(Lambda(global_average_pooling,
                     output_shape=global_average_pooling_shape))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
model=base_model()
#callbacks
callback=[]
csvlogger=callbacks.CSVLogger
checkpt=ModelCheckpoint('vgg_models/40X/weights.{epoch:02d}_CAM.hdf5', monitor='val_acc',verbose=1,save_best_only=True)
callback.append(csvlogger)
callback.append(checkpt)
#training
'''model.fit_generator(
        train_generator,
        steps_per_epoch=80,
        verbose=1,
        callbacks=[checkpt],
        validation_data=test_generator,
        validation_steps=27,
        epochs=50)'''
#testing
model.load_weights('D:\\Rishal\\PycharmProjects\\sop\\vgg_models\\40X\\weights.02_CAM.hdf5')
scores = model.evaluate_generator(val_generator,steps=26,verbose=1)
print("Accuracy = ", scores[1])
