from __future__ import print_function,division

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import *
from keras.models import *
import keras
from keras.layers import SeparableConv2D
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import sys
import os
apath = os.path.abspath(os.path.dirname(sys.argv[0]))
import cv2 as cv

HEIGHT = 224
WIDTH = 224
CHANNELS = 3
EPOCHES = 2

def MobileNet(input_height,input_width):
    img_input = Input(shape=(input_height,input_width,CHANNELS))
    #x = Conv2D(32,kernel_size=3,activation='relu',strides=1,padding='SAME')(img_input)
    x = SeparableConv2D(filters=32,kernel_size=3,strides=1,padding='SAME',activation='relu')(img_input)
    x = BatchNormalization()(x)
    f1 = x
    x = SeparableConv2D(filters=32,kernel_size=3,strides=2,padding='SAME',activation='relu')(img_input)
    x = BatchNormalization()(x)
    f2 = x
    #112*112*32
    x = SeparableConv2D(filters=64,kernel_size=3,strides=2,padding='SAME',activation='relu')(x)
    x = BatchNormalization()(x)
    f3 = x
    #56*56*64
    x = SeparableConv2D(filters=128,kernel_size=3,strides=1,padding='SAME',activation='relu')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(filters=128,kernel_size=3,strides=2,padding='SAME',activation='relu')(x)
    x = BatchNormalization()(x)
    f4 = x
    #28*28*128
    x = SeparableConv2D(filters=256,kernel_size=3,strides=1,padding='SAME',activation='relu')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(filters=256,kernel_size=3,strides=2,padding='SAME',activation='relu')(x)
    x = BatchNormalization()(x)
    f5 = x
    #14*14*256
    x = SeparableConv2D(filters=512,kernel_size=3,strides=1,padding='SAME',activation='relu')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(filters=512,kernel_size=3,strides=2,padding='SAME',activation='relu')(x)
    x = BatchNormalization()(x)
    #f5 = x
    #7*7*512
    x = SeparableConv2D(filters=1024,kernel_size=3,strides=1,padding='SAME',activation='relu')(x)
    x = BatchNormalization()(x)
    #7*7*1024
    x = AveragePooling2D(pool_size=(7,7),strides=1,padding='valid')(x)
    #1*1*1024
    x = Dense(1000,activation='relu')(x)
    #1*1*1000
    return img_input,[f1,f2,f3,f4,f5]



def get_decoder(decoder_input,n_classes):
    x = decoder_input

    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(512,(3,3),padding='valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2,2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256,(3,3),padding='valid')(x)
    x = BatchNormalization()(x)    

    x = UpSampling2D(size=(2,2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(128,(3,3),padding='valid')(x)
    x = BatchNormalization()(x)   

    x = UpSampling2D(size=(2,2))(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(64,(3,3),padding='valid')(x)
    x = BatchNormalization()(x)

    x = Conv2D(n_classes,(3,3),padding='SAME')(x)

    return x

def SegNet(n_classes,input_height=224,input_width=224,encoder_level=4):
    img_input,levels = MobileNet(input_height=input_height,input_width=input_width)
    #获取压缩4次时的结果，赋值为feat
    feat = levels[encoder_level]

    x = get_decoder(decoder_input=feat,n_classes=n_classes)

    x = Reshape((int(input_height/2)*int(input_width/2),-1))(x)
    #x = Softmax()(x)

    model = Model(img_input,x)
    
    return model

mobilenet = SegNet(n_classes=2)
mobilenet.summary()