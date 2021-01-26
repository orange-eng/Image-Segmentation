from net.SegNet import SegNet
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from PIL import Image
import keras
from keras import backend as K 
import numpy as np 
from keras.utils.data_utils import get_file
import sys
import os
apath = os.path.abspath(os.path.dirname(sys.argv[0]))
import cv2 as cv 

NCLASSES = 2
HEIGHT = 416
WIDTH = 416
BATCHES = 1
EPOCHES = 2

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0

    X_train = []
    Y_train = []
    # 获取一个batch_size大小的数据
    for _ in range(batch_size):
        if i==0:
            np.random.shuffle(lines)
        name = lines[i].split(';')[0]
        # 从文件中读取图像
        img = Image.open(apath+"\dataset\jpg" + '/' + name)
        img = img.resize((WIDTH,HEIGHT))
        img = np.array(img)
        img = img/255
        X_train.append(img)

        name = (lines[i].split(';')[1]).replace("\n", "")
        # 从文件中读取图像
        img = Image.open(apath+"\dataset\png" + '/' + name)
        img = img.resize((int(WIDTH/2),int(HEIGHT/2)))
        img = np.array(img)
        seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))
        for c in range(NCLASSES):
            seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
        seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
        Y_train.append(seg_labels)

        # 读完一个周期后重新开始
        i = (i+1) % n
    return (np.array(X_train),np.array(Y_train))

def loss(y_true,y_pred):
    loss = K.categorical_crossentropy(y_true,y_pred)
    return loss

if __name__ == "__main__":
    log_dir = apath+"/logs/"

    model = SegNet(n_classes=NCLASSES,input_height=HEIGHT,input_width=WIDTH)
    model.summary()

    with open(apath+"/dataset/train.txt") as f:
        lines = f.readlines()
    print(lines)
    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    batch_size = 4
    (X_train,Y_train) = generate_arrays_from_file(lines[:num_train], batch_size)

    model.compile(loss = loss,
        optimizer = Adam(lr=1e-3),
        metrics = ['accuracy'])
    
    model.fit(x=X_train,y=Y_train,batch_size=BATCHES,epochs=EPOCHES)
    model.save_weights(apath+"/logs/weight_%s.h5"% EPOCHES)

