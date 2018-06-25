import numpy as np
import pandas as pd
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam

batch_size = 128
epochs = 500
train=np.load('train.npy')
label=np.load('label.npy')
train=train.reshape(9473,40,391,1)
x=train
y=label


kfold=10
n=9473
split=np.append(np.array(np.repeat(int(n/kfold),kfold-1)),(n-(kfold-1)*int(n/kfold)))
cumsplit=np.cumsum(split)
for i in range(kfold):
    X_train,y_train=np.delete(x,np.arange(0+i*split[0],cumsplit[i]),axis=0),np.delete(y,np.arange(0+i*split[0],cumsplit[i]),axis=0)
    X_test,y_test=x[np.arange(0+i*split[0],cumsplit[i]),:],y[np.arange(0+i*split[0],cumsplit[i]),:]
    model = Sequential()
    model.add(Conv2D(32, (4, 10), input_shape=(40,391,1),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))
    model.add(Conv2D(32, (4, 10),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))
    model.add(Conv2D(32, (4, 10),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(units = 50, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units = 41, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    ckpt = ModelCheckpoint(filepath='kfold%i.h5'%i, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    print('train_________')
    model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,validation_data=(X_test, y_test),callbacks=[ckpt])




    print('\nTesting______')
    loss, accuracy=model.evaluate(X_test,y_test)
    print('\ntest loss:',loss)
    print('\ntest accuracy:',accuracy)







