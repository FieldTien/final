import os
import shutil
import platform
import shutil
import platform
import scipy
import numpy as np 
import pandas as pd
import librosa
from keras.models import load_model
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.utils import Sequence, to_categorical
from sklearn.cross_validation import StratifiedKFold

np.random.seed(1001)
input_length=200000
train_label=pd.read_csv('train (3).csv')


#['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock', 'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard', 'Keys_jangling', 'Snare_drum', 'Writing', 'Laughter', 'Tearing', 'Fart', 'Oboe', 'Flute', 'Cough', 'Telephone', 'Bark', 'Chime', 'Bass_drum', 'Bus', 'Squeak', 'Scissors', 'Harmonica','Gong', 'Microwave_oven', 'Burping_or_eructation', 'Double_bass', 'Shatter', 'Fireworks', 'Tambourine', 'Cowbell', 'Electric_piano', 'Meow', 'Drawer_open_or_close', 'Applause', 'Acoustic_guitar', 'Violin_or_fiddle', 'Finger_snapping']
sample_submit=pd.read_csv('sample_submission.csv')
id=sample_submit['fname'].tolist()


filefold='audio_test/'



def data_prepocess(id,filefold):
    a=np.zeros((9400,40,391))
    step=0
    for i in id:
        wave, fs = librosa.load(filefold+'%s'%i, sr=None)
    #reference https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data/code
        if len(wave) > input_length:  
           max_offset = len(wave) - input_length
           offset = np.random.randint(max_offset)
           wave = wave[offset:(input_length+offset)]
        elif input_length > len(wave):
           max_offset = input_length - len(wave)
           offset = np.random.randint(max_offset)
           wave = np.pad(wave, (offset, input_length - len(wave) - offset), "constant")

        chroma = librosa.feature.mfcc(wave, sr=44100, n_mfcc=40)
    
    #time=time+[chroma.shape[1]]
    #    print(step)
        a[step,:,:]=chroma
        step=step+1
    a=a.reshape(9400,40,391,1)
    return(a)

for i in range(4):
    
    np.random.seed(1001*(i+1)+1000)
    a=data_prepocess(id,filefold)
    
    np.save('2D_data_pre%s.npy'%i,a) 

