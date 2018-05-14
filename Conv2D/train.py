import numpy as np 
import pandas as pd
import librosa 
import librosa.display
import matplotlib.pyplot as plt

input_length=200000 #medain size is 179046 and mean suze is 299350.68827193073


train_data=pd.read_csv('train (3).csv')
id=train_data['fname'].tolist()

filefold='audio_train/'
encoded=pd.get_dummies(train_data['label'])
encoded=np.array(encoded)
np.save('label.npy',encoded)
a=np.zeros((9473,40,391))
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
    
    
    print(step)
    a[step,:,:]=chroma
    step=step+1




np.save('train.npy',a)
print(a[12,:,:])



    
    

