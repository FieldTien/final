import numpy as np 
import pandas as pd
import librosa
from keras.models import load_model
input_length=200000
train_label=pd.read_csv('train (3).csv')
encoded=pd.get_dummies(train_label['label'])
encoded=list(encoded.columns)

sample_submit=pd.read_csv('sample_submission.csv')
id=sample_submit['fname'].tolist()

filefold='audio_test/'

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
    
    
    print(step)
    a[step,:,:]=chroma
    step=step+1

model = load_model('best.h5')
a=a.reshape(9400,40,391,1)
y_hat=model.predict(a)
y_hat=np.argsort(y_hat, axis=1)

for i in range(9400):
    predicted_labels=[' '.join([encoded[y_hat[i,-1]],encoded[y_hat[i,-2]],encoded[y_hat[i,-3]]])]
    sample_submit['label'][i]=predicted_labels[0]
sample_submit.to_csv('result3.csv',index=False)
print(sample_submit)    

