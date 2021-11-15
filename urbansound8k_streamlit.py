import os
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
import sklearn

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title(
'UrbanSound8K  Classification'
)

st.write('This model will classify sound files')

st.subheader('Dataset')
df = pd.read_csv('metadata/UrbanSound8K.csv')
st.write(df.head(10))

#MFCC feature extractor function


def features_extractor(file):
    audio,sample_rate=librosa.load(file,res_type='kaiser_fast')
    mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features=np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

#Load the model
from tensorflow.keras.models import load_model
new_model=load_model('urbansound8k.h5')

#Load the onehotencoder
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from joblib import load
enc=load('enc.joblib')


#Upload the audio file
filename = st.file_uploader("Choose a file")


submit = st.button('Classify')
if submit:
	prediction_features=features_extractor(filename).reshape(1,-1)
	pred_class=new_model.predict(prediction_features)
	class_name=enc.inverse_transform(pred_class)
	st.write(class_name)
