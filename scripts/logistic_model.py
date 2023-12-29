import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , f1_score
from keras import models
import pickle
import os
import json
import pickle
import librosa
import soundfile as sf
from keras.models import load_model
from collections import Counter
from scipy import signal
import noisereduce as nr
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from pyannote.audio import Model
from sklearn.preprocessing import StandardScaler


model_pretrained = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")


from pyannote.audio import Inference
inference = Inference(model_pretrained, window="whole")


def load_file(path):

    signal , sr =librosa.load(path,sr=16000)
    voice_df=pd.DataFrame()
    for i in range(150):
        start=i*sr
        end=start+sr
        segment=signal[start:end]
        filename = os.path.join("embeddings_test", f'segment_{i+1}.wav')
        sf.write(filename, segment, sr)
        embedding = inference(filename)
        sig_df=pd.DataFrame(embedding)
        sig_df=sig_df.T
        voice_df = pd.concat([voice_df,sig_df])
    return voice_df
voice_df = pd.read_csv('demo_embedding.csv')


X=voice_df.drop('label',axis=1)
y=voice_df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
st = StandardScaler()
st.fit(X_train)
X_train=st.transform(X_train)
X_val=st.transform(X_val)

best_params ={'C': 0.0001, 'penalty': 'l2', 'solver': 'liblinear'}

model = LogisticRegression(**best_params,random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f"Classification_report(y_val, y_pred): \n{classification_report(y_val, y_pred)}\n")


with open('models\\logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)