import os
import json
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
import librosa
import soundfile as sf
from keras.models import load_model
from collections import Counter
from scipy.signal import butter, lfilter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , f1_score , precision_score, recall_score
import noisereduce as nr
import random
import shutil
from sklearn.preprocessing import LabelEncoder
from pyannote.audio import Model
model_predrained = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")


from pyannote.audio import Inference
inference = Inference(model_predrained, window="whole")



def load_file(path):

    signal , sr =librosa.load(path,sr=16000)
    #signal = reduce_noise(signal, sr)
    voice_df=pd.DataFrame()
    for i in range(5):
        start=i*sr
        end=start+sr
        segment=signal[start:end]
        if i==0:
            output_dir = "embeddings_test"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        filename = os.path.join("embeddings_test", f'segment_{i+1}.wav')
        sf.write(filename, segment, sr)
        embedding = inference(filename)
        sig_df=pd.DataFrame(embedding)
        sig_df=sig_df.T
        voice_df = pd.concat([voice_df,sig_df])
    shutil.rmtree("embeddings_test")
    return voice_df




if __name__ == "__main__":
    with open("demo.json", "r") as fp:
        data = json.load(fp)
        
    classes = data["classes"]
    pred_arr=[]
    y_true = []

    print(f"Speakers: \n")
    for i in os.listdir("test_demo"):
        y_true.append(classes.index(i.replace(".wav","").split("_")[0]))
        DATA_PATH = f"test_demo\\{i}"
        X= load_file(DATA_PATH)
        
        log_model = pickle.load(open('models\\logistic_model.pkl', 'rb'))


        pred = log_model.predict(X)
        
        
                
        #print(Counter(pred))
        lab= Counter(pred).most_common(1)[0][0]
        pred_arr.append(lab)
        
        
        print(classes[lab])
    #print(f"F1_score: {f1_score(y_true, pred_arr, average='macro'):.4f}")
    #print(f"Recall: {recall_score(y_true, pred_arr, average='macro'):.4f}")
    #print(f"Precision: {precision_score(y_true, pred_arr, average='macro'):.4f}")