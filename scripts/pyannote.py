from pyannote.audio import Pipeline
import os
import pandas as pd
import numpy as np
import librosa
import json
import soundfile as sf
from audiomentations import Compose , AddGaussianNoise, AddBackgroundNoise , AddGaussianSNR,TimeMask,SpecFrequencyMask,HighPassFilter,LowPassFilter
from tqdm import tqdm
import shutil 






from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")


from pyannote.audio import Inference
inference = Inference(model, window="whole")

from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

lab ={
    "classes" :[],
}




voice_df=pd.DataFrame()
for dirpath,dirnames,filenames in os.walk("train_demo"):
    for j,f in enumerate(tqdm(filenames, desc="Processing Files")):
        lab["classes"].append(f.replace(".wav","").split("_")[0])
        print(f"processing in {f} \n")
        file_name = os.path.join(dirpath,f)
        signal, sr = librosa.load(file_name, sr=16000)
        for i in range(30):
            start=i*sr
            end=start+sr
            segment=signal[start:end]
            if i==0:
                output_dir = "embeddings_train"
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
            filename = os.path.join(output_dir, f'segment_{f.replace(".wav","")}_{i+1}.wav')
            sf.write(filename, segment, sr)            
            embedding = inference(filename)
            sig_df=pd.DataFrame(embedding)
            sig_df=sig_df.T
            sig_df = pd.concat([sig_df,pd.DataFrame([j],columns=['label'])],axis=1)
            voice_df = pd.concat([voice_df,sig_df])
    
    
with open("demo.json", "w") as fp:
    json.dump(lab, fp)
    
voice_df.to_csv('demo_embedding.csv',index=False)
shutil.rmtree("embeddings_train")