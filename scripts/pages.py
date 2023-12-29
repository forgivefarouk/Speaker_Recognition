import streamlit as st
from pydub import AudioSegment
import sounddevice as sd
import soundfile as sf
import os
fs=16000
def page1():
    def generator():
        for i in range(10000):
            yield i

    def record_voice(path):
        # Set recording parameters
        fs = 16000  # Sampling frequency
        seconds = 30  # Recording duration in seconds


        
        # Record audio
        recorded_audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        

        if os.path.isdir(os.path.join("train_demo")):
            pass
        else:
            os.mkdir(os.path.join("train_demo"))
            
            
        
        file_name = f"train_demo/{path}.wav"
        sf.write(file_name, recorded_audio, fs)

    st.title("Train Voice Recorder")

    text_input = st.text_input("Enter your name:")
    st.write("Plese record 30 Second:")
    st.button("Record Voice", on_click=record_voice, args=(text_input,))


def page2():
    def generator2():
        for i in range(10000):
            yield i

    def record_voice(path):
        # Set recording parameters
        fs = 16000  # Sampling frequency
        seconds = 5  # Recording duration in seconds

        
        
        # Record audio
        recorded_audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        

        
        if os.path.isdir(os.path.join("test_demo")):
            pass
        else:
            os.mkdir(os.path.join("test_demo"))
            
            
        file_name = f"test_demo/{path}.wav"
        sf.write(file_name, recorded_audio, fs)

    st.title("Test Voice Recorder")

    text_input = st.text_input("Enter your name:")
    st.write("Plese record 5 Second:")
    st.button("Record Voice", on_click=record_voice, args=(text_input,))


st.sidebar.title("Navigation")
page_names = ["Train", "Test"]
selected_page = st.sidebar.radio("Select a page:", page_names)

if selected_page == "Train":
    page1()
elif selected_page == "Test":
    page2()
