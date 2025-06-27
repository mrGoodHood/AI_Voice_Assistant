import streamlit as st
import openai
from elevenlabs.client import ElevenLabs
import torch
import soundfile as sf
import io

st.title("Голосовой AI-ассистент")
движок = st.radio("Выберите движок синтеза речи:", ["ElevenLabs (платно)", "Silero TTS (бесплатно)"])

st.write("Нажмите и удерживайте кнопку, чтобы записать вопрос голосом:")
аудио_запись = st.audio_input("Запись аудио")

if аудио_запись:
    with open("вопрос.wav", "wb") as f:
        f.write(аудио_запись.getbuffer())

    аудио_файл = open("вопрос.wav", "rb")
    транскрипт = openai.Audio.transcribe("whisper-1", аудио_файл, language="ru")
    текст_вопроса = транскрипт.text
    st.subheader("Распознанный текст:")
    st.write(текст_вопроса)