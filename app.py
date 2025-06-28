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


сообщения = [
    {"role": "system", "content": "Вы — голосовой ассистент, отвечайте понятно и по существу."},
    {"role": "user", "content": текст_вопроса}
    ]
ответ = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=сообщения)
текст_ответа = ответ["choices"][0]["message"]["content"]
st.subheader("Ответ ассистента:")
st.write(текст_ответа)


st.write("Производим синтез речи...")
if движок.startswith("ElevenLabs"):
    client = ElevenLabs(api_key="ВАШ_ELEVENLABS_API_KEY")
    voice_id = "JBFqnCBsd6RMkjVDRZzb"  # пример: английский голос "George"
    аудиоданные = client.text_to_speech.convert(
        text=текст_ответа,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    st.audio(аудиоданные, format="audio/mp3")
else:
        model, _ = torch.hub.load(
        'snakers4/silero-models', 'silero_tts',
        language='ru', speaker='v4_ru'
    )
model.to('cpu')
аудио_массив = model.apply_tts(
    text=текст_ответа,
    speaker='xenia',
    sample_rate=48000
)
буфер = io.BytesIO()
sf.write(буфер, аудио_массив, 48000, format='WAV')
буфер.seek(0)
st.audio(буфер, format="audio/wav")