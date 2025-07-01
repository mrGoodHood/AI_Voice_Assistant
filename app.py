import streamlit as st
import openai
from elevenlabs.client import ElevenLabs
import torch
import soundfile as sf
import io

from jinja2.compiler import generate

st.title("Голосовой AI-ассистент")
# Выбор движка синтеза речи
engine = st.radio("Выберите движок синтеза речи:", ["ElevenLabs (платно)", "Silero TTS (бесплатно)"])

st.write("Нажмите и удерживайте кнопку, чтобы записать вопрос голосом:")
# Запись аудио с микрофона пользователя
audio = st.audio_input("Запись аудио")

if audio:
    # Сохраняем записанное аудио во временный WAV-файл
    with open("вопрос.wav", "wb") as f:
        f.write(audio.getbuffer())

    # Распознаём речь через Whisper API
    audio_file = open("вопрос.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="ru")
    text = transcript.text
    st.subheader("Распознанный текст:")
    st.write(text)

    # Генерируем ответ через ChatGPT (gpt-3.5-turbo)
    messages = [
        {"role": "system", "content": "Вы — голосовой ассистент, отвечайте понятно и по существу."},
        {"role": "user", "content": text}
    ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    response_text = response["choices"][0]["message"]["content"]
    st.subheader("Ответ ассистента:")
    st.write(response_text)

    # Синтезируем ответ в речь
    st.write("Производим синтез речи...")
    if engine.startswith("ElevenLabs"):
        audio_data = generate(
            text=response_text,
            voice="George",  # можно заменить на подходящий голос
            model="eleven_multilingual_v2"
        )
        st.audio(audio_data, format="audio/mp3")
    else:
        @st.cache_resource
        def load_silero_model():
            model, _ = torch.hub.load(
                'snakers4/silero-models', 'silero_tts',
                language='ru', speaker='v4_ru'
            )
            model.to('cpu')
            return model

        model = load_silero_model()
        audio_massive = model.apply_tts(
            text=response_text,
            speaker='xenia',
            sample_rate=48000
        )

        # Сохраняем аудио в буфер и воспроизводим
        buffer = io.BytesIO()
        sf.write(buffer, audio_massive, 48000, format='WAV')
        buffer.seek(0)
        st.audio(buffer, format="audio/wav")