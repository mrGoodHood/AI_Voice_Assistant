import streamlit as st
import openai
from elevenlabs.client import ElevenLabs
import torch
import soundfile as sf
import io

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
        # Используем ElevenLabs API (платно)
        client = ElevenLabs(api_key="ВАШ_ELEVENLABS_API_KEY")
        # Пример использования: voice_id можно получить через client.voices.search()
        voice_id = "JBFqnCBsd6RMkjVDRZzb"  # пример: английский голос "George"
        audio_data = client.text_to_speech.convert(
            text=response_text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        st.audio(audio_data, format="audio/mp3")
    else:
        # Используем Silero TTS (бесплатно)
        model, _ = torch.hub.load(
            'snakers4/silero-models', 'silero_tts',
            language='ru', speaker='v4_ru'
        )
        model.to('cpu')
        audio_massive = model.apply_tts(
            text=response_text,
            speaker='xenia',        # одна из русскоязычных актрис Silero
            sample_rate=48000
        )
        # Сохраняем аудио в буфер и воспроизводим
        buffer = io.BytesIO()
        sf.write(buffer, audio_massive, 48000, format='WAV')
        buffer.seek(0)
        st.audio(buffer, format="audio/wav")