import streamlit as st
import pandas as pd
from openai import OpenAI
from gtts import gTTS
from playsound import playsound
import os
import uuid
import tempfile

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import speech_recognition as sr
import av

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# OpenAI API
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Convert text to speech
def speak_text(text):
    filename = f"{uuid.uuid4()}.mp3"
    tts = gTTS(text)
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# GPT to generate pandas code
def generate_pandas_code(question):
    prompt = f"""
Given this DataFrame df with columns:
student_id, school_code, grade, student_name, gender, academic_year, subject_name,
subject_marks, teacher_name, subject_grade, total_marks, average_score, result_status, stream

Write pandas code to answer:
{question}

Use df['student_id'].nunique() for counts.
Filter strings with .str.upper().str.strip().
Assign final output to `result`.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# GPT to summarize results
def summarize_result(question, result_df):
    prompt = f"""User asked: {question}
Data:
{result_df.to_string(index=False)}
Give a 1-line natural language summary."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# UI layout
st.title("🎙️ CBSE Copilot — Ask with Voice or Text")

# Voice input
st.subheader("🎤 Speak your question")

# Setup WebRTC for audio
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype('int16')
        recognizer = sr.Recognizer()
        with sr.AudioData(audio.tobytes(), frame.sample_rate, 2) as source:
            try:
                text = recognizer.recognize_google(source)
                st.session_state["voice_question"] = text
            except sr.UnknownValueError:
                st.warning("Could not understand voice")
        return frame

webrtc_streamer(
    key="audio",
    mode="SENDONLY",
    audio_processor_factory=AudioProcessor,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

# Capture voice result
question = st.text_input("💬 Or type your question:", value=st.session_state.get("voice_question", ""))

if question:
    with st.spinner("🤖 Thinking..."):
        try:
            code = generate_pandas_code(question)
            st.subheader("📄 Generated Code")
            st.code(code, language="python")

            # Execute code safely
            local_vars = {}
            exec(code, {"df": df, "pd": pd}, local_vars)
            result = local_vars.get("result")

            if result is None:
                result = next(reversed(local_vars.values()))

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame({"Result": [result]})

            st.subheader("📊 Result")
            st.dataframe(result)

            if not result.empty:
                insight = summarize_result(question, result)
                st.success("💬 Insight: " + insight)
                speak_text(insight)
            else:
                st.warning("⚠️ No results.")
                speak_text("No results were found.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
