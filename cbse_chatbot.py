import streamlit as st
import pandas as pd
from openai import OpenAI
import speech_recognition as sr
from gtts import gTTS
import os
import uuid
from playsound import playsound

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Generate pandas code from GPT
def generate_pandas_code(question):
    prompt = f"""
You are a data analyst. Given this DataFrame `df` with columns:
student_id, school_code, grade, student_name, gender, academic_year, subject_name,
subject_marks, teacher_name, subject_grade, total_marks, average_score, result_status, stream

User question: {question}

Write a valid Python pandas code using df to answer this question.

Rules:
- Use `subject_marks` for subject-specific stats.
- Use `average_score` for overall student stats.
- Count students using df['student_id'].nunique().
- Normalize string filters with .str.upper().str.strip().
- Assign output to a variable `result`.
- If nothing is returned, assign result = pd.DataFrame().
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Generate 1-line summary using GPT
def summarize_result(question, data):
    prompt = f"""User asked: {question}
Data:
{data.to_string(index=False)}

Give a short, 1-line summary of the insight."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Convert text to speech using gTTS
def speak_text(text):
    filename = f"temp_{uuid.uuid4()}.mp3"
    tts = gTTS(text)
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# Streamlit UI
st.title("🎙️ CBSE Copilot with Voice")

# Voice input
st.subheader("🎤 Speak your question (or type below):")
use_voice = st.button("Start Voice Input")

question = st.text_input("Or type your question:")

if use_voice:
    st.info("🎧 Listening... please speak clearly.")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        question = r.recognize_google(audio)
        st.success(f"You said: {question}")
    except Exception as e:
        st.error(f"Voice input error: {e}")
        question = ""

if question:
    with st.spinner("🤖 Thinking..."):
        try:
            code = generate_pandas_code(question)
            st.subheader("🧠 Generated Code")
            st.code(code, language="python")

            # Execute generated code
            local_vars = {}
            exec(code, {"df": df, "pd": pd}, local_vars)
            result = local_vars.get("result")

            if result is None:
                result = next(reversed(local_vars.values()))

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame({"Result": [result]})

            st.subheader("📊 Result")
            st.dataframe(result)

            # Voice + Text insight
            if result is not None and not result.empty:
                insight = summarize_result(question, result)
                st.success("💬 Insight: " + insight)
                speak_text(insight)
            else:
                st.warning("⚠️ No result found.")
                speak_text("No result was found for your question.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
