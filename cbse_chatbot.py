import streamlit as st
import pandas as pd
from openai import OpenAI

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# Setup OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Generate pandas code from question
def generate_pandas_code(question):
    prompt = f"""
You are a data analyst. Given this DataFrame `df` with columns:
student_id, school_code, grade, student_name, gender, academic_year, subject_name,
subject_marks, teacher_name, subject_grade, total_marks, average_score, result_status, stream

User question: {question}

Write a valid pandas code using df to answer this. Only return the code, no explanation.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Summarize the result
def summarize_result(question, data):
    prompt = f"""User asked: {question}
Data:
{data.to_string(index=False)}

Give a 1-line summary of the result."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("📊 CBSE Copilot (GPT + Streamlit)")
question = st.text_input("Ask a question about CBSE data:")

if question:
    with st.spinner("Generating response..."):
        try:
            code = generate_pandas_code(question)
            st.subheader("📄 Generated Code")
            st.code(code, language="python")

            # Execute generated code safely
            result = eval(code)

            st.subheader("📊 Result")
            st.dataframe(result)

            if not result.empty:
                insight = summarize_result(question, result)
                st.success("💬 Insight: " + insight)
            else:
                st.warning("No data matched.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
