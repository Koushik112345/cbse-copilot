import streamlit as st
import pandas as pd
from openai import OpenAI

# Load CSV data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# OpenAI client (using API key from Streamlit Secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Generate pandas code from user's question
def generate_pandas_code(question):
    prompt = f"""
You are a data analyst. Given this DataFrame `df` with columns:
student_id, school_code, grade, student_name, gender, academic_year, subject_name,
subject_marks, teacher_name, subject_grade, total_marks, average_score, result_status, stream

User question: {question}

Write a valid Python pandas code using df to answer this question.
Always return the result as a DataFrame using pd.DataFrame(...) — even if it's a single value.
Only return code, no explanations.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# Summarize the result in plain language
def summarize_result(question, data):
    prompt = f"""User asked: {question}
Data:
{data.to_string(index=False)}

Provide a 1-line summary of the result in simple language."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("📊 CBSE Copilot — Ask Your Data Anything")
question = st.text_input("💬 Ask a question about CBSE results:")

if question:
    with st.spinner("🤖 Thinking..."):
        try:
            code = generate_pandas_code(question)
            st.subheader("📄 Generated Code")
            st.code(code, language="python")

            # Evaluate the code to get the result
            result = eval(code)

            st.subheader("📊 Result")
            st.dataframe(result)

            if not result.empty:
                insight = summarize_result(question, result)
                st.success("💬 Insight: " + insight)
            else:
                st.warning("⚠️ No data matched your query.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
