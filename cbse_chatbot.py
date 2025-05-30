import streamlit as st
import pandas as pd
from openai import OpenAI

# Load CSV data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# OpenAI client using Streamlit secret key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Generate pandas code from question
def generate_pandas_code(question):
    prompt = f"""
You are a data analyst. Given this DataFrame `df` with columns:
student_id, school_code, grade, student_name, gender, academic_year, subject_name,
subject_marks, teacher_name, subject_grade, total_marks, average_score, result_status, stream

User question: {question}

Write a valid Python pandas code using df to answer this question.
Always assign the final result to a variable named result.
The result must be a pandas DataFrame (pd.DataFrame).
Do not print or explain anything. Only return Python code.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Summarize the result using GPT
def summarize_result(question, data):
    prompt = f"""User asked: {question}
Data:
{data.to_string(index=False)}

Provide a short, clear 1-line summary of the result."""
    
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

            # Use exec to run multi-line GPT code
            local_vars = {}
            exec(code, {"df": df, "pd": pd}, local_vars)

            result = local_vars.get("result")

            if result is None:
                result = next(reversed(local_vars.values()))

            # Ensure result is a DataFrame
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame({"Result": [result]})

            st.subheader("📊 Result")
            st.dataframe(result)

            if not result.empty:
                insight = summarize_result(question, result)
                st.success("💬 Insight: " + insight)
            else:
                st.warning("⚠️ No data matched your query.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
