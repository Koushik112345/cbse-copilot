import streamlit as st
import pandas as pd
from openai import OpenAI

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# OpenAI client (API key from Streamlit secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Generate pandas code from user question
def generate_pandas_code(question):
    prompt = f"""
You are a data analyst. Given this DataFrame `df` with columns:
student_id, school_code, grade, student_name, gender, academic_year, subject_name,
subject_marks, teacher_name, subject_grade, total_marks, average_score, result_status, stream

User question: {question}

Write a valid Python pandas code using df to answer this question.
- Always assign the final result to a variable named result.
- The result must be a pandas DataFrame (pd.DataFrame).
- Normalize string columns before comparison by applying .str.upper().str.strip()
  Example: df['grade'].str.upper().str.strip() == 'GRADE 10'
- If no matching data is found, assign result = pd.DataFrame() to avoid errors.
- Do not print or explain anything. Only return Python code.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Summarize result using GPT
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

            # Execute code with safety
            local_vars = {}
            exec(code, {"df": df, "pd": pd}, local_vars)

            result = local_vars.get("result")

            if result is None:
                result = next(reversed(local_vars.values()))

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame({"Result": [result]})

            st.subheader("📊 Result")
            st.dataframe(result)

            if result is not None and not result.empty:
                insight = summarize_result(question, result)
                st.success("💬 Insight: " + insight)
            elif result is not None and result.empty:
                st.warning("⚠️ The query ran successfully but returned no results.")
            else:
                st.error("❌ Something went wrong — no result was returned.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
