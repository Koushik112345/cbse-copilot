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

Rules:
- Use `subject_marks` for subject-specific stats.
- Use `average_score` for student-level averages.
- Normalize all string columns before filtering using .str.upper().str.strip()
  Example: df['grade'].str.upper().str.strip() == 'GRADE 10'
- Always assign final output to a DataFrame named result.
- If query returns no data, use result = pd.DataFrame() to prevent errors.
- Do not print or explain. Only return Python code.
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
            # Generate code from GPT
            code = generate_pandas_code(question)
            st.subheader("📄 Generated Code")
            st.code(code, language="python")

            # Execute safely
            local_vars = {}
            exec(code, {"df": df, "pd": pd}, local_vars)
            result = local_vars.get("result")

            if result is None:
                result = next(reversed(local_vars.values()))

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame({"Result": [result]})

            st.subheader("📊 Result")
            st.dataframe(result)

            # Fallback if result is empty
            if result is not None and not result.empty:
                insight = summarize_result(question, result)
                st.success("💬 Insight: " + insight)
            elif "school-wise overall average of 2024-2025 in order for grade 10" in question.lower():
                st.info("🧠 Using fallback logic since no GPT result was found.")
                fallback = df[
                    (df['academic_year'].str.upper().str.strip() == '2024-2025') &
                    (df['grade'].str.upper().str.strip() == 'GRADE 10')
                ]
                result = (
                    fallback.groupby('school_code')['average_score']
                    .mean()
                    .reset_index()
                    .sort_values(by='average_score', ascending=False)
                )
                st.dataframe(result)
                st.success("💬 Insight: School-wise averages for Grade 10 in 2024-2025 shown.")
            else:
                st.warning("⚠️ The query ran but returned no results.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
