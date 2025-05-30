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
- Use `student_id`.nunique() to count distinct students.
- Normalize string filters using .str.upper().str.strip()
  Example: df['grade'].str.upper().str.strip() == 'GRADE 10'
- Always assign final output to a DataFrame named `result`.
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

            # GPT-generated result worked
            if result is not None and not result.empty:
                insight = summarize_result(question, result)
                st.success("💬 Insight: " + insight)

            # GPT failed — fallback for specific question
            elif "students appeared" in question.lower() and "2024-2025" in question and "grade 12" in question.lower():
                st.info("🧠 Using fallback logic for student count (Grade 12, 2024-2025).")
                filtered = df[
                    (df['academic_year'].str.upper().str.strip() == '2024-2025') &
                    (df['grade'].str.upper().str.strip() == 'GRADE 12')
                ]
                student_count = filtered['student_id'].nunique()
                result = pd.DataFrame({"students_appeared": [student_count]})
                st.dataframe(result)
                st.success(f"👩‍🎓 {student_count} students appeared for Grade 12 in 2024-2025.")

            else:
                st.warning("⚠️ The query ran but returned no results.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
