
import os

import streamlit as st

from crew_setup import run_admission_crew


st.set_page_config(
    page_title="AI University Admission Advisor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 AI University Admission Advisor")

st.write(
    "A simple multi-agent admission system "
    "powered by CrewAI, Groq and Streamlit."
)

st.info(
    "Demo only: This app uses fictional universities "
    "and sample admission requirements."
)

st.subheader("Enter Student Details")

with st.form("student_form"):
    qualification = st.selectbox(
        "Previous Qualification",
        [
            "FSc Pre-Engineering",
            "ICS",
            "FSc Pre-Medical",
            "A Levels",
            "Other",
        ],
    )

    percentage = st.number_input(
        "Marks Percentage",
        min_value=0.0,
        max_value=100.0,
        value=82.0,
        step=0.5,
    )

    interest = st.selectbox(
        "Interested Field",
        [
            "Computer Science / AI",
            "Software Engineering",
            "Data Science",
            "Engineering",
            "Business",
            "Other",
        ],
    )

    mathematics = st.selectbox(
        "Have you studied Mathematics?",
        ["Yes", "No"],
    )

    submit = st.form_submit_button(
        "Check Admission Eligibility",
        type="primary",
        use_container_width=True,
    )


if submit:
    student = {
        "qualification": qualification,
        "percentage": percentage,
        "interest": interest,
        "mathematics": mathematics,
    }

    # Read Groq API key from Streamlit Secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        st.error(
            "Groq API key not found. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )
        st.stop()

    os.environ["GROQ_API_KEY"] = api_key

    try:
        with st.spinner(
            "Running the three admission agents..."
        ):
            report = run_admission_crew(student)

        st.success("Admission analysis completed!")

        st.subheader("Your Admission Report")

        st.markdown(report)

        st.download_button(
            label="Download Admission Report",
            data=report,
            file_name="admission_report.md",
            mime="text/markdown",
        )

    except Exception as error:
        st.error("The admission analysis could not finish.")
        st.exception(error)


st.divider()

st.caption(
    "Built using Streamlit + CrewAI + Groq | "
    "Demo university data only."
)
