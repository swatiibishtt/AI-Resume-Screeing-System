
from pypdf import PdfReader

def extract_resume_text(pdf_file):
    """Extract text from an uploaded PDF resume."""
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


import re

SKILLS = [
    "Python",
    "Java",
    "SQL",
    "MySQL",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Django",
    "Flask",
    "Spring Boot",
    "Streamlit",
    "Scikit-learn",
    "TensorFlow",
    "Pandas",
    "NumPy",
    "Git",
    "REST API",
    "Data Structures",
    "OOP"
]


def extract_skills(text):
    found_skills = []

    text = text.lower()

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


def analyze_skills(resume_text, job_description):

    resume_skills = extract_skills(resume_text)
    required_skills = extract_skills(job_description)

    matched_skills = [
        skill for skill in required_skills
        if skill in resume_skills
    ]

    missing_skills = [
        skill for skill in required_skills
        if skill not in resume_skills
    ]

    return matched_skills, missing_skills


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(resume_text, job_description):

    documents = [job_description, resume_text]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    score = round(similarity * 100, 2)

    return score


import streamlit as st
import pandas as pd


# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Application heading
st.title("📄 AI Resume Screening System")

st.write(
    "Upload candidate resumes and compare them "
    "against a job description using NLP."
)

st.divider()

# Job description input
st.subheader("1. Enter Job Description")

job_description = st.text_area(
    "Paste the job description here:",
    height=200,
    placeholder="Example: We are looking for a Python "
                "developer with knowledge of SQL, Java, "
                "Machine Learning and REST API."
)

# Resume upload
st.subheader("2. Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

# Analyze button
if st.button("Analyze Resumes", type="primary"):

    if not job_description.strip():
        st.error("Please enter a job description.")

    elif not uploaded_files:
        st.error("Please upload at least one resume.")

    else:

        results = []

        with st.spinner("Analyzing candidate resumes..."):

            for pdf_file in uploaded_files:

                try:
                    resume_text = extract_resume_text(pdf_file)

                    if not resume_text.strip():
                        st.warning(
                            f"No readable text found in "
                            f"{pdf_file.name}"
                        )
                        continue

                    matched, missing = analyze_skills(
                        resume_text,
                        job_description
                    )

                    score = calculate_similarity(
                        resume_text,
                        job_description
                    )

                    results.append({
                        "Candidate": pdf_file.name,
                        "Similarity Score (%)": score,
                        "Matched Skills": matched,
                        "Missing Skills": missing
                    })

                except Exception as error:
                    st.warning(
                        f"Could not process {pdf_file.name}: "
                        f"{error}"
                    )

        # Rank candidates
        results.sort(
            key=lambda x: x["Similarity Score (%)"],
            reverse=True
        )

        if results:

            st.success("Resume analysis completed!")

            st.divider()

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Resumes",
                len(uploaded_files)
            )

            col2.metric(
                "Successfully Analyzed",
                len(results)
            )

            col3.metric(
                "Average Similarity",
                f"{sum(r['Similarity Score (%)'] for r in results) / len(results):.2f}%"
            )

            # Candidate ranking
            st.subheader("3. Candidate Ranking")

            ranking_data = []

            for result in results:

                ranking_data.append({
                    "Candidate": result["Candidate"],
                    "Similarity Score (%)":
                        result["Similarity Score (%)"],
                    "Matched Skills":
                        ", ".join(result["Matched Skills"]),
                    "Missing Skills":
                        ", ".join(result["Missing Skills"])
                })

            df = pd.DataFrame(ranking_data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # Individual candidate details
            st.subheader("4. Candidate Details")

            for index, result in enumerate(results):

                with st.expander(
                    f"Rank {index + 1}: "
                    f"{result['Candidate']} - "
                    f"{result['Similarity Score (%)']}%"
                ):

                    st.write("**Matched Skills**")

                    if result["Matched Skills"]:
                        st.success(
                            ", ".join(result["Matched Skills"])
                        )
                    else:
                        st.info("No matching skills detected.")

                    st.write("**Missing Skills**")

                    if result["Missing Skills"]:
                        st.warning(
                            ", ".join(result["Missing Skills"])
                        )
                    else:
                        st.success("No missing listed skills.")

            # Download results
            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Screening Results",
                data=csv_data,
                file_name="resume_screening_results.csv",
                mime="text/csv"
            )

        else:
            st.error(
                "No resumes could be analyzed. "
                "Check whether the PDFs contain readable text."
            )

st.divider()

st.caption(
    "AI Resume Screening System | Python, NLP, "
    "Scikit-learn and Streamlit"
)

st.caption(
    "Screening scores are informational aids only. "
    "They should not be used as automatic hiring decisions."
)
