import streamlit as st
from PyPDF2 import PdfReader
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Student Skill Gap Analyzer",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------

st.title("🎓 AI Student Skill Gap Analyzer")

st.caption(
    "AI-powered resume analysis, skill gap detection & personalized learning"
)

st.divider()

# -----------------------------
# RAG Search Function
# -----------------------------

def rag_search(question):

    with open(
        "learning_material.txt",
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    if not chunks:
        return "No learning material found."

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        chunks + [question]
    )

    scores = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )[0]

    best_index = scores.argmax()

    return chunks[best_index]


# -----------------------------
# Upload Resume
# -----------------------------

st.header("📄 Upload Your Resume")

uploaded_file = st.file_uploader(
    "Choose your resume PDF",
    type=["pdf"]
)

resume_text = ""

if uploaded_file:

    st.success(
        "Resume uploaded successfully! ✅"
    )

    # -----------------------------
    # Extract Resume Text
    # -----------------------------

    pdf_reader = PdfReader(
        uploaded_file
    )

    for page in pdf_reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text

    st.subheader(
        "📃 Extracted Resume Text"
    )

    st.text_area(
        "Resume Content",
        resume_text,
        height=250
    )

    # -----------------------------
    # AI Skill Extraction
    # -----------------------------

    st.subheader(
        "🤖 AI Skill Extraction"
    )

    if st.button(
        "🔍 Extract My Skills"
    ):

        with st.spinner(
            "AI is analyzing your resume..."
        ):

            prompt = f"""
Analyze the following resume.

Extract only the technical and professional
skills mentioned in the resume.

Return the skills as a simple bullet-point list.

Resume:
{resume_text}
"""

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            skills = response[
                "message"
            ]["content"]

            st.session_state[
                "skills"
            ] = skills

        st.success(
            "Skills extracted successfully! ✅"
        )

    # -----------------------------
    # Show Skills
    # -----------------------------

    if "skills" in st.session_state:

        st.subheader(
            "💡 Your Skills"
        )

        st.write(
            st.session_state["skills"]
        )

        st.divider()

        # -----------------------------
        # Target Job
        # -----------------------------

        st.subheader(
            "🎯 Select Your Target Job"
        )

        target_job = st.selectbox(
            "Choose your target job:",
            [
                "Full Stack Developer",
                "Python Developer",
                "Java Developer",
                "Data Analyst",
                "AI/ML Engineer"
            ]
        )

        st.write(
            "Selected Job:",
            target_job
        )

        # -----------------------------
        # Skill Match Percentage
        # -----------------------------

        job_skills = {

            "Full Stack Developer": [
                "Python",
                "JavaScript",
                "React",
                "Node.js",
                "Express",
                "MongoDB",
                "SQL",
                "Git",
                "HTML",
                "CSS"
            ],

            "Python Developer": [
                "Python",
                "SQL",
                "Git",
                "Django",
                "Flask",
                "REST API"
            ],

            "Java Developer": [
                "Java",
                "SQL",
                "Spring Boot",
                "REST API",
                "Git",
                "Maven"
            ],

            "Data Analyst": [
                "Python",
                "SQL",
                "Excel",
                "Power BI",
                "Statistics",
                "Pandas"
            ],

            "AI/ML Engineer": [
                "Python",
                "NumPy",
                "Pandas",
                "scikit-learn",
                "TensorFlow",
                "SQL",
                "Machine Learning"
            ]
        }

        required_skills = job_skills[
            target_job
        ]

        student_skills_text = (
            st.session_state[
                "skills"
            ].lower()
        )

        matched_skills = [
            skill
            for skill in required_skills
            if skill.lower()
            in student_skills_text
        ]

        match_percentage = int(
            len(matched_skills)
            / len(required_skills)
            * 100
        )

        st.subheader(
            "📊 Skill Match Percentage"
        )

        st.progress(
            match_percentage / 100
        )

        st.metric(
            "Your Skill Match",
            f"{match_percentage}%"
        )

        # -----------------------------
        # Matched Skills
        # -----------------------------

        if matched_skills:

            st.write(
                "### ✅ Matched Skills"
            )

            for skill in matched_skills:

                st.write(
                    f"✔️ {skill}"
                )

        st.divider()

        # -----------------------------
        # Skill Gap Analysis
        # -----------------------------

        if st.button(
            "🔍 Analyze Skill Gap"
        ):

            with st.spinner(
                "Analyzing your skill gap..."
            ):

                prompt = f"""
Student Skills:
{st.session_state["skills"]}

Target Job:
{target_job}

Find the important missing skills.

Give the result in this format:

Missing Skills:
- skill 1
- skill 2
- skill 3

Learning Roadmap:
1. skill - what to learn
2. skill - what to learn
3. skill - what to learn
"""

                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                gap_result = response[
                    "message"
                ]["content"]

                st.session_state[
                    "gap_result"
                ] = gap_result

            st.success(
                "Skill Gap Analysis Completed! ✅"
            )

        # -----------------------------
        # Display Skill Gap
        # -----------------------------

        if "gap_result" in st.session_state:

            st.subheader(
                "📊 Your Skill Gap"
            )

            st.write(
                st.session_state[
                    "gap_result"
                ]
            )

            st.divider()

            # -----------------------------
            # Personalized Resources
            # -----------------------------

            st.subheader(
                "📚 Personalized Learning Resources"
            )

            resources = {

                "JavaScript":
                    "Learn JavaScript basics, ES6, functions and async/await.",

                "React":
                    "Learn React components, props, state and hooks.",

                "Angular":
                    "Learn Angular components, services and routing.",

                "SQL":
                    "Learn SQL queries, joins, tables and databases.",

                "MongoDB":
                    "Learn MongoDB collections, documents and CRUD operations.",

                "Git":
                    "Learn Git commands, GitHub and version control.",

                "Docker":
                    "Learn Docker containers, Dockerfiles and basic Docker commands.",

                "Kubernetes":
                    "Learn Kubernetes pods, deployments and services.",

                "AWS":
                    "Learn AWS fundamentals, EC2, S3 and RDS.",

                "Azure":
                    "Learn Azure fundamentals, storage and cloud services.",

                "Python":
                    "Learn Python fundamentals, functions, OOP and libraries.",

                "Java":
                    "Learn Java fundamentals, OOP and collections.",

                "Spring Boot":
                    "Learn Spring Boot, REST APIs and backend development.",

                "Excel":
                    "Learn Excel formulas, charts and pivot tables.",

                "Power BI":
                    "Learn Power BI dashboards and data visualization.",

                "Pandas":
                    "Learn Pandas DataFrames, data cleaning and analysis.",

                "Machine Learning":
                    "Learn machine learning algorithms and model evaluation."
            }

            found_resources = False

            for skill, resource in resources.items():

                if skill.lower() in st.session_state[
                    "gap_result"
                ].lower():

                    found_resources = True

                    st.markdown(
                        f"### 🎯 {skill}"
                    )

                    st.write(
                        resource
                    )

            if not found_resources:

                st.info(
                    "No matching resources found."
                )

            st.divider()

            # -----------------------------
            # RAG Learning Assistant
            # -----------------------------

            st.subheader(
                "🤖 RAG Learning Assistant"
            )

            st.write(
                "Ask questions about the skills "
                "you need to learn."
            )

            question = st.text_input(
                "💬 Ask your learning question:"
            )

            if st.button(
                "🚀 Ask RAG Assistant"
            ):

                if question.strip():

                    with st.spinner(
                        "Searching learning material..."
                    ):

                        context = rag_search(
                            question
                        )

                        rag_prompt = f"""
You are an AI learning assistant.

Use the following learning material
to answer the student's question.

Learning Material:
{context}

Student Question:
{question}

Give a simple beginner-friendly answer.
"""

                        response = ollama.chat(
                            model="llama3.2",
                            messages=[
                                {
                                    "role": "user",
                                    "content": rag_prompt
                                }
                            ]
                        )

                        answer = response[
                            "message"
                        ]["content"]

                    st.success(
                        "RAG Answer Generated! ✅"
                    )

                    st.subheader(
                        "📖 Answer"
                    )

                    st.write(
                        answer
                    )

                else:

                    st.warning(
                        "Please enter a question."
                    )

else:

    st.info(
        "👆 Please upload your resume PDF first."
    )