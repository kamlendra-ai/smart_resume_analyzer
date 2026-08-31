from flask import Flask, render_template, request, g
from PyPDF2 import PdfReader
import sqlite3
import datetime
import re

# AI / ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

DB_NAME = "bbduanalyzer.db"


# ============================================================
# DATABASE
# ============================================================

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_NAME)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    # Development ke liye fresh table
    db.execute("DROP TABLE IF EXISTS analysis")

    db.execute("""
        CREATE TABLE analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            reg_no TEXT,
            email TEXT,
            target_role TEXT,
            target_name TEXT,
            match_percent INTEGER,
            ai_best_role TEXT,
            ai_best_score REAL,
            analyzed_at TEXT
        )
    """)

    db.commit()


with app.app_context():
    init_db()


# ============================================================
# CAREER PATHS / ROLE SKILLS
# ============================================================

CAREER_PATHS = {

    "general_it": {
        "name": "General IT / Fresher",
        "skills": [
            "C",
            "C++",
            "Java",
            "Python",
            "HTML",
            "CSS",
            "JavaScript",
            "Data Structures",
            "Algorithms",
            "Git",
            "GitHub",
            "Operating Systems",
            "Computer Networks",
            "SQL"
        ]
    },

    "web_dev": {
        "name": "Full Stack Web Developer",
        "skills": [
            "HTML",
            "CSS",
            "JavaScript",
            "React",
            "Node.js",
            "Express",
            "MongoDB",
            "SQL",
            "REST API",
            "Git",
            "GitHub",
            "Responsive Design",
            "Bootstrap"
        ]
    },

    "frontend": {
        "name": "Frontend Developer",
        "skills": [
            "HTML",
            "CSS",
            "JavaScript",
            "React",
            "Bootstrap",
            "Responsive Design",
            "Git",
            "GitHub"
        ]
    },

    "backend": {
        "name": "Backend Developer",
        "skills": [
            "Python",
            "Java",
            "Node.js",
            "Express",
            "Django",
            "Flask",
            "SQL",
            "MongoDB",
            "REST API",
            "Git",
            "GitHub"
        ]
    },

    "python_dev": {
        "name": "Python Developer",
        "skills": [
            "Python",
            "Django",
            "Flask",
            "FastAPI",
            "SQL",
            "REST API",
            "Git",
            "GitHub"
        ]
    },

    "data_analyst": {
        "name": "Data Analyst",
        "skills": [
            "Python",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Statistics",
            "SQL",
            "Excel",
            "Power BI",
            "Tableau"
        ]
    },

    "data_science": {
        "name": "Data Scientist",
        "skills": [
            "Python",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Seaborn",
            "Statistics",
            "Probability",
            "Machine Learning",
            "SQL",
            "Scikit-learn"
        ]
    },

    "ai_ml": {
        "name": "AI / ML Engineer",
        "skills": [
            "Python",
            "Machine Learning",
            "Deep Learning",
            "TensorFlow",
            "PyTorch",
            "Scikit-learn",
            "NumPy",
            "Pandas",
            "Statistics",
            "Git"
        ]
    },

    "android": {
        "name": "Android App Developer",
        "skills": [
            "Java",
            "Kotlin",
            "Android Studio",
            "XML Layouts",
            "REST API",
            "Firebase",
            "Git",
            "GitHub"
        ]
    },

    "devops": {
        "name": "DevOps Engineer",
        "skills": [
            "Linux",
            "Git",
            "GitHub",
            "Docker",
            "Kubernetes",
            "AWS",
            "CI/CD",
            "Jenkins",
            "Python"
        ]
    }
}


# ============================================================
# AI CAREER PROFILES
# ============================================================

career_profiles = {

    "General IT / Fresher": """
        basic programming, c language, c++, java, python basics,
        operating system, dbms, computer networks,
        communication skills, problem solving, fresher,
        data structures, algorithms
    """,

    "Full Stack Web Developer": """
        html, css, javascript, react, nodejs, express, mongodb,
        sql, frontend, backend, full stack, git, github,
        django, flask, rest api, responsive design, bootstrap
    """,

    "Frontend Developer": """
        html, css, javascript, react, bootstrap,
        responsive design, frontend development,
        user interface, ui, git, github
    """,

    "Backend Developer": """
        python, java, nodejs, express, django, flask,
        backend development, sql, mongodb, rest api,
        server, database, git, github
    """,

    "Python Developer": """
        python, django, flask, fastapi, sql,
        rest api, backend, programming, git, github
    """,

    "Data Analyst": """
        python, pandas, numpy, matplotlib,
        statistics, sql, excel, power bi, tableau,
        data analysis, data visualization, eda
    """,

    "Data Scientist": """
        python, pandas, numpy, matplotlib, seaborn,
        statistics, probability, machine learning,
        sql, sklearn, data science, eda
    """,

    "AI / ML Engineer": """
        python, machine learning, deep learning,
        tensorflow, pytorch, scikit learn,
        numpy, pandas, statistics,
        artificial intelligence, neural networks, git
    """,

    "Android App Developer": """
        java, kotlin, android studio, xml layouts,
        firebase, rest api, mobile app development,
        ui design, android
    """,

    "DevOps Engineer": """
        linux, git, github, docker, kubernetes,
        aws, ci cd, jenkins, python,
        cloud, deployment, devops
    """
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = (text or "").lower()

    # Technology aliases
    text = text.replace("node.js", "nodejs")
    text = text.replace("scikit-learn", "scikit learn")
    text = text.replace("ci/cd", "ci cd")

    text = re.sub(
        r"[^a-z0-9\s+#.-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        text += (
            page.extract_text() or ""
        ) + "\n"

    return text


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(resume_text):

    resume_clean = clean_text(
        resume_text
    )

    all_skills = set()

    # All career paths ke skills collect karo
    for path in CAREER_PATHS.values():

        all_skills.update(
            path["skills"]
        )

    found = set()

    for skill in all_skills:

        skill_clean = clean_text(
            skill
        )

        if skill_clean in resume_clean:

            found.add(skill)

    return found


# ============================================================
# AI CAREER FIT
# ============================================================

def compute_career_fit(resume_text):

    resume_clean = clean_text(
        resume_text
    )

    labels = list(
        career_profiles.keys()
    )

    corpus = (
        [resume_clean]
        +
        [
            clean_text(
                career_profiles[k]
            )
            for k in labels
        ]
    )

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        corpus
    )

    resume_vec = tfidf_matrix[0:1]

    profiles_vec = tfidf_matrix[1:]

    sims = cosine_similarity(
        resume_vec,
        profiles_vec
    )[0]

    scores = {}

    for label, sim in zip(
        labels,
        sims
    ):

        scores[label] = round(
            sim * 100,
            1
        )

    best_role = max(
        scores,
        key=scores.get
    )

    best_score = scores[
        best_role
    ]

    return (
        best_role,
        best_score,
        scores
    )


# ============================================================
# MULTIPLE ROLE RECOMMENDATION
# ============================================================

def get_role_recommendations(
    resume_text,
    top_n=5,
    exclude_role=None
):

    """
    Resume ke skills ke basis par
    multiple career roles recommend karta hai.

    exclude_role:
    User ne jo target role select kiya hai,
    usko recommendations me repeat nahi karega.
    """

    found_skills = extract_skills(
        resume_text
    )

    recommendations = []

    for role_id, role_data in CAREER_PATHS.items():

        # ----------------------------------------------------
        # Selected target role ko skip karo
        # ----------------------------------------------------

        if role_id == exclude_role:
            continue

        role_name = role_data[
            "name"
        ]

        required_skills = set(
            role_data[
                "skills"
            ]
        )

        matched_skills = sorted(
            found_skills
            &
            required_skills
        )

        missing_skills = sorted(
            required_skills
            -
            found_skills
        )

        # ----------------------------------------------------
        # Match Percentage
        # ----------------------------------------------------

        if required_skills:

            skill_score = round(
                (
                    len(matched_skills)
                    /
                    len(required_skills)
                )
                * 100
            )

        else:

            skill_score = 0

        recommendations.append({

            "role_id": role_id,

            "role": role_name,

            "score": skill_score,

            "matched_skills": matched_skills,

            "missing_skills": missing_skills,

            "total_skills": len(
                required_skills
            ),

            "matched_count": len(
                matched_skills
            )

        })

    # --------------------------------------------------------
    # Highest matching role first
    # --------------------------------------------------------

    recommendations.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return recommendations[:top_n]


# ============================================================
# LEARNING ROADMAP
# ============================================================

def generate_learning_roadmap(
    missing_skills,
    target_name
):

    if not missing_skills:

        return [

            f"Your resume already matches most core skills for {target_name}.",

            "Focus on building 2–3 strong projects in this domain.",

            "Contribute to GitHub, participate in hackathons and improve problem solving (DSA)."

        ]

    n = len(
        missing_skills
    )

    part1 = missing_skills[
        :max(1, n // 3)
    ]

    part2 = missing_skills[
        max(1, n // 3):
        max(2, 2 * n // 3)
    ]

    part3 = missing_skills[
        max(2, 2 * n // 3):
    ]

    roadmap = []

    if part1:

        roadmap.append(

            "Step 1: Basics (1–2 months) → Learn: "
            +
            ", ".join(part1)

        )

    if part2:

        roadmap.append(

            "Step 2: Intermediate (1–2 months) → Practice: "
            +
            ", ".join(part2)

        )

    if part3:

        roadmap.append(

            "Step 3: Advanced / Projects (1–2 months) → Work on: "
            +
            ", ".join(part3)

        )

    roadmap.append(

        "Final Step: Build 2–3 good projects, "
        "upload on GitHub, and update these skills clearly "
        "in your resume."

    )

    return roadmap


# ============================================================
# HOME / RESUME ANALYSIS
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    if request.method == "POST":

        # ----------------------------------------------------
        # STUDENT DETAILS
        # ----------------------------------------------------

        student_name = request.form.get(
            "student_name"
        )

        reg_no = request.form.get(
            "reg_no"
        )

        email = request.form.get(
            "email"
        )

        target_role = request.form.get(
            "target_role",
            "general_it"
        )

        # ----------------------------------------------------
        # RESUME FILE
        # ----------------------------------------------------

        resume_file = request.files.get(
            "resume"
        )

        if not resume_file:

            return render_template(

                "index.html",

                career_paths=CAREER_PATHS,

                error="Please upload a PDF resume."

            )

        # ----------------------------------------------------
        # PDF TEXT
        # ----------------------------------------------------

        try:

            resume_text = extract_text_from_pdf(
                resume_file
            )

        except Exception as e:

            return render_template(

                "index.html",

                career_paths=CAREER_PATHS,

                error=(
                    f"Unable to read PDF: {str(e)}"
                )

            )

        if not resume_text.strip():

            return render_template(

                "index.html",

                career_paths=CAREER_PATHS,

                error=(
                    "Could not extract text from this PDF. "
                    "Please upload a text-based PDF."
                )

            )

        # ----------------------------------------------------
        # VALIDATE TARGET ROLE
        # ----------------------------------------------------

        if target_role not in CAREER_PATHS:

            target_role = "general_it"

        # ----------------------------------------------------
        # EXTRACT SKILLS
        # ----------------------------------------------------

        found_skills = extract_skills(
            resume_text
        )

        # ----------------------------------------------------
        # SELECTED TARGET ROLE
        # ----------------------------------------------------

        target_name = CAREER_PATHS[
            target_role
        ]["name"]

        ideal = set(
            CAREER_PATHS[
                target_role
            ]["skills"]
        )

        matched = sorted(
            list(
                found_skills
                &
                ideal
            )
        )

        missing = sorted(
            list(
                ideal
                -
                found_skills
            )
        )

        # ----------------------------------------------------
        # TARGET ROLE MATCH %
        # ----------------------------------------------------

        if ideal:

            match_percent = round(

                (
                    len(matched)
                    /
                    len(ideal)
                )
                * 100

            )

        else:

            match_percent = 0

        # ----------------------------------------------------
        # LEARNING ROADMAP
        # ----------------------------------------------------

        roadmap = generate_learning_roadmap(

            missing,

            target_name

        )

        # ----------------------------------------------------
        # AI CAREER SCAN
        # ----------------------------------------------------

        (
            ai_best_role,
            ai_best_score,
            ai_scores
        ) = compute_career_fit(
            resume_text
        )

        # ----------------------------------------------------
        # MULTIPLE ROLE RECOMMENDATION
        # ----------------------------------------------------

        role_recommendations = (
            get_role_recommendations(

                resume_text,

                top_n=5,

                # IMPORTANT:
                # Selected role repeat nahi hoga
                exclude_role=target_role

            )
        )

        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        db = get_db()

        db.execute(
            """
            INSERT INTO analysis
            (
                student_name,
                reg_no,
                email,
                target_role,
                target_name,
                match_percent,
                ai_best_role,
                ai_best_score,
                analyzed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (

                student_name,

                reg_no,

                email,

                target_role,

                target_name,

                match_percent,

                ai_best_role,

                ai_best_score,

                datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                )

            )
        )

        db.commit()

        # ----------------------------------------------------
        # RESULT PAGE
        # ----------------------------------------------------

        return render_template(

            "result.html",

            student_name=student_name,

            reg_no=reg_no,

            email=email,

            # Selected target role
            target_role=target_role,

            target_name=target_name,

            # Target role skills
            matched_skills=matched,

            missing_skills=missing,

            match_percent=match_percent,

            # AI
            ai_best_role=ai_best_role,

            ai_best_score=ai_best_score,

            ai_scores=ai_scores,

            # Roadmap
            roadmap=roadmap,

            # Multiple role recommendation
            role_recommendations=role_recommendations

        )

    # --------------------------------------------------------
    # GET REQUEST
    # --------------------------------------------------------

    return render_template(

        "index.html",

        career_paths=CAREER_PATHS,

        error=None

    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    rows = get_db().execute(

        """
        SELECT *
        FROM analysis
        ORDER BY id DESC
        """

    ).fetchall()

    return render_template(

        "history.html",

        rows=rows

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )