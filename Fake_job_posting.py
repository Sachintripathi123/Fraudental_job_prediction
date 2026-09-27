import streamlit as st
import pandas as pd
import re
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🕵️ Fake Job Posting Detector")

st.write(
    "Enter a job posting below to check whether "
    "it may be fraudulent."
)

st.divider()


# ============================================================
# NLTK STOPWORDS
# ============================================================

try:
    stop_words = set(
        stopwords.words("english")
    )

except LookupError:

    nltk.download(
        "stopwords",
        quiet=True
    )

    stop_words = set(
        stopwords.words("english")
    )


stemmer = PorterStemmer()


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

dataset_path = (
    BASE_DIR
    / "datasets"
    / "fake_job_postings.csv"
)


# ============================================================
# CHECK DATASET
# ============================================================

if not dataset_path.exists():

    st.error(
        "Dataset not found.\n\n"
        f"Expected location:\n{dataset_path}"
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        dataset_path
    )


df = load_dataset()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# REMOVE STOPWORDS
# ============================================================

def remove_stopwords(text):

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================================
# STEMMING
# ============================================================

def stemming(text):

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


# ============================================================
# PREPROCESS DATASET
# ============================================================

@st.cache_data
def prepare_dataset(data):

    data = data.copy()

    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    for column in text_columns:

        data[column] = data[column].fillna("")

    # Combine all important text fields
    data["text"] = (
        data["title"] + " " +
        data["company_profile"] + " " +
        data["description"] + " " +
        data["requirements"] + " " +
        data["benefits"]
    )

    # Text preprocessing
    data["text"] = data["text"].apply(
        clean_text
    )

    data["text"] = data["text"].apply(
        remove_stopwords
    )

    data["text"] = data["text"].apply(
        stemming
    )

    return data


df = prepare_dataset(df)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000
    )

    X = tfidf.fit_transform(
        data["text"]
    )

    y = data["fraudulent"]

    # Lightweight SVM
    model = LinearSVC(
        C=1.0
    )

    model.fit(
        X,
        y
    )

    return tfidf, model


# ============================================================
# TRAINING
# ============================================================

with st.spinner(
    "Loading machine learning model..."
):

    tfidf, model = train_model(df)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Project Information")

    st.write(
        "### Model"
    )

    st.write(
        "Linear Support Vector Machine"
    )

    st.write(
        "### Feature Extraction"
    )

    st.write(
        "TF-IDF"
    )

    st.write(
        "### Dataset"
    )

    st.write(
        "Fake Job Postings"
    )

    st.divider()

    st.write(
        f"Dataset records: **{len(df)}**"
    )

    st.write(
        f"TF-IDF features: **{len(tfidf.vocabulary_)}**"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("📋 Job Posting Details")


job_title = st.text_input(
    "Job Title",
    placeholder="Example: Data Scientist"
)


company_name = st.text_input(
    "Company Name",
    placeholder="Example: ABC Technologies"
)


job_description = st.text_area(
    "Job Description",
    height=180,
    placeholder="Paste the job description here..."
)


requirements = st.text_area(
    "Requirements",
    height=120,
    placeholder="Enter skills, qualifications and experience..."
)


benefits = st.text_area(
    "Benefits",
    height=100,
    placeholder="Enter salary, benefits and perks..."
)


# ============================================================
# SAMPLE JOB
# ============================================================

st.subheader("💡 Quick Test")


if st.button(
    "Load Sample Job"
):

    st.session_state.job_title = (
        "Junior Data Analyst"
    )

    st.session_state.company_name = (
        "ABC Technologies Pvt Ltd"
    )

    st.session_state.job_description = """
    We are looking for a Junior Data Analyst
    to join our analytics team.

    The candidate will work with business data,
    prepare reports, create dashboards and
    support data-driven decision making.

    The position follows a structured interview
    and selection process.
    """

    st.session_state.requirements = """
    Bachelor's degree in Computer Science,
    Information Technology, Data Science,
    Statistics or a related field.

    Basic knowledge of Python and SQL.

    Knowledge of Excel and data analysis.

    Good analytical and communication skills.

    Freshers can apply.
    """

    st.session_state.benefits = """
    Competitive salary based on skills and experience.

    Paid training and learning opportunities.

    Professional development opportunities.

    Health insurance according to company policy.
    """

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Check Job Posting",
    use_container_width=True
):

    # Check input
    if not job_title and not job_description:

        st.warning(
            "Please enter at least a Job Title "
            "or Job Description."
        )

    else:

        # Combine user input
        new_job = (
            job_title + " " +
            company_name + " " +
            job_description + " " +
            requirements + " " +
            benefits
        )

        # Preprocess input
        new_job = clean_text(
            new_job
        )

        new_job = remove_stopwords(
            new_job
        )

        new_job = stemming(
            new_job
        )

        # Convert to TF-IDF
        job_vector = tfidf.transform(
            [new_job]
        )

        # Prediction
        prediction = model.predict(
            job_vector
        )

        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🔎 Prediction Result"
        )


        if prediction[0] == 1:

            st.error(
                "🚨 POTENTIALLY FRAUDULENT JOB POSTING"
            )

            st.warning(
                "The model detected patterns associated "
                "with fraudulent job postings."
            )

            st.info(
                "Do not send money or sensitive personal "
                "information until the employer is independently verified."
            )

        else:

            st.success(
                "✅ NOT DETECTED AS FRAUDULENT"
            )

            st.info(
                "The model did not classify this posting "
                "as fraudulent. However, this does not guarantee "
                "that the employer is legitimate."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake Job Posting Detection | "
    "Machine Learning + TF-IDF + Linear SVM"
)