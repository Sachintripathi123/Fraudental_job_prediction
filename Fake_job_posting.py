import streamlit as st
import pandas as pd
import re
import nltk

from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🕵️ Fake Job Posting Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect whether a job posting is potentially fraudulent using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# NLTK
# ============================================================

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


stemmer = PorterStemmer()


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

dataset_path = BASE_DIR / "datasets" / "fake_job_postings.csv"


# ============================================================
# CHECK DATASET
# ============================================================

if not dataset_path.exists():

    st.error(
        f"Dataset not found!\n\n"
        f"Expected location:\n{dataset_path}"
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(dataset_path)


df = load_data()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

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

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


# ============================================================
# STEMMING
# ============================================================

def stemming(text):

    words = text.split()

    stemmed_words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(stemmed_words)


# ============================================================
# PREPROCESS DATASET
# ============================================================

@st.cache_data
def prepare_data(data):

    data = data.copy()

    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    for col in text_columns:

        data[col] = data[col].fillna("")

    data["text"] = (
        data["title"] + " " +
        data["company_profile"] + " " +
        data["description"] + " " +
        data["requirements"] + " " +
        data["benefits"]
    )

    data["text"] = data["text"].str.lower()

    data["text"] = data["text"].apply(clean_text)

    data["text"] = data["text"].apply(remove_stopwords)

    data["text"] = data["text"].apply(stemming)

    return data


df = prepare_data(df)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    tfidf = TfidfVectorizer(
        max_features=5000
    )

    X = tfidf.fit_transform(
        data["text"]
    )

    y = data["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    param_grid = {
        "C": [
            0.01,
            0.1,
            1,
            10,
            100
        ]
    }

    grid = GridSearchCV(
        LinearSVC(),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    return tfidf, grid


# ============================================================
# MODEL
# ============================================================

with st.spinner("Training machine learning model..."):

    tfidf, model = train_model(df)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Project Information")

    st.write(
        """
        **Model:** Tuned Support Vector Machine

        **Feature Extraction:** TF-IDF

        **Dataset:** Fake Job Postings

        **Task:** Binary Classification
        """
    )

    st.divider()

    st.write(
        f"📄 Dataset Records: **{len(df)}**"
    )

    st.write(
        f"🔤 TF-IDF Features: **{len(tfidf.vocabulary_)}**"
    )


# ============================================================
# JOB POST INPUT
# ============================================================

st.subheader("📋 Enter Job Posting")

job_title = st.text_input(
    "Job Title",
    placeholder="e.g. Data Scientist"
)


company = st.text_input(
    "Company Name",
    placeholder="e.g. ABC Technologies"
)


job_description = st.text_area(
    "Job Description",
    height=180,
    placeholder=(
        "Paste the complete job description here..."
    )
)


requirements = st.text_area(
    "Requirements",
    height=120,
    placeholder=(
        "Enter required skills, qualifications, experience..."
    )
)


benefits = st.text_area(
    "Benefits",
    height=100,
    placeholder=(
        "Enter salary, benefits, perks, etc..."
    )
)


# ============================================================
# SAMPLE JOB BUTTON
# ============================================================

st.subheader("💡 Quick Test")

if st.button("Load Sample Fraudulent Job"):

    job_title = "Work From Home Data Entry"

    company = "Global Online Company"

    job_description = """
    Congratulations! You have been selected without applying.
    Earn ₹2,50,000 per month.
    No experience required.
    Immediate joining.
    """

    requirements = """
    No skills required.
    No experience required.
    """

    benefits = """
    High salary.
    Work from home.
    """

    st.info(
        "Sample job loaded. Click 'Check Job Posting' below."
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Check Job Posting",
    use_container_width=True
):

    # Check input
    if not job_title and not job_description:

        st.warning(
            "Please enter at least a Job Title or Job Description."
        )

    else:

        # Combine user input
        new_job_post = (
            job_title + " " +
            company + " " +
            job_description + " " +
            requirements + " " +
            benefits
        )

        # Preprocess
        processed_job = new_job_post.lower()

        processed_job = clean_text(
            processed_job
        )

        processed_job = remove_stopwords(
            processed_job
        )

        processed_job = stemming(
            processed_job
        )

        # TF-IDF
        job_vector = tfidf.transform(
            [processed_job]
        )

        # Prediction
        prediction = model.predict(
            job_vector
        )

        # Decision score
        score = model.decision_function(
            job_vector
        )[0]


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader("🔎 Prediction Result")


        if prediction[0] == 1:

            st.error(
                "🚨 FRAUDULENT JOB POSTING"
            )

            st.markdown(
                """
                <div class="result-box">
                ⚠️ This job posting has characteristics
                associated with fraudulent job postings.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.warning(
                "Avoid sharing sensitive information or "
                "making payments until the employer is verified."
            )

        else:

            st.success(
                "✅ NOT DETECTED AS FRAUDULENT"
            )

            st.markdown(
                """
                <div class="result-box">
                ✅ The model did not classify this posting
                as fraudulent.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "This prediction is not a guarantee that the "
                "job or employer is legitimate. Verify the employer independently."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake Job Posting Detection | Machine Learning Project"
)