import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import validators

# Sample Dataset (expanded with remote and benefits info)
data = {
    "text": [
        "Software engineer needed with experience in Python and Django. Apply on company website. Remote work available. Health insurance provided.",
        "Work from home, earn $500 daily. No experience required. Just sign up! No benefits.",
        "Looking for marketing intern at a reputed firm. Must know SEO tools. On-site position. 401k benefits.",
        "Data analyst position. Requires knowledge of SQL and Tableau. Hybrid work. Insurance and paid leave.",
        "Earn big! No skills needed. Start today and become rich fast! Work from home.",
        "HR role with remote flexibility. Good communication skills needed. Comprehensive health insurance."
    ],
    "label": [0, 1, 0, 0, 1, 0],  # 0 = Genuine, 1 = Fake
    "remote": [1, 1, 0, 1, 1, 1],   # 1 = Remote/Hybrid, 0 = On-site
    "benefits": [1, 0, 1, 1, 0, 1]   # 1 = Benefits mentioned, 0 = None
}
df = pd.DataFrame(data)

# Train model (using only text for now)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression())
])
model.fit(df["text"], df["label"])

# Headers to mimic browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Validate URL format
def is_valid_url(url):
    return validators.url(url) is True

# Check if the URL is a valid and job-related page
def is_valid_job_url(url):
    try:
        res = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        if res.status_code != 200:
            return False, "❌ URL is not reachable", None
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True).lower()
        keywords = [
            "job", "careers", "apply", "recruit", "position", "hiring", 
            "description", "responsibilities", "qualifications"
        ]
        if any(word in text for word in keywords):
            return True, "✅ URL is reachable and job-related", soup
        else:
            return False, "⚠️ URL reachable but doesn't look like a job page", soup
    except requests.RequestException as e:
        return False, f"❌ Error fetching URL: {str(e)}", None

# Extract job description text and check for remote/benefits
def extract_job_text(soup):
    job_text_parts = []

    selectors = [
        {"tag": "div", "class": "jobDescriptionText"},  # Indeed
        {"tag": "div", "class": "description"},         # LinkedIn/Glassdoor
        {"tag": "section", "class": "job-desc"},        # Generic
        {"tag": "div", "id": "jobDetails"},             # Naukri-style
        {"tag": "meta", "attr": "description"},         # Meta fallback
    ]

    for sel in selectors:
        if "attr" in sel:
            tag = soup.find(sel["tag"], attrs={"name": sel["attr"]})
            if tag and tag.get("content"):
                job_text_parts.append(tag["content"])
        else:
            tag = soup.find(sel["tag"], class_=sel.get("class"), id=sel.get("id"))
            if tag:
                job_text_parts.append(tag.get_text(strip=True))

    # Fallback to full visible text
    if not job_text_parts:
        all_text = soup.find_all(["p", "div", "span"])
        job_text_parts = [el.get_text(strip=True) for el in all_text]

    final_text = " ".join(job_text_parts).strip()
    
    # Check for remote/hybrid and benefits
    remote_keywords = ["remote", "work from home", "hybrid", "telecommute"]
    benefits_keywords = ["insurance", "healthcare", "401k", "benefits", "paid leave", "pension"]
    is_remote = any(keyword in final_text.lower() for keyword in remote_keywords)
    has_benefits = any(keyword in final_text.lower() for keyword in benefits_keywords)
    
    return final_text[:3000], is_remote, has_benefits

# Streamlit App with Multi-Page Flow
st.set_page_config(page_title="Fake Job Detector", layout="centered")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "user_details"
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

# User Details Page
if st.session_state.page == "user_details":
    st.title("👤 User Details")
    st.markdown("Please enter your details to proceed with the job URL analysis.")
    
    with st.form("user_form"):
        name = st.text_input("Name", placeholder="Enter your full name")
        email = st.text_input("Email", placeholder="Enter your email")
        submitted = st.form_submit_button("Proceed")
        
        if submitted:
            if not name or not email:
                st.error("Please fill in all fields.")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Please enter a valid email address.")
            else:
                st.session_state.user_data = {"name": name, "email": email}
                st.session_state.page = "job_analysis"
                st.rerun()

# Job Analysis Page
elif st.session_state.page == "job_analysis":
    st.title("🕵️‍♂️ Fake Job Detector")
    st.markdown(f"Welcome, {st.session_state.user_data['name']}! Enter a job URL to analyze.")
    st.markdown("""
    This tool checks if a job URL is:
    1. Reachable and job-related
    2. Real or fake using a machine learning model
    3. Remote/hybrid or on-site, and if it offers benefits like insurance

    Supports company pages and third-party portals like **Indeed, LinkedIn, Naukri**, etc.
    """)

    url = st.text_input("🔗 Enter Job URL", placeholder="https://example.com/job")

    if st.button("Analyze Job"):
        if not is_valid_url(url):
            st.warning("Please enter a valid URL starting with http or https and in correct format.")
        else:
            with st.spinner("Analyzing URL..."):
                valid, message, soup = is_valid_job_url(url)
                st.info(message)
                if valid and soup:
                    job_text, is_remote, has_benefits = extract_job_text(soup)
                    if len(job_text) < 50:
                        st.warning("⚠️ Not enough content extracted from the page.")
                    else:
                        prediction = model.predict([job_text])[0]
                        prob = model.predict_proba([job_text])[0][prediction]
                        if prediction == 1:
                            st.error(f"🚩 This job post is likely **FAKE** with {prob*100:.2f}% confidence.")
                        else:
                            st.success(f"✅ This job post is likely **GENUINE** with {prob*100:.2f}% confidence.")
                        
                        # Display remote and benefits info
                        st.markdown("### Additional Details")
                        st.write(f"**Work Type**: {'Remote/Hybrid' if is_remote else 'On-site'}")
                        st.write(f"**Benefits Offered**: {'Yes' if has_benefits else 'No'}")