import re
import ipaddress
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Phishing URL Detection System")
st.write(
    "A Machine Learning demonstration that analyzes URL patterns "
    "to identify potentially suspicious links."
)

st.info(
    "Educational demo only. This model uses illustrative synthetic "
    "training examples and cannot reliably determine whether a real "
    "website is safe."
)


# ---------------- URL FEATURES ----------------
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure",
    "account", "banking", "confirm", "password",
    "signin", "wallet", "free", "gift"
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co",
    "is.gd", "cutt.ly", "rb.gy"
]


def extract_features(url):
    url = url.strip()

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    lower_url = url.lower()

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "dot_count": hostname.count("."),
        "hyphen_count": hostname.count("-"),
        "at_symbol": int("@" in url),
        "https": int(parsed.scheme == "https"),
        "ip_address": has_ip,
        "subdomain_count": max(0, hostname.count(".") - 1),
        "suspicious_word_count": sum(
            word in lower_url for word in SUSPICIOUS_WORDS
        ),
        "digit_count": sum(char.isdigit() for char in url),
        "shortener": int(
            any(hostname == domain for domain in SHORTENERS)
        ),
        "has_port": int(parsed.port is not None)
        if parsed.port else 0,
    }

    return features


# ---------------- DEMO TRAINING DATA ----------------
# These examples are illustrative, not a real-world dataset.

legitimate_examples = [
    "https://www.wikipedia.org",
    "https://www.python.org",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.bbc.com",
    "https://www.github.com",
    "https://www.openai.com",
    "https://www.nationalgeographic.com",
    "https://www.adobe.com",
    "https://www.nasa.gov",
    "https://www.un.org",
]

phishing_examples = [
    "http://paypal-login-verify.example.com",
    "http://secure-account-update.example.net",
    "http://192.0.2.10/verify-account",
    "http://banking-confirm-password.example.org",
    "http://free-gift-login.example.com",
    "http://account-security-update.example.net",
    "http://signin-wallet-confirm.example.org",
    "http://verify-user-account.example.com",
    "http://secure-login-password.example.net",
    "http://update-banking-details.example.org",
    "http://confirm-account-access.example.com",
    "http://login-verify-user.example.net",
]

rows = []

for url in legitimate_examples:
    row = extract_features(url)
    row["label"] = 0
    rows.append(row)

for url in phishing_examples:
    row = extract_features(url)
    row["label"] = 1
    rows.append(row)

df = pd.DataFrame(rows)

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "dot_count",
    "hyphen_count",
    "at_symbol",
    "https",
    "ip_address",
    "subdomain_count",
    "suspicious_word_count",
    "digit_count",
    "shortener",
    "has_port",
]

X = df[FEATURE_COLUMNS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))


# ---------------- DASHBOARD ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Training Examples", len(X_train))
col2.metric("Model", "Random Forest")
col3.metric("Demo Test Accuracy", f"{accuracy:.0%}")

st.divider()

st.subheader("🔎 Check a URL")

url_input = st.text_input(
    "Enter a website URL",
    placeholder="https://example.com"
)

if st.button("Analyze URL", type="primary"):
    if not url_input.strip():
        st.warning("Please enter a URL first.")

    else:
        candidate = url_input.strip()

        if re.search(r"\s", candidate):
            st.error("Please enter a URL without spaces.")

        else:
            try:
                normalized = (
                    candidate
                    if re.match(r"^https?://", candidate, re.IGNORECASE)
                    else "http://" + candidate
                )

                parsed = urlparse(normalized)

                if (
                    parsed.scheme not in ("http", "https")
                    or not parsed.hostname
                    or "." not in parsed.hostname
                    or parsed.hostname.startswith(".")
                    or parsed.hostname.endswith(".")
                 ):
                    st.error("Please enter a valid-looking website URL.")

                else:
                    features = extract_features(candidate)
                    input_df = pd.DataFrame(
                        [features],
                        columns=FEATURE_COLUMNS
                    )

                    prediction = int(model.predict(input_df)[0])
                    probabilities = model.predict_proba(input_df)[0]
                    phishing_probability = float(probabilities[1])

                    st.subheader("Analysis Result")

                    if prediction == 1:
                        st.error(
                            "⚠️ The demo model flags this URL "
                            "as potentially suspicious."
                        )
                    else:
                        st.success(
                            "The demo model classifies this URL "
                            "as likely legitimate."
                        )

                    st.metric(
                        "Model's Phishing Score",
                        f"{phishing_probability:.1%}"
                    )

                    st.caption(
                        "This score is the model's output, not a "
                        "verified probability that the website is malicious."
                    )

                    st.write("**Extracted URL features**")
                    st.dataframe(
                        pd.DataFrame(
                            features.items(),
                            columns=["Feature", "Value"]
                        ),
                        hide_index=True,
                        use_container_width=True
                    )

            except Exception:
                st.error(
                    "Could not analyze this input. "
                    "Please check the URL and try again."
                )

with st.expander("About this project"):
    st.write(
        "This beginner project extracts URL patterns and uses a "
        "Random Forest classifier. The training examples are synthetic "
        "and very small. For a meaningful cybersecurity model, train "
        "and evaluate it on a properly labelled, representative dataset."
    )

    st.write(
        "The app does not visit the URL or verify the actual website. "
        "Do not enter passwords or other sensitive information."
    )
    #conda activate phishing_env
    #cd /d "C:\Users\pc\Favorites\Downloads\phishing_URL_Detection"
    #streamlit run app.py