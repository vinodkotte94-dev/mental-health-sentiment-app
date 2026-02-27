<<<<<<< HEAD
import streamlit as st
import pandas as pd
import datetime
from transformers import pipeline

# -----------------------------
# Load Pretrained Models
# -----------------------------
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)
sentiment_classifier = pipeline("sentiment-analysis")

# -----------------------------
# Map transformer emotions to Positive/Negative
# -----------------------------
emotion_mapping = {
    "joy": "Positive",
    "anger": "Negative",
    "sadness": "Negative",
    "fear": "Negative",
    "neutral": "Negative"   # Neutral is treated as Negative
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mental Health Emotion Journal", layout="wide")
st.title("🧠 Mental Health Emotion Journal")
st.write(
    "Share your thoughts, a journal entry, or just how your day went. "
    "I'm here to help you understand your emotions."
)
st.markdown("---")

user_input = st.text_area("✍️ Write your thoughts here...")

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("⚠️ Please write something before analyzing.")
    else:
        with st.spinner("Analyzing emotions..."):
            # -----------------------------
            # Emotion Detection
            # -----------------------------
            emotion_scores_raw = emotion_classifier(user_input)

            if isinstance(emotion_scores_raw, list) and isinstance(emotion_scores_raw[0], list):
                emotion_scores = emotion_scores_raw[0]
            elif isinstance(emotion_scores_raw, list) and isinstance(emotion_scores_raw[0], dict):
                emotion_scores = emotion_scores_raw
            else:
                emotion_scores = [emotion_scores_raw]

            # Sort by score descending
            emotion_scores_sorted = sorted(emotion_scores, key=lambda x: x["score"], reverse=True)
            top_emotion = emotion_scores_sorted[0]["label"]
            top_score = emotion_scores_sorted[0]["score"]

            # Sentiment detection
            sentiment_output = sentiment_classifier(user_input)[0]
            sentiment = sentiment_output["label"]
            sentiment_score = sentiment_output["score"]

        # -----------------------------
        # Journal Entry Display
        # -----------------------------
        st.markdown(
            f"**{emotion_mapping.get(top_emotion, 'Negative').upper()} • "
            f"{datetime.datetime.now().strftime('%B %d, %Y • %I:%M %p')}**"
        )
        st.markdown(f"> {user_input}")

        # -----------------------------
        # Display Detected Emotion & Sentiment (No Suggestions)
        # -----------------------------
        st.write("### 💡 Emotional Insight")
        st.write(f"**Detected Emotion:** {top_emotion.capitalize()} ({emotion_mapping.get(top_emotion, 'Negative')})")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Emotion Confidence:** {round(top_score*100,2)}%")
        st.write(f"**Sentiment Confidence:** {round(sentiment_score*100,2)}%")

        # -----------------------------
        # Emotion Scores Table Only
        # -----------------------------
        breakdown_df = pd.DataFrame({
            "Emotion": [e["label"] for e in emotion_scores_sorted],
            "Intensity": [round(e["score"]*100,2) for e in emotion_scores_sorted]
        })

        st.write("#### Emotion Scores Table")
        st.dataframe(breakdown_df)

        st.markdown("---")
=======
import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Mental Health Sentiment App", page_icon="🧠")

# -----------------------------
# Dataset (Balanced & Improved)
# -----------------------------
data = {
    "text": [
        # Positive
        "I am happy",
        "I feel happy",
        "I am very happy today",
        "Life is beautiful",
        "I feel amazing",
        "I feel great",
        "I am good",
        "I am feeling good",
        "I am excited",
        "I feel confident",
        "I am proud",
        "I am satisfied",
        "Everything is going well",
        "I feel peaceful",
        "I feel relaxed",
        "Today is a wonderful day",
        "I feel positive",
        "I am hopeful",
        "I feel motivated",
        "I am not sad anymore",

        # Negative
        "I am sad",
        "I feel sad",
        "I am very sad",
        "I feel depressed",
        "I am depressed",
        "I feel anxious",
        "I feel stressed",
        "I feel lonely",
        "I feel hopeless",
        "I am not happy",
        "I am not good",
        "I am not feeling well",
        "I feel terrible",
        "I feel bad",
        "Nothing makes me happy",
        "I am tired of everything",
        "I feel negative",
        "I am frustrated",
        "I feel upset",
        "I feel miserable"
    ],
    "label": ["Positive"]*20 + ["Negative"]*20
}

df = pd.DataFrame(data)

# -----------------------------
# Preprocessing (Simple & Safe)
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["clean_text"] = df["text"].apply(preprocess)

# -----------------------------
# Vectorizer (Important Improvement)
# -----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),      # handles "not happy"
    stop_words=None
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# -----------------------------
# Model (Better Settings)
# -----------------------------
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"   # helps improve fairness
)

model.fit(X, y)

accuracy = model.score(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Mental Health Sentiment Detection App")
st.write("This app detects whether the entered text expresses **Positive** or **Negative** mental sentiment.")
st.write(f"### 📊 Model Accuracy: {round(accuracy*100,2)}%")

st.markdown("---")

user_input = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():

        cleaned = preprocess(user_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized).max()

        if prediction == "Positive":
            st.success(f"✅ Prediction: {prediction}")
        else:
            st.error(f"⚠️ Prediction: {prediction}")

        st.info(f"📈 Confidence Score: {round(probability*100,2)}%")

    else:
        st.warning("⚠️ Please enter some text.")
>>>>>>> f666e1dc816c1ba59ea29972c59f48f2795714c8
