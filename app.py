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