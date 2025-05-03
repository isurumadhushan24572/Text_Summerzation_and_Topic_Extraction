from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import streamlit as st
from typing import Dict, List

@st.cache_resource
def initialize_topic_model():
    """Initialize BERTopic with stopword removal and clustering settings."""
    embedding_model = SentenceTransformer("paraphrase-albert-small-v2")

    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=False
    )

    return topic_model


def extract_topics(text: str, topic_model) -> Dict:
    """Extract clean topics and representative sentences from input text."""
    docs = [sent.strip() for sent in text.split('.') if len(sent.strip().split()) > 5]

    if not docs or len(docs) < 3:
        return {
            "name": "Extracted Topics",
            "summary": "Not enough meaningful content to extract topics.",
            "topics": [],
            "descriptions": []
        }

    try:
        topics, _ = topic_model.fit_transform(docs)
        topic_info = topic_model.get_topic_info()
        valid_topics = topic_info[topic_info.Topic != -1]

        topic_labels = []
        topic_descriptions = []

        for topic_id in valid_topics['Topic']:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                clean_words = [w[0] for w in topic_words if w[0].lower() not in ENGLISH_STOP_WORDS]
                topic_labels.append(", ".join(clean_words[:5]))

            try:
                rep_docs = topic_model.get_representative_docs(topic_id)
                if rep_docs:
                    topic_descriptions.append(rep_docs[0])
                else:
                    topic_descriptions.append("No representative text available.")
            except:
                topic_descriptions.append("No representative text available.")

        summary = _generate_topic_summary(topic_labels)

        return {
            "name": "Extracted Topics",
            "summary": summary,
            "topics": topic_labels,
            "descriptions": topic_descriptions
        }

    except Exception as e:
        st.error(f"Topic extraction failed: {e}")
        return {
            "name": "Extracted Topics",
            "summary": "Topic extraction failed.",
            "topics": [],
            "descriptions": []
        }


def _generate_topic_summary(topic_labels: List[str]) -> str:
    if not topic_labels:
        return "No significant topics identified."
    elif len(topic_labels) == 1:
        return f"The main topic is about: {topic_labels[0]}"
    else:
        return f"Main topics include: {', '.join(topic_labels)}"
