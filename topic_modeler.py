# Import necessary modules from BERTopic and supporting libraries
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import streamlit as st
from typing import Dict, List

# Cache the model so it doesn't reload every time the app runs
@st.cache_resource
def initialize_topic_model():
    """Initialize BERTopic with stopword removal and clustering settings."""

    # Load a lightweight sentence transformer model for embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # Configure HDBSCAN for clustering the embeddings
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        prediction_data=True
    )

    # Use a basic vectorizer with English stopword removal
    vectorizer_model = CountVectorizer(stop_words="english")

    # Combine everything into the BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=False
    )

    return topic_model  # Return the initialized topic model


def extract_topics(text: str, topic_model) -> Dict:
    """Extract clean topics and representative sentences from input text."""

    # Preprocess: split input text into sentences with > 5 words
    docs = [sent.strip() for sent in text.split('.') if len(sent.strip().split()) > 5]

    # If not enough valid text segments, return a fallback message
    if not docs or len(docs) < 3:
        return {
            "name": "Extracted Topics",
            "summary": "Not enough meaningful content to extract topics.",
            "topics": [],
            "descriptions": []
        }

    try:
        # Fit BERTopic model and extract topic assignments
        topics, _ = topic_model.fit_transform(docs)

        # Get the topic summary table
        topic_info = topic_model.get_topic_info()

        # Filter out the outlier topic (-1)
        valid_topics = topic_info[topic_info.Topic != -1]

        topic_labels = []        # Store topic names (labels)
        topic_descriptions = []  # Store a sample sentence for each topic

        for topic_id in valid_topics['Topic']:
            # Get top words from each topic
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                # Filter out stopwords and pick top 5
                clean_words = [w[0] for w in topic_words if w[0].lower() not in ENGLISH_STOP_WORDS]
                topic_labels.append(", ".join(clean_words[:5]))

            try:
                # Try getting a representative document (text segment) for the topic
                rep_docs = topic_model.get_representative_docs(topic_id)
                if rep_docs:
                    topic_descriptions.append(rep_docs[0])
                else:
                    topic_descriptions.append("No representative text available.")
            except:
                topic_descriptions.append("No representative text available.")

        # Generate an overall topic summary sentence
        summary = _generate_topic_summary(topic_labels)

        return {
            "name": "Extracted Topics",
            "summary": summary,
            "topics": topic_labels,
            "descriptions": topic_descriptions
        }

    except Exception as e:
        # If anything goes wrong, show an error and return fallback
        st.error(f"Topic extraction failed: {e}")
        return {
            "name": "Extracted Topics",
            "summary": "Topic extraction failed.",
            "topics": [],
            "descriptions": []
        }


# Generate a natural-language summary sentence of the topics
def _generate_topic_summary(topic_labels: List[str]) -> str:
    if not topic_labels:
        return "No significant topics identified."
    elif len(topic_labels) == 1:
        return f"The main topic is about: {topic_labels[0]}"
    else:
        return f"Main topics include: {', '.join(topic_labels)}"
