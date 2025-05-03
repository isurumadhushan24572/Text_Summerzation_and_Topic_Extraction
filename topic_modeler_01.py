# Import necessary libraries
from bertopic import BERTopic  # Core topic modeling library
from sentence_transformers import SentenceTransformer  # For generating document embeddings
from umap import UMAP  # Dimensionality reduction tool used in BERTopic
from hdbscan import HDBSCAN  # Clustering algorithm used to find topics
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS  # Vectorizer and stopword list
import streamlit as st  # For UI and caching in the Streamlit app
from typing import Dict, List  # Type annotations for function signatures

# Caches the model so it's loaded only once per session
@st.cache_resource
def initialize_topic_model():
    """Initialize BERTopic with stopword removal and clustering settings."""

    # Load a small, fast transformer model for embedding text
    embedding_model = SentenceTransformer("paraphrase-albert-small-v2")

    # Configure UMAP for dimensionality reduction before clustering
    umap_model = UMAP(
        n_neighbors=10,       # Local neighborhood size
        n_components=5,       # Reduced dimensional space
        min_dist=0.0,         # Minimum distance between points
        metric='cosine',      # Distance metric
        random_state=42       # Reproducibility
    )

    # HDBSCAN is used for clustering the reduced embeddings
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,   # Minimum documents per cluster
        prediction_data=True  # Required for getting representative docs
    )

    # Basic count vectorizer with stopword removal
    vectorizer_model = CountVectorizer(stop_words="english")

    # Construct the BERTopic model with all components
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,  # Useful for downstream filtering
        verbose=False  # Suppress console output
    )

    return topic_model


def extract_topics(text: str, topic_model) -> Dict:
    """Extract clean topics and representative sentences from input text."""

    # Split text into candidate "documents" using periods
    docs = [sent.strip() for sent in text.split('.') if len(sent.strip().split()) > 5]

    # If not enough content, return a placeholder
    if not docs or len(docs) < 3:
        return {
            "name": "Extracted Topics",
            "summary": "Not enough meaningful content to extract topics.",
            "topics": [],
            "descriptions": []
        }

    try:
        # Apply BERTopic model on the document segments
        topics, _ = topic_model.fit_transform(docs)

        # Get full topic metadata
        topic_info = topic_model.get_topic_info()

        # Filter out noise (topic ID -1)
        valid_topics = topic_info[topic_info.Topic != -1]

        topic_labels = []       # Store human-readable topic names
        topic_descriptions = [] # Store representative sentences

        for topic_id in valid_topics['Topic']:
            topic_words = topic_model.get_topic(topic_id)  # List of tuples: (word, relevance score)
            if topic_words:
                # Remove common English stopwords and limit to top 5
                clean_words = [w[0] for w in topic_words if w[0].lower() not in ENGLISH_STOP_WORDS]
                topic_labels.append(", ".join(clean_words[:5]))

            # Try to get a representative document for the topic
            try:
                rep_docs = topic_model.get_representative_docs(topic_id)
                if rep_docs:
                    topic_descriptions.append(rep_docs[0])
                else:
                    topic_descriptions.append("No representative text available.")
            except:
                topic_descriptions.append("No representative text available.")

        # Generate a readable summary from topic labels
        summary = _generate_topic_summary(topic_labels)

        return {
            "name": "Extracted Topics",
            "summary": summary,
            "topics": topic_labels,
            "descriptions": topic_descriptions
        }

    # Catch any runtime errors during modeling
    except Exception as e:
        st.error(f"Topic extraction failed: {e}")
        return {
            "name": "Extracted Topics",
            "summary": "Topic extraction failed.",
            "topics": [],
            "descriptions": []
        }


# Utility function to generate a summary string from topic labels
def _generate_topic_summary(topic_labels: List[str]) -> str:
    if not topic_labels:
        return "No significant topics identified."
    elif len(topic_labels) == 1:
        return f"The main topic is about: {topic_labels[0]}"
    else:
        return f"Main topics include: {', '.join(topic_labels)}"
"
