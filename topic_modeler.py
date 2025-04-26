from bertopic import BERTopic

def initialize_topic_model():
    return BERTopic(embedding_model="all-MiniLM-L6-v2")

def extract_topics(text, topic_model):
    # Split text into sentences for better topic modeling
    sentences = [sentence.strip() for sentence in text.split('.') if sentence]
    topics, _ = topic_model.fit_transform(sentences)
    return topic_model.get_topic_info()