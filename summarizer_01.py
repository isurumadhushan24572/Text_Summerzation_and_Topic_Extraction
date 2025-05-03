from transformers import pipeline

def initialize_summarizer():
    return pipeline("summarization", model="Falconsai/text_summarization")  # Light and fast

def chunk_text(text, max_chunk=512):
    return [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

def generate_summary(text, summarizer, max_length=150):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(
                chunk,
                max_length=min(max_length, 512),  # Safely cap length
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"[Error in summarization: {e}]")
    return " ".join(summaries)
