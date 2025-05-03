from transformers import pipeline

def initialize_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def chunk_text(text, max_chunk=1024):
    return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

def generate_summary(text, summarizer, default_max_length=150):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        input_length = len(chunk.split())  # Approximate token length using words
        effective_max_length = min(default_max_length, int(input_length * 0.8))

        try:
            summary = summarizer(
                chunk,
                max_length=effective_max_length,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"[Summarization failed: {e}]")

    return " ".join(summaries)
