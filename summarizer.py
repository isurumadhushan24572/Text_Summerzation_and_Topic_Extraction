# Import the Hugging Face summarization pipeline
from transformers import pipeline

# Initialize the summarizer pipeline using a pre-trained DistilBART model
def initialize_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


# Split long text into manageable chunks for the summarizer
def chunk_text(text, max_chunk=1024):
    # Slices the text into pieces of `max_chunk` characters
    return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]


# Generate a summary from long input text using the summarizer pipeline
def generate_summary(text, summarizer, default_max_length=150):
    chunks = chunk_text(text)  # Break the text into smaller parts
    summaries = []  # Store summaries for each chunk

    for chunk in chunks:
        # Estimate the number of words in the chunk
        input_length = len(chunk.split())

        # Dynamically set the max summary length to 80% of input, capped at default
        effective_max_length = min(default_max_length, int(input_length * 0.8))

        try:
            # Generate the summary using the transformer pipeline
            summary = summarizer(
                chunk,
                max_length=effective_max_length,  # Cap summary length
                min_length=30,                    # Ensure a minimum summary size
                do_sample=False                   # Use greedy decoding for deterministic output
            )[0]['summary_text']
            summaries.append(summary)  # Add the summary of the chunk
        except Exception as e:
            # If something goes wrong, store the error message in the output
            summaries.append(f"[Summarization failed: {e}]")

    # Join all chunk summaries into a single string
    return " ".join(summaries)
