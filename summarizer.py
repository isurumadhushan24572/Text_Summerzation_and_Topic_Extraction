# from transformers import pipeline

# def initialize_summarizer():
#     return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# def chunk_text(text, max_chunk=1024):
#     return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

# def generate_summary(text, summarizer, max_length=150):
#     chunks = chunk_text(text)
#     summaries = []
#     for chunk in chunks:
#         summary = summarizer(
#             chunk,
#             max_length=max_length,
#             min_length=30,
#             do_sample=False
#         )[0]['summary_text']
#         summaries.append(summary)
#     return " ".join(summaries)



from transformers import pipeline

def initialize_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def chunk_text(text, max_chunk=1024):
    return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

def generate_summary(text, summarizer, max_length=150):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)
