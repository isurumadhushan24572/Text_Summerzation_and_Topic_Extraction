import streamlit as st
from pdf_processor import extract_text_from_pdf
from summarizer import initialize_summarizer, generate_summary
from topic_modeler import initialize_topic_model, extract_topics
import time

# Initialize models once
@st.cache_resource
def load_models():
    summarizer = initialize_summarizer()
    topic_model = initialize_topic_model()
    return summarizer, topic_model

def main():
    st.title("Text Summarization & Topic Extraction Tool")
    
    summarizer, topic_model = load_models()
    
    # File upload or text input
    input_method = st.radio("Choose input method:", ("Upload PDF", "Paste Text"))
    
    text = ""
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
    else:
        text = st.text_area("Paste your text here:", height=300)
    
    if text:
        with st.spinner("Processing..."):
            # Generate Summary
            start_time = time.time()
            summary = generate_summary(text, summarizer)
            st.subheader("Summary")
            st.write(summary)
            
            # Topic Extraction
            st.subheader("Key Topics")
            topic_info = extract_topics(text, topic_model)
            
            # Display topics with representative words
            for index, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Filter out outliers
                    st.markdown(f"**Topic {row['Topic']}**: {', '.join(row['Representative_Docs'])}")
            
            st.write(f"Processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()