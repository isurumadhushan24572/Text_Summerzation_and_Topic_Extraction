import streamlit as st
from pdf_processor import extract_text_from_pdf
from summarizer_01 import initialize_summarizer, generate_summary
from topic_modeler_01 import initialize_topic_model, extract_topics
import time

st.set_page_config(
    page_title="Summarizer & Topic Extractor 🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_models():
    summarizer = initialize_summarizer()
    topic_model = initialize_topic_model()
    return summarizer, topic_model

def main():
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        input_method = st.radio("Choose input method:", ("📄 Upload PDF", "✍️ Paste Text"))
        st.markdown("---")
        st.info("Need help? Scroll to the bottom ➡️ 📚 About Section")

    st.title("Smart Text Summarization & Topic Extraction")
    st.caption("An AI tool to quickly **summarize** and **understand** your documents.")

    summarizer, topic_model = load_models()
    text = ""

    if input_method == "📄 Upload PDF":
        uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
    else:
        text = st.text_area("Paste your text below 👇", height=300,
                            placeholder="Paste large articles, research papers, notes...")

    if text and st.button("✨ Process Text"):
        with st.spinner("AI is working on it... 🛠️"):
            start_time = time.time()

            summary = generate_summary(text, summarizer)  # max_length handled inside summarizer_01 now
            topic_info = extract_topics(text, topic_model)

            processing_time = time.time() - start_time

        st.success(f"✅ Done! Processed in {processing_time:.2f} seconds.")
        st.markdown("---")

        st.header("📃 Summary Result")
        with st.expander("🔎 View Summary", expanded=True):
            st.info(summary)
        st.download_button("📥 Download Summary", summary, file_name="summary.txt")

        st.markdown("---")
        st.header("🗂 Topic Extraction")

        st.subheader("Main Themes")
        st.write(topic_info['summary'])

        if topic_info['topics']:
            st.subheader("Topic Labels")
            for i, (label, desc) in enumerate(zip(topic_info['topics'], topic_info['descriptions']), 1):
                st.markdown(f"""
                **Topic {i}:** {label}  
                *Representative segment:*  
                `{desc}`
                """)
                st.markdown("---")
        else:
            st.warning("No meaningful topics could be extracted from the text.")

    elif not text:
        st.warning("🚨 Please upload a PDF or paste some text to begin.")

    st.markdown("---")
    with st.expander("📚 About This App"):
        st.markdown("""
        ## Features
        - **AI-Powered Summarization**: Transformers-based text condensation
        - **Smart Topic Extraction**: BERTopic-driven semantic clustering
        - **Contextual Topic Descriptions**: Human-friendly summaries using real sentences
        - **Download Support**: Save summaries for offline use

        ## Tips
        - Use texts longer than 500 words for better topic extraction
        - Click 'Process Text' after uploading or pasting content
        """)

if __name__ == "__main__":
    main()
