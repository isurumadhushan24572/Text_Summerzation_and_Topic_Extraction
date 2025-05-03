import streamlit as st
from pdf_processor import extract_text_from_pdf
from summarizer import initialize_summarizer, generate_summary
from topic_modeler import initialize_topic_model, extract_topics
import time


# --------- Setup Page Config ---------
st.set_page_config(
    page_title="Summarizer & Topic Extractor 🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------- Cache Models ---------
=======
# Initialize models once
]
@st.cache_resource
def load_models():
    summarizer = initialize_summarizer()
    topic_model = initialize_topic_model()
    return summarizer, topic_model


# --------- Main App ---------
def main():
    # ---------- Sidebar ----------
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")

        # Input Method (Choose between PDF or Text)
        input_method = st.radio("Choose input method:", ("📄 Upload PDF", "✍️ Paste Text"))

        # Summary Length Slider
        max_summary_length = st.slider("Summary Length (words)", 50, 500, 150, 10)
        st.markdown("---")

        # Show Topics Checkbox
        show_topics = st.checkbox("Show Topics after Summary", value=True)
        st.markdown("---")
        st.info("Need help? Scroll to the bottom ➡️ 📚 About Section")

    # ---------- Header ----------
    st.title("Smart Text Summarization & Topic Extraction")
    st.caption("An AI tool to quickly **summarize** and **understand** your documents.")

    summarizer, topic_model = load_models()

    # ---------- Input Section ----------
    text = ""

    # File upload option (PDF)
    if input_method == "📄 Upload PDF":
        uploaded_file = st.file_uploader("Upload your PDF here", type="pdf", accept_multiple_files=False)
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)

    # Text input option (Paste Text)
    else:
        text = st.text_area("Paste your text below 👇", height=300, placeholder="Paste large articles, research papers, notes...")

    # ---------- Process Button ----------
    if text:
        if st.button("✨ Process Text"):
            with st.spinner("AI is working on it... 🛠️"):
                start_time = time.time()

                # Generate Summary
                summary = generate_summary(text, summarizer, max_length=max_summary_length)
                processing_time = time.time() - start_time

            # Success Message after processing
            st.success(f"✅ Done! Processed in {processing_time:.2f} seconds.")

            # ---------- Results Display ----------
            st.markdown("---")
            st.header("📃 Summary Result")

            # Expandable summary section
            with st.expander("🔎 View Summary"):
                st.info(summary)

            # Download Button for the Summary
            st.download_button("📥 Download Summary", summary, file_name="summary.txt", mime="text/plain")

            # Show Topics if selected
            if show_topics:
                st.markdown("---")
                st.header("🗂 Extracted Topics")

                # Extract and display topics
                topic_info = extract_topics(text, topic_model)

                for index, row in topic_info.iterrows():
                    if row['Topic'] != -1:  # Ignore outliers
                        with st.container():
                            st.subheader(f"Topic {row['Topic']}")
                            st.write(", ".join(row['Representative_Docs']))
                            st.markdown("---")

    else:
        st.warning("🚨 Please upload a PDF or paste some text to begin.")

    # ---------- Footer ----------
    st.markdown("---")
    with st.expander("📚 About This App"):
        st.markdown("""
        - **Text Summarization** using Transformers
        - **Topic Extraction** using BERTopic
        - Built with Python & Streamlit
        - Perfect for summarizing research papers, business documents, long articles, and more!
        """)
=======
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