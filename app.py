# Importing required libraries
import streamlit as st  # For building the web app interface
from pdf_processor import extract_text_from_pdf  # Custom function to extract text from PDF files
from summarizer import initialize_summarizer, generate_summary  # Summarizer setup and execution
from topic_modeler import initialize_topic_model, extract_topics  # Topic model setup and topic extraction
import time  # For measuring processing time
from collections import defaultdict  # (Not used here, but useful for mapping structures)

# Streamlit page configuration
st.set_page_config(
    page_title="Summarizer & Topic Extractor 🚀",  # Title shown in the browser tab
    layout="wide",  # Use the full width of the browser
    initial_sidebar_state="expanded",  # Sidebar starts expanded
)

# Load and cache the models to avoid reloading them every time the app runs
@st.cache_resource
def load_models():
    summarizer = initialize_summarizer()  # Load the summarization model
    topic_model = initialize_topic_model()  # Load the topic modeling pipeline
    return summarizer, topic_model  # Return both models as a tuple

# Main function that runs the Streamlit app
def main():
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Settings")  # Sidebar title
        st.markdown("---")  # Horizontal separator

        # Input method radio buttons for choosing PDF upload or text input
        input_method = st.radio("Choose input method:", ("📄 Upload PDF", "✍️ Paste Text"))

        st.markdown("---")  # Another separator
        st.info("Need help? Scroll to the bottom ➡️ 📚 About Section")  # Info message in sidebar

    # Main page title and caption
    st.title("Smart Text Summarization & Topic Extraction")
    st.caption("An AI tool to quickly **summarize** and **understand** your documents.")

    summarizer, topic_model = load_models()  # Load cached models
    text = ""  # Initialize an empty text variable

    # Input section: either upload a PDF or paste text manually
    if input_method == "📄 Upload PDF":
        uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")  # File uploader UI
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):  # Show spinner while processing
                text = extract_text_from_pdf(uploaded_file)  # Extract text from the uploaded PDF
    else:
        # Text area for pasting input directly
        text = st.text_area("Paste your text below 👇", height=300,
                            placeholder="Paste large articles, research papers, notes...")

    # When user clicks the button to process the input
    if text and st.button("✨ Process Text"):
        with st.spinner("AI is working on it... 🛠️"):  # Show spinner while processing
            start_time = time.time()  # Start timer

            summary = generate_summary(text, summarizer)  # Generate summary (max_length is handled inside the function)
            topic_info = extract_topics(text, topic_model)  # Extract topics and theme descriptions

            processing_time = time.time() - start_time  # Calculate how long the processing took

        # Show success message and how long it took
        st.success(f"✅ Done! Processed in {processing_time:.2f} seconds.")
        st.markdown("---")  # Separator

        # Summary output
        st.header("📃 Summary Result")
        with st.expander("🔎 View Summary", expanded=True):
            st.info(summary)  # Display the summary in an info box
        st.download_button("📥 Download Summary", summary, file_name="summary.txt")  # Download button for summary

        # Topic extraction output
        st.markdown("---")
        st.header("🗂 Topic Extraction")

        st.subheader("Main Themes")
        st.write(topic_info['summary'])  # Display general topic summary

        if topic_info['topics']:  # If topic labels were extracted
            st.subheader("Topic Labels")
            for i, (label, desc) in enumerate(zip(topic_info['topics'], topic_info['descriptions']), 1):
                # Show each topic label and a representative segment
                st.markdown(f"""
                **Topic {i}:** {label}  
                *Representative segment:*  
                `{desc}`
                """)
                st.markdown("---")  # Separator for each topic
        else:
            st.warning("No meaningful topics could be extracted from the text.")  # Fallback warning

    elif not text:
        st.warning("🚨 Please upload a PDF or paste some text to begin.")  # Warning if no text was provided

    # About section at the bottom of the page
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

# Run the main function when the script is executed
if __name__ == "__main__":
    main()

