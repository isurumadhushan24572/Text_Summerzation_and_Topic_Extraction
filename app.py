import streamlit as st
from pdf_processor import extract_text_from_pdf
from summarizer import initialize_summarizer, generate_summary
from topic_modeler import initialize_topic_model, extract_topics
import time
from collections import defaultdict

# --------- Setup Page Config ---------
st.set_page_config(
    page_title="Summarizer & Topic Extractor 🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------- Cache Models ---------
@st.cache_resource
def load_models():
    summarizer = initialize_summarizer()
    topic_model = initialize_topic_model()
    return summarizer, topic_model

# --------- Cache PDF Text ---------
@st.cache_data
def cached_extract_text(uploaded_file):
    return extract_text_from_pdf(uploaded_file)

# --------- Text Processing Utilities ---------
def split_into_sentences(text):
    return [sent.strip() for sent in text.split('. ') if sent.strip()]

def map_keywords_to_sentences(sentences, keywords):
    keyword_map = defaultdict(list)
    for idx, sentence in enumerate(sentences):
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                keyword_map[keyword].append({
                    "sentence": sentence,
                    "position": idx+1
                })
    return keyword_map

# --------- Main App ---------
def main():
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")

        input_method = st.radio("Choose input method:", ("📄 Upload PDF", "✍️ Paste Text"))
        max_summary_length = st.slider("Summary Length (words)", 50, 1000, 150, 10)
        num_keywords = st.slider("Number of Keywords", 5, 20, 10)
        st.markdown("---")
        st.info("Need help? Scroll to the bottom ➡️ 📚 About Section")

    st.title("Smart Text Summarization & Topic Extraction")
    st.caption("An AI tool to quickly **summarize** and **understand** your documents.")

    summarizer, topic_model = load_models()
    text = ""

    # ---------- Input Section ----------
    if input_method == "📄 Upload PDF":
        uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")
        if uploaded_file:
            with st.spinner("Extracting text from PDF... ⏳"):
                text = cached_extract_text(uploaded_file)
    else:
        text = st.text_area("Paste your text below 👇", height=300, 
                          placeholder="Paste large articles, research papers, notes...")

    # ---------- Processing Section ----------
    if text and st.button("✨ Process Text"):
        with st.spinner("AI is working on it... 🛠️"):
            start_time = time.time()
            
            # Generate Summary
            summary = generate_summary(text, summarizer, max_length=max_summary_length)
            
            # Extract Topics
            topic_info = extract_topics(text, topic_model)
            sentences = split_into_sentences(text)
            keyword_map = map_keywords_to_sentences(sentences, topic_info['keywords'][:num_keywords])
            
            processing_time = time.time() - start_time

        # ---------- Display Results ----------
        st.success(f"✅ Done! Processed in {processing_time:.2f} seconds.")
        st.markdown("---")
        
        # Summary Section
        st.header("📃 Summary Result")
        with st.expander("🔎 View Summary", expanded=True):
            st.info(summary)
        st.download_button("📥 Download Summary", summary, file_name="summary.txt")

        # Topics Section
        st.markdown("---")
        st.header("🗂 Extracted Topics")
        
        # Topic Summary
        st.subheader("Main Themes")
        st.write(topic_info['summary'])

        # Topic Labels
        st.subheader("Topic Labels")
        for topic in topic_info.get("topics", []):
            st.markdown(f"**{topic['label']}**: {', '.join(topic['words'])}")

        # Interactive Keywords
        st.subheader("Key Terms")
        cols = st.columns(4)
        keywords = topic_info['keywords'][:num_keywords]
        
        for idx, keyword in enumerate(keywords):
            with cols[idx % 4]:
                if st.button(keyword, key=f"kw_{idx}"):
                    st.session_state.selected_keyword = keyword

        if 'selected_keyword' in st.session_state:
            st.markdown("---")
            st.subheader(f"Text segments related to: {st.session_state.selected_keyword}")
            related_sentences = keyword_map.get(st.session_state.selected_keyword, [])
            
            for i, sent in enumerate(related_sentences, 1):
                st.markdown(f"""
                **Segment {i}**  
                {sent['sentence']}  
                *Position in text: {sent['position']}*
                """)
                st.markdown("---")

    elif not text:
        st.warning("🚨 Please upload a PDF or paste some text to begin.")

    # ---------- Footer ----------
    st.markdown("---")
    with st.expander("📚 About This App"):
        st.markdown("""
        ## Features
        - **AI-Powered Summarization**: Transformers-based text condensation
        - **Smart Topic Extraction**: BERTopic-driven theme analysis
        - **Keyword Insights**: Interactive exploration of key terms
        - **Contextual Analysis**: Direct links to original text segments

        ## Usage Tips
        - For best results, use texts longer than 500 words
        - Adjust summary length based on needs
        - Click keywords to explore context
        - Download summaries for later use
        """)

if __name__ == "__main__":
    main()
