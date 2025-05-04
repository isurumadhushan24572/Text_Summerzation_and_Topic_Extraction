# 📝 Text Summarization and Topic Extraction

## 💡 Group Project - Neural Network Module (Group 02)

#### This project is a Streamlit-based web application that allows users to upload or paste text, summarize it using a Transformer-based model, and extract key topics using BERTopic.

![My Logo](QR_CODE.jpeg)

## Steps Followed
## 1.Create a virtual environment 
 ##### 1.1)  Make Virtual Environment as nlp-env
 
  ```
  python -m venv nlp-env
  ```
 ##### 1.2) Activate the virtual environment

  ```
  cd nlp-env\Scripts\activate
  ```
##### 1.3) creating python gitignore file

##### 1.4) Install required libraries

  ```
  pip install -r requirements.txt
  ```
## 2.Create pdf_processor python File

 ##### Used Function

 ```
 def extract_text_from_pdf(uploaded_file):
 ```
* This Function was used for extract the text data from uploaded the PDF Files
* That mean when the function called return the the text data of uploaded PDF Files.
   
## 3.Create summarizer python File

 ##### Used Libraries
   ```
 from transformers import pipeline
   ```
   * Pipeline - use for tasks like summarization, sentiment analysis, translation, etc.

 ##### Used Functions
  ###### **initialize_summarizer()**
   ```
def initialize_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
   ```
   * Loads a pre-trained summarization pipeline using Hugging Face's transformers library.
   * Uses the DistilBART model (sshleifer/distilbart-cnn-12-6), which is a lighter, faster version of BART optimized for summarizing long texts
  ###### **chunk_text(text, max_chunk=1024)**
   ```
def chunk_text(text, max_chunk=1024):
    return [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
   ```
   * Splits the input text into smaller chunks (default max 1024 characters).
   * This is necessary because most transformer models have a limit on input length (often 1024 tokens or fewer).
   * Ensures that long documents can be processed safely without exceeding the model's input limit.

   ###### **generate_summary(text, summarizer, default_max_length=150)**
   ```
def generate_summary(text, summarizer, default_max_length=150):
   ```
   * First calls chunk_text to split the long text.
   * Then loops over each chunk and:
        * Calculates an appropriate max_length based on the input length.
        * Calls the summarizer model to summarize each chunk.
        * Handles exceptions and appends all summaries together.
          
   ###### **Then,returns a single combined summary of the full text**

## 4.Create topic_modeler python File
 #### Steps:
  **1. Input Text**
     
* Start with a long paragraph of text, which the user provides.


**2. Sentence Split**

* The text is split into individual sentences.

Code:
     
    docs = [sent.strip() for sent in text.split('.') if len(sent.strip().split()) > 5]


  **3. Sentence Embeddings (BERT)**         

* Each sentence is then transformed into a numerical form, called an "embedding." We use a BERT-based model (SentenceTransformer) to capture the meaning of each sentence in numbers.

Code:

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

           
  **4. Dimensionality Reduction (UMAP)**

* These sentence embeddings are large, we use a technique called UMAP to reduce their size, making them easier to group.

* This helps the next step (clustering) to work more efficiently.
     
Code:

    umap_model = UMAP(n_neighbors=10, n_components=5, ...)

                      
  **5. Clustering (HDBSCAN)**

* We group the sentences that are similar in meaning into clusters, which we call topics. Any sentences that don't fit well into any cluster are marked as "noise".

Code:

    hdbscan_model = HDBSCAN(min_cluster_size=5, ...)

                  
  **6. Topic Extraction (BERTopic)**

* Use BERTopic to extract the topics. This combines all the earlier steps and identifies the key topics based on the clustered sentences. It also pulls out the most important words related to each topic.

Code: 

    topic_model = BERTopic(...)
    topics, _ = topic_model.fit_transform(docs)

                  
  **7. Representative Sentences + Summary**

* Extract sentences for each topic & give the summary.

Code: 

    rep_docs = topic_model.get_representative_docs(topic_id)
    summary = _generate_topic_summary(topic_labels)



## 5.Create app python File

* Import created Python Files functions required libraries
```
import streamlit as st
from pdf_processor import extract_text_from_pdf
from summarizer import initialize_summarizer, generate_summary
from topic_modeler import initialize_topic_model, extract_topics
import time
```

* Streamlit page configuration
* Load and cache the models to avoid reloading them every time the app runs
  * summarization model
  * topic modeling pipeline
* Set Input methods as PDF Files or Text input
* Generate Summarry
```
summary = generate_summary(text, summarizer)
```
* Extract Keywords & Topics
```
topic_info = extract_topics(text, topic_model)
```
* Display the precessing Time
* Summary output
* Topic extraction output
* About section at the bottom of the page
  
## ⚠️ Challenges Faced

* Topic Extraction: Selecting relevant key topics was challenging due to it's only display the key words.
* Time Consumption: Processing time increased with larger texts specially in PDF files,topic modeling.

## How Solved the challeges

* For the summerization used light weight and fast model called : Falconsai/text_summarization (summarizer_01.py)
* Foor the SentenceTransformer used light weight and fast model called : paraphrase-albert-small-v2
* Remove Stop Words when extracting the topics
  
 ## Used Technologies & Tools
 
* ***Python 3.10+*** – Core programming language
* ***Streamlit*** – Interactive web app framework
* ***Transformers (Hugging Face)*** – For text summarization using sshleifer/distilbart-cnn-12-6
* ***BERTopic*** – Topic modeling using transformer-based embeddings, UMAP, and HDBSCAN
* ***Sentence-Transformers*** – Embedding model (all-MiniLM-L6-v2) for semantic understanding
* ***UMAP*** – Non-linear dimensionality reduction for visualization and clustering
* ***HDBSCAN*** – Hierarchical density-based clustering for topic discovery
* ***Scikit-learn*** – CountVectorizer for word frequency analysis and stopword removal
* ***pdfplumber*** – High-fidelity PDF text extraction

## 👥 Contribution        
M K I M Rohana - 24572 - pdf_processor.py, app.py   

M R K Karunathilaka - 24490 - topic_modeler.py 

G A A S Ganegoda - 24614 - summarizer.py    



