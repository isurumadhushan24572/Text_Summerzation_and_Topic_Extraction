# 📝 Text Summarization and Topic Extraction

## 💡 Group Project - Neural Network Module (Group 02)

#### This project is a Streamlit-based web application that allows users to upload or paste text, summarize it using a Transformer-based model, and extract key topics using BERTopic.

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
 ##### 2.1) 

## 4.Create topic_modeler python File
 ##### 2.1) 

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
  

   pdf_processor.py — For extracting text from PDF files.

   summarizer.py — For generating text summaries using Transformer models.

   topic_modeler.py — For extracting topics using BERTopic.

   app.py — Main Streamlit application scrip

 ## Used Technologies & Tools
 #####

## 👥 Contribution        
Pdf_processor.py, app.py - M.K.I.M. Rohana - 24572   

Topic_modeler.py - M.R.K. Karunathilaka - 24490

Summarizer.py    - G.A.A.S. Ganegoda - 24614



