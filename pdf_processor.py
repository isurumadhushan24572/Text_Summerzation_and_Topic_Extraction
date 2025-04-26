import PyPDF2 # reading and manipulating PDF files

# This is the function we used to extract text from the PDF file
def extract_text_from_pdf(uploaded_file):   
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# This function return entire text from the PDF file
