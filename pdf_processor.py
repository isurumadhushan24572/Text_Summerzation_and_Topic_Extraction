import pdfplumber

def extract_text_from_pdf(uploaded_file):
    """Extracts text from PDF using pdfplumber (faster and cleaner than PyPDF2)"""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text
