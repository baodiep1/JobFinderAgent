import os
import pdfplumber
import re
import nltk
from nltk.tokenize import word_tokenize
from smolagents import tool

def format_extracted_text(text):
    """Enhances extracted text readability by applying better formatting."""
    text = re.sub(r'\s+', ' ', text)  # Removes excessive whitespace
    text = text.replace(" | ", "\n")  # Separates sections properly
    text = text.replace("Education ", "\n📚 Education\n")  # Highlights education section
    text = text.replace("Experience ", "\n💼 Experience\n")  # Highlights experience section
    text = text.replace("Skills ", "\n🛠 Skills\n")  # Highlights skills section
    return text.strip()

def extract_skills_from_text(text):
    """Uses NLP to extract potential skill keywords from resume text."""
    words = word_tokenize(text.lower())
    skills = ["python", "java", "c++", "assembly", "machine learning", "data analysis"]  # Expand this list
    extracted_skills = list(set(word for word in words if word in skills))  # Remove duplicates
    return extracted_skills

@tool
def extract_text_from_pdf(pdf_path: str) -> dict:
    """Extracts text from a PDF and identifies skills for job matching.
    
    Args:
        pdf_path (str): The path to the PDF file.
        
    Returns:
        dict: The extracted text and a list of detected skills.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Processing PDF with {len(pdf.pages)} pages...")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text += page_text + "\n"

    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {str(e)}")

    formatted_text = format_extracted_text(text)
    extracted_skills = extract_skills_from_text(formatted_text)  # New NLP function
    
    return {"text": formatted_text, "skills": extracted_skills}