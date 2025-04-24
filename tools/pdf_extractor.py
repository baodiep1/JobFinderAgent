import os
import PyPDF2
from .tool import Tool

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path: Relative or absolute path to the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        PyPDF2.PdfReadError: If PDF is corrupted
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        raise PyPDF2.PdfReadError(f"Error reading PDF: {str(e)}")
    
    return text.strip()

pdf_extractor_tool = Tool(
    name="PDF Text Extractor",
    description="Extracts text content from PDF files.",
    func=extract_text_from_pdf,
    arguments=[("pdf_path", "str")],
    outputs="str"
)
