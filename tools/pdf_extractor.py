import os
import PyPDF2
from .tool import Tool

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            print(f"Processing PDF with {len(reader.pages)} pages...")
            
            for i, page in enumerate(reader.pages):
                try:
                    # Remove x_tolerance parameter for compatibility
                    page_text = page.extract_text() or ""
                    print(f"Page {i+1}: {len(page_text)} chars")
                    text += page_text
                except Exception as page_error:
                    print(f"Warning: Error on page {i+1}: {str(page_error)}")
                    continue
                    
    except Exception as e:
        error_type = getattr(PyPDF2, "PdfError", PyPDF2.PdfReadError)
        raise error_type(f"Error reading PDF: {str(e)}")

    cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    print(f"\nTotal extracted characters: {len(cleaned_text)}")
    return cleaned_text

pdf_extractor_tool = Tool(
    name="PDF Text Extractor",
    description="Extracts text content from PDF files.",
    func=extract_text_from_pdf,
    arguments=[("pdf_path", "str")],
    outputs="str"
)
