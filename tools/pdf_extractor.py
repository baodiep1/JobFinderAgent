import os
import pdfplumber
import re
import nltk
from nltk.tokenize import word_tokenize
from smolagents import tool

def format_extracted_text(text):
    """Enhances extracted text readability by applying better formatting."""
    # Basic cleanup
    text = re.sub(r'\s+', ' ', text) # Removes excessive whitespace
    
    # Fix common spacing issues
    text = text.replace(" | ", "\n") # Separates sections properly
    
    # Add spacing between words that were merged during PDF extraction
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) # Insert space between camelCase
    
    # Improve section headers
    text = text.replace("Education", "\n📚 Education\n") # Highlights education section
    text = text.replace("Experience", "\n💼 Experience\n") # Highlights experience section
    text = text.replace("Skills", "\n🛠 Skills\n") # Highlights skills section
    text = text.replace("Projects", "\n🔧 Projects\n") # Highlights projects section
    
    # Fix common spacing problems
    text = text.replace(",", ", ") # Add space after commas
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text) # Space between numbers and letters
    
    return text.strip()

def extract_skills_from_text(text):
    """Uses NLP to extract potential skill keywords from resume text."""
    # Comprehensive list of technical skills
    technical_skills = [
        # Programming languages
        "python", "java", "javascript", "c++", "c#", "c", "php", "ruby", "swift", 
        "kotlin", "go", "rust", "typescript", "html", "css", "sql", "r", 
        "matlab", "scala", "perl", "bash", "shell", "powershell", "assembly",
        
        # Frameworks and libraries
        "react", "angular", "vue", "django", "flask", "spring", "hibernate",
        "node.js", "express", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "bootstrap", "jquery", "laravel", "symfony",
        
        # Databases
        "mysql", "postgresql", "mongodb", "oracle", "sql server", "sqlite",
        "cassandra", "redis", "dynamodb", "firebase", "mariadb",
        
        # Cloud platforms
        "aws", "amazon web services", "azure", "google cloud", "gcp", 
        "heroku", "digitalocean", "alibaba cloud",
        
        # DevOps
        "docker", "kubernetes", "jenkins", "git", "github", "gitlab", 
        "bitbucket", "ci/cd", "terraform", "ansible", "chef", "puppet",
        
        # Other technical
        "machine learning", "deep learning", "artificial intelligence", "ai", 
        "data science", "big data", "data mining", "data analysis",
        "nlp", "natural language processing", "computer vision",
        "object-oriented programming", "oop", "algorithms", "data structures"
    ]
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Extract words with fallback mechanism
    try:
        # Use NLTK's word_tokenize function
        words = word_tokenize(text_lower)
    except (LookupError, ImportError, AttributeError):
        # Fallback tokenization if NLTK resources aren't available
        # Simple regex to split on non-word characters
        words = re.findall(r'\b\w+\b', text_lower)
    
    # Find skills mentioned in the text
    extracted_skills = []
    
    # Single word skills
    for skill in technical_skills:
        if " " not in skill and skill in words:
            extracted_skills.append(skill)
    
    # Multi-word skills
    for skill in technical_skills:
        if " " in skill and skill in text_lower:
            extracted_skills.append(skill)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(extracted_skills))

def extract_education_section(text):
    """Extract education information from formatted CV text"""
    education = []
    
    # Try to find the education section
    education_section = None
    sections = text.split("\n\n")
    
    for i, section in enumerate(sections):
        if "📚 Education" in section:
            # Get the next section as the education content
            if i+1 < len(sections):
                education_section = sections[i+1]
                break
    
    # If found, extract details
    if education_section:
        # Split by lines and clean up
        lines = education_section.split("\n")
        for line in lines:
            if line.strip() and "Skills" not in line:
                education.append(line.strip())
    else:
        # Fallback: Look for education keywords throughout the text
        education_keywords = ["university", "college", "bachelor", "master", "phd", "degree", "gpa"]
        lines = text.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in education_keywords):
                education.append(line.strip())
    
    return education

def extract_experience_section(text):
    """Extract work experience information from formatted CV text"""
    experience = []
    
    # Try to find the experience section
    experience_section = None
    sections = text.split("\n\n")
    
    for i, section in enumerate(sections):
        if "💼 Experience" in section:
            # Get the next section as the experience content
            if i+1 < len(sections):
                experience_section = sections[i+1]
                break
    
    # If found, extract details
    if experience_section:
        # Split by lines and clean up
        lines = experience_section.split("\n")
        for line in lines:
            if line.strip() and "Education" not in line and "Skills" not in line:
                experience.append(line.strip())
    else:
        # Fallback: Look for experience keywords throughout the text
        work_keywords = ["experience", "employment", "job", "internship", "work"]
        lines = text.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in work_keywords):
                # Check if not part of a header
                if "section" not in line_lower and len(line) > 15:
                    experience.append(line.strip())
    
    # If no formal experience found, look for projects
    if not experience:
        projects_section = None
        for i, section in enumerate(sections):
            if "🔧 Projects" in section or "Projects" in section:
                # Get the next section as the projects content
                if i+1 < len(sections):
                    projects_section = sections[i+1]
                    break
        
        if projects_section:
            lines = projects_section.split("\n")
            for line in lines:
                if line.strip() and len(line) > 15:
                    experience.append("Project: " + line.strip())
    
    return experience

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
    
    raw_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Processing PDF with {len(pdf.pages)} pages...")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                raw_text += page_text + "\n"
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {str(e)}")
    
    # Format the text to be more readable
    formatted_text = format_extracted_text(raw_text)
    
    # Extract skills from the formatted text
    try:
        # First try the normal extraction
        extracted_skills = extract_skills_from_text(formatted_text)
    except Exception as e:
        # If any error occurs, use a simple fallback approach
        print(f"Warning: Skill extraction failed, using fallback method. Error: {str(e)}")
        extracted_skills = []
        text_lower = formatted_text.lower()
        technical_skills = ["python", "java", "javascript", "html", "css", "react", "node.js", 
                          "sql", "data science", "machine learning", "aws", "docker"]
        
        for skill in technical_skills:
            if skill in text_lower:
                extracted_skills.append(skill)
    
    # Parse sections with error handling
    try:
        education = extract_education_section(formatted_text)
    except Exception:
        education = []
    
    try:
        experience = extract_experience_section(formatted_text)
    except Exception:
        experience = []
    
    return {
        "text": formatted_text, 
        "skills": extracted_skills,
        "education": education,
        "experience": experience
    }
