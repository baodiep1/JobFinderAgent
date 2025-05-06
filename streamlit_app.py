import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="JobFinderAgent - Resume Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import tempfile
import logging
import sys
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('jobfinder')

# Starting log message
logger.info("Starting JobFinderAgent")

# Custom CSS to improve the UI with dark theme
st.markdown("""
<style>
    /* Base theme colors with dark background */
    :root {
        --background-color: #121212;
        --card-background: #1E1E1E;
        --primary-color: #1DB954;
        --secondary-color: #1ED760;
        --text-color: #EAEAEA;
        --light-text-color: #B3B3B3;
        --border-color: #333333;
        --accent-color: #1DB954;
    }
    
    /* Main content background */
    .main .block-container {
        background-color: var(--background-color);
        padding: 2rem;
        border-radius: 10px;
    }
    
    /* Adjust content width for better readability */
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Main title styling */
    .main-title {
        color: var(--text-color);
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: var(--light-text-color);
        font-size: 1.2rem;
        font-style: italic;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Card-like container */
    .card-container {
        background-color: var(--card-background);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
    }
    
    /* Skills tag styling */
    .skill-tag {
        display: inline-block;
        background-color: var(--primary-color);
        color: #000000;
        border-radius: 15px;
        padding: 5px 10px;
        margin: 5px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Job result styling */
    .job-result {
        background-color: var(--card-background);
        border-left: 4px solid var(--secondary-color);
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Job result title */
    .job-title {
        color: var(--secondary-color);
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
    
    /* Job source */
    .job-source {
        color: var(--light-text-color);
        font-size: 0.8rem;
        margin-bottom: 10px;
    }
    
    /* Fix formatting of extracted text */
    .extracted-text {
        white-space: pre-wrap;
        line-height: 1.6;
        font-family: 'Open Sans', sans-serif;
        color: var(--text-color);
    }
    
    /* Improved formatting for resume content */
    .resume-content {
        letter-spacing: 0.02em;
        word-spacing: 0.1em;
    }
    
    /* Logo styling */
    .logo-text {
        font-family: 'Arial', sans-serif;
        font-weight: 800;
        color: var(--primary-color);
        font-size: 1.2rem;
        letter-spacing: 1px;
    }
    
    /* Card divider */
    hr.card-divider {
        margin: 15px 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(29, 185, 84, 0.75), rgba(0, 0, 0, 0));
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: var(--light-text-color);
        font-size: 0.8rem;
        margin-top: 50px;
        padding-top: 10px;
        border-top: 1px solid var(--border-color);
    }
    
    /* Layout fixes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--background-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        background-color: #2A2A2A;
        color: var(--text-color);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: var(--card-background);
        border-radius: 0 0 10px 10px;
        padding: 15px;
        border: 1px solid var(--border-color);
        border-top: none;
    }
    
    /* Improve spacing between words in extracted text */
    .fixed-spaces {
        letter-spacing: 0.02em;
        word-spacing: 0.12em;
    }
    
    /* Better file uploader styling */
    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        background-color: #2A2A2A;
        border: 2px dashed var(--primary-color);
        border-radius: 10px;
        padding: 30px 20px;
        text-align: center;
    }
    
    .stFileUploader div[data-testid="stFileUploadDropzone"]:hover {
        background-color: #333333;
        border-color: var(--secondary-color);
    }
    
    /* Improve button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: black;
        font-weight: 600;
        border-radius: 5px;
        border: none;
        padding: 0.5em 1em;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Enhance expander styling */
    .streamlit-expanderHeader {
        background-color: #2A2A2A;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #333333;
    }
    
    /* Improve sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1A1A1A;
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Fix text input styling */
    div[data-baseweb="input"] {
        background-color: #2A2A2A;
        border-radius: 5px;
        border: 1px solid #333333;
    }
    
    div[data-baseweb="input"] input {
        color: var(--text-color);
    }
    
    /* Improve text formatting for specific sections */
    .formatted-edu-exp {
        line-height: 1.6;
        padding: 5px 0;
        word-spacing: 0.1em;
    }
    
    /* Better section styling */
    .section-content {
        padding: 5px 10px;
        margin-bottom: 10px;
    }
    
    /* Add spaces between words in specific sections */
    .add-word-spacing {
        word-spacing: 0.15em;
    }
    
    /* Info message styling */
    .info-message {
        background-color: #1A3741;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        color: #90CAF9;
    }
    
    /* Warning message styling */
    .warning-message {
        background-color: #3D3223;
        border-left: 4px solid #FFC107;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        color: #FFECB3;
    }
    
    /* Improve tab panel styling for dark theme */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1E1E1E;
        color: var(--text-color);
    }
    
    /* Fix header colors */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
    }
    
    /* Fix paragraph text colors */
    p {
        color: var(--text-color);
    }
    
    /* Markdown text color fix */
    .stMarkdown {
        color: var(--text-color);
    }
    
    /* Fix expander content background */
    .streamlit-expanderContent {
        background-color: var(--card-background);
        border-radius: 0 0 5px 5px;
    }
    
    /* Fix select box styling */
    div[data-baseweb="select"] {
        background-color: #2A2A2A;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #2A2A2A;
        color: var(--text-color);
    }
    
    /* Fixing checkbox style */
    label[data-baseweb="checkbox"] {
        color: var(--text-color);
    }
    
    /* Fix stDataFrame backgrounds */
    .stDataFrame {
        background-color: var(--card-background);
    }
    
    .stDataFrame th {
        background-color: #2A2A2A;
        color: var(--text-color);
    }
    
    .stDataFrame td {
        color: var(--text-color);
    }
    
    /* Radio buttons fix */
    .stRadio label {
        color: var(--text-color);
    }
    
    /* Fix code blocks */
    .stCodeBlock {
        background-color: #2A2A2A;
    }
    
    /* Fix json display */
    .element-container pre {
        background-color: #2A2A2A;
        color: var(--text-color);
    }
    
    /* Success message styling */
    .success-message {
        background-color: #1A3D29;
        border-left: 4px solid var(--primary-color);
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        color: #A5D6A7;
    }
    
    /* Remove white background from various components */
    .stTextInput, .stTextArea, .stSelectbox, .stMultiselect, .stDateInput {
        background-color: transparent !important;
    }
    
    .stTextInput > div, .stTextArea > div, .stSelectbox > div, .stMultiselect > div, .stDateInput > div {
        background-color: #2A2A2A !important;
    }
    
    /* Fix for main container background */
    .main .block-container {
        background-color: #121212 !important;
    }
    
    /* Fix modal backgrounds if any */
    div[data-modal-container="true"] > div {
        background-color: var(--card-background);
    }
    
    /* Fix iframe background if any */
    iframe {
        background-color: var(--card-background);
    }
</style>
""", unsafe_allow_html=True)

# Create a data directory in a writable location for NLTK
nltk_data_dir = Path("./nltk_data")
nltk_data_dir.mkdir(exist_ok=True)
nltk.data.path.append(str(nltk_data_dir))

# Display a loading message while SpaCy loads
loading_message = st.empty()
loading_message.info("Loading NLP models... This may take a moment.")

# Force reload numpy to ensure compatibility
st.markdown("### Setting up environment...")
try:
    import numpy
    importlib.reload(numpy)
    logger.info(f"NumPy version: {numpy.__version__}")
    st.write(f"NumPy version: {numpy.__version__}")
except Exception as e:
    logger.error(f"Error reloading NumPy: {str(e)}")

# Now try to import tools
try:
    from tools.pdf_extractor import extract_text_from_pdf
    from tools.google_search import search_google_jobs
    logger.info("Tools imported successfully")
except Exception as e:
    logger.error(f"Error importing tools: {str(e)}")
    st.error(f"Error importing tools: {str(e)}")

# Import and load SpaCy
logger.info("About to load SpaCy model")
try:
    import spacy
    # Try loading the model with detailed error reporting
    try:
        nlp = spacy.load("en_core_web_sm")
        loading_message.empty()  # Clear the loading message when done
        logger.info("SpaCy model loaded successfully")
    except OSError as e:
        logger.error(f"Error loading SpaCy model: {str(e)}")
        loading_message.error(f"Error loading SpaCy model: {str(e)}")
        
        # Add fallback with minimal NLP functionality
        logger.info("Creating blank SpaCy model as fallback")
        nlp = spacy.blank("en")
        loading_message.warning("Using simplified language model. Some features may be limited.")
        
except Exception as e:
    logger.error(f"Unable to load SpaCy: {str(e)}")
    loading_message.error(f"Unable to load SpaCy: {str(e)}")
    
    # Ultimate fallback - create a dummy NLP function to avoid halting the app
    class DummyNLP:
        def __call__(self, text):
            class DummyDoc:
                def __init__(self, text):
                    self.text = text
                    self.ents = []
            return DummyDoc(text)
    
    nlp = DummyNLP()
    st.warning("NLP features are unavailable. App will run with limited functionality.")
    logger.warning("Using dummy NLP function")

# Download necessary NLTK data with better error handling
logger.info("Checking NLTK resources")
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    logger.info("NLTK resources found")
except LookupError:
    logger.info("NLTK resources not found, downloading...")
    with st.spinner("Downloading language resources..."):
        nltk.download('punkt', download_dir=str(nltk_data_dir))
        nltk.download('stopwords', download_dir=str(nltk_data_dir))
    logger.info("NLTK resources downloaded")

def check_requirements():
    """Check if all required components are available"""
    try:
        # Check SpaCy
        import spacy
        try:
            models = spacy.util.get_installed_models()
            st.sidebar.success(f"✅ SpaCy installed. Models: {', '.join(models)}")
        except:
            st.sidebar.warning("⚠️ SpaCy installed but couldn't list models")
        
        # Check NLTK
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            st.sidebar.success("✅ NLTK data available")
        except LookupError:
            st.sidebar.warning("⚠️ NLTK data missing, will download")
        
        # Check PDF processing
        import pdfplumber
        st.sidebar.success("✅ PDF processing available")
        
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Requirements check failed: {str(e)}")
        return False

def extract_skills_from_cv(cv_text):
    """Extract skills from CV text using NLP"""
    technical_skills = [
        # Programming languages
        "python", "java", "javascript", "c++", "c#", "php", "ruby", "swift", 
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
        "blockchain", "iot", "internet of things", "augmented reality", "ar",
        "virtual reality", "vr", "cybersecurity", "network security",
        "full stack", "frontend", "backend", "mobile development",
        "web development", "object-oriented programming", "oop", "agile",
        "scrum", "kanban", "tdd", "test driven development",
        "linux", "unix", "windows", "macos", "android", "ios",
        "api", "rest", "graphql", "soap", "microservices", "serverless",
        "hadoop", "spark", "kafka", "ethereum", "solidity", "cryptography"
    ]
    
    # Prepare text
    cv_text_lower = cv_text.lower()
    
    # Extract skills - simpler algorithm with fallback options
    skills = set()
    
    try:
        # Tokenize the text
        tokens = word_tokenize(cv_text_lower)
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        except:
            # Fallback if stopwords fail
            logger.warning("Stopwords failed, using basic filtering")
            filtered_tokens = [token for token in tokens if len(token) > 2]
    except:
        # Fallback if tokenization fails
        logger.warning("Tokenization failed, using simple split")
        filtered_tokens = [word for word in cv_text_lower.split() if len(word) > 2]
    
    # Direct matching with our predefined list (will work even with dummy NLP)
    for skill in technical_skills:
        if skill in cv_text_lower or any(skill == token for token in filtered_tokens):
            skills.add(skill)
    
    # Only try SpaCy features if we have a real NLP model
    if not isinstance(nlp, DummyNLP):
        try:
            # Process with SpaCy for more advanced extraction
            doc = nlp(cv_text)
            
            # Using SpaCy's entity recognition for additional skills
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 2:
                    # Check if this organization/product could be a technology or tool
                    potential_skill = ent.text.lower()
                    if potential_skill in technical_skills:
                        skills.add(potential_skill)
        except Exception as e:
            logger.warning(f"SpaCy processing failed: {str(e)}")
    
    # Extract multi-word technical skills like "machine learning"
    for skill in technical_skills:
        if " " in skill:
            if skill in cv_text_lower:
                skills.add(skill)
    
    # Look for sections that might contain skills
    try:
        sections = re.split(r'\n\s*\n', cv_text)
        for section in sections:
            section_lower = section.lower()
            if any(header in section_lower for header in ["skill", "technical", "technology", "tools", "proficiency"]):
                lines = section.split('\n')
                for line in lines:
                    words = line.split()
                    for word in words:
                        word_clean = word.lower().strip(',:;()[]{}').strip()
                        if word_clean in technical_skills and len(word_clean) > 2:
                            skills.add(word_clean)
    except Exception as e:
        logger.warning(f"Section extraction failed: {str(e)}")
    
    return list(skills)

def extract_location_from_cv(cv_text, nlp_model):
    """Extract location information from CV text using NLP"""
    # Default location if nothing is found
    default_location = "United States"
    
    # Simple extraction with fallbacks
    try:
        # Look for standard location pattern "City, State" in the text
        # This pattern matches "City, ST" format where ST is a two-letter state code
        city_state_pattern = re.compile(r'([A-Za-z\s]+),\s*([A-Z]{2})')
        matches = city_state_pattern.findall(cv_text)
        
        if matches:
            # Found a match like "City, State"
            before_comma, state = matches[0]
            return f"{before_comma.strip()}, {state}"
        
        # Check if we have a real NLP model
        if not isinstance(nlp_model, DummyNLP):
            # Process the text with SpaCy for NER
            doc = nlp_model(cv_text)
            
            # Find potential locations using SpaCy's entity recognition
            locations = []
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or location
                    locations.append(ent.text)
            
            if locations:
                return locations[0]
    except Exception as e:
        logger.warning(f"Location extraction error: {str(e)}")
        
    # If all else fails, return default
    return default_location

def clean_location_for_search(location_input):
    """Clean up location format for optimal searching"""
    # Handle empty case
    if not location_input:
        return "United States"
    
    # Remove any extra spaces
    location_input = location_input.strip()
    
    # Handle "City, State" format with potential extra words
    if "," in location_input:
        parts = location_input.split(",")
        if len(parts) == 2:
            city_part = parts[0].strip()
            state_part = parts[1].strip()
            
            # Check if the city part has multiple words
            city_words = city_part.split()
            if len(city_words) > 1:
                # Take just the last word as the likely city name
                # This handles cases where name might be included before city
                city = city_words[-1]
                return f"{city}, {state_part}"
            else:
                # Simple City, State format - just clean up spaces
                return f"{city_part}, {state_part}"
    
    # Return as is if doesn't match expected format
    return location_input

def generate_search_query(skills, education, experience):
    """Generate a relevant search query based on CV analysis"""
    # Extract potential job titles from experience
    job_titles = []
    common_titles = ["developer", "engineer", "programmer", "analyst", "scientist", 
                    "designer", "manager", "consultant", "specialist", "technician"]
    
    for exp in experience:
        exp_lower = exp.lower()
        for title in common_titles:
            if title in exp_lower:
                # Extract phrases that might be job titles
                matches = re.findall(r'([A-Za-z]+\s+' + title + r'|' + title + r'\s+[A-Za-z]+)', exp_lower)
                if matches:
                    job_titles.extend(matches)
                else:
                    job_titles.append(title)
    
    # Default job titles if none found
    if not job_titles:
        # Extract from education to determine field
        edu_text = " ".join(education).lower()
        if "computer" in edu_text and "science" in edu_text:
            job_titles = ["software developer", "programmer"]
        elif "data" in edu_text:
            job_titles = ["data analyst", "data scientist"]
        elif "design" in edu_text:
            job_titles = ["ui designer", "ux designer"]
        # Extract from skills
        elif any(lang in skills for lang in ["python", "java", "javascript", "c++", "c#"]):
            job_titles = ["software developer", "programmer"]
        elif "data" in " ".join(skills).lower():
            job_titles = ["data analyst", "data scientist"]
        elif any(design in skills for design in ["ui", "ux", "design"]):
            job_titles = ["ui designer", "ux designer"]
        else:
            job_titles = ["entry level", "graduate"]
    
    # Select top job title and skills
    primary_job = job_titles[0] if job_titles else "entry level"
    top_skills = skills[:3] if len(skills) >= 3 else skills
    
    # Create query combining job title and skills
    query = f"{primary_job} jobs {' '.join(top_skills)}"
    
    return query

# Function to format resume text - fixing spacing and formatting issues
def format_resume_text(text):
    """Improve resume text formatting by adding proper spacing between words"""
    # Fix words that are run together
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Add space after colon if not present
    text = re.sub(r':([A-Za-z])', r': \1', text)
    
    # Add space after comma if not present
    text = re.sub(r',([A-Za-z])', r', \1', text)
    
    # Fix specific formatting issues commonly found in resumes
    text = text.replace("GPA:", "GPA: ")
    text = text.replace("Skills:", "Skills: ")
    text = text.replace("Experience:", "Experience: ")
    text = text.replace("Education:", "Education: ")
    
    # Space between number and word
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    
    return text

def run_streamlit_app():
    # Add a success message to confirm the app is running
    st.sidebar.success("App loaded successfully!")
    
    # Enhanced title and introduction with professional header
    header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
    
    with header_col2:
        st.markdown('''
        <div style="display: flex; align-items: center; justify-content: center; margin: 2rem 0;">
            <div style="background-color: #1DB954; color: black; width: 60px; height: 60px; border-radius: 50%; 
                     display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 2rem;">
                📝
            </div>
            <div>
                <h1 class="main-title">JobFinderAgent</h1>
                <p class="subtitle">Resume Analysis & Job Matching Platform</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Introduction card with better styling
    st.markdown('''
    <div class="card-container" style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            <span style="color: #1DB954; font-weight: 600;">Intelligent resume parsing and job search</span> powered by Natural Language Processing
        </p>
        <p>Upload your resume to extract skills, experience, and education, then find matching job opportunities</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Initialize session state
    if 'alt_search' not in st.session_state:
        st.session_state.alt_search = None
    if 'alt_location' not in st.session_state:
        st.session_state.alt_location = None
    
    # Sidebar for API key
    st.sidebar.title("Settings")
    
    # Get API key from secrets if available, otherwise use input field
    api_key = None
    try:
        # Try to get API key from secrets
        api_key = st.secrets["serpapi"]["key"]
        st.sidebar.success("✅ SerpAPI key loaded from secrets")
        logger.info("SerpAPI key loaded from secrets")
    except Exception as e:
        logger.warning(f"Could not load SerpAPI key from secrets: {str(e)}")
        # Fall back to user input if secrets are not available
        api_key = st.sidebar.text_input("SerpAPI Key", type="password")
        if api_key:
            st.sidebar.success("✅ SerpAPI key provided")
    
    if api_key:
        os.environ["SERPAPI_KEY"] = api_key
    
    # Debug mode toggle - default to True for troubleshooting
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Check requirements if in debug mode
    if debug_mode:
        requirements_ok = check_requirements()
        if not requirements_ok:
            st.warning("Some requirements failed the check. App may not function correctly.")
    
    # Enhanced file upload section with better styling
    st.markdown('<h2 class="section-header">Upload Your Resume</h2>', unsafe_allow_html=True)
    
    upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])
    
    with upload_col2:
        # Add instructions above the file uploader
        st.markdown('''
        <div style="text-align: center; margin-bottom: 1rem;">
            <p>Upload your resume in PDF format to analyze skills and find matching jobs</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Custom styled file uploader
        uploaded_file = st.file_uploader("Upload your resume/CV (PDF format)", type="pdf")
    
    if uploaded_file:
        # Save uploaded file to a temporary file with better error handling
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            
            logger.info(f"Uploaded file saved to temporary path: {temp_path}")
            
            # Extract text
            try:
                # Get result from PDF extractor - now a dictionary with multiple keys
                logger.info("Extracting text from PDF")
                pdf_result = extract_text_from_pdf(temp_path)
                
                # Extract the actual text string from the dictionary
                cv_text = pdf_result['text']
                logger.info(f"Extracted {len(cv_text)} characters from PDF")
                
                # Format the extracted text for better readability
                formatted_cv_text = format_resume_text(cv_text)
                
                # Display extracted text in a better formatted way
                with st.expander("Extracted CV Text"):
                    st.markdown(f'<div class="card-container extracted-text fixed-spaces resume-content">{formatted_cv_text}</div>', 
                              unsafe_allow_html=True)
                
                # Process the CV
                with st.spinner("Analyzing your resume..."):
                    logger.info("Starting CV analysis")
                    # Use pre-extracted information if available
                    if 'skills' in pdf_result and pdf_result['skills']:
                        skills = pdf_result['skills']
                        logger.info(f"Using {len(skills)} pre-extracted skills")
                    else:
                        logger.info("Extracting skills from CV text")
                        skills = extract_skills_from_cv(cv_text)
                        logger.info(f"Extracted {len(skills)} skills")
                    
                    if 'education' in pdf_result and pdf_result['education']:
                        education = pdf_result['education']
                        logger.info(f"Using {len(education)} pre-extracted education items")
                    else:
                        education = []
                        logger.info("No education information extracted")
                    
                    if 'experience' in pdf_result and pdf_result['experience']:
                        experience = pdf_result['experience']
                        logger.info(f"Using {len(experience)} pre-extracted experience items")
                    else:
                        experience = []
                        logger.info("No experience information extracted")
                    
                    # Extract and clean location using NLP
                    logger.info("Extracting location from CV")
                    raw_location = extract_location_from_cv(cv_text, nlp)
                    location = clean_location_for_search(raw_location)
                    logger.info(f"Extracted location: {location}")
                    
                    # Display extracted information in a tabbed interface with better formatting
                    st.markdown('<h2 class="section-header">Extracted Information</h2>', unsafe_allow_html=True)
                    
                    # Create tabs for different sections of information
                    info_tabs = st.tabs(["📊 Overview", "🛠️ Skills", "📚 Education", "💼 Experience"])
                    
                    with info_tabs[0]:  # Overview tab
                        st.markdown('<div class="card-container">', unsafe_allow_html=True)
                        
                        # Create a clean overview layout
                        overview_col1, overview_col2 = st.columns([1, 1])
                        
                        with overview_col1:
                            st.markdown('<p style="color:#1DB954; font-weight:600; font-size:1.1rem;">Location</p>', unsafe_allow_html=True)
                            st.markdown(f'<p style="font-size:1.2rem;">{location}</p>', unsafe_allow_html=True)
                            
                            st.markdown('<p style="color:#1DB954; font-weight:600; font-size:1.1rem; margin-top:20px;">Skills Summary</p>', unsafe_allow_html=True)
                            # Display top skills as tags with space between
                            skill_tags_html = ""
                            for skill in skills[:6]:  # Show top 6 skills in overview
                                skill_tags_html += f'<span class="skill-tag">{skill}</span> '
                            st.markdown(f'<div style="margin-bottom:20px;">{skill_tags_html}</div>', unsafe_allow_html=True)
                        
                        with overview_col2:
                            st.markdown('<p style="color:#1DB954; font-weight:600; font-size:1.1rem;">Education Summary</p>', unsafe_allow_html=True)
                            if education:
                                # Format education summary with proper spacing
                                edu_summary = format_resume_text(education[0][:100])
                                st.markdown(f'<p class="add-word-spacing">{edu_summary}...</p>', unsafe_allow_html=True)
                            else:
                                st.markdown('<p>No education details detected</p>', unsafe_allow_html=True)
                            
                            st.markdown('<p style="color:#1DB954; font-weight:600; font-size:1.1rem; margin-top:20px;">Experience Summary</p>', unsafe_allow_html=True)
                            if experience:
                                # Format experience summary with proper spacing
                                exp_summary = format_resume_text(experience[0][:100])
                                st.markdown(f'<p class="add-word-spacing">{exp_summary}...</p>', unsafe_allow_html=True)
                            else:
                                st.markdown('<p>No experience details detected</p>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with info_tabs[1]:  # Skills tab
                        st.markdown('<div class="card-container">', unsafe_allow_html=True)
                        
                        # Display skills as tags
                        if skills:
                            skill_tags_html = ""
                            for skill in skills:
                                skill_tags_html += f'<span class="skill-tag">{skill}</span> '
                            st.markdown(f'<div style="margin: 10px 0;">{skill_tags_html}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p>No skills detected in the resume</p>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with info_tabs[2]:  # Education tab
                        st.markdown('<div class="card-container">', unsafe_allow_html=True)
                        
                        if education:
                            for edu in education:
                                # Format education with proper spacing
                                formatted_edu = format_resume_text(edu)
                                st.markdown(f'<div class="formatted-edu-exp section-content add-word-spacing" style="margin-bottom:15px; padding-bottom:10px; border-bottom:1px solid #333333;">{formatted_edu}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p>No education details detected in the resume</p>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with info_tabs[3]:  # Experience tab
                        st.markdown('<div class="card-container">', unsafe_allow_html=True)
                        
                        if experience:
                            for exp in experience:
                                # Format experience with proper spacing
                                formatted_exp = format_resume_text(exp)
                                st.markdown(f'<div class="formatted-edu-exp section-content add-word-spacing" style="margin-bottom:15px; padding-bottom:10px; border-bottom:1px solid #333333;">{formatted_exp}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p>No experience details detected in the resume</p>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Generate search query
                    logger.info("Generating search query")
                    default_query = generate_search_query(skills, education, experience)
                    logger.info(f"Generated query: {default_query}")
                
                # Search query input - use alternative search if set
                search_query = st.session_state.alt_search if st.session_state.alt_search else default_query
                location_input = st.session_state.alt_location if st.session_state.alt_location else location
                
                # Reset alternative search after using it
                st.session_state.alt_search = None
                st.session_state.alt_location = None
                
                # Improved search parameters section with better styling
                st.markdown('<h2 class="section-header">Search Parameters</h2>', unsafe_allow_html=True)
                
                # Wrap in a card container for better visual appearance
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                
                search_col1, search_col2 = st.columns([3, 1])
                
                with search_col1:
                    search_query = st.text_input("Search Query", value=search_query, placeholder="Enter job search keywords...", 
                                               help="Specify job title and skills to search for")
                
                with search_col2:
                    location_input = st.text_input("Location", value=location_input, placeholder="City, State or Country",
                                                 help="Enter the location to search for jobs")
                
                # Better styled find jobs button
                search_button_col1, search_button_col2, search_button_col3 = st.columns([1, 1, 1])
                with search_button_col2:
                    find_jobs_button = st.button("🔍 Find Jobs", use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if find_jobs_button:
                    if not api_key:
                        st.markdown('<div class="warning-message">Please enter your SerpAPI key in the sidebar</div>', unsafe_allow_html=True)
                    else:
                        with st.spinner("Searching for jobs..."):
                            try:
                                # Clean location for search
                                search_location = clean_location_for_search(location_input)
                                
                                # Log the search parameters for debugging
                                if debug_mode:
                                    st.markdown(f'<div class="info-message">Searching for: \'{search_query}\' in \'{search_location}\'</div>', unsafe_allow_html=True)
                                
                                logger.info(f"Searching for jobs with query: '{search_query}' in '{search_location}'")
                                
                                # Execute the search
                                results = search_google_jobs(query=search_query, location=search_location)
                                
                                logger.info(f"Search completed, found results: {bool(results)}")
                                
                                # Display debug info if enabled
                                if debug_mode and results:
                                    with st.expander("Debug: Raw Search Results"):
                                        st.json(results)
                                
                                # Display results with improved styling
                                st.markdown('<h2 class="section-header">Job Listings</h2>', unsafe_allow_html=True)
                                
                                # Check for errors in the search response
                                if "error" in results:
                                    logger.error(f"Search error: {results['error']}")
                                    st.markdown(f'<div class="warning-message"><strong>Search Error:</strong> {results["error"]}<br>Please try modifying your search query or location.</div>', unsafe_allow_html=True)
                                    
                                    # Suggest fixes for location issues
                                    if "location" in results.get("error", "").lower():
                                        st.warning("Location format may be causing the issue.")
                                        st.markdown('<div class="card-container" style="border-left: 4px solid #FFC107; padding: 15px;"><p style="font-weight: 600;">Try these location formats instead:</p></div>', unsafe_allow_html=True)
                                        
                                        # Extract just city if location contains comma
                                        if "," in search_location:
                                            city = search_location.split(",")[0].strip()
                                            state = search_location.split(",")[1].strip() if len(search_location.split(",")) > 1 else ""
                                            
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                if st.button(f"Use '{city}' only"):
                                                    st.session_state.alt_location = city
                                                    st.experimental_rerun()
                                            with col2:
                                                if state and st.button(f"Use '{state}' only"):
                                                    st.session_state.alt_location = state
                                                    st.experimental_rerun()
                                            with col3:
                                                if st.button("Use 'United States'"):
                                                    st.session_state.alt_location = "United States"
                                                    st.experimental_rerun()
                                        else:
                                            if st.button("Use 'United States'"):
                                                st.session_state.alt_location = "United States"
                                                st.experimental_rerun()
                                
                                # Process the results
                                elif "organic_results" in results and results["organic_results"]:
                                    logger.info(f"Processing {len(results['organic_results'])} organic results")
                                    jobs = results["organic_results"][:8] # Top 8 results
                                    
                                    job_count = 0
                                    for i, job in enumerate(jobs, 1):
                                        # Filter out non-job listing results
                                        title = job.get('title', '').lower()
                                        link = job.get('link', '').lower()
                                        snippet = job.get('snippet', '').lower()
                                        
                                        # Check if result is likely a job posting
                                        is_job = (
                                            'job' in title or 'career' in title or 'position' in title or
                                            'developer' in title or 'engineer' in title or 'programmer' in title or
                                            'linkedin.com/jobs' in link or 'indeed.com/job' in link or 
                                            'glassdoor.com/job' in link or 'apply' in snippet
                                        )
                                        
                                        if is_job or debug_mode: # Show all results in debug mode
                                            job_count += 1
                                            # Enhanced job listing display with card styling
                                            st.markdown(f'''
                                            <div class="job-result">
                                                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                                    <div style="background-color: #1DB954; color: black; border-radius: 50%; width: 30px; height: 30px; 
                                                               display: flex; align-items: center; justify-content: center; margin-right: 15px; font-weight: 600;">
                                                        {job_count}
                                                    </div>
                                                    <div class="job-title">{job.get('title', 'Unknown Title')}</div>
                                                </div>
                                                <div class="job-source">Source: {job.get('source', 'Unknown')}</div>
                                                <p style="margin-bottom: 15px; line-height: 1.5;">{job.get('snippet', 'No description')[:200]}...</p>
                                                <a href="{job.get('link', '#')}" target="_blank" style="display: inline-block; background-color: #1DB954; 
                                                   color: black; padding: 5px 15px; border-radius: 4px; text-decoration: none; font-weight: 600;">
                                                   View Job Listing →
                                                </a>
                                            </div>
                                            ''', unsafe_allow_html=True)
                                    
                                    if job_count == 0:
                                        logger.warning("Found search results, but none appear to be job listings")
                                        st.markdown('<div class="warning-message">Found search results, but none appear to be job listings. Showing all results:</div>', unsafe_allow_html=True)
                                        # Show all results regardless of job filtering
                                        for i, job in enumerate(jobs, 1):
                                            st.markdown(f'''
                                            <div class="job-result">
                                                <div class="job-title">{job.get('title', 'Unknown Title')}</div>
                                                <div class="job-source">Source: {job.get('source', 'Unknown')}</div>
                                                <p style="margin-bottom: 15px; line-height: 1.5;">{job.get('snippet', 'No description')[:200]}...</p>
                                                <a href="{job.get('link', '#')}" target="_blank" style="display: inline-block; background-color: #1DB954; 
                                                   color: black; padding: 5px 15px; border-radius: 4px; text-decoration: none; font-weight: 600;">
                                                   View Link →
                                                </a>
                                            </div>
                                            ''', unsafe_allow_html=True)
                                else:
                                    logger.warning("No job listings found in search results")
                                    st.markdown('<div class="warning-message">No job listings found. Try these suggestions:</div>', unsafe_allow_html=True)
                                    
                                    # Improved suggestions box
                                    st.markdown('''
                                    <div class="card-container" style="border-left: 4px solid #FFC107; padding: 15px;">
                                        <p style="margin-bottom: 10px; font-weight: 600;">Try these approaches to improve your search:</p>
                                        <ol style="margin-left: 20px; margin-bottom: 15px; line-height: 1.6;">
                                            <li>Simplify your search query (e.g., 'software developer' instead of using multiple languages)</li>
                                            <li>Try a broader location (e.g., just the state name or 'United States')</li>
                                            <li>Check different job titles (e.g., 'programmer' or 'software engineer')</li>
                                        </ol>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                    
                                    # Provide alternative search buttons
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        if st.button("Try 'Entry Level Developer'"):
                                            st.session_state.alt_search = "Entry Level Developer"
                                            st.experimental_rerun()
                                    with col2:
                                        if st.button("Try State-wide Search"):
                                            # Extract state if location has comma
                                            if "," in search_location:
                                                state = search_location.split(",")[1].strip()
                                                st.session_state.alt_location = state
                                            else:
                                                st.session_state.alt_location = "United States"
                                            st.experimental_rerun()
                                    with col3:
                                        if st.button("Use Broader Location"):
                                            st.session_state.alt_location = "United States"
                                            st.experimental_rerun()
                            
                            except Exception as e:
                                logger.error(f"Error during job search: {str(e)}")
                                st.markdown(f'<div class="warning-message"><strong>Error during job search:</strong> {str(e)}<br>Please try again with a different search query or check your API key.</div>', unsafe_allow_html=True)
                                
                                # Display traceback in debug mode
                                if debug_mode:
                                    import traceback
                                    error_details = traceback.format_exc()
                                    logger.error(f"Detailed error: {error_details}")
                                    st.expander("Debug: Error Details").code(error_details)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logger.info(f"Temporary file removed: {temp_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file: {str(e)}")
                        if debug_mode:
                            st.warning(f"Could not remove temporary file: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                st.markdown(f'<div class="warning-message"><strong>Error processing PDF:</strong> {str(e)}</div>', unsafe_allow_html=True)
                
                # Display traceback in debug mode
                if debug_mode:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Detailed error: {error_details}")
                    st.expander("Debug: Error Details").code(error_details)
        
        except Exception as e:
            logger.error(f"Error handling uploaded file: {str(e)}")
            st.markdown(f'<div class="warning-message"><strong>Error handling uploaded file:</strong> {str(e)}</div>', unsafe_allow_html=True)
            if debug_mode:
                import traceback
                error_details = traceback.format_exc()
                logger.error(f"Detailed error: {error_details}")
                st.expander("Debug: File Handling Error").code(error_details)
    
    else:
        # Improved empty state message
        st.markdown('''
        <div class="card-container" style="text-align: center; padding: 30px; margin: 30px 0; background-color: #1E1E1E;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>
            <h3 style="margin-bottom: 1rem; color: #1DB954;">Ready to Find Your Next Job?</h3>
            <p style="margin-bottom: 1.5rem;">Upload your resume (PDF format) to analyze your skills and find matching job opportunities</p>
            <div style="color: #B3B3B3; font-size: 0.9rem;">Your data remains private and is not stored permanently</div>
        </div>
        ''', unsafe_allow_html=True)

# DummyNLP class definition for global scope
class DummyNLP:
    def __call__(self, text):
        class DummyDoc:
            def __init__(self, text):
                self.text = text
                self.ents = []
        return DummyDoc(text)

if __name__ == "__main__":
    try:
        run_streamlit_app()
        logger.info("App completed running successfully")
        
        # Add a professional footer
        st.markdown('''
        <div class="footer">
            <p>JobFinderAgent - Resume Analysis & Job Matching Platform</p>
            <p>Powered by Streamlit, NLP, and SerpAPI</p>
            <p>© 2025 JobFinderAgent</p>
        </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        logger.critical(f"Critical app error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        if "debug_mode" in locals() and debug_mode:
            import traceback
            error_details = traceback.format_exc()
            logger.critical(f"Detailed error: {error_details}")
            st.expander("Debug: Critical Error Details").code(error_details)