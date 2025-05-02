import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(page_title="JobFinderAgent", layout="wide")

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

def run_streamlit_app():
    # Add a success message to confirm the app is running
    st.sidebar.success("App loaded successfully!")
    
    # Display title and introduction - st.set_page_config is already called at the top
    st.title("JobFinderAgent - Resume Analysis & Job Matching Platform")
    st.markdown("*Intelligent resume parsing and job search powered by NLP*")
    
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
    
    # File upload
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
                
                # Display extracted text
                with st.expander("Extracted CV Text"):
                    st.text(cv_text)
                
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
                    
                    # Display extracted information
                    st.subheader("Extracted Information")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**Skills:**")
                        st.write(", ".join(skills) if skills else "No skills detected")
                        
                        st.markdown("**Education:**")
                        for edu in education:
                            st.write(f"- {edu}")
                        if not education:
                            st.write("No education details detected")
                    
                    with col2:
                        st.markdown("**Experience:**")
                        for exp in experience:
                            st.write(f"- {exp}")
                        if not experience:
                            st.write("No experience details detected")
                    
                    with col3:
                        st.markdown("**Location:**")
                        st.write(location)
                    
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
                
                st.subheader("Search Parameters")
                search_query = st.text_input("Search Query", value=search_query)
                location_input = st.text_input("Location", value=location_input)
                
                if st.button("Find Jobs"):
                    if not api_key:
                        st.error("Please enter your SerpAPI key in the sidebar")
                    else:
                        with st.spinner("Searching for jobs..."):
                            try:
                                # Clean location for search
                                search_location = clean_location_for_search(location_input)
                                
                                # Log the search parameters for debugging
                                if debug_mode:
                                    st.info(f"Searching for: '{search_query}' in '{search_location}'")
                                
                                logger.info(f"Searching for jobs with query: '{search_query}' in '{search_location}'")
                                
                                # Execute the search
                                results = search_google_jobs(query=search_query, location=search_location)
                                
                                logger.info(f"Search completed, found results: {bool(results)}")
                                
                                # Display debug info if enabled
                                if debug_mode and results:
                                    with st.expander("Debug: Raw Search Results"):
                                        st.json(results)
                                
                                # Display results
                                st.subheader("Job Listings")
                                
                                # Check for errors in the search response
                                if "error" in results:
                                    logger.error(f"Search error: {results['error']}")
                                    st.error(f"Search Error: {results['error']}")
                                    st.write("Please try modifying your search query or location.")
                                    
                                    # Suggest fixes for location issues
                                    if "location" in results.get("error", "").lower():
                                        st.warning("Location format may be causing the issue.")
                                        st.write("Try these location formats instead:")
                                        
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
                                            with st.container():
                                                col1, col2 = st.columns([1, 3])
                                                with col1:
                                                    st.write(f"**#{job_count}**")
                                                
                                                with col2:
                                                    st.write(f"**{job.get('title', 'Unknown Title')}**")
                                                    st.write(f"Source: {job.get('source', 'Unknown')}")
                                                    st.write(f"Description: {job.get('snippet', 'No description')[:200]}...")
                                                    if "link" in job:
                                                        st.write(f"[View Job Listing]({job['link']})")
                                                
                                                st.divider()
                                    
                                    if job_count == 0:
                                        logger.warning("Found search results, but none appear to be job listings")
                                        st.warning("Found search results, but none appear to be job listings. Showing all results:")
                                        # Show all results regardless of job filtering
                                        for i, job in enumerate(jobs, 1):
                                            with st.container():
                                                st.write(f"**{job.get('title', 'Unknown Title')}**")
                                                st.write(f"Source: {job.get('source', 'Unknown')}")
                                                st.write(f"Description: {job.get('snippet', 'No description')[:200]}...")
                                                if "link" in job:
                                                    st.write(f"[View Link]({job['link']})")
                                                st.divider()
                                else:
                                    logger.warning("No job listings found in search results")
                                    st.warning("No job listings found. Try these suggestions:")
                                    st.write("1. Simplify your search query (e.g., 'software developer' instead of using multiple languages)")
                                    st.write("2. Try a broader location (e.g., just the state name or 'United States')")
                                    st.write("3. Check different job titles (e.g., 'programmer' or 'software engineer')")
                                    
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
                                st.error(f"Error during job search: {str(e)}")
                                st.write("Please try again with a different search query or check your API key.")
                                
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
                st.error(f"Error processing PDF: {str(e)}")
                
                # Display traceback in debug mode
                if debug_mode:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Detailed error: {error_details}")
                    st.expander("Debug: Error Details").code(error_details)
        
        except Exception as e:
            logger.error(f"Error handling uploaded file: {str(e)}")
            st.error(f"Error handling uploaded file: {str(e)}")
            if debug_mode:
                import traceback
                error_details = traceback.format_exc()
                logger.error(f"Detailed error: {error_details}")
                st.expander("Debug: File Handling Error").code(error_details)
    
    else:
        st.info("Please upload your CV to get started")

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
    except Exception as e:
        logger.critical(f"Critical app error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        if "debug_mode" in locals() and debug_mode:
            import traceback
            error_details = traceback.format_exc()
            logger.critical(f"Detailed error: {error_details}")
            st.expander("Debug: Critical Error Details").code(error_details)