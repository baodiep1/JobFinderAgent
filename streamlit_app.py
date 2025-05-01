import streamlit as st
import os
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tools.pdf_extractor import extract_text_from_pdf
from tools.google_search import search_google_jobs

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("Downloading SpaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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
    
    # Tokenize the text
    tokens = word_tokenize(cv_text_lower)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Process with SpaCy for more advanced extraction
    doc = nlp(cv_text)
    
    # Extract skills
    skills = set()
    
    # Method 1: Direct matching with our predefined list
    for skill in technical_skills:
        if skill in cv_text_lower or any(skill == token for token in filtered_tokens):
            skills.add(skill)
    
    # Method 2: Using SpaCy's entity recognition for additional skills
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 2:
            # Check if this organization/product could be a technology or tool
            potential_skill = ent.text.lower()
            if potential_skill in technical_skills:
                skills.add(potential_skill)
    
    # Method 3: Extract multi-word technical skills like "machine learning"
    for skill in technical_skills:
        if " " in skill:
            if skill in cv_text_lower:
                skills.add(skill)
    
    # Look for sections that might contain skills
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
    
    return list(skills)

def extract_education(cv_text):
    """Extract education information from CV text"""
    education_keywords = ["education", "university", "college", "school", "degree", "bachelor", 
                         "master", "phd", "mba", "bsc", "msc", "b.a.", "m.a.", "major"]
    
    doc = nlp(cv_text)
    education_info = []
    
    # Look for education section
    sections = re.split(r'\n\s*\n', cv_text)
    for section in sections:
        section_lower = section.lower()
        if any(keyword in section_lower for keyword in education_keywords):
            lines = section.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in education_keywords):
                    education_info.append(line.strip())
    
    # If no structured section found, use NER to find educational organizations
    if not education_info:
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(keyword in ent.text.lower() for keyword in ["university", "college", "school"]):
                education_info.append(ent.text)
    
    return education_info

def extract_experience(cv_text):
    """Extract work experience information from CV text"""
    experience_keywords = ["experience", "work", "employment", "job", "career", "position", "role"]
    
    experience_info = []
    
    # Look for experience section
    sections = re.split(r'\n\s*\n', cv_text)
    for section in sections:
        section_lower = section.lower()
        if any(keyword in section_lower for keyword in experience_keywords):
            lines = section.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ["position", "title", "role"]) or re.search(r'\b\d{4}\b', line):
                    experience_info.append(line.strip())
    
    return experience_info

def generate_search_query(skills, education, experience):
    """Generate a relevant search query based on CV analysis"""
    # Extract potential job titles from experience
    job_titles = []
    common_titles = ["developer", "engineer", "programmer", "analyst", "scientist", 
                    "designer", "manager", "consultant", "specialist", "technician"]
    
    for exp in experience:
        for title in common_titles:
            if title in exp.lower():
                # Extract phrases that might be job titles
                matches = re.findall(r'([A-Za-z]+\s+' + title + r'|' + title + r'\s+[A-Za-z]+)', exp.lower())
                if matches:
                    job_titles.extend(matches)
                else:
                    job_titles.append(title)
    
    # Default job titles if none found
    if not job_titles:
        # Extract from skills to determine field
        if any(lang in skills for lang in ["python", "java", "javascript", "c++", "c#"]):
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
    
    query = f"{primary_job} jobs {' '.join(top_skills)}"
    
    return query

def run_streamlit_app():
    st.set_page_config(page_title="JobFinderAI", layout="wide")
    st.title("JobFinderAI - CV-based Job Search")
    
    # Sidebar for API key
    st.sidebar.title("Settings")
    
    api_key = st.sidebar.text_input("SerpAPI Key", type="password")
    if api_key:
        os.environ["SERPAPI_KEY"] = api_key
    
    # File upload
    uploaded_file = st.file_uploader("Upload your resume/CV (PDF format)", type="pdf")
    
    if uploaded_file:
        # Save temp file
        with open("temp_cv.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text
        try:
            cv_text = extract_text_from_pdf("temp_cv.pdf")
            
            # Show extracted text
            with st.expander("Extracted CV Text"):
                st.text(cv_text)
            
            # Process the CV
            with st.spinner("Analyzing your resume..."):
                skills = extract_skills_from_cv(cv_text)
                education = extract_education(cv_text)
                experience = extract_experience(cv_text)
                
                # Display extracted information
                st.subheader("Extracted Information")
                
                col1, col2 = st.columns(2)
                
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
                
                # Generate search query
                default_query = generate_search_query(skills, education, experience)
            
            # Search query input
            st.subheader("Search Parameters")
            search_query = st.text_input("Search Query", value=default_query)
            location = st.text_input("Location", value="United States")
            
            if st.button("Find Jobs"):
                if not api_key:
                    st.error("Please enter your SerpAPI key in the sidebar")
                else:
                    with st.spinner("Searching for jobs..."):
                        try:
                            # Modify query to focus on job listings
                            job_focused_query = search_query + " job posting site:linkedin.com OR site:indeed.com OR site:glassdoor.com"
                            
                            results = search_google_jobs(query=job_focused_query, location=location)
                            
                            # Display results
                            st.subheader("Job Listings")
                            
                            if "organic_results" in results and results["organic_results"]:
                                jobs = results["organic_results"][:5]  # Top 5 results
                                
                                job_count = 0
                                for i, job in enumerate(jobs, 1):
                                    # Filter out non-job listing results
                                    title = job.get('title', '').lower()
                                    link = job.get('link', '').lower()
                                    
                                    # Check if result is likely a job posting
                                    is_job = (
                                        'job' in title or 
                                        'career' in title or 
                                        'position' in title or
                                        'linkedin.com/jobs' in link or
                                        'indeed.com/job' in link or
                                        'glassdoor.com/job' in link
                                    )
                                    
                                    if is_job:
                                        job_count += 1
                                        with st.container():
                                            col1, col2 = st.columns([1, 3])
                                            with col1:
                                                st.write(f"**#{job_count}**")
                                            
                                            with col2:
                                                st.write(f"**{job.get('title', 'Unknown Title')}**")
                                                st.write(f"Company: {job.get('source', 'Unknown')}")
                                                st.write(f"Description: {job.get('snippet', 'No description')[:200]}...")
                                                if "link" in job:
                                                    st.write(f"[Apply Here]({job['link']})")
                                            
                                            st.divider()
                                
                                if job_count == 0:
                                    st.info("No specific job listings found. Try modifying your search query.")
                            else:
                                st.info("No job listings found. Try a different search query.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Clean up
            if os.path.exists("temp_cv.pdf"):
                os.remove("temp_cv.pdf")
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    
    else:
        st.info("Please upload your CV to get started")

if __name__ == "__main__":
    run_streamlit_app()
