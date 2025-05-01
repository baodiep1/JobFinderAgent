import subprocess
import sys
import streamlit as st

def install_dependencies():
    """Install additional dependencies needed for the app."""
    try:
        st.info("Setting up NLP components...")
        # Download spacy model with pip - use user mode
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl"
        ])
        st.success("Setup completed successfully!")
    except Exception as e:
        st.warning(f"Setup encountered an issue: {str(e)}")
        st.info("We'll try to continue with limited functionality.")
    
    # Install NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        st.warning("Could not download NLTK data. Some features may not work properly.")

if __name__ == "__main__":
    install_dependencies()
