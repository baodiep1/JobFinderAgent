import os
import json
from huggingface_hub import login
from pathlib import Path
from dotenv import load_dotenv
from tools.google_search import search_google_jobs
from tools.pdf_extractor import extract_text_from_pdf
from smolagents import CodeAgent, HfApiModel


env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


def run():

    # Read system prompt from file
    prompt_path = Path(__file__).parent / "system_prompt.txt"
    with open(prompt_path, "r") as f:
        system_prompt = f.read()

    cv_path = "test_files/sample_cv.pdf"
    if not os.path.exists(cv_path):
        cv_path = "test_files/real_resume.pdf"
            
    cv_text = extract_text_from_pdf(cv_path)



    agent = CodeAgent(tools=[search_google_jobs], model=HfApiModel())

    agent.run(system_prompt + "cv_text: " + cv_text)


    """
    try:
        cv_path = "test_files/resume-sample.pdf"
        if not os.path.exists(cv_path):
            cv_path = "test_files/real_resume.pdf"
            
        cv_text = pdf_extractor_tool(cv_path)
        print(f"\nExtracted CV ({len(cv_text)} chars):")
        print("-" * 50)
        print(cv_text)
        print("-" * 50)
        print()

        # Generate smarter query from CV text
        skills = ["python", "nlp", "machine learning", "ai"]
        query = " ".join(
            ["Software Engineer", "Remote"] + 
            [word for word in cv_text.lower().split() 
             if word in skills]
        )
        print(f"\nGenerated search query: {query}")

        # Execute search with environment check
        if not os.getenv("SERPAPI_KEY"):
            raise ValueError("SERPAPI_KEY not found in environment")
            
        results = google_search_tool(query, "United States")

        # Save and show results
        with open("search_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("\nSearch successful! Top result:")
        print(results.get("jobs_results", [{}])[0].get("title", "No jobs found"))

    except Exception as e:
        print(f"\nError: {str(e)}")
        if "SERPAPI_KEY" in str(e):
            print("Please ensure:")
            print("1. Your .env file exists in the project root")
            print("2. It contains: SERPAPI_KEY=your_actual_key")
            print("3. The file isn't listed in .gitignore")

               """
    
if __name__ == "__main__":


    login()
    run()
