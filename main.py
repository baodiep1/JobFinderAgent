import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tools.google_search import google_search_tool
from tools.pdf_extractor import pdf_extractor_tool

# Load environment variables with explicit path
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Debug environment loading
print(f"\nEnvironment Debug:")
print(f"Loading .env from: {env_path.absolute()}")
print(f"SERPAPI_KEY exists: {os.getenv('SERPAPI_KEY') is not None}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}\n")

def run():
    system_message = """You are an AI assistant designed to help users find job opportunities that best match their experience and skills. Your primary goal is to analyze the user's CV and retrieve job listings that are highly relevant.

You have access to the following tools:
Tool Name: google_search_tool, Description: Searches for job postings based on extracted keywords and location on google., Arguments: keywords: list, location: str, Outputs: list of job listings

You should think step by step in order to fulfill the objective with a reasoning process divided into Thought/Action/Observation steps that can be repeated multiple times if needed.

You should first reflect on the current situation using `Thought: {your_thoughts}`, then (if necessary), call a tool with the proper JSON formatting `Action: {JSON_BLOB}`, or print your final answer starting with the prefix `Final Answer:`
"""

    try:
        cv_path = "test_files/resume-sample.pdf"
        if not os.path.exists(cv_path):
            cv_path = "test_files/real_resume.pdf"
            
        cv_text = pdf_extractor_tool(cv_path)
        print(f"\nExtracted CV ({len(cv_text)} chars):")
        print("-" * 50)
        print(cv_text[:300] + "...")
        print("-" * 50)

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

if __name__ == "__main__":
    run()
