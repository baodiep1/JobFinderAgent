import os
import json
from dotenv import load_dotenv
from tools.google_search import google_search_tool

load_dotenv()

def run():

    query = "Software Engineer NLP Remote"
    results = google_search_tool(query, "United States")

    with open("search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(google_search_tool.to_string())

if __name__ == "__main__":
    run()
