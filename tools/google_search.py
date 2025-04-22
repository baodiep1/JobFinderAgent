import os
from serpapi import GoogleSearch
from .tool import Tool

def search_google_jobs(query: str, location: str = "United States") -> dict:
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise EnvironmentError("SERPAPI_KEY not set in environment.")
    
    params = {
        "engine": "google",
        "q": query,
        "location": location,
        "api_key": api_key
    }
    search = GoogleSearch(params)
    return search.get_dict()

google_search_tool = Tool(
    name="Google Job Search Tool",
    description="Searches Google for job postings with a query and location.",
    func=search_google_jobs,
    arguments=[("query", "str"), ("location", "str")],
    outputs="dict"
)
