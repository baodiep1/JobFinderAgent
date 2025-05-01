import os
from serpapi import GoogleSearch

from smolagents import tool


@tool
def search_google_jobs(query: str, location: str = "United States") -> dict:
    """Searches for job postings based on the provided query and location.

    Args:
        query: The search query to find job postings.
        location: The location to search for jobs in. Defaults to "United States".

    Returns:
        A dictionary containing the search results from Google.
    """
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


"""
google_search_tool = Tool(
    name="Google Job Search Tool",
    description="Searches Google for job postings with a query and location.",
    func=search_google_jobs,
    arguments=[("query", "str"), ("location", "str")],
    outputs="dict"
)
"""