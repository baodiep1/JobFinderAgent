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
    
    # Clean up the location string - remove any extraneous characters
    location = location.strip()
    
    # Make sure the query is job-focused
    if "job" not in query.lower():
        query = query + " jobs"
    
    # Set up the search parameters
    params = {
        "engine": "google",
        "q": query,
        "location": location,
        "api_key": api_key,
        "google_domain": "google.com",
        "gl": "us",          # Country to use for the Google search
        "hl": "en",          # Language to use for the Google search
        "num": 10            # Number of results
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        print(f"SerpAPI Search completed. Found {len(results.get('organic_results', []))} results.")
        return results
    except Exception as e:
        print(f"Error in SerpAPI search: {str(e)}")
        return {
            "error": str(e),
            "query": query,
            "location": location,
            "organic_results": []
        }
