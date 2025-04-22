

from serpapi import GoogleSearch
import os
import json 

params = {
  "engine": "google",
  "q": "Python Software Engineer NLP remote",
  "location": "United States",
  "api_key": "561af63ae039653cca273b9efa016b2bb44f5536499eb1b770eb1371df27f3f6"
}

search = GoogleSearch(params)
results = search.get_dict()
results = search.get_dict()

# Save to a JSON file next to the script
file_path = os.path.join(os.path.dirname(__file__), "search_results.json")

with open(file_path, "w") as f:
    json.dump(results, f, indent=2)
